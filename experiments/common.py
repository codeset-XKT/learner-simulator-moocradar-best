from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from learner_simulator.data import (  # noqa: E402
    AGENT4EDU_HISTORY_STEPS,
    AGENT4EDU_TARGET_STEPS,
    clean_sequence,
    load_questions,
    split_rows_agent4edu,
    summarize_sequences,
    take_sequence_rows,
)
from learner_simulator.evaluation import evaluate_steps, flatten_simulations  # noqa: E402
from learner_simulator.evaluation_views import layered_metric_view  # noqa: E402
from learner_simulator.llm import load_json  # noqa: E402
from learner_simulator.process_verification import verify_process_steps  # noqa: E402
from learner_simulator.simulators import (  # noqa: E402
    Agent4EduBaselineSimulator,
    LLMLearnerSimulator,
    MultiRoleLearnerSimulator,
    RandomLearnerSimulator,
)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        default="E:/yyx/KT数据集/XES3G5M/XES3G5M",
    )
    parser.add_argument("--source-rows", type=int, default=1000)
    parser.add_argument("--max-users", type=int, default=3)
    parser.add_argument("--history-steps", type=int, default=AGENT4EDU_HISTORY_STEPS)
    parser.add_argument("--target-steps", type=int, default=AGENT4EDU_TARGET_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--llm-config", default="configs/llm.example.json")
    parser.add_argument(
        "--dneuralcdm-proficiency",
        default=None,
        help=(
            "Optional path to DNeuralCDM stu_know_proficiency.json or its directory. "
            "Used only by legacy simulators and baselines; multi-role Full ignores it."
        ),
    )
    parser.add_argument(
        "--dneuralcdm-checkpoint",
        default=None,
        help=(
            "DNeuralCDM best_model.pt checkpoint. Multi-role Full uses this single "
            "artifact for history-conditioned concept state; its item-response "
            "probability is retained only for independent baseline evaluation."
        ),
    )
    parser.add_argument(
        "--mikt-proficiency",
        default=None,
        help=(
            "Optional path to MIKT mikt_know_proficiency.json or its directory. "
            "When provided, it initializes concept mastery before target simulation "
            "and takes precedence over DKT and DNeuralCDM."
        ),
    )
    parser.add_argument(
        "--dkt-proficiency",
        default=None,
        help=(
            "Optional path to DKT dkt_know_proficiency.json or its directory. "
            "When provided, it initializes concept mastery before target simulation "
            "and takes precedence over DNeuralCDM unless MIKT is also provided."
        ),
    )
    parser.add_argument(
        "--include-prompt",
        dest="include_prompt",
        action="store_true",
        default=True,
        help="Save the complete user and system prompts for every simulated step (default).",
    )
    parser.add_argument(
        "--no-include-prompt",
        dest="include_prompt",
        action="store_false",
        help="Do not save prompts. Intended only for short local debugging runs.",
    )
    parser.add_argument(
        "--save-steps",
        dest="save_steps",
        action="store_true",
        default=True,
        help="Save every per-step simulation trace in all_steps (default).",
    )
    parser.add_argument(
        "--no-save-steps",
        dest="save_steps",
        action="store_false",
        help="Save only sample steps. Intended only for short local debugging runs.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--parallel-learners",
        type=int,
        default=1,
        help=(
            "Number of learner sequences to simulate concurrently inside each "
            "variant. Supported by multi-role and Agent4Edu; steps within one learner "
            "remain sequential. Agent4Edu makes an action and a reflection call per step."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--cohort-file",
        default=None,
        help="Load an exact saved fixed-history cohort instead of selecting learners again.",
    )
    parser.add_argument(
        "--cohort-user-limit",
        type=int,
        default=None,
        help="Use only the first N learners from a saved cohort.",
    )
    parser.add_argument(
        "--simulate-uids",
        default=None,
        help=(
            "Comma-separated learner UIDs to simulate after fitting on the complete "
            "loaded cohort. Intended for fit-consistent sequence repair runs."
        ),
    )
    parser.add_argument(
        "--save-cohort",
        default=None,
        help="Save the selected exact fixed-history cohort for later fair comparisons.",
    )
    parser.add_argument(
        "--feedback-mode",
        choices=["rollout", "teacher-forcing"],
        default="rollout",
        help=(
            "How target-step state and memory are updated. rollout uses simulated "
            "responses; teacher-forcing uses ground-truth target responses after scoring."
        ),
    )


def load_fixed_cohort(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    initialize_experiment_run(args)
    dataset_root = Path(args.dataset_root)
    if args.cohort_file:
        cohort_path = resolve_project_path(args.cohort_file)
        cohort = load_json(cohort_path)
        stored_dataset_root = cohort.get("dataset_root")
        if not dataset_root.exists() and stored_dataset_root:
            dataset_root = Path(stored_dataset_root)
        questions = load_questions(dataset_root / "metadata" / "questions.json")
        history_rows = list(cohort["history_rows"])
        target_rows = list(cohort["target_rows"])
        source_rows_scanned = int(cohort.get("source_rows_scanned", args.source_rows))
        normalization_source = take_sequence_rows(
            dataset_root / "kc_level" / "train_valid_sequences.csv",
            source_rows_scanned,
        )
        args.profile_normalization_rows = exclude_cohort_users(
            normalization_source,
            history_rows,
        )
        if not args.profile_normalization_rows:
            raise ValueError(
                "Profile normalization requires source learners outside the fixed cohort"
            )
        validate_cohort(
            history_rows,
            target_rows,
            history_steps=args.history_steps,
            target_steps=args.target_steps,
        )
        if args.cohort_user_limit is not None:
            if args.cohort_user_limit <= 0:
                raise ValueError("--cohort-user-limit must be positive")
            history_rows = history_rows[: args.cohort_user_limit]
            target_rows = target_rows[: args.cohort_user_limit]
        return questions, history_rows, target_rows

    questions = load_questions(dataset_root / "metadata" / "questions.json")
    source = take_sequence_rows(
        dataset_root / "kc_level" / "train_valid_sequences.csv",
        args.source_rows,
    )
    history_rows, target_rows = split_rows_agent4edu(
        source,
        history_steps=args.history_steps,
        target_steps=args.target_steps,
        max_users=args.max_users,
    )
    if not target_rows:
        raise RuntimeError(
            "No eligible learners found. Increase --source-rows; every learner needs at least "
            f"{args.history_steps + args.target_steps} interactions."
        )
    args.profile_normalization_rows = exclude_cohort_users(source, history_rows)
    if not args.profile_normalization_rows:
        raise ValueError(
            "Profile normalization requires source learners outside the sampled cohort"
        )
    if args.save_cohort:
        save_cohort(
            history_rows,
            target_rows,
            args.save_cohort,
            dataset_root=dataset_root,
            source_rows=args.source_rows,
            history_steps=args.history_steps,
            target_steps=args.target_steps,
        )
    return questions, history_rows, target_rows


def save_cohort(
    history_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    output: str | Path,
    dataset_root: Path,
    source_rows: int,
    history_steps: int,
    target_steps: int,
) -> Path:
    validate_cohort(
        history_rows,
        target_rows,
        history_steps=history_steps,
        target_steps=target_steps,
    )
    payload = {
        "protocol": f"fixed_history_{history_steps}_{target_steps}",
        "dataset_root": str(dataset_root),
        "source_rows_scanned": source_rows,
        "history_steps": history_steps,
        "target_steps": target_steps,
        "uids": [row["uid"] for row in target_rows],
        "history_rows": history_rows,
        "target_rows": target_rows,
    }
    return save_report(payload, output)


def validate_cohort(
    history_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    history_steps: int = AGENT4EDU_HISTORY_STEPS,
    target_steps: int = AGENT4EDU_TARGET_STEPS,
) -> None:
    history_uids = [row["uid"] for row in history_rows]
    target_uids = [row["uid"] for row in target_rows]
    if history_uids != target_uids:
        raise ValueError("Cohort history and target UID order must match exactly")
    if any(len(clean_sequence(row)) != history_steps for row in history_rows):
        raise ValueError(
            f"Every cohort history row must contain exactly {history_steps} interactions"
        )
    if any(len(clean_sequence(row)) != target_steps for row in target_rows):
        raise ValueError(
            f"Every cohort target row must contain exactly {target_steps} interactions"
        )


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def exclude_cohort_users(
    source_rows: list[dict[str, str]],
    cohort_history_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    cohort_uids = {str(row["uid"]) for row in cohort_history_rows}
    return [row for row in source_rows if str(row["uid"]) not in cohort_uids]


def simulator_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "user_weight": 0.30,
        "item_weight": 0.25,
        "concept_weight": 0.25,
        "mastery_weight": 0.20,
        "irt_weight": 0.20,
        "irt_epochs": 8,
        "irt_learning_rate": 0.04,
        "irt_l2": 0.001,
        "learning_rate": 0.12,
        "short_window": 5,
        "long_threshold": 3,
        "behavior_control": True,
        "dneuralcdm_proficiency_path": args.dneuralcdm_proficiency,
        "dneuralcdm_checkpoint_path": args.dneuralcdm_checkpoint,
        "mikt_proficiency_path": args.mikt_proficiency,
        "dkt_proficiency_path": args.dkt_proficiency,
    }


def run_experiment(
    name: str,
    args: argparse.Namespace,
    questions: dict[str, dict[str, Any]],
    history_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    modules: dict[str, bool] | None = None,
    progress_name: str | None = None,
) -> dict[str, Any]:
    initialize_experiment_run(args)
    modules = modules or {}
    kwargs = simulator_kwargs(args)
    llm_config = None
    if name == "random":
        simulator = RandomLearnerSimulator(**kwargs)
    elif name == "full":
        simulator = LLMLearnerSimulator(**kwargs)
        llm_config = load_json(ROOT / args.llm_config)
    elif name == "multi-role":
        # Full computes its NCDM state directly from one checkpoint. Do not
        # silently inject legacy DKT/MIKT or exported-proficiency artifacts.
        kwargs["dneuralcdm_proficiency_path"] = None
        kwargs["mikt_proficiency_path"] = None
        kwargs["dkt_proficiency_path"] = None
        simulator = MultiRoleLearnerSimulator(**kwargs)
        llm_config = load_json(ROOT / args.llm_config)
    elif name == "agent4edu":
        # The official memory reinforcement threshold is five.
        kwargs["long_threshold"] = 5
        simulator = Agent4EduBaselineSimulator(**kwargs)
        llm_config = load_json(ROOT / args.llm_config)
    else:
        raise ValueError(f"Unknown experiment: {name}")

    simulator.fit(
        history_rows,
        questions=questions,
        profile_normalization_rows=getattr(args, "profile_normalization_rows", None),
    )
    history_by_uid = {row["uid"]: row for row in history_rows}
    simulation_target_rows = _select_simulation_targets(
        target_rows,
        getattr(args, "simulate_uids", None),
    )
    simulation_uids = {str(row["uid"]) for row in simulation_target_rows}
    simulation_history_rows = [
        row for row in history_rows if str(row["uid"]) in simulation_uids
    ]
    total_steps = sum(len(clean_sequence(row)) for row in simulation_target_rows)
    progress = make_progress(progress_name or name, total_steps, enabled=args.progress)
    learner_workers = int(getattr(args, "parallel_learners", 1))
    if learner_workers <= 0:
        raise ValueError("--parallel-learners must be positive")
    if learner_workers > 1 and name not in {"multi-role", "agent4edu"}:
        raise ValueError(
            "--parallel-learners > 1 is supported only by multi-role and agent4edu"
        )
    started = time.perf_counter()

    def simulate_row(row: dict[str, str]) -> dict[str, Any]:
        history_row = history_by_uid[row["uid"]]
        if name == "random":
            return simulator.simulate_sequence(
                row,
                questions=questions,
                history_row=history_row,
            )
        elif name == "agent4edu":
            return simulator.simulate_sequence(
                row,
                questions=questions,
                llm_config=llm_config,
                history_row=history_row,
                include_prompt=args.include_prompt,
                progress_callback=progress,
                enable_reflection=modules.get("reflection", True),
            )
        else:
            extra_module_kwargs = {}
            if name == "multi-role":
                extra_module_kwargs["include_ncdm_evidence"] = modules.get(
                    "ncdm_evidence",
                    True,
                )
                extra_module_kwargs["include_dynamic_state_evolution"] = modules.get(
                    "dynamic_state_evolution",
                    True,
                )
                extra_module_kwargs["include_irt_evidence"] = modules.get(
                    "irt_evidence",
                    True,
                )
                extra_module_kwargs["include_evidence_representation"] = modules.get(
                    "evidence_representation",
                    True,
                )
                extra_module_kwargs["include_state_item_alignment"] = modules.get(
                    "state_item_alignment",
                    True,
                )
            return simulator.simulate_sequence(
                row,
                questions=questions,
                llm_config=llm_config,
                call_llm=True,
                include_prompt=args.include_prompt,
                progress_callback=progress,
                history_row=history_row,
                response_format=(
                    "four_tier"
                    if modules.get("structured_response", True)
                    else "reduced_response"
                ),
                include_profile=modules.get("profile", True),
                include_memory=modules.get("memory", True),
                include_cognitive_profile=modules.get("cognitive_profile", True),
                include_ability_profile=modules.get("ability_profile", True),
                feedback_mode=args.feedback_mode,
                **extra_module_kwargs,
            )
    if learner_workers == 1:
        simulations = [simulate_row(row) for row in simulation_target_rows]
    else:
        # executor.map preserves cohort order while each learner's internal
        # target steps remain strictly sequential inside simulate_sequence.
        with ThreadPoolExecutor(max_workers=learner_workers) as executor:
            simulations = list(executor.map(simulate_row, simulation_target_rows))

    elapsed = time.perf_counter() - started
    steps = flatten_simulations(simulations)
    metric_steps, excluded_uids = valid_metric_steps(name, steps)
    report = {
        "experiment": name,
        "archive": build_archive_metadata(args, llm_config),
        "protocol": {
            "name": "agent4edu_fixed_history",
            "grouping_unit": "unique_uid",
            "history_steps": args.history_steps,
            "target_steps": args.target_steps,
            "feedback_mode": (
                "official_observed_outcome" if name == "agent4edu" else args.feedback_mode
            ),
        },
        "modules": modules,
        "unique_simulated_users": len(simulation_target_rows),
        "history_summary": summarize_sequences(simulation_history_rows),
        "target_summary": summarize_sequences(simulation_target_rows),
        "fit_cohort_summary": summarize_sequences(history_rows),
        "simulator_summary": simulator.summary(),
        "metrics": evaluate_steps(metric_steps, threshold=args.threshold),
        "runtime": runtime_summary(steps, elapsed),
        "validity": validity_summary(name, steps),
        "metric_population": {
            "included_steps": len(metric_steps),
            "excluded_steps": len(steps) - len(metric_steps),
            "excluded_uids": excluded_uids,
            "policy": "exclude_entire_learner_sequence_after_any_llm_failure",
        },
        "sample_steps": steps[:20],
        "process_verification": verify_process_steps(metric_steps),
    }
    report["metric_layers"] = layered_metric_view(
        {
            "name": name,
            "source": "current_run",
            "report": report,
            "metrics": report["metrics"],
            "validity": report["validity"],
            "runtime": report["runtime"],
        }
    )
    if name == "agent4edu":
        report["agent4edu_reproduction"] = {
            "official_four_task_action_prompt": True,
            "task4_used_as_response_prediction": True,
            "post_response_reflection": modules.get("reflection", True),
            "reference_answer_exposed": True,
            "reference_analysis_exposed": True,
            "proficiency_adapter": "dneuralcdm_checkpoint_current_concept_state",
            "irt_profile_adapter": "history_fitted_rasch1pl_ability",
            "feedback_mode": "official_observed_outcome",
            "parallel_learners": learner_workers,
        }
    if args.save_steps:
        report["all_steps"] = steps
    return report


def _select_simulation_targets(
    target_rows: list[dict[str, str]],
    requested_uids: str | None,
) -> list[dict[str, str]]:
    if not requested_uids:
        return target_rows
    requested = {
        item.strip() for item in requested_uids.split(",") if item.strip()
    }
    selected = [row for row in target_rows if str(row["uid"]) in requested]
    found = {str(row["uid"]) for row in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Requested simulation UIDs are absent from the cohort: {missing}")
    if not selected:
        raise ValueError("--simulate-uids selected no learners")
    return selected


def save_report(report: dict[str, Any], output: str | Path) -> Path:
    requested_path = Path(output)
    if not requested_path.is_absolute():
        requested_path = ROOT / requested_path
    requested_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    path = requested_path
    while True:
        try:
            with path.open("x", encoding="utf-8") as file:
                file.write(payload)
            return path
        except FileExistsError:
            path = versioned_report_path(requested_path)


def initialize_experiment_run(args: argparse.Namespace) -> None:
    """Attach one stable run identity before any parallel variants start."""

    if getattr(args, "experiment_run_id", None):
        return
    started = datetime.now(timezone.utc)
    args.experiment_run_id = (
        f"{started.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    )
    args.experiment_started_at_utc = started.isoformat()


def build_archive_metadata(
    args: argparse.Namespace,
    llm_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe everything needed to audit a saved run without storing secrets."""

    return {
        "schema_version": "experiment-trace-v2",
        "implementation_fingerprint": current_implementation_fingerprint(),
        "run_id": args.experiment_run_id,
        "started_at_utc": args.experiment_started_at_utc,
        "report_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "arguments": sanitized_arguments(args),
        "llm_config": redact_secrets(llm_config),
        "storage_policy": {
            "all_steps_saved": bool(args.save_steps),
            "prompts_saved": bool(args.include_prompt),
            "existing_files_overwritten": False,
        },
    }


def current_implementation_fingerprint() -> dict[str, Any]:
    """Fingerprint response semantics so repairs cannot mix code revisions."""
    paths: list[Path] = []
    paths.extend(sorted((ROOT / "src").rglob("*.py")))
    paths.extend([
        ROOT / "experiments" / "common.py",
        ROOT / "experiments" / "ablation" / "run_ablation.py",
        ROOT / "scripts" / "repair_failed_steps.py",
    ])
    digest = hashlib.sha256()
    included: list[str] = []
    for path in paths:
        if not path.exists() or "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(ROOT))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        included.append(relative)
    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "files": included,
    }


def sanitized_arguments(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"profile_normalization_rows"}
    values = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }
    return redact_secrets(values)


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            secret_key = (
                "api_key" in normalized
                or "secret" in normalized
                or "password" in normalized
                or "authorization" in normalized
                or normalized in {"token", "access_token", "refresh_token"}
                or normalized.endswith("_token")
            )
            if secret_key:
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = redact_secrets(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def versioned_report_path(requested_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = requested_path.suffix or ".json"
    stem = requested_path.stem if requested_path.suffix else requested_path.name
    return requested_path.with_name(f"{stem}__{timestamp}{suffix}")


def metric_view(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    validity = report.get("validity", {})
    mastery_response = metrics.get("mastery_response_monotonicity") or {}
    mastery_confidence = metrics.get("mastery_confidence_monotonicity") or {}
    return {
        "valid": validity.get("valid", True),
        "validity_reason": validity.get("reason", ""),
        "count": metrics["count"],
        "acc": metrics.get("llm_response_acc", metrics["sample_match_acc"]),
        "f1": metrics.get("llm_response_f1", metrics["sample_f1"]),
        "balanced_acc": metrics.get("response_balanced_accuracy"),
        "specificity": metrics.get("response_specificity"),
        "mcc": metrics.get("response_mcc"),
        "learner_distribution_error": metrics.get("learner_distribution_error"),
        "concept_distribution_error": metrics.get("concept_distribution_error"),
        "mastery_response_monotonicity": mastery_response.get("score"),
        "mastery_confidence_monotonicity": mastery_confidence.get("score"),
        "four_tier_answer_confidence_ece": metrics.get(
            "four_tier_answer_confidence_ece"
        ),
        "prob_acc": metrics.get("prob_acc_at_threshold"),
        "prob_f1": metrics.get("prob_f1_at_threshold"),
        "llm_valid_count": metrics.get("llm_response_count"),
        "task2_concept_accuracy": metrics.get("task2_concept_accuracy"),
        "task3_response_acc": metrics.get("task3_response_acc"),
        "task3_response_balanced_accuracy": metrics.get(
            "task3_response_balanced_accuracy"
        ),
        "task4_mean_abs_mastery_delta": metrics.get(
            "task4_mean_abs_mastery_delta"
        ),
        "runtime": report["runtime"]["total_human"],
    }


def runtime_summary(steps: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    call_times = [
        float(step["llm_elapsed_seconds"])
        for step in steps
        if step.get("llm_elapsed_seconds") is not None
    ]
    return {
        "total_seconds": round(elapsed, 3),
        "total_human": format_duration(elapsed),
        "steps": len(steps),
        "llm_called_steps": len(call_times),
        "llm_error_count": sum("llm_error" in step for step in steps),
        "avg_seconds_per_target": round(elapsed / len(steps), 3) if steps else 0.0,
        "avg_seconds_per_llm_step": (
            round(sum(call_times) / len(call_times), 3) if call_times else None
        ),
    }


def validity_summary(name: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    if name == "random":
        return {"valid": True, "reason": "non_llm_baseline"}
    expected = len(steps)
    errors = sum("llm_error" in step for step in steps)
    parsed = sum(isinstance(step.get("llm_parsed_action"), dict) for step in steps)
    valid = errors == 0 and parsed == expected
    reason = "ok" if valid else f"llm_errors={errors}; parsed={parsed}/{expected}"
    return {
        "valid": valid,
        "reason": reason,
        "expected_llm_steps": expected,
        "parsed_llm_steps": parsed,
        "llm_error_count": errors,
    }


def valid_metric_steps(
    name: str,
    steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Prevent fallback values from entering LLM-method performance metrics."""

    if name == "random":
        return steps, []
    invalid_uids = {
        str(step["uid"])
        for step in steps
        if step.get("llm_error") or not step.get("prediction_valid", True)
    }
    return (
        [step for step in steps if str(step["uid"]) not in invalid_uids],
        sorted(invalid_uids),
    )


def make_progress(name: str, total: int, enabled: bool):
    state = {"done": 0, "started": time.perf_counter()}
    lock = threading.Lock()

    def callback(step: dict[str, Any]) -> None:
        with lock:
            state["done"] += 1
            if not enabled:
                return
            elapsed = time.perf_counter() - state["started"]
            avg = elapsed / state["done"]
            eta = avg * max(0, total - state["done"])
            status = "error" if step.get("llm_error") else "ok"
            print(
                f"[{name} {state['done']}/{total}] status={status} "
                f"uid={step.get('uid')} qid={step.get('qid')} "
                f"call={float(step.get('llm_elapsed_seconds', 0.0)):.2f}s "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                flush=True,
            )

    return callback


def format_duration(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    minutes, sec = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"
