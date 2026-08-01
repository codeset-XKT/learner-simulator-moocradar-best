from __future__ import annotations

import argparse
import json
import sys
import time
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
            "When provided, it initializes concept mastery before target simulation."
        ),
    )
    parser.add_argument(
        "--dneuralcdm-checkpoint",
        default=None,
        help=(
            "Optional DNeuralCDM best_model.pt checkpoint. Multi-role Full uses it "
            "to compute current-item response probability before each simulated step."
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
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--save-steps", action="store_true")
    parser.add_argument("--progress", action="store_true")
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
    dataset_root = Path(args.dataset_root)
    if args.cohort_file:
        cohort_path = resolve_project_path(args.cohort_file)
        cohort = load_json(cohort_path)
        dataset_root = Path(cohort.get("dataset_root") or dataset_root)
        questions = load_questions(dataset_root / "metadata" / "questions.json")
        history_rows = list(cohort["history_rows"])
        target_rows = list(cohort["target_rows"])
        source_rows_scanned = int(cohort.get("source_rows_scanned", args.source_rows))
        args.profile_normalization_rows = take_sequence_rows(
            dataset_root / "kc_level" / "train_valid_sequences.csv",
            source_rows_scanned,
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
    args.profile_normalization_rows = source
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
    modules = modules or {}
    kwargs = simulator_kwargs(args)
    llm_config = None
    if name == "random":
        simulator = RandomLearnerSimulator(**kwargs)
    elif name == "full":
        simulator = LLMLearnerSimulator(**kwargs)
        llm_config = load_json(ROOT / args.llm_config)
    elif name == "multi-role":
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
    total_steps = sum(len(clean_sequence(row)) for row in target_rows)
    progress = make_progress(progress_name or name, total_steps, enabled=args.progress)
    simulations: list[dict[str, Any]] = []
    started = time.perf_counter()

    for row in target_rows:
        history_row = history_by_uid[row["uid"]]
        if name == "random":
            simulation = simulator.simulate_sequence(
                row,
                questions=questions,
                history_row=history_row,
            )
        elif name == "agent4edu":
            simulation = simulator.simulate_sequence(
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
                extra_module_kwargs["include_item_conditioned_ability"] = modules.get(
                    "item_conditioned_ability",
                    True,
                )
            simulation = simulator.simulate_sequence(
                row,
                questions=questions,
                llm_config=llm_config,
                call_llm=True,
                include_prompt=args.include_prompt,
                progress_callback=progress,
                history_row=history_row,
                response_format=(
                    "four_tier" if modules.get("four_tier", True) else "answer_only"
                ),
                include_profile=modules.get("profile", True),
                include_memory=modules.get("memory", True),
                include_proficiency=modules.get("proficiency", True),
                include_behavior=modules.get("behavior", True),
                include_cognitive_strategy=modules.get("cognitive_strategy", True),
                include_cognitive_profile=modules.get("cognitive_profile", True),
                include_ability_profile=modules.get("ability_profile", True),
                include_irt_evidence=modules.get("irt_evidence", True),
                include_learning_tool_state=modules.get("learning_tool_state", True),
                include_historical_reflection=modules.get(
                    "historical_reflection",
                    True,
                ),
                include_dkt_predictor=modules.get("dkt_predictor", False),
                feedback_mode=args.feedback_mode,
                **extra_module_kwargs,
            )
        simulations.append(simulation)

    elapsed = time.perf_counter() - started
    steps = flatten_simulations(simulations)
    report = {
        "experiment": name,
        "protocol": {
            "name": "agent4edu_fixed_history",
            "grouping_unit": "unique_uid",
            "history_steps": args.history_steps,
            "target_steps": args.target_steps,
            "feedback_mode": args.feedback_mode,
        },
        "modules": modules,
        "unique_simulated_users": len(target_rows),
        "history_summary": summarize_sequences(history_rows),
        "target_summary": summarize_sequences(target_rows),
        "simulator_summary": simulator.summary(),
        "metrics": evaluate_steps(steps, threshold=args.threshold),
        "runtime": runtime_summary(steps, elapsed),
        "validity": validity_summary(name, steps),
        "sample_steps": steps[:20],
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
            "proficiency_adapter": "project_dynamic_mastery",
            "rollout_feedback": "simulated_response",
        }
    if args.save_steps:
        report["all_steps"] = steps
    return report


def save_report(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


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
        "task2_kt_anchor_acc": metrics.get("task2_kt_anchor_acc"),
        "task2_kt_anchor_balanced_accuracy": metrics.get(
            "task2_kt_anchor_balanced_accuracy"
        ),
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


def make_progress(name: str, total: int, enabled: bool):
    state = {"done": 0, "started": time.perf_counter()}

    def callback(step: dict[str, Any]) -> None:
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
