from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.common import (
    current_implementation_fingerprint,
    metric_view,
    runtime_summary,
    valid_metric_steps,
    validity_summary,
)
from learner_simulator.action import parse_agent_response
from learner_simulator.educational_multi_agent_prompt import build_four_tier_response_record
from learner_simulator.evaluation import evaluate_steps
from learner_simulator.evaluation_views import layered_metric_view
from learner_simulator.four_tier import (
    assess_four_tier_response,
    parse_reduced_response,
)
from learner_simulator.llm import call_openai_compatible_chat
from learner_simulator.process_evidence import build_process_consistency
from learner_simulator.process_verification import verify_process_steps
from learner_simulator.simulators.multi_role_simulator import (
    enforce_learner_correct_answer_consistency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry only failed LLM steps while preserving complete traces."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Bounded parallel LLM repair requests (default: 8).",
    )
    parser.add_argument(
        "--commit-batch-size",
        type=int,
        default=8,
        help="Atomically persist the complete trace after this many repairs (default: 8).",
    )
    parser.add_argument(
        "--allow-legacy-implementation",
        action="store_true",
        help="Explicitly permit a legacy report without an implementation fingerprint.",
    )
    args = parser.parse_args()
    if args.max_workers < 1 or args.commit_batch_size < 1:
        parser.error("max-workers and commit-batch-size must be positive")
    return args


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _atomic_write(path: str | Path, payload: dict[str, Any]) -> None:
    """Commit a complete checkpoint, never a partial multi-hundred-MB trace."""

    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def _assert_implementation_compatible(payload: dict[str, Any], allow_legacy: bool) -> str:
    current = current_implementation_fingerprint()["digest"]
    reports = payload.get("reports") or {}
    for variant, report in reports.items():
        archived = (report.get("archive") or {}).get("implementation_fingerprint") or {}
        archived_digest = archived.get("digest")
        if archived_digest == current:
            continue
        if archived_digest is None and allow_legacy:
            continue
        reason = "missing legacy fingerprint" if archived_digest is None else "different implementation fingerprint"
        raise RuntimeError(
            f"Refusing targeted repair for {variant}: {reason}. "
            "Repair with the exact generating implementation, or create a new v6 run."
        )
    return current


def _reference_concepts(step: dict[str, Any], alignment: dict[str, Any]) -> list[str]:
    values = alignment.get("current_concepts") or step.get("kc_routes") or [step.get("cid")]
    cleaned = [str(value) for value in values if value is not None and str(value).strip()]
    # Targeted repairs must follow the same primary-route contract as fresh
    # simulation, including for an archived step missing its alignment field.
    return cleaned[:1]


def _materialize_repaired_step(
    step: dict[str, Any],
    raw: str,
    attempt: int,
    repair_source: str,
) -> dict[str, Any]:
    original_error = str(step["llm_error"])
    started = time.time()
    response_format = (step.get("enabled_modules") or {}).get("response_contract")
    if response_format == "four_tier":
        action = parse_agent_response(raw)
    elif response_format == "reduced_response":
        action = parse_reduced_response(raw)
    else:
        raise RuntimeError(f"Unsupported v6 repair response contract: {response_format!r}")
    if action is None or action.get("learner_correct") not in {0, 1}:
        raise RuntimeError(
            f"repair parse failed uid={step.get('uid')} qid={step.get('qid')}"
        )

    action["raw"] = raw
    rendered, final, decision_source = enforce_learner_correct_answer_consistency(action, step)
    step.update(
        llm_result=raw,
        simulated_response=final,
        prediction_valid=True,
        prediction_source="llm_learner_correct",
        llm_parsed_action=action,
        agent_action=action,
        response_learner_correct=final,
        response_declared_learner_correct=action.get("declared_learner_correct"),
        learner_correct_answer_consistent=action["learner_correct_consistent_with_answer"],
        rendered_answer_correct=rendered,
        response_decision_source=decision_source,
    )
    if response_format == "four_tier":
        # The archived response has already been materialized. Score precisely
        # that response, matching the fresh v6 execution path.
        four_tier = assess_four_tier_response(
            action,
            reference_answers=step.get("answer"),
            reference_reasoning=str(step.get("analysis", "")),
        )
        alignment = step.get("cognitive_state_item_alignment") or {}
        action["state_alignment"] = alignment
        consistency = build_process_consistency(
            action,
            alignment,
            _reference_concepts(step, alignment),
        )
        step["process_consistency"] = consistency
        step["four_tier_assessment"] = four_tier
        step["four_tier_response_module"] = build_four_tier_response_record(
            action,
            four_tier,
            irt_evidence=step.get("irt_ability_difficulty_evidence"),
            state_item_alignment=alignment,
            process_consistency=consistency,
        )

    tasks = step.get("simulation_tasks") or {}
    task2 = tasks.get("module2_cognitive_state_item_alignment") or {}
    task3 = tasks.get("module3_structured_response_generation") or {}
    if task2:
        refs = task2.get("reference_concepts") or [task2.get("true_concept")]
        task2["concept_match"] = str(action.get("identified_concept") or "") in {
            str(value) for value in refs if value is not None
        }
    task3.update(
        learner_correct=final,
        student_answer=action.get("student_answer"),
        student_reasoning=action.get("student_reasoning"),
        answer_confidence=action.get("answer_confidence"),
        reasoning_confidence=action.get("reasoning_confidence"),
        answer_correct=rendered,
    )
    # Preserve llm_raw_on_error for audit; the successful raw response is kept
    # in llm_result and both are linked by repair_audit.
    step["repair_audit"] = {
        "attempt": attempt,
        "original_error": original_error,
        "original_raw_response_preserved": "llm_raw_on_error" in step,
        "repair_source": repair_source,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    step.pop("llm_error", None)
    return {
        "uid": step.get("uid"),
        "qid": step.get("qid"),
        "step_index": step.get("step_index"),
        "original_error": original_error,
    }


def _recover_archived_step(step: dict[str, Any], attempt: int) -> dict[str, Any]:
    """Use a valid raw response already saved at the failed step; no API call."""

    raw = step.get("llm_raw_on_error") or step.get("llm_result")
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("no archived LLM response available for local recovery")
    return _materialize_repaired_step(step, raw, attempt, "archived_local_parse")


def _repair_step(
    step: dict[str, Any],
    config: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    raw = call_openai_compatible_chat(
        config,
        step["response_agent_prompt"],
        system_prompt=step["response_agent_system_prompt"],
    )
    return _materialize_repaired_step(step, raw, attempt, "api_retry")


def _refresh_report(report: dict[str, Any]) -> None:
    steps = report.get("all_steps") or []
    by_key = {(s.get("uid"), s.get("step_index"), s.get("qid")): s for s in steps}
    report["sample_steps"] = [
        copy.deepcopy(by_key.get((s.get("uid"), s.get("step_index"), s.get("qid")), s))
        for s in report.get("sample_steps", [])
    ]
    name = report.get("experiment", "multi-role")
    metric_steps, excluded = valid_metric_steps(name, steps)
    threshold = float((report.get("metrics") or {}).get("threshold", 0.5))
    report["metrics"] = evaluate_steps(metric_steps, threshold=threshold)
    report["validity"] = validity_summary(name, steps)
    report["metric_population"] = {
        "included_steps": len(metric_steps),
        "excluded_steps": len(steps) - len(metric_steps),
        "excluded_uids": excluded,
        "policy": "targeted_failed_step_repair",
    }
    report["runtime"] = runtime_summary(
        steps, float((report.get("runtime") or {}).get("total_seconds", 0))
    )
    report["process_verification"] = verify_process_steps(metric_steps)
    report["metric_layers"] = layered_metric_view(
        {
            "name": name,
            "source": "targeted_step_repaired_run",
            "report": report,
            "metrics": report["metrics"],
            "validity": report["validity"],
            "runtime": report["runtime"],
        }
    )


def main() -> None:
    args = parse_args()
    source = Path(args.output) if Path(args.output).exists() else Path(args.base)
    payload = _load(source)
    config = _load(args.config)
    fingerprint = _assert_implementation_compatible(payload, args.allow_legacy_implementation)
    repairs: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    pending = [
        (variant, step)
        for variant, report in (payload.get("reports") or {}).items()
        for step in report.get("all_steps") or []
        if step.get("llm_error")
    ]
    api_pending: list[tuple[str, dict[str, Any]]] = []
    for variant, step in pending:
        try:
            record = _recover_archived_step(step, args.attempt)
        except Exception:
            # A genuinely absent or malformed archived response needs an API
            # retry.  Contract-only parser misses should never spend a token.
            api_pending.append((variant, step))
            continue
        record["variant"] = variant
        repairs.append(record)
        payload.setdefault("targeted_step_repairs_history", []).append(
            {"attempt": args.attempt, **record}
        )
        print(json.dumps({"status": "recovered_locally", **record}, ensure_ascii=False), flush=True)
    if repairs:
        _atomic_write(args.output, payload)

    pending = api_pending
    for start in range(0, len(pending), args.commit_batch_size):
        batch = pending[start : start + args.commit_batch_size]
        # Every archived teacher-forcing prompt is self-contained, so failed
        # steps can be safely retried in parallel.  The LLM client retains its
        # provider-wide request gate; this only removes needless serial work.
        with ThreadPoolExecutor(max_workers=min(args.max_workers, len(batch))) as pool:
            futures = {
                pool.submit(_repair_step, step, config, args.attempt): (variant, step)
                for variant, step in batch
            }
            for future in as_completed(futures):
                variant, step = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "variant": variant,
                        "uid": step.get("uid"),
                        "qid": step.get("qid"),
                        "step_index": step.get("step_index"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failed.append(record)
                    print(json.dumps({"status": "failed", **record}, ensure_ascii=False), flush=True)
                    continue
                record["variant"] = variant
                repairs.append(record)
                payload.setdefault("targeted_step_repairs_history", []).append(
                    {"attempt": args.attempt, **record}
                )
                print(json.dumps({"status": "repaired", **record}, ensure_ascii=False), flush=True)
        # Do not serialize ~315MB after every single success.  A committed
        # batch is atomic; interruption loses at most this small batch.
        _atomic_write(args.output, payload)

    for variant, report in (payload.get("reports") or {}).items():
        _refresh_report(report)
        payload.setdefault("summary", {})[variant] = metric_view(report)
    # Recompute the fair comparison table after repairs, on the intersection of
    # all currently valid learner sequences.
    from experiments.ablation.run_ablation import _formal_metrics_on_common_population
    payload["formal_metrics_v6"] = _formal_metrics_on_common_population(
        payload.get("reports") or {}, list((payload.get("reports") or {}).keys())
    )
    payload["targeted_step_repairs"] = {
        "attempt": args.attempt,
        "count": len(repairs),
        "records": repairs,
        "failed_count": len(failed),
        "failed": failed,
        "policy": "retry_only_failed_variant_uid_qid_step_parallel_checkpointed",
        "max_workers": args.max_workers,
        "commit_batch_size": args.commit_batch_size,
        "implementation_fingerprint": fingerprint,
    }
    _atomic_write(args.output, payload)
    print(json.dumps({"ok": not failed, "repaired": len(repairs), "failed": len(failed), "output": args.output}))


if __name__ == "__main__":
    main()
