from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
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
from learner_simulator.llm import load_json  # noqa: E402
from learner_simulator.simulators import LLMLearnerSimulator, RandomLearnerSimulator  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learner simulators on XES3G5M.")
    parser.add_argument("--dataset-root", default="E:/yyx/KT数据集/XES3G5M/XES3G5M")
    parser.add_argument("--simulator", choices=["random", "llm"], default="random")
    parser.add_argument(
        "--ablation",
        choices=["none", "no-four-tier"],
        default="none",
    )
    parser.add_argument(
        "--source-rows",
        type=int,
        default=1000,
        help="Maximum source sequence rows scanned before learner aggregation.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=50,
        help="Maximum number of unique learners to simulate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--user-weight", type=float, default=0.30)
    parser.add_argument("--item-weight", type=float, default=0.25)
    parser.add_argument("--concept-weight", type=float, default=0.25)
    parser.add_argument("--mastery-weight", type=float, default=0.20)
    parser.add_argument("--irt-weight", type=float, default=0.20)
    parser.add_argument("--irt-epochs", type=int, default=8)
    parser.add_argument("--irt-learning-rate", type=float, default=0.04)
    parser.add_argument("--irt-l2", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--long-threshold", type=int, default=3)
    parser.add_argument("--disable-behavior-control", action="store_true")
    parser.add_argument("--llm-config", default="configs/llm.example.json")
    parser.add_argument("--call-api", action="store_true")
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Print real-time LLM call timing.")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--save-steps", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    questions = load_questions(dataset_root / "metadata" / "questions.json")
    train_rows, test_rows = load_eval_rows(args, dataset_root)

    simulator_kwargs = {
        "seed": args.seed,
        "user_weight": args.user_weight,
        "item_weight": args.item_weight,
        "concept_weight": args.concept_weight,
        "mastery_weight": args.mastery_weight,
        "irt_weight": args.irt_weight,
        "irt_epochs": args.irt_epochs,
        "irt_learning_rate": args.irt_learning_rate,
        "irt_l2": args.irt_l2,
        "learning_rate": args.learning_rate,
        "short_window": args.short_window,
        "long_threshold": args.long_threshold,
        "behavior_control": not args.disable_behavior_control,
    }

    if args.simulator == "random":
        simulator = RandomLearnerSimulator(**simulator_kwargs)
        llm_config = None
    else:
        simulator = LLMLearnerSimulator(**simulator_kwargs)
        llm_config = load_json(ROOT / args.llm_config) if args.call_api else None

    simulator.fit(train_rows, questions=questions)
    history_by_uid = {row["uid"]: row for row in train_rows}

    simulations = []
    start_time = time.perf_counter()
    total_expected_steps = sum(len(clean_sequence(row)) for row in test_rows)
    progress_state = {"done": 0, "success": 0, "errors": 0}
    show_progress = args.call_api and (args.progress or args.simulator == "llm")

    def progress_callback(step: dict[str, object]) -> None:
        progress_state["done"] += 1
        if "llm_parsed_action" in step and "llm_error" not in step:
            progress_state["success"] += 1
            status = "ok"
        else:
            progress_state["errors"] += 1
            status = "error"

        done = progress_state["done"]
        if args.progress_every <= 0 or done % args.progress_every != 0:
            return

        elapsed = time.perf_counter() - start_time
        avg = elapsed / done if done else 0.0
        remaining = max(0, total_expected_steps - done)
        eta = remaining * avg
        call_elapsed = step.get("llm_elapsed_seconds", 0.0)
        print(
            "[LLM {done}/{total}] status={status} uid={uid} step={step_index} qid={qid} "
            "call={call_elapsed:.2f}s elapsed={elapsed} avg={avg:.2f}s/step eta={eta} "
            "ok={ok} err={err}".format(
                done=done,
                total=total_expected_steps,
                status=status,
                uid=step.get("uid"),
                step_index=step.get("step_index"),
                qid=step.get("qid"),
                call_elapsed=float(call_elapsed or 0.0),
                elapsed=format_duration(elapsed),
                avg=avg,
                eta=format_duration(eta),
                ok=progress_state["success"],
                err=progress_state["errors"],
            ),
            flush=True,
        )

    for row in test_rows:
        if args.simulator == "random":
            simulation = simulator.simulate_sequence(
                row,
                questions=questions,
                history_row=history_by_uid.get(row["uid"]),
            )
        else:
            simulation = simulator.simulate_sequence(
                row,
                questions=questions,
                llm_config=llm_config,
                call_llm=args.call_api,
                include_prompt=args.include_prompt,
                progress_callback=progress_callback if show_progress else None,
                history_row=history_by_uid.get(row["uid"]),
                response_format=(
                    "answer_only"
                    if args.ablation == "no-four-tier"
                    else "four_tier"
                ),
            )
        simulations.append(simulation)

    total_elapsed = time.perf_counter() - start_time
    steps = flatten_simulations(simulations)
    metrics = evaluate_steps(steps, threshold=args.threshold)
    runtime = build_runtime_summary(steps, total_elapsed)

    report = {
        "dataset_root": str(dataset_root),
        "simulator": args.simulator,
        "called_api": args.call_api,
        "ablation": args.ablation,
        "protocol": {
            "name": "agent4edu_fixed_history",
            "grouping_unit": "unique_uid",
            "history_steps": AGENT4EDU_HISTORY_STEPS,
            "target_steps": AGENT4EDU_TARGET_STEPS,
            "history_initializes_profile": True,
            "history_initializes_memory": True,
            "history_initializes_mastery": True,
        },
        "unique_simulated_users": len({row["uid"] for row in test_rows}),
        "train_sample_summary": summarize_sequences(train_rows),
        "test_sample_summary": summarize_sequences(test_rows),
        "simulator_summary": simulator.summary(),
        "metrics": metrics,
        "runtime": runtime,
        "sample_steps": steps[:20],
    }
    if args.save_steps:
        report["all_steps"] = steps

    output = args.output or f"outputs/eval_{args.simulator}.json"
    output_path = ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "simulator": args.simulator,
                "called_api": args.call_api,
                "ablation": args.ablation,
                "protocol": "agent4edu_fixed_history_90_10",
                "count": metrics["count"],
                "prob_acc_at_threshold": metrics["prob_acc_at_threshold"],
                "prob_f1_at_threshold": metrics["prob_f1_at_threshold"],
                "sample_match_acc": metrics["sample_match_acc"],
                "sample_f1": metrics["sample_f1"],
                "llm_response_acc": metrics.get("llm_response_acc"),
                "llm_response_f1": metrics.get("llm_response_f1"),
                "runtime": runtime,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_eval_rows(args: argparse.Namespace, dataset_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_rows = take_sequence_rows(
        dataset_root / "kc_level" / "train_valid_sequences.csv",
        args.source_rows,
    )
    return split_rows_agent4edu(
        source_rows,
        history_steps=AGENT4EDU_HISTORY_STEPS,
        target_steps=AGENT4EDU_TARGET_STEPS,
        max_users=args.max_users,
    )


def build_runtime_summary(steps: list[dict[str, object]], total_elapsed: float) -> dict[str, object]:
    llm_times = [
        float(step["llm_elapsed_seconds"])
        for step in steps
        if step.get("llm_elapsed_seconds") is not None
    ]
    return {
        "total_seconds": round(total_elapsed, 3),
        "total_human": format_duration(total_elapsed),
        "steps": len(steps),
        "avg_seconds_per_step": round(total_elapsed / len(steps), 3) if steps else 0.0,
        "llm_called_steps": len(llm_times),
        "llm_success_count": sum(
            1
            for step in steps
            if step.get("llm_parsed_action") is not None
            and step.get("llm_error") is None
        ),
        "llm_error_count": sum(1 for step in steps if step.get("llm_error") is not None),
        "llm_avg_seconds": round(sum(llm_times) / len(llm_times), 3) if llm_times else None,
        "llm_min_seconds": round(min(llm_times), 3) if llm_times else None,
        "llm_max_seconds": round(max(llm_times), 3) if llm_times else None,
    }


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


if __name__ == "__main__":
    main()
