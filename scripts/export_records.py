from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from learner_simulator.data import (  # noqa: E402
    AGENT4EDU_HISTORY_STEPS,
    AGENT4EDU_TARGET_STEPS,
    load_questions,
    split_rows_agent4edu,
    take_sequence_rows,
)
from learner_simulator.llm import load_json  # noqa: E402
from learner_simulator.simulators import LLMLearnerSimulator, RandomLearnerSimulator  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full simulated learner records.")
    parser.add_argument("--dataset-root", default="E:/yyx/KT数据集/XES3G5M/XES3G5M")
    parser.add_argument("--simulator", choices=["random", "llm"], default="random")
    parser.add_argument("--source-rows", type=int, default=1000)
    parser.add_argument("--max-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--call-api", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    questions = load_questions(dataset_root / "metadata" / "questions.json")
    train_rows, test_rows = load_rows(args, dataset_root)

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

    llm_config = None
    if args.simulator == "random":
        simulator = RandomLearnerSimulator(**simulator_kwargs)
    else:
        simulator = LLMLearnerSimulator(**simulator_kwargs)
        llm_config = load_json(ROOT / args.llm_config) if args.call_api else None

    simulator.fit(train_rows, questions=questions)
    history_by_uid = {row["uid"]: row for row in train_rows}

    output = args.output or f"outputs/{args.simulator}_simulated_records.jsonl"
    output_path = ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record_count = 0
    with output_path.open("w", encoding="utf-8") as f:
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
                    history_row=history_by_uid.get(row["uid"]),
                )
            for record in simulation["steps"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_count += 1

    print(
        json.dumps(
            {
                "ok": True,
                "simulator": args.simulator,
                "protocol": "agent4edu_fixed_history_90_10",
                "called_api": args.call_api,
                "output": str(output_path),
                "records": record_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_rows(args: argparse.Namespace, dataset_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = take_sequence_rows(
        dataset_root / "kc_level" / "train_valid_sequences.csv",
        args.source_rows,
    )
    return split_rows_agent4edu(
        rows,
        history_steps=AGENT4EDU_HISTORY_STEPS,
        target_steps=AGENT4EDU_TARGET_STEPS,
        max_users=args.max_users,
    )


if __name__ == "__main__":
    main()
