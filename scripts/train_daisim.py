#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.daisim import (  # noqa: E402
    DAISimConfig,
    DAISimDiscriminator,
    DAISimGenerator,
    build_index,
    checkpoint_payload,
    encode_steps,
    history_feature,
    train_epoch,
)
from learner_simulator.data import merge_steps_by_uid, take_sequence_rows  # noqa: E402
from learner_simulator.llm import load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an official-code-derived DAISim baseline.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--exclude-cohort-file",
        required=True,
        nargs="+",
        help="One or more fixed-cohort JSON files. All listed users are excluded.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs-direct", type=int, default=3)
    parser.add_argument("--epochs-adversarial", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-users", type=int, default=0)
    parser.add_argument("--max-sequence-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_root = Path(args.dataset_root)
    cohort_paths = [Path(value) for value in args.exclude_cohort_file]
    cohort_paths = [path if path.is_absolute() else ROOT / path for path in cohort_paths]
    cohorts = [load_json(path) for path in cohort_paths]
    excluded = {str(uid) for cohort in cohorts for uid in cohort.get("uids", [])}
    excluded.update(
        str(row["uid"])
        for cohort in cohorts
        for row in cohort.get("history_rows", [])
    )
    if not excluded:
        raise ValueError("exclude cohort has no users")

    source = take_sequence_rows(dataset_root / "kc_level" / "train_valid_sequences.csv")
    sequences = merge_steps_by_uid(source)
    train = [(uid, steps[: args.max_sequence_steps]) for uid, steps in sequences.items() if uid not in excluded and len(steps) >= 2]
    if args.max_train_users > 0:
        train = train[: args.max_train_users]
    if not train:
        raise ValueError("no non-held-out DAISim training sequences")
    item_index = build_index(step["qid"] for _, steps in train for step in steps)
    concept_index = build_index(step["cid"] for _, steps in train for step in steps)
    examples = []
    for _, steps in train:
        item_ids, concept_ids, responses = encode_steps(steps, item_index, concept_index)
        examples.append((
            history_feature(steps, concept_index),
            torch.tensor(item_ids, dtype=torch.long),
            torch.tensor(concept_ids, dtype=torch.long),
            torch.tensor(responses, dtype=torch.float32),
        ))
    config = DAISimConfig(len(item_index) + 1, len(concept_index) + 1, args.embedding_dim, args.hidden_dim, args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = DAISimGenerator(config).to(device)
    discriminator = DAISimDiscriminator(config).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    history = []
    for phase, epochs, weight in (("direct", args.epochs_direct, 0.0), ("direct_adversarial", args.epochs_adversarial, args.adversarial_weight)):
        for epoch in range(epochs):
            random.shuffle(examples)
            metrics = train_epoch(
                generator,
                discriminator,
                examples,
                generator_optimizer,
                discriminator_optimizer,
                weight,
                run_adversarial=(phase == "direct_adversarial"),
                device=device,
                batch_size=args.batch_size,
            )
            row = {"phase": phase, "epoch": epoch + 1, **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "leakage_safe": True,
        "cohort_history_used_for_training": False,
        "excluded_user_ids": sorted(excluded),
        "excluded_cohort_users": len(excluded),
        "excluded_users_present_in_source": len(set(sequences) & excluded),
        "cohort_files": [str(path) for path in cohort_paths],
        "cohort_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in cohort_paths
        },
        "training_user_count": len(train),
        "max_sequence_steps": args.max_sequence_steps,
        "batch_size": args.batch_size,
        "protocol": "MAIL-style direct imitation followed by pairwise adversarial imitation",
        "official_code_limitations": "Official public release lacked data module and complete model sources; this checkpoint is an official-code-derived reproduction.",
    }
    torch.save(checkpoint_payload(generator, discriminator, config, item_index, concept_index, metadata), output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps({"ok": True, "device": str(device), "history": history, **metadata}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
