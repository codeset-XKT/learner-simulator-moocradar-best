#!/usr/bin/env python3
"""Train Deep-IRT, SAKT, or the official AKT architecture on project sequences.

All target cohort users supplied through ``--exclude-cohort-file`` are removed
*before* question-id maps, examples, or validation splits are built.  The
checkpoint therefore carries an auditable leakage-safe training manifest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import clean_sequence, merge_steps_by_uid, sequence_row_from_steps, take_sequence_rows
from learner_simulator.neural_kt import DeepIRT, NeuralKTConfig, SAKT, checkpoint_payload, require_torch, torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("deepirt", "sakt", "akt"), required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-cohort-file", nargs="+", default=[])
    parser.add_argument("--source-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum epochs; training may stop earlier with --early-stopping-patience.")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Stop after this many consecutive validation evaluations without a meaningful AUC improvement; 0 disables early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4,
                        help="Minimum validation AUC improvement required to reset early-stopping patience.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=20)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _cohort_uids(files: list[str]) -> set[str]:
    uids: set[str] = set()
    for raw in files:
        data = json.loads(Path(raw).read_text(encoding="utf-8-sig"))
        uids.update(str(uid) for uid in data.get("uids", []))
    return uids


def _prepare_rows(dataset_root: Path, limit: int, excluded: set[str]):
    rows = take_sequence_rows(dataset_root / "kc_level" / "train_valid_sequences.csv", None if limit <= 0 else limit)
    grouped = merge_steps_by_uid(rows)
    kept = [sequence_row_from_steps(uid, steps) for uid, steps in grouped.items() if uid not in excluded and len(steps) >= 2]
    leaked = sorted({str(row["uid"]) for row in kept} & excluded)
    if leaked:
        raise AssertionError(f"Excluded cohort users remain in neural-KT training rows: {len(leaked)}")
    return kept, len(rows), len(grouped)


def _cohort_item_ids(files: list[str]) -> set[str]:
    """Register cohort item identities without ever consuming their labels."""
    items: set[str] = set()
    for raw in files:
        cohort = json.loads(Path(raw).read_text(encoding="utf-8-sig"))
        for row in list(cohort.get("history_rows", [])) + list(cohort.get("target_rows", [])):
            items.update(str(step["qid"]) for step in clean_sequence(row))
    return items


def _item_map(rows: list[dict[str, str]], extra_item_ids: set[str] | None = None) -> dict[str, int]:
    items = {str(step["qid"]) for row in rows for step in clean_sequence(row)}
    items.update(extra_item_ids or set())
    return {item: index + 1 for index, item in enumerate(sorted(items, key=int))}


def _examples(rows: list[dict[str, str]], item_map: dict[str, int], max_len: int):
    result: list[tuple[str, list[int], list[int]]] = []
    for row in rows:
        values = [(item_map.get(str(step["qid"])), int(step["response"])) for step in clean_sequence(row)]
        values = [(item, response) for item, response in values if item is not None]
        for start in range(0, len(values), max_len):
            chunk = values[start:start + max_len]
            if len(chunk) >= 2:
                result.append((str(row["uid"]), [x[0] for x in chunk], [x[1] for x in chunk]))
    return result


def _collate(batch):
    max_len = max(len(items) for _, items, _ in batch)
    items = torch.zeros((len(batch), max_len), dtype=torch.long)
    responses = torch.zeros((len(batch), max_len), dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for index, (_, item_values, response_values) in enumerate(batch):
        length = len(item_values)
        lengths[index] = length
        items[index, :length] = torch.tensor(item_values, dtype=torch.long)
        responses[index, :length] = torch.tensor(response_values, dtype=torch.long)
    return items, responses, lengths


def _loader(examples, batch_size, shuffle):
    return torch.utils.data.DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)


def _load_official_akt(device):
    source = ROOT / "third_party" / "AKT_official" / "akt.py"
    if not source.exists():
        raise RuntimeError("Official AKT source is missing at third_party/AKT_official/akt.py")
    spec = importlib.util.spec_from_file_location("learner_simulator_official_akt", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import official AKT source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.device = device
    return module


def _make_model(args, config, device):
    if args.method == "deepirt":
        return DeepIRT(config).to(device), None
    if args.method == "sakt":
        return SAKT(config).to(device), None
    official = _load_official_akt(device)
    model = official.AKT(
        n_question=config.num_items,
        n_pid=config.num_items,
        d_model=config.embed_dim,
        n_blocks=1,
        kq_same=1,
        dropout=config.dropout,
        model_type="akt",
        final_fc_dim=max(64, config.embed_dim * 2),
        n_heads=config.num_heads,
        d_ff=max(128, config.embed_dim * 4),
        l2=1e-5,
        separate_qa=False,
    ).to(device)
    return model, official


def _forward(method, model, items, responses, lengths):
    valid = torch.arange(items.shape[1], device=items.device).unsqueeze(0) < lengths.unsqueeze(1)
    # t=0 has no observed interaction history, so it is never a supervised target.
    loss_mask = valid & (torch.arange(items.shape[1], device=items.device).unsqueeze(0) > 0)
    if method in {"deepirt", "sakt"}:
        if method == "deepirt":
            probabilities, _, _ = model(items, responses)
        else:
            probabilities = model(items, responses)
        loss = torch.nn.functional.binary_cross_entropy(
            probabilities[loss_mask], responses.float()[loss_mask]
        )
        return loss, probabilities, loss_mask
    qa = items + responses * model.n_question
    target = responses.float().masked_fill(~valid, -1.0)
    raw_loss, flat_probabilities, _ = model(items, qa, target, items)
    probabilities = flat_probabilities.reshape_as(items)
    # Official model returns summed BCE + Rasch regularization. Normalization
    # makes its optimization scale independent of batch sequence length.
    return raw_loss / loss_mask.sum().clamp_min(1), probabilities, loss_mask


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = sorted(range(len(labels)), key=lambda i: scores[i])
    ranks = [0.0] * len(labels)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        rank = (start + 1 + end + 1) / 2
        for index in range(start, end + 1):
            ranks[order[index]] = rank
        start = end + 1
    return (sum(rank for rank, label in zip(ranks, labels) if label) - positive * (positive + 1) / 2) / (positive * negative)


def _evaluate(args, model, loader, device):
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for items, responses, lengths in loader:
            items, responses, lengths = items.to(device), responses.to(device), lengths.to(device)
            loss, probabilities, mask = _forward(args.method, model, items, responses, lengths)
            losses.append(float(loss.item()))
            labels.extend(int(x) for x in responses[mask].cpu().tolist())
            scores.extend(float(x) for x in probabilities[mask].cpu().tolist())
    prediction = [int(value >= 0.5) for value in scores]
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "count": len(labels),
        "accuracy": sum(int(y == p) for y, p in zip(labels, prediction)) / max(len(labels), 1),
        "auc": _auc(labels, scores),
    }


def main() -> None:
    args = parse_args()
    require_torch()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    excluded = _cohort_uids(args.exclude_cohort_file)
    rows, source_row_count, source_user_count = _prepare_rows(Path(args.dataset_root), args.source_rows, excluded)
    cohort_item_ids = _cohort_item_ids(args.exclude_cohort_file)
    item_map = _item_map(rows, cohort_item_ids)
    examples = _examples(rows, item_map, args.max_seq_len)
    users = sorted({uid for uid, _, _ in examples})
    random.Random(args.seed).shuffle(users)
    val_users = set(users[max(1, int(len(users) * 0.9)):])
    train_examples = [example for example in examples if example[0] not in val_users]
    val_examples = [example for example in examples if example[0] in val_users]
    if not train_examples or not val_examples:
        raise RuntimeError("Need non-empty train and validation examples")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = NeuralKTConfig(
        num_items=len(item_map), embed_dim=args.embed_dim, memory_size=args.memory_size,
        dropout=args.dropout, num_heads=args.num_heads, max_seq_len=args.max_seq_len,
    )
    model, _ = _make_model(args, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = _loader(train_examples, args.batch_size, True)
    val_loader = _loader(val_examples, args.batch_size, False)
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be non-negative")
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    epoch_summaries = []
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        train_losses = []
        for items, responses, lengths in train_loader:
            items, responses, lengths = items.to(device), responses.to(device), lengths.to(device)
            optimizer.zero_grad()
            loss, _, _ = _forward(args.method, model, items, responses, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val = _evaluate(args, model, val_loader, device)
        summary = {"epoch": epoch, "train_loss": sum(train_losses) / max(len(train_losses), 1), **val, "seconds": perf_counter() - started}
        epoch_summaries.append(summary)
        print(json.dumps({"method": args.method, **summary}, ensure_ascii=False), flush=True)
        criterion = val["auc"] if val["auc"] is not None else val["accuracy"]
        # Keep the best state even for a tiny numerical gain, but only reset
        # patience when the gain is substantively larger than min_delta.
        if criterion > best_auc:
            best_auc = criterion
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
        if criterion > (max((item["auc"] if item["auc"] is not None else item["accuracy"]) for item in epoch_summaries[:-1]) if len(epoch_summaries) > 1 else -1.0) + args.early_stopping_min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            print(json.dumps({"method": args.method, "early_stopping": True, "epoch": epoch,
                              "best_epoch": best_epoch, "best_validation_criterion": best_auc,
                              "patience": args.early_stopping_patience,
                              "min_delta": args.early_stopping_min_delta}, ensure_ascii=False), flush=True)
            break
    metadata = {
        "leakage_safe": True,
        "excluded_user_ids": sorted(excluded),
        "excluded_cohort_users": len(excluded),
        "cohort_history_used_for_training": False,
        "source_sequence_rows": source_row_count,
        "source_users_before_exclusion": source_user_count,
        "train_users": len({uid for uid, _, _ in train_examples}),
        "validation_users": len(val_users),
        "cohort_question_ids_registered_without_labels": len(cohort_item_ids),
        "max_epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "seed": args.seed,
    }
    payload = checkpoint_payload(
        args.method, best_state or model.state_dict(), config, item_map, metadata,
        official_reference=("third_party/DeepIRT_official" if args.method == "deepirt" else "third_party/AKT_official" if args.method == "akt" else "SAKT_2019"),
        epoch_summaries=epoch_summaries,
    )
    checkpoint = output / "best_model.pt"
    torch.save(payload, checkpoint)
    summary = {"ok": True, "method": args.method, "checkpoint": str(checkpoint), "training_metadata": metadata, "best_validation_criterion": best_auc, "best_epoch": best_epoch, "stopped_early": stopped_early, "epochs": epoch_summaries}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
