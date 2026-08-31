from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from learner_simulator.data import clean_sequence

try:  # Optional dependency: most simulator code can run without torch.
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None


def require_torch() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError(
            "DKT requires PyTorch. Run this script in a Python environment where torch is available."
        )


if nn is not None:

    class DKT(nn.Module):
        """Standard Deep Knowledge Tracing model over knowledge concepts."""

        def __init__(self, num_concepts: int, hidden_dim: int = 100, dropout: float = 0.2) -> None:
            super().__init__()
            self.num_concepts = num_concepts
            self.lstm = nn.LSTM(2 * num_concepts, hidden_dim, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_dim, num_concepts)

        def forward(self, interactions):
            hidden, _ = self.lstm(interactions.float())
            return torch.sigmoid(self.output(self.dropout(hidden)))


class DKTSequenceDataset:
    def __init__(
        self,
        rows: list[dict[str, str]],
        concept_id_map: dict[str, int] | None = None,
    ) -> None:
        require_torch()
        self.concept_id_map = concept_id_map or _build_concept_id_map(rows)
        self.num_concepts = len(self.concept_id_map)
        self.samples = []
        for row in rows:
            concept_indices: list[int] = []
            labels: list[float] = []
            for step in clean_sequence(row):
                concept_index = self.concept_id_map.get(str(step["cid"]))
                if concept_index is None:
                    continue
                response = int(step["response"])
                concept_indices.append(concept_index)
                labels.append(float(response))
            if len(concept_indices) >= 2:
                self.samples.append((concept_indices, labels))

    def split_by_student(self, train_ratio: float = 0.8, val_ratio: float = 0.2):
        train_data, val_data, all_data = [], [], []
        for concept_indices, labels in self.samples:
            split1 = max(2, int(len(concept_indices) * train_ratio))
            split2 = max(2, int(split1 * (1.0 - val_ratio)))
            train_data.append((concept_indices[:split2], labels[:split2]))
            val_data.append((concept_indices[split2:split1], labels[split2:split1]))
            all_data.append((concept_indices, labels))
        return train_data, val_data, all_data


def make_dkt_loader(data, batch_size: int, shuffle: bool, num_concepts: int):
    require_torch()
    filtered = [item for item in data if len(item[0]) >= 2]
    return DataLoader(
        filtered,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_make_collate_fn(num_concepts),
    )


def train_dkt(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_dim: int = 100,
    dropout: float = 0.2,
    seed: int = 42,
    log_every: int = 20,
    training_metadata: dict[str, Any] | None = None,
) -> Path:
    require_torch()
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = DKTSequenceDataset(rows)
    train_data, val_data, _ = dataset.split_by_student()
    train_loader = make_dkt_loader(train_data, batch_size, True, dataset.num_concepts)
    val_loader = make_dkt_loader(val_data, batch_size, False, dataset.num_concepts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DKT(dataset.num_concepts, hidden_dim=hidden_dim, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = None
    best_val_auc = -1.0
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss, _, _ = _step_loss(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1
            if log_every > 0 and batch_index % log_every == 0:
                print(
                    "DKT "
                    f"epoch={epoch}/{epochs} "
                    f"batch={batch_index}/{len(train_loader)} "
                    f"train_loss={train_loss / max(train_batches, 1):.4f}",
                    flush=True,
                )

        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_labels: list[int] = []
        val_scores: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                loss, outputs, target = _step_loss(model, batch, criterion, device)
                val_loss += float(loss.item())
                val_batches += 1
                val_labels.extend(int(x) for x in target.detach().cpu().reshape(-1).tolist())
                val_scores.extend(float(x) for x in outputs.detach().cpu().reshape(-1).tolist())
        val_pred = [1 if score >= 0.5 else 0 for score in val_scores]
        val_acc = sum(1 for y, p in zip(val_labels, val_pred) if y == p) / max(len(val_labels), 1)
        val_auc = _auc(val_labels, val_scores) or 0.0
        print(
            "DKT "
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_loss / max(train_batches, 1):.4f} "
            f"val_loss={val_loss / max(val_batches, 1):.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_auc={val_auc:.4f} "
            f"train_batches={train_batches} "
            f"val_batches={val_batches}",
            flush=True,
        )
        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_state = model.state_dict()

    checkpoint_path = output / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state or model.state_dict(),
            "num_concepts": dataset.num_concepts,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "concept_id_map": dataset.concept_id_map,
            "best_val_acc": best_val_acc,
            "best_val_auc": best_val_auc,
            "training_metadata": training_metadata or {},
        },
        checkpoint_path,
    )
    return checkpoint_path


class DKTProficiency:
    """Runtime loader for DKT-exported student concept proficiency."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = {}
        self.students: dict[str, Any] = {}
        self.concept_id_map: dict[str, int] = {}
        self.index_to_concept: dict[int, int] = {}
        if self.path is not None:
            self._load(self.path)

    def _load(self, path: Path) -> None:
        if path.is_dir():
            file_path = path / "dkt_know_proficiency.json"
            if not file_path.exists():
                file_path = path / "stu_know_proficiency.json"
        else:
            file_path = path
        if not file_path.exists():
            return
        self.data = json.loads(file_path.read_text(encoding="utf-8"))
        meta = self.data.get("_meta", {}) if isinstance(self.data, dict) else {}
        self.students = self.data.get("students", self.data)
        self.concept_id_map = {
            str(key): int(value)
            for key, value in meta.get("concept_id_map", {}).items()
        }
        self.index_to_concept = {
            int(value): int(key)
            for key, value in self.concept_id_map.items()
            if str(key).lstrip("-").isdigit()
        }

    def available(self) -> bool:
        return bool(self.students)

    def _values(self, uid: str) -> list[float] | None:
        value = self.students.get(str(uid))
        if value is None:
            return None
        if value and isinstance(value[0], list):
            return value[-1]
        return value

    def value(self, uid: str, cid: int) -> float | None:
        values = self._values(uid)
        if not values:
            return None
        concept_index = self.concept_id_map.get(str(cid), int(cid))
        if concept_index < 0 or concept_index >= len(values):
            return None
        return float(values[concept_index])

    def latest_values(self, uid: str) -> dict[int, float]:
        values = self._values(uid)
        if not values:
            return {}
        return {
            int(self.index_to_concept.get(index, index)): float(value)
            for index, value in enumerate(values)
        }


def export_dkt_proficiency(
    rows: list[dict[str, str]],
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export per-learner post-history DKT concept probabilities."""

    require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    concept_id_map = {str(k): int(v) for k, v in checkpoint["concept_id_map"].items()}
    model = DKT(
        checkpoint["num_concepts"],
        hidden_dim=int(checkpoint.get("hidden_dim", 100)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    students: dict[str, list[float]] = {}
    with torch.no_grad():
        for row in rows:
            encoded: list[tuple[int, int]] = []
            for step in clean_sequence(row):
                concept_index = concept_id_map.get(str(step["cid"]))
                if concept_index is None:
                    continue
                encoded.append((concept_index, int(step["response"])))
            if not encoded:
                continue
            history = torch.zeros((1, len(encoded), checkpoint["num_concepts"] * 2), dtype=torch.float32)
            for index, (concept_index, response) in enumerate(encoded):
                history[0, index, (2 * concept_index) + (0 if response == 1 else 1)] = 1.0
            probabilities = model(history)[0, -1, :].detach().cpu().tolist()
            students[str(row["uid"])] = [float(value) for value in probabilities]

    payload = {
        "_meta": {
            "format": "learner_simulator_dkt_proficiency_v1",
            "concept_id_map": checkpoint["concept_id_map"],
        },
        "students": students,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return output


def _step_loss(model, batch, criterion, device):
    interactions, concept_indices, labels, lengths = batch
    interactions = interactions.to(device)
    concept_indices = concept_indices.to(device)
    labels = labels.to(device)
    lengths = lengths.to(device)
    probabilities = model(interactions[:, :-1, :])
    next_concepts = concept_indices[:, 1:]
    outputs = probabilities.gather(dim=-1, index=next_concepts.unsqueeze(-1)).squeeze(-1)
    target = labels[:, 1:]
    positions = torch.arange(target.shape[1], device=device).unsqueeze(0)
    valid = positions < (lengths.unsqueeze(1) - 1)
    valid_outputs = outputs[valid]
    valid_target = target[valid]
    return criterion(valid_outputs, valid_target), valid_outputs, valid_target


def _make_collate_fn(num_concepts: int):
    def _collate_fn(batch):
        concept_indices, labels = zip(*batch)
        max_length = max(len(sequence) for sequence in concept_indices)
        padded_sequences = torch.zeros(len(batch), max_length, num_concepts * 2, dtype=torch.float32)
        padded_concepts = torch.zeros(len(batch), max_length, dtype=torch.long)
        padded_labels = torch.zeros(len(batch), max_length, dtype=torch.float32)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        for index, (sequence, label) in enumerate(zip(concept_indices, labels)):
            length = len(sequence)
            lengths[index] = length
            for step_index, (concept_index, response) in enumerate(zip(sequence, label)):
                padded_sequences[index, step_index, (2 * concept_index) + (0 if response == 1 else 1)] = 1.0
                padded_concepts[index, step_index] = concept_index
            padded_labels[index, :length] = torch.tensor(label, dtype=torch.float32)
        return padded_sequences, padded_concepts, padded_labels, lengths

    return _collate_fn

def _build_concept_id_map(rows: list[dict[str, str]]) -> dict[str, int]:
    values: set[str] = set()
    for row in rows:
        for step in clean_sequence(row):
            values.add(str(step["cid"]))
    return {value: index for index, value in enumerate(sorted(values, key=lambda x: int(x)))}


def _auc(y_true: list[int], y_score: list[float]) -> float | None:
    positive = sum(y_true)
    negative = len(y_true) - positive
    if positive == 0 or negative == 0:
        return None
    order = sorted(range(len(y_true)), key=lambda index: y_score[index])
    ranks = [0.0] * len(y_true)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and y_score[order[end + 1]] == y_score[order[index]]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2.0
        for rank_index in range(index, end + 1):
            ranks[order[rank_index]] = average_rank
        index = end + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)
