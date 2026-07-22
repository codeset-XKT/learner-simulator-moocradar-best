from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from learner_simulator.data import clean_sequence

try:  # Optional dependency: normal simulation can run without torch.
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover - exercised on no-torch envs.
    torch = None
    nn = None
    DataLoader = None


class DNeuralCDMProficiency:
    """Runtime loader for DNeuralCDM-exported student knowledge proficiency."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = {}
        self.students: dict[str, Any] = {}
        self.concept_id_map: dict[str, int] = {}
        self.index_to_concept: dict[int, int] = {}
        if self.path is not None:
            self._load(self.path)

    def _load(self, path: Path) -> None:
        file_path = path / "stu_know_proficiency.json" if path.is_dir() else path
        if not file_path.exists():
            return
        with file_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
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

    def _sequence(self, uid: str) -> list[list[float]] | None:
        value = self.students.get(str(uid))
        if value is None:
            return None
        if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
            return value[0]
        return value

    def value(self, uid: str, cid: int, time_step: int | None = None) -> float | None:
        seq = self._sequence(uid)
        if not seq:
            return None
        concept_index = self.concept_id_map.get(str(cid), int(cid))
        index = len(seq) - 1 if time_step is None else max(0, min(len(seq) - 1, time_step))
        if concept_index < 0 or concept_index >= len(seq[index]):
            return None
        return float(seq[index][concept_index])

    def latest_values(self, uid: str) -> dict[int, float]:
        seq = self._sequence(uid)
        if not seq:
            return {}
        latest = seq[-1]
        values: dict[int, float] = {}
        for index, value in enumerate(latest):
            cid = self.index_to_concept.get(index, index)
            values[int(cid)] = float(value)
        return values

    @staticmethod
    def tier(value: float | None) -> str:
        if value is None:
            return "unknown"
        if value > 0.66:
            return "high"
        if value > 0.33:
            return "medium"
        return "low"


def require_torch() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError(
            "DNeuralCDM requires PyTorch. Install torch or run this script in a "
            "Python environment where torch is available."
        )


if nn is not None:

    class DNeuralCDM(nn.Module):
        """DNeuralCDM-style recurrent cognitive diagnosis model.

        The forward structure follows Agent4Edu's DNeuralCDM tool: response-coded
        concept inputs are embedded, passed through an LSTM, transformed into a
        student knowledge state, and diagnosed with exercise difficulty and
        discrimination terms.
        """

        def __init__(
            self,
            num_exercises: int,
            num_know: int,
            embedding_dim: int,
            hidden_dim: int,
        ) -> None:
            super().__init__()
            self.embedding = nn.Linear(2 * num_know, embedding_dim)
            self.exer_diff = nn.Linear(num_exercises, num_know)
            self.exer_disc = nn.Linear(num_exercises, 1)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_know)
            self.pre1 = nn.Linear(num_know, 512)
            self.pre2 = nn.Linear(512, 256)
            self.pre3 = nn.Linear(256, 1)
            self.sigmoid = nn.Sigmoid()
            for name, param in self.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)

        def apply_clipper(self) -> None:
            clipper = _NoneNegClipper()
            self.pre1.apply(clipper)
            self.pre2.apply(clipper)
            self.pre3.apply(clipper)

        def forward(self, x, exercise_id, mask):
            x_reshaped = x.reshape(-1, x.shape[2]).float()
            embedded = self.embedding(x_reshaped).reshape(x.shape[0], x.shape[1], -1)
            lstm_out, _ = self.lstm(embedded)
            exercise_flat = exercise_id.reshape(-1, exercise_id.shape[2]).float()
            exer_diff = torch.sigmoid(self.exer_diff(exercise_flat)).reshape(
                exercise_id.shape[0],
                exercise_id.shape[1],
                -1,
            )
            exer_disc = self.exer_disc(exercise_flat).reshape(exercise_id.shape[0], exercise_id.shape[1], -1)
            stu_emb = torch.sigmoid(self.fc(lstm_out))
            input_x = (stu_emb - exer_diff) * mask * exer_disc
            input_x = torch.tanh(self.pre1(input_x)).squeeze(-1)
            input_x = torch.tanh(self.pre2(input_x)).squeeze(-1)
            probs = self.sigmoid(self.pre3(input_x)).squeeze(-1)
            return probs, lstm_out, stu_emb


    class _NoneNegClipper:
        def __call__(self, module) -> None:
            if hasattr(module, "weight"):
                weight = module.weight.data
                weight.add_(torch.relu(torch.neg(weight)))


class DNeuralCDMSequenceDataset:
    def __init__(
        self,
        rows: list[dict[str, str]],
        exercise_id_map: dict[str, int] | None = None,
        concept_id_map: dict[str, int] | None = None,
    ) -> None:
        require_torch()
        self.exercise_id_map = exercise_id_map or _build_id_map(rows, "qid")
        self.concept_id_map = concept_id_map or _build_id_map(rows, "cid")
        self.num_exercises = len(self.exercise_id_map)
        self.num_know = len(self.concept_id_map)
        self.samples = []
        for row in rows:
            sequence = []
            labels = []
            masks = []
            exercise_vectors = []
            user_ids = []
            for step in clean_sequence(row):
                qid = self.exercise_id_map.get(str(step["qid"]))
                cid = self.concept_id_map.get(str(step["cid"]))
                if qid is None or cid is None:
                    continue
                response = int(step["response"])
                encoded = [0.0] * (2 * self.num_know)
                mask = [0.0] * self.num_know
                exercise_vec = [0.0] * self.num_exercises
                encoded[(2 * cid) + (0 if response == 1 else 1)] = 1.0
                mask[cid] = 1.0
                exercise_vec[qid] = 1.0
                sequence.append(encoded)
                labels.append(float(response))
                masks.append(mask)
                exercise_vectors.append(exercise_vec)
                user_ids.append(str(row["uid"]))
            if len(sequence) >= 2:
                self.samples.append((sequence, masks, exercise_vectors, labels, user_ids))

    def split_by_student(self, train_ratio: float = 0.8, val_ratio: float = 0.2):
        train_data, val_data, all_data = [], [], []
        for seq, mask, exe, labels, uids in self.samples:
            split1 = max(2, int(len(seq) * train_ratio))
            split2 = max(2, int(split1 * (1.0 - val_ratio)))
            train_data.append((seq[:split2], mask[:split2], exe[:split2], labels[:split2], uids[:split2]))
            val_data.append((seq[split2:split1], mask[split2:split1], exe[split2:split1], labels[split2:split1], uids[split2:split1]))
            all_data.append((seq, mask, exe, labels, uids))
        return train_data, val_data, all_data


def make_dneuralcdm_loader(data, batch_size: int, shuffle: bool):
    require_torch()
    filtered = [item for item in data if len(item[0]) >= 2]
    return DataLoader(filtered, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)


def train_dneuralcdm(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 0.001,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    seed: int = 42,
    log_every: int = 20,
) -> Path:
    require_torch()
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = DNeuralCDMSequenceDataset(rows)
    train_data, val_data, _ = dataset.split_by_student()
    train_loader = make_dneuralcdm_loader(train_data, batch_size, True)
    val_loader = make_dneuralcdm_loader(val_data, batch_size, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNeuralCDM(dataset.num_exercises, dataset.num_know, embedding_dim, hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = None
    best_val = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss, _, _ = _step_loss(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            model.apply_clipper()
            train_loss += float(loss.item())
            train_batches += 1
            if log_every > 0 and batch_index % log_every == 0:
                print(
                    "DNeuralCDM "
                    f"epoch={epoch}/{epochs} "
                    f"batch={batch_index}/{len(train_loader)} "
                    f"train_loss={train_loss / max(train_batches, 1):.4f}",
                    flush=True,
                )
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss, outputs, target = _step_loss(model, batch, criterion, device)
                val_loss += float(loss.item())
                val_batches += 1
                pred = (outputs >= 0.5).float()
                correct += (pred == target).sum().item()
                total += target.numel()
        val_acc = correct / max(total, 1)
        print(
            "DNeuralCDM "
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_loss / max(train_batches, 1):.4f} "
            f"val_loss={val_loss / max(val_batches, 1):.4f} "
            f"val_acc={val_acc:.4f} "
            f"train_batches={train_batches} "
            f"val_batches={val_batches}",
            flush=True,
        )
        if val_acc >= best_val:
            best_val = val_acc
            best_state = model.state_dict()
    checkpoint_path = output / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state or model.state_dict(),
            "num_exercises": dataset.num_exercises,
            "num_know": dataset.num_know,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "exercise_id_map": dataset.exercise_id_map,
            "concept_id_map": dataset.concept_id_map,
            "best_val_acc": best_val,
        },
        checkpoint_path,
    )
    return checkpoint_path


def export_dneuralcdm_proficiency(
    rows: list[dict[str, str]],
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset = DNeuralCDMSequenceDataset(
        rows,
        exercise_id_map={str(k): int(v) for k, v in checkpoint["exercise_id_map"].items()},
        concept_id_map={str(k): int(v) for k, v in checkpoint["concept_id_map"].items()},
    )
    _, _, all_data = dataset.split_by_student()
    loader = make_dneuralcdm_loader(all_data, batch_size=1, shuffle=False)
    model = DNeuralCDM(
        checkpoint["num_exercises"],
        checkpoint["num_know"],
        checkpoint["embedding_dim"],
        checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    students: dict[str, Any] = {}
    with torch.no_grad():
        for sequences, masks, exercise_ids, labels, user_ids, lengths in loader:
            _, _, stu_emb = model(sequences[:, :-1, :], exercise_ids[:, 1:, :], masks[:, 1:, :])
            uid = str(user_ids[0][0])
            students[uid] = stu_emb.cpu().tolist()[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "format": "learner_simulator_dneuralcdm_proficiency_v1",
            "concept_id_map": checkpoint["concept_id_map"],
            "exercise_id_map": checkpoint["exercise_id_map"],
        },
        "students": students,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return output


def _build_id_map(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    values: set[str] = set()
    for row in rows:
        for step in clean_sequence(row):
            values.add(str(step[key]))
    return {value: index for index, value in enumerate(sorted(values, key=lambda x: int(x)))}


def _collate_fn(batch):
    sequences, masks, exercise_ids, labels, user_ids = zip(*batch)
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = torch.zeros(len(batch), max_length, len(sequences[0][0]), dtype=torch.float32)
    padded_masks = torch.zeros(len(batch), max_length, len(masks[0][0]), dtype=torch.float32)
    padded_exercise_ids = torch.zeros(len(batch), max_length, len(exercise_ids[0][0]), dtype=torch.float32)
    padded_labels = torch.zeros(len(batch), max_length, dtype=torch.float32)
    padded_user_ids: list[list[str]] = []
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for index, (seq, mask, exe, label, uid) in enumerate(zip(sequences, masks, exercise_ids, labels, user_ids)):
        length = len(seq)
        lengths[index] = length
        padded_sequences[index, :length] = torch.tensor(seq, dtype=torch.float32)
        padded_masks[index, :length] = torch.tensor(mask, dtype=torch.float32)
        padded_exercise_ids[index, :length] = torch.tensor(exe, dtype=torch.float32)
        padded_labels[index, :length] = torch.tensor(label, dtype=torch.float32)
        padded_user_ids.append(list(uid))
    return padded_sequences, padded_masks, padded_exercise_ids, padded_labels, padded_user_ids, lengths


def _step_loss(model, batch, criterion, device):
    sequences, masks, exercise_ids, labels, user_ids, lengths = batch
    sequences = sequences.to(device)
    masks = masks.to(device)
    exercise_ids = exercise_ids.to(device)
    labels = labels.to(device)
    outputs, _, _ = model(sequences[:, :-1, :], exercise_ids[:, 1:, :], masks[:, 1:, :])
    target = labels[:, 1:]
    return criterion(outputs, target), outputs, target
