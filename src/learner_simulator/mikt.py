from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from learner_simulator.data import clean_sequence

try:  # Optional dependency: runtime loading should not require torch.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError(
            "MIKT requires PyTorch. Run this script in a Python environment where torch is available."
        )


@dataclass
class MIKTConfig:
    ques_num: int
    skill_num: int
    embed_dim: int
    max_seq_len: int


if nn is not None:

    class MIKT(nn.Module):
        """Minimal local MIKT runtime for proficiency export."""

        def __init__(
            self,
            config: MIKTConfig,
            problem_skill_tensor,
            p: float = 0.2,
        ) -> None:
            super().__init__()
            self.skill_max = int(config.skill_num)
            self.pro_max = int(config.ques_num)
            self.embed_dim = int(config.embed_dim)
            self.max_seq_len = int(config.max_seq_len)

            if problem_skill_tensor.shape != (self.pro_max, self.skill_max):
                raise ValueError(
                    "problem_skill_tensor shape must match "
                    f"({self.pro_max}, {self.skill_max}); got {tuple(problem_skill_tensor.shape)}"
                )
            self.register_buffer("pro2skill", problem_skill_tensor.float())

            d = self.embed_dim
            state_d = self.embed_dim

            self.pro_embed = nn.Parameter(torch.rand(self.pro_max, d))
            nn.init.xavier_uniform_(self.pro_embed)

            self.skill_embed = nn.Parameter(torch.rand(self.skill_max, d))
            nn.init.xavier_uniform_(self.skill_embed)

            self.var = nn.Parameter(torch.rand(self.pro_max, d))
            self.change = nn.Parameter(torch.rand(self.pro_max, 1))

            self.pos_embed = nn.Parameter(torch.rand(self.max_seq_len, d))
            nn.init.xavier_uniform_(self.pos_embed)

            self.skill_state = nn.Parameter(torch.rand(self.skill_max, state_d))
            self.time_state = nn.Parameter(torch.rand(self.max_seq_len, state_d))
            self.all_state = nn.Parameter(torch.rand(1, state_d))

            self.all_forget = nn.Sequential(
                nn.Linear(2 * state_d, state_d),
                nn.ReLU(),
                nn.Linear(state_d, state_d),
                nn.Sigmoid(),
            )

            self.ans_embed = nn.Embedding(2, d)
            self.lstm = nn.LSTM(2 * d, d, batch_first=True)

            self.now_obtain = nn.Sequential(
                nn.Linear(d, state_d),
                nn.Tanh(),
                nn.Linear(state_d, state_d),
                nn.Tanh(),
            )

            self.pro_diff_embed = nn.Parameter(torch.rand(self.pro_max, d))
            self.pro_diff = nn.Embedding(self.pro_max, 1)

            self.pro_linear = nn.Linear(d, d)
            self.skill_linear = nn.Linear(d, d)
            self.pro_change = nn.Linear(d, d)

            self.pro_guess = nn.Embedding(self.pro_max, 1)
            self.pro_divide = nn.Embedding(self.pro_max, 1)

            self.pro_ability = nn.Sequential(
                nn.Linear(3 * d, d),
                nn.ReLU(),
                nn.Linear(d, 1),
            )

            self.obtain1_linear = nn.Linear(d, d)
            self.obtain2_linear = nn.Linear(d, d)
            self.pro_diff_judge = nn.Linear(d, 1)
            self.all_obtain = nn.Linear(d, d)

            self.skill_forget = nn.Sequential(
                nn.Linear(3 * d, d),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(d, d),
            )

            self.do_attn = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(d, 1),
            )

            self.predict_attn = nn.Linear(3 * d, d)
            self.dropout = nn.Dropout(p=p)

            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.xavier_uniform_(module.weight)

        def forward(self, last_problem, last_ans, next_problem, next_ans):
            device = last_problem.device
            seq = next_problem.shape[1]
            batch = last_problem.shape[0]

            pro2skill = self.pro2skill.to(device=device)
            skill_embed = self.skill_embed
            pro_embed = self.pro_embed

            skill_mean = torch.matmul(pro2skill, skill_embed) / (
                torch.sum(pro2skill, dim=-1, keepdim=True) + 1e-8
            )

            pro_idx = torch.arange(self.pro_max, device=device)
            pro_diff = torch.sigmoid(self.pro_diff(pro_idx))

            q_pro = self.pro_linear(pro_embed)
            q_skill = self.skill_linear(skill_embed)
            attn = torch.matmul(q_pro, q_skill.transpose(-1, -2)) / math.sqrt(
                q_pro.shape[-1]
            )
            attn = torch.masked_fill(attn, pro2skill == 0, -1e9)
            attn = torch.softmax(attn, dim=-1)
            skill_attn = torch.matmul(attn, skill_embed)

            now_embed = skill_attn + pro_diff * self.pro_change(skill_mean)
            pro_embed = self.dropout(now_embed)

            next_pro_rasch = F.embedding(next_problem, pro_embed)
            next_pro_guess = torch.sigmoid(self.pro_guess(next_problem))
            next_pro_divide = self.pro_divide(next_problem)

            last_pro_rasch = F.embedding(last_problem, pro_embed)
            next_x = next_pro_rasch + self.ans_embed(next_ans.long())

            last_all_time = torch.ones((batch,), device=device, dtype=torch.long)
            time_embed = self.time_state
            all_gap_embed = F.embedding(last_all_time, time_embed)

            res_p: list[Any] = []
            last_skill_time = torch.zeros((batch, self.skill_max), device=device, dtype=torch.long)
            skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)
            all_state = self.all_state.repeat(batch, 1)

            for now_step in range(seq):
                now_pro = next_problem[:, now_step]
                now_pro2skill = F.embedding(now_pro, pro2skill).unsqueeze(1)
                now_pro_embed = next_pro_rasch[:, now_step]

                skill_time_gap = now_step - now_pro2skill.squeeze(1) * last_skill_time
                skill_time_gap_embed = F.embedding(skill_time_gap.long(), time_embed)

                forget_now_all_state = all_state * self.all_forget(
                    self.dropout(torch.cat([all_state, all_gap_embed], dim=-1))
                )
                effect_all_state = forget_now_all_state.unsqueeze(1).repeat(
                    1, skill_state.shape[1], 1
                )

                skill_forget = torch.sigmoid(
                    self.skill_forget(
                        self.dropout(
                            torch.cat(
                                [skill_state, skill_time_gap_embed, effect_all_state],
                                dim=-1,
                            )
                        )
                    )
                )
                skill_forget = torch.masked_fill(
                    skill_forget,
                    now_pro2skill.transpose(-1, -2) == 0,
                    1,
                )
                skill_state = skill_state * skill_forget

                now_pro_skill_attn = torch.matmul(
                    now_pro_embed.unsqueeze(1),
                    skill_state.transpose(-1, -2),
                ) / now_pro_embed.shape[-1]
                now_pro_skill_attn = torch.masked_fill(
                    now_pro_skill_attn,
                    now_pro2skill == 0,
                    -1e9,
                )
                now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)
                now_need_state = torch.matmul(now_pro_skill_attn, skill_state).squeeze(1)

                all_attn = torch.sigmoid(
                    self.predict_attn(
                        self.dropout(
                            torch.cat(
                                [now_need_state, forget_now_all_state, now_pro_embed],
                                dim=-1,
                            )
                        )
                    )
                )
                now_need_state = torch.cat(
                    [
                        (1 - all_attn) * now_need_state,
                        all_attn * forget_now_all_state,
                    ],
                    dim=-1,
                )

                last_skill_time = torch.masked_fill(
                    last_skill_time,
                    now_pro2skill.squeeze(1) == 1,
                    now_step,
                )

                now_ability = torch.sigmoid(
                    self.pro_ability(torch.cat([now_need_state, now_pro_embed], dim=-1))
                )
                _ = next_pro_guess[:, now_step]
                _ = next_pro_divide[:, now_step]
                now_diff = F.embedding(now_pro, pro_diff)
                now_output = torch.sigmoid(5 * (now_ability - now_diff)).squeeze(-1)
                res_p.append(now_output)

                now_x = next_x[:, now_step]
                all_state = forget_now_all_state + torch.tanh(
                    self.all_obtain(self.dropout(now_x))
                ).squeeze(1)
                to_get = torch.tanh(self.now_obtain(self.dropout(now_x))).unsqueeze(1)
                now_pro_skill_attn = torch.matmul(
                    to_get,
                    skill_state.transpose(-1, -2),
                ) / to_get.shape[-1]
                now_pro_skill_attn = torch.masked_fill(
                    now_pro_skill_attn,
                    now_pro2skill == 0,
                    -1e9,
                )
                now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)
                now_get = torch.matmul(now_pro_skill_attn.transpose(-1, -2), to_get)
                skill_state = skill_state + now_get

            probability = torch.vstack(res_p).T
            return probability, 0.0


class MIKTSequenceDataset:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        require_torch()
        self.question_id_map = _build_raw_id_map(rows, "qid")
        self.concept_id_map = _build_raw_id_map(rows, "cid")
        self.index_to_question = {
            int(value): int(key)
            for key, value in self.question_id_map.items()
        }
        self.index_to_concept = {
            int(value): int(key)
            for key, value in self.concept_id_map.items()
        }
        self.num_questions = len(self.question_id_map)
        self.num_concepts = len(self.concept_id_map)
        self.question_to_concepts: dict[int, set[int]] = {}
        self.samples: list[tuple[list[int], list[float]]] = []
        self.max_seq_len = 0

        for row in rows:
            question_indices: list[int] = []
            labels: list[float] = []
            for step in clean_sequence(row):
                question_index = self.question_id_map.get(str(step["qid"]))
                concept_index = self.concept_id_map.get(str(step["cid"]))
                if question_index is None or concept_index is None:
                    continue
                response = int(step["response"])
                question_indices.append(question_index)
                labels.append(float(response))
                self.question_to_concepts.setdefault(question_index, set()).add(concept_index)
            if len(question_indices) >= 2:
                self.samples.append((question_indices, labels))
                self.max_seq_len = max(self.max_seq_len, len(question_indices) - 1)

    def split_by_student(self, train_ratio: float = 0.8, val_ratio: float = 0.2):
        train_data, val_data, all_data = [], [], []
        for question_indices, labels in self.samples:
            split1 = max(2, int(len(question_indices) * train_ratio))
            split2 = max(2, int(split1 * (1.0 - val_ratio)))
            train_data.append((question_indices[:split2], labels[:split2]))
            val_data.append((question_indices[split2:split1], labels[split2:split1]))
            all_data.append((question_indices, labels))
        return train_data, val_data, all_data


def make_mikt_loader(data, batch_size: int, shuffle: bool):
    require_torch()
    filtered = [item for item in data if len(item[0]) >= 2]
    return torch.utils.data.DataLoader(
        filtered,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_make_mikt_collate_fn(),
    )


def train_mikt(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    hidden_dim: int = 100,
    dropout: float = 0.2,
    seed: int = 42,
    log_every: int = 20,
    device_name: str | None = None,
) -> Path:
    require_torch()
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = MIKTSequenceDataset(rows)
    train_data, val_data, _ = dataset.split_by_student()
    train_loader = make_mikt_loader(train_data, batch_size, True)
    val_loader = make_mikt_loader(val_data, batch_size, False)

    config = MIKTConfig(
        ques_num=dataset.num_questions,
        skill_num=dataset.num_concepts,
        embed_dim=hidden_dim,
        max_seq_len=max(dataset.max_seq_len, 2),
    )
    problem_skill_tensor = _build_problem_skill_tensor(dataset.question_to_concepts, config)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MIKT(config, problem_skill_tensor, p=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = None
    best_val_auc = -1.0
    best_val_acc = -1.0
    epoch_summaries: list[dict[str, Any]] = []

    print(
        "MIKT "
        f"train_sequences={len(train_data)} "
        f"val_sequences={len(val_data)} "
        f"questions={dataset.num_questions} "
        f"concepts={dataset.num_concepts} "
        f"max_seq_len={config.max_seq_len} "
        f"device={device}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        epoch_start = perf_counter()
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss, _, _ = _step_mikt_loss(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1
            if log_every > 0 and batch_index % log_every == 0:
                print(
                    "MIKT "
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
                loss, outputs, target = _step_mikt_loss(model, batch, criterion, device)
                val_loss += float(loss.item())
                val_batches += 1
                val_labels.extend(int(x) for x in target.detach().cpu().reshape(-1).tolist())
                val_scores.extend(float(x) for x in outputs.detach().cpu().reshape(-1).tolist())
        val_pred = [1 if score >= 0.5 else 0 for score in val_scores]
        val_acc = sum(1 for y, p in zip(val_labels, val_pred) if y == p) / max(len(val_labels), 1)
        val_auc = _auc(val_labels, val_scores) or 0.0
        epoch_seconds = perf_counter() - epoch_start
        print(
            "MIKT "
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_loss / max(train_batches, 1):.4f} "
            f"val_loss={val_loss / max(val_batches, 1):.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_auc={val_auc:.4f} "
            f"train_batches={train_batches} "
            f"val_batches={val_batches} "
            f"epoch_seconds={epoch_seconds:.2f}",
            flush=True,
        )
        epoch_summaries.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_batches, 1),
                "val_loss": val_loss / max(val_batches, 1),
                "val_acc": val_acc,
                "val_auc": val_auc,
                "train_batches": train_batches,
                "val_batches": val_batches,
                "epoch_seconds": epoch_seconds,
            }
        )
        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    checkpoint_path = output / "best_model.pt"
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "config": {
                "ques_num": config.ques_num,
                "skill_num": config.skill_num,
                "embed_dim": config.embed_dim,
                "max_seq_len": config.max_seq_len,
            },
            "question_id_map": dataset.question_id_map,
            "concept_id_map": dataset.concept_id_map,
            "best_val_acc": best_val_acc,
            "best_val_auc": best_val_auc,
            "epoch_summaries": epoch_summaries,
        },
        checkpoint_path,
    )
    return checkpoint_path


class MIKTProficiency:
    """Runtime loader for MIKT-exported student concept proficiency."""

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
            file_path = path / "mikt_know_proficiency.json"
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


def export_mikt_proficiency(
    rows: list[dict[str, str]],
    checkpoint_path: str | Path,
    output_path: str | Path,
    reference_rows: list[dict[str, str]] | None = None,
) -> Path:
    """Export per-learner post-history concept proficiency from a trained MIKT model."""

    require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise RuntimeError("Unsupported MIKT checkpoint format.")

    config = _infer_mikt_config(state, checkpoint if isinstance(checkpoint, dict) else None)
    question_id_map = _load_checkpoint_id_map(
        checkpoint if isinstance(checkpoint, dict) else None,
        "question_id_map",
    )
    concept_id_map = _load_checkpoint_id_map(
        checkpoint if isinstance(checkpoint, dict) else None,
        "concept_id_map",
    )
    reference = list(rows) + list(reference_rows or [])
    question_to_skills, skill_to_questions = _build_question_skill_maps(
        reference,
        ques_num=config.ques_num,
        skill_num=config.skill_num,
        question_id_map=question_id_map or None,
        concept_id_map=concept_id_map or None,
    )
    model = MIKT(config, _build_problem_skill_tensor(question_to_skills, config))
    missing, unexpected = model.load_state_dict(state, strict=False)
    required = {"pro_embed", "skill_embed", "pro_diff.weight", "skill_state", "time_state"}
    missing_required = [name for name in missing if name in required]
    if missing_required:
        raise RuntimeError(f"MIKT checkpoint missing required tensors: {missing_required}")
    if unexpected:
        unexpected = list(unexpected)
    model.eval()

    if concept_id_map:
        sorted_raw_concepts = sorted(concept_id_map, key=lambda value: int(value))
        exported_concept_id_map = {
            str(cid): index for index, cid in enumerate(sorted_raw_concepts)
        }
    else:
        sorted_raw_concepts = [str(cid) for cid in sorted(skill_to_questions, key=int)]
        exported_concept_id_map = {
            str(cid): index for index, cid in enumerate(sorted_raw_concepts)
        }
    students: dict[str, list[float]] = {}
    with torch.no_grad():
        for row in rows:
            history = clean_sequence(row)
            encoded: list[tuple[int, int]] = []
            for step in history:
                question_index = _remap_id(
                    step["qid"],
                    question_id_map or None,
                    upper_bound=config.ques_num,
                )
                if question_index is None:
                    continue
                encoded.append((question_index, int(step["response"])))
            if not encoded:
                continue
            last_problem = torch.tensor(
                [[qid for qid, _ in encoded]],
                dtype=torch.long,
            )
            last_ans = torch.tensor(
                [[response for _, response in encoded]],
                dtype=torch.long,
            )

            concept_values: list[float] = []
            for raw_cid in sorted_raw_concepts:
                concept_index = _remap_id(
                    raw_cid,
                    concept_id_map or None,
                    upper_bound=config.skill_num,
                )
                if concept_index is None:
                    concept_values.append(0.5)
                    continue
                predictions: list[float] = []
                for qid in sorted(skill_to_questions.get(concept_index, [])):
                    if qid < 0 or qid >= config.ques_num:
                        continue
                    next_problem = torch.tensor([[qid]], dtype=torch.long)
                    next_ans = torch.zeros((1, 1), dtype=torch.float32)
                    probability, _ = model(last_problem, last_ans, next_problem, next_ans)
                    predictions.append(float(probability[0, 0].item()))
                value = sum(predictions) / len(predictions) if predictions else 0.5
                concept_values.append(float(min(1.0, max(0.0, value))))
            students[str(row["uid"])] = concept_values

    payload = {
        "_meta": {
            "format": "learner_simulator_mikt_proficiency_v1",
            "concept_id_map": exported_concept_id_map,
            "question_skill_pairs": sum(len(value) for value in question_to_skills.values()),
            "checkpoint_unexpected_keys": unexpected,
        },
        "students": students,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return output


def _infer_mikt_config(
    state: dict[str, Any],
    checkpoint: dict[str, Any] | None = None,
) -> MIKTConfig:
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(config, dict):
        return MIKTConfig(
            ques_num=int(config["ques_num"]),
            skill_num=int(config["skill_num"]),
            embed_dim=int(config["embed_dim"]),
            max_seq_len=int(config["max_seq_len"]),
        )
    pro_embed = state.get("pro_embed")
    skill_embed = state.get("skill_embed")
    time_state = state.get("time_state")
    if time_state is None:
        time_state = state.get("pos_embed")
    if pro_embed is None or skill_embed is None or time_state is None:
        raise RuntimeError(
            "Unable to infer MIKT config from checkpoint. Expected pro_embed, skill_embed, and time_state/pos_embed."
        )
    return MIKTConfig(
        ques_num=int(pro_embed.shape[0]),
        skill_num=int(skill_embed.shape[0]),
        embed_dim=int(pro_embed.shape[1]),
        max_seq_len=int(time_state.shape[0]),
    )


def _build_question_skill_maps(
    rows: list[dict[str, str]],
    ques_num: int,
    skill_num: int,
    question_id_map: dict[str, int] | None = None,
    concept_id_map: dict[str, int] | None = None,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    question_to_skills: dict[int, set[int]] = {}
    skill_to_questions: dict[int, set[int]] = {}
    for row in rows:
        for step in clean_sequence(row):
            qid = _remap_id(step["qid"], question_id_map, upper_bound=ques_num)
            cid = _remap_id(step["cid"], concept_id_map, upper_bound=skill_num)
            if qid is None or cid is None:
                continue
            question_to_skills.setdefault(qid, set()).add(cid)
            skill_to_questions.setdefault(cid, set()).add(qid)
    return question_to_skills, skill_to_questions


def _build_problem_skill_tensor(
    question_to_skills: dict[int, set[int]],
    config: MIKTConfig,
):
    require_torch()
    tensor = torch.zeros((config.ques_num, config.skill_num), dtype=torch.float32)
    for qid, skills in question_to_skills.items():
        if qid < 0 or qid >= config.ques_num:
            continue
        for cid in skills:
            if 0 <= cid < config.skill_num:
                tensor[qid, cid] = 1.0
    return tensor


def _step_mikt_loss(model, batch, criterion, device):
    last_problem, last_ans, next_problem, next_ans, lengths = batch
    last_problem = last_problem.to(device)
    last_ans = last_ans.to(device)
    next_problem = next_problem.to(device)
    next_ans = next_ans.to(device)
    lengths = lengths.to(device)
    outputs, _ = model(last_problem, last_ans, next_problem, next_ans)
    positions = torch.arange(outputs.shape[1], device=device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    valid_outputs = outputs[valid]
    valid_target = next_ans[valid]
    return criterion(valid_outputs, valid_target), valid_outputs, valid_target


def _make_mikt_collate_fn():
    def _collate_fn(batch):
        max_length = max(len(sequence) - 1 for sequence, _ in batch)
        batch_size = len(batch)
        last_problem = torch.zeros((batch_size, max_length), dtype=torch.long)
        last_ans = torch.zeros((batch_size, max_length), dtype=torch.float32)
        next_problem = torch.zeros((batch_size, max_length), dtype=torch.long)
        next_ans = torch.zeros((batch_size, max_length), dtype=torch.float32)
        lengths = torch.zeros(batch_size, dtype=torch.long)

        for index, (sequence, labels) in enumerate(batch):
            length = len(sequence) - 1
            lengths[index] = length
            last_problem[index, :length] = torch.tensor(sequence[:-1], dtype=torch.long)
            last_ans[index, :length] = torch.tensor(labels[:-1], dtype=torch.float32)
            next_problem[index, :length] = torch.tensor(sequence[1:], dtype=torch.long)
            next_ans[index, :length] = torch.tensor(labels[1:], dtype=torch.float32)
        return last_problem, last_ans, next_problem, next_ans, lengths

    return _collate_fn


def _load_checkpoint_id_map(
    checkpoint: dict[str, Any] | None,
    key: str,
) -> dict[str, int]:
    if not isinstance(checkpoint, dict):
        return {}
    raw = checkpoint.get(key, {})
    if not isinstance(raw, dict):
        return {}
    return {str(item_key): int(item_value) for item_key, item_value in raw.items()}


def _build_raw_id_map(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    values: set[str] = set()
    for row in rows:
        for step in clean_sequence(row):
            values.add(str(step[field]))
    return {
        value: index for index, value in enumerate(sorted(values, key=lambda item: int(item)))
    }


def _remap_id(
    raw_value: Any,
    id_map: dict[str, int] | None,
    upper_bound: int,
) -> int | None:
    if id_map:
        mapped = id_map.get(str(raw_value))
        if mapped is None:
            return None
        return mapped if 0 <= mapped < upper_bound else None
    value = int(raw_value)
    return value if 0 <= value < upper_bound else None


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
