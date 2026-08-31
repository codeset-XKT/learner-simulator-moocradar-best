"""Interpretable Knowledge Tracing (IKT) reference implementation.

The implementation preserves IKT's three separable, human-readable inputs:
item difficulty decile, per-skill BKT mastery decile, and a cross-skill
ability-profile cluster.  A Tree-Augmented Naive Bayes (TAN) classifier then
combines them.  The feature calculation at a step only uses earlier feedback.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from learner_simulator.data import clean_sequence

try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError:  # pragma: no cover
    KMeans = None


@dataclass(frozen=True)
class BKTParameters:
    initial: float = 0.5
    learn: float = 0.15
    guess: float = 0.2
    slip: float = 0.1


def _clip(value: float, low: float = 1e-5, high: float = 1.0 - 1e-5) -> float:
    return min(high, max(low, value))


def _bkt_update(mastery: float, response: int, params: BKTParameters) -> float:
    if response:
        posterior = mastery * (1.0 - params.slip) / (mastery * (1.0 - params.slip) + (1.0 - mastery) * params.guess)
    else:
        posterior = mastery * params.slip / (mastery * params.slip + (1.0 - mastery) * (1.0 - params.guess))
    return _clip(posterior + (1.0 - posterior) * params.learn)


@dataclass
class IKTStudentState:
    mastery: dict[str, float]
    correct: int = 0
    attempts: int = 0
    recent: list[int] | None = None

    @classmethod
    def empty(cls) -> "IKTStudentState":
        return cls(mastery={}, recent=[])

    def profile_vector(self) -> list[float]:
        recent = (self.recent or [])[-10:]
        return [
            self.correct / max(self.attempts, 1),
            min(self.attempts / 20.0, 1.0),
            sum(recent) / max(len(recent), 1),
            sum(self.mastery.values()) / max(len(self.mastery), 1),
        ]


class TANClassifier:
    """Small categorical Tree-Augmented Naive Bayes classifier."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.parents: list[int | None] = []
        self.cardinalities: list[int] = []
        self.class_counts = [0, 0]
        self.counts: list[dict[tuple[int, int, int | None], int]] = []

    def fit(self, rows: list[list[int]], labels: list[int], cardinalities: list[int]) -> None:
        if not rows or len(rows) != len(labels):
            raise ValueError("TAN requires equally-sized non-empty feature rows and labels")
        self.cardinalities = list(cardinalities)
        self.class_counts = [labels.count(0), labels.count(1)]
        n_features = len(cardinalities)
        weights = [[0.0] * n_features for _ in range(n_features)]
        for left in range(n_features):
            for right in range(left + 1, n_features):
                weight = _conditional_mutual_information(rows, labels, left, right, cardinalities)
                weights[left][right] = weights[right][left] = weight
        # Maximum spanning tree; deterministic root keeps serialized models stable.
        self.parents = [None] * n_features
        selected = {0}
        while len(selected) < n_features:
            edge = max(
                ((weights[parent][child], parent, child) for parent in selected for child in range(n_features) if child not in selected),
                key=lambda value: (value[0], -value[1], -value[2]),
            )
            _, parent, child = edge
            self.parents[child] = parent
            selected.add(child)
        self.counts = [defaultdict(int) for _ in range(n_features)]
        for values, label in zip(rows, labels):
            for index, value in enumerate(values):
                parent = self.parents[index]
                parent_value = values[parent] if parent is not None else None
                self.counts[index][(int(label), int(value), parent_value)] += 1

    def predict_proba(self, values: list[int]) -> float:
        if not self.cardinalities:
            raise RuntimeError("TAN classifier is not fitted")
        totals = []
        total = sum(self.class_counts)
        for label in (0, 1):
            logp = math.log((self.class_counts[label] + self.alpha) / (total + 2 * self.alpha))
            for index, value in enumerate(values):
                parent = self.parents[index]
                parent_value = values[parent] if parent is not None else None
                numerator = self.counts[index].get((label, int(value), parent_value), 0) + self.alpha
                if parent is None:
                    denominator = self.class_counts[label] + self.alpha * self.cardinalities[index]
                else:
                    parent_count = sum(
                        self.counts[index].get((label, possible, parent_value), 0)
                        for possible in range(self.cardinalities[index])
                    )
                    denominator = parent_count + self.alpha * self.cardinalities[index]
                logp += math.log(numerator / denominator)
            totals.append(logp)
        maximum = max(totals)
        values_exp = [math.exp(value - maximum) for value in totals]
        return values_exp[1] / sum(values_exp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha, "parents": self.parents, "cardinalities": self.cardinalities,
            "class_counts": self.class_counts,
            "counts": [[[*key, value] for key, value in table.items()] for table in self.counts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TANClassifier":
        result = cls(float(data["alpha"]))
        result.parents = [None if value is None else int(value) for value in data["parents"]]
        result.cardinalities = [int(value) for value in data["cardinalities"]]
        result.class_counts = [int(value) for value in data["class_counts"]]
        result.counts = []
        for table in data["counts"]:
            values = defaultdict(int)
            for label, value, parent, count in table:
                values[(int(label), int(value), None if parent is None else int(parent))] = int(count)
            result.counts.append(values)
        return result


def _conditional_mutual_information(rows, labels, left, right, cardinalities) -> float:
    count_xyz = Counter((row[left], row[right], label) for row, label in zip(rows, labels))
    count_xy = Counter((row[left], label) for row, label in zip(rows, labels))
    count_yz = Counter((row[right], label) for row, label in zip(rows, labels))
    count_y = Counter(labels)
    total = len(rows)
    result = 0.0
    for (x, z, label), count in count_xyz.items():
        p_xyz = count / total
        ratio = (count * count_y[label]) / max(count_xy[(x, label)] * count_yz[(z, label)], 1)
        result += p_xyz * math.log(max(ratio, 1e-12))
    return result


class IKTModel:
    """Serializable IKT state model with BKT + ability profile + TAN."""

    feature_cardinalities = [10, 10, 5]

    def __init__(self, bkt: BKTParameters | None = None) -> None:
        self.bkt = bkt or BKTParameters()
        self.item_difficulty_bin: dict[str, int] = {}
        self.profile_centers: list[list[float]] = []
        self.tan = TANClassifier()

    def _profile_group(self, state: IKTStudentState) -> int:
        vector = state.profile_vector()
        if not self.profile_centers:
            return 0
        return min(range(len(self.profile_centers)), key=lambda index: sum((a - b) ** 2 for a, b in zip(vector, self.profile_centers[index])))

    def features(self, state: IKTStudentState, step: dict[str, Any]) -> list[int]:
        cid = str(step["cid"])
        qid = str(step["qid"])
        mastery = state.mastery.get(cid, self.bkt.initial)
        return [
            self.item_difficulty_bin.get(qid, 4),
            min(9, int(_clip(mastery, 0.0, 0.999999) * 10)),
            min(4, self._profile_group(state)),
        ]

    def predict(self, state: IKTStudentState, step: dict[str, Any]) -> float:
        return self.tan.predict_proba(self.features(state, step))

    def update(self, state: IKTStudentState, step: dict[str, Any], response: int) -> None:
        cid = str(step["cid"])
        state.mastery[cid] = _bkt_update(state.mastery.get(cid, self.bkt.initial), int(response), self.bkt)
        state.correct += int(response)
        state.attempts += 1
        (state.recent or []).append(int(response))

    def fit(self, rows: list[dict[str, str]], seed: int = 42) -> dict[str, Any]:
        item_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            for step in clean_sequence(row):
                item_stats[str(step["qid"])][0] += int(step["response"])
                item_stats[str(step["qid"])][1] += 1
        # Higher bin means harder item; ten ordered bins as in IKT.
        rates = {qid: (correct + 1) / (count + 2) for qid, (correct, count) in item_stats.items()}
        ordered = sorted(rates, key=lambda qid: rates[qid])
        self.item_difficulty_bin = {qid: min(9, int(index * 10 / max(len(ordered), 1))) for index, qid in enumerate(ordered)}
        snapshots = []
        raw_features = []
        labels = []
        for row in rows:
            state = IKTStudentState.empty()
            for step in clean_sequence(row):
                snapshots.append(state.profile_vector())
                raw_features.append((state, step))
                labels.append(int(step["response"]))
                self.update(state, step, int(step["response"]))
        if KMeans is None:
            raise RuntimeError("IKT requires scikit-learn for its ability-profile clustering")
        clusters = min(5, max(1, len(snapshots)))
        kmeans = KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit(snapshots)
        self.profile_centers = [[float(value) for value in center] for center in kmeans.cluster_centers_.tolist()]
        # Rebuild the states so profile group uses the fitted cluster centers.
        feature_rows = []
        labels = []
        for row in rows:
            state = IKTStudentState.empty()
            for step in clean_sequence(row):
                feature_rows.append(self.features(state, step))
                labels.append(int(step["response"]))
                self.update(state, step, int(step["response"]))
        self.tan.fit(feature_rows, labels, [10, 10, clusters])
        return {"interactions": len(labels), "items": len(self.item_difficulty_bin), "ability_groups": clusters, "tan_parents": self.tan.parents}

    def to_dict(self) -> dict[str, Any]:
        return {"format": "learner_simulator_ikt_v1", "bkt": asdict(self.bkt), "item_difficulty_bin": self.item_difficulty_bin, "profile_centers": self.profile_centers, "tan": self.tan.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IKTModel":
        result = cls(BKTParameters(**data["bkt"]))
        result.item_difficulty_bin = {str(key): int(value) for key, value in data["item_difficulty_bin"].items()}
        result.profile_centers = [[float(x) for x in values] for values in data["profile_centers"]]
        result.tan = TANClassifier.from_dict(data["tan"])
        return result

    def save(self, path: str | Path, metadata: dict[str, Any]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["training_metadata"] = metadata
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output


def load_ikt(path: str | Path) -> tuple[IKTModel, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return IKTModel.from_dict(data), dict(data.get("training_metadata") or {})
