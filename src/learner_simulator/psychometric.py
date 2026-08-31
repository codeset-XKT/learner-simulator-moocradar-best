"""Leakage-safe classical IRT and multidimensional IRT reference models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def require_torch():
    if torch is None or F is None:
        raise RuntimeError("Psychometric reference training requires PyTorch")


def _sigmoid(value: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


@dataclass
class IRTReference:
    item_id_map: dict[str, int]
    beta: list[float]
    prior_theta: float = 0.0
    learning_rate: float = 0.1

    def state_from_history(self, history: list[dict[str, Any]]) -> float:
        theta = self.prior_theta
        for step in history:
            theta = self.update(theta, step, int(step["response"]))
        return theta

    def predict(self, theta: float, step: dict[str, Any]) -> float:
        index = self.item_id_map.get(str(step["qid"]))
        beta = self.beta[index] if index is not None else 0.0
        return _sigmoid(theta - beta)

    def update(self, theta: float, step: dict[str, Any], response: int) -> float:
        return max(-5.0, min(5.0, theta + self.learning_rate * (int(response) - self.predict(theta, step)) - 0.002 * theta))

    def to_dict(self):
        return {"method": "irt_1pl", "item_id_map": self.item_id_map, "beta": self.beta, "prior_theta": self.prior_theta, "learning_rate": self.learning_rate}

    @classmethod
    def from_dict(cls, data):
        return cls({str(k): int(v) for k, v in data["item_id_map"].items()}, [float(x) for x in data["beta"]], float(data.get("prior_theta", 0.0)), float(data.get("learning_rate", 0.1)))


@dataclass
class MIRTReference:
    item_id_map: dict[str, int]
    discrimination: list[list[float]]
    difficulty: list[float]
    learning_rate: float = 0.08

    def state_from_history(self, history: list[dict[str, Any]]) -> list[float]:
        theta = [0.0] * len(self.discrimination[0])
        for step in history:
            theta = self.update(theta, step, int(step["response"]))
        return theta

    def predict(self, theta: list[float], step: dict[str, Any]) -> float:
        index = self.item_id_map.get(str(step["qid"]))
        if index is None:
            return 0.5
        return _sigmoid(sum(a * t for a, t in zip(self.discrimination[index], theta)) - self.difficulty[index])

    def update(self, theta: list[float], step: dict[str, Any], response: int) -> list[float]:
        index = self.item_id_map.get(str(step["qid"]))
        if index is None:
            return theta
        error = int(response) - self.predict(theta, step)
        return [max(-5.0, min(5.0, value + self.learning_rate * (a * error - 0.002 * value))) for value, a in zip(theta, self.discrimination[index])]

    def to_dict(self):
        return {"method": "mirt_2pl", "item_id_map": self.item_id_map, "discrimination": self.discrimination, "difficulty": self.difficulty, "learning_rate": self.learning_rate}

    @classmethod
    def from_dict(cls, data):
        return cls({str(k): int(v) for k, v in data["item_id_map"].items()}, [[float(v) for v in row] for row in data["discrimination"]], [float(v) for v in data["difficulty"]], float(data.get("learning_rate", 0.08)))


def fit_psychometric(interactions: list[tuple[str, str, int]], dimensions: int = 1, epochs: int = 30, batch_size: int = 4096, lr: float = 0.03, seed: int = 42):
    require_torch()
    torch.manual_seed(seed)
    users = {uid: index for index, uid in enumerate(sorted({x[0] for x in interactions}))}
    item_map = {qid: index for index, qid in enumerate(sorted({x[1] for x in interactions}, key=int))}
    uid = torch.tensor([users[x[0]] for x in interactions], dtype=torch.long)
    item = torch.tensor([item_map[x[1]] for x in interactions], dtype=torch.long)
    label = torch.tensor([x[2] for x in interactions], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uid, item, label = uid.to(device), item.to(device), label.to(device)
    theta = torch.nn.Embedding(len(users), dimensions).to(device)
    difficulty = torch.nn.Embedding(len(item_map), 1).to(device)
    params = list(theta.parameters()) + list(difficulty.parameters())
    discrimination = None
    if dimensions > 1:
        discrimination = torch.nn.Embedding(len(item_map), dimensions).to(device)
        torch.nn.init.normal_(discrimination.weight, mean=0.5, std=0.1)
        params += list(discrimination.parameters())
    torch.nn.init.zeros_(theta.weight); torch.nn.init.zeros_(difficulty.weight)
    optimizer = torch.optim.Adam(params, lr=lr)
    order = torch.arange(len(interactions), device=device)
    for _ in range(epochs):
        order = order[torch.randperm(len(order), device=device)]
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            if dimensions == 1:
                logits = theta(uid[ids]).squeeze(-1) - difficulty(item[ids]).squeeze(-1)
            else:
                logits = (theta(uid[ids]) * discrimination(item[ids])).sum(-1) - difficulty(item[ids]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, label[ids]) + 1e-4 * sum((p ** 2).mean() for p in params)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    beta = difficulty.weight.detach().cpu().squeeze(-1).tolist()
    if dimensions == 1:
        return IRTReference(item_map, [float(v) for v in beta]), {"users": len(users), "items": len(item_map), "interactions": len(interactions), "dimensions": 1}
    return MIRTReference(item_map, discrimination.weight.detach().cpu().tolist(), [float(v) for v in beta]), {"users": len(users), "items": len(item_map), "interactions": len(interactions), "dimensions": dimensions}
