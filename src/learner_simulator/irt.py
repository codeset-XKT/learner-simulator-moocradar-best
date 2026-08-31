from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from learner_simulator.data import clean_sequence


@dataclass
class IRTModel:
    """Small Rasch/1PL IRT estimator.

    It estimates learner ability theta_u and item difficulty beta_i from binary
    response logs:

        p(correct | u, i) = sigmoid(theta_u - beta_i)

    This implementation is intentionally dependency-free so the project remains
    easy to run. It is a baseline estimator, not a replacement for a full IRT
    training pipeline with validation and convergence diagnostics.
    """

    epochs: int = 8
    learning_rate: float = 0.04
    l2: float = 0.001
    seed: int = 42
    clip_value: float = 4.0
    theta: dict[str, float] = field(default_factory=dict)
    beta: dict[int, float] = field(default_factory=dict)
    global_rate: float = 0.5
    fitted_interactions: int = 0

    def fit(self, rows: list[dict[str, str]]) -> None:
        interactions = _collect_interactions(rows)
        self.fitted_interactions = len(interactions)
        if not interactions:
            self.global_rate = 0.5
            return

        self.global_rate = sum(response for _, _, response in interactions) / len(interactions)
        rng = random.Random(self.seed)
        for uid, qid, _ in interactions:
            self.theta.setdefault(uid, _logit(self.global_rate))
            self.beta.setdefault(qid, 0.0)

        for epoch in range(max(0, self.epochs)):
            rng.shuffle(interactions)
            lr = self.learning_rate / math.sqrt(epoch + 1)
            for uid, qid, response in interactions:
                theta = self.theta[uid]
                beta = self.beta[qid]
                prediction = _sigmoid(theta - beta)
                error = response - prediction

                theta += lr * (error - self.l2 * theta)
                beta += lr * (-error - self.l2 * beta)
                self.theta[uid] = _clip(theta, self.clip_value)
                self.beta[qid] = _clip(beta, self.clip_value)

        self._center_parameters()

    def predict(self, uid: str, qid: int) -> float:
        theta = self.theta.get(uid, _logit(self.global_rate))
        beta = self.beta.get(qid, 0.0)
        return min(0.98, max(0.02, _sigmoid(theta - beta)))

    def ability_score(self, uid: str) -> float:
        theta = self.theta.get(uid, _logit(self.global_rate))
        return min(0.98, max(0.02, _sigmoid(theta)))

    def item_difficulty_score(self, qid: int) -> float:
        beta = self.beta.get(qid, 0.0)
        return min(0.98, max(0.02, _sigmoid(beta)))

    def user_theta(self, uid: str) -> float:
        return self.theta.get(uid, _logit(self.global_rate))

    def item_beta(self, qid: int) -> float:
        return self.beta.get(qid, 0.0)

    def summary(self) -> dict[str, Any]:
        return {
            "type": "Rasch1PL",
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "fitted_interactions": self.fitted_interactions,
            "observed_users": len(self.theta),
            "observed_items": len(self.beta),
            "global_rate": round(self.global_rate, 4),
        }

    def _center_parameters(self) -> None:
        if not self.beta:
            return
        mean_beta = sum(self.beta.values()) / len(self.beta)
        for qid in list(self.beta):
            self.beta[qid] = _clip(self.beta[qid] - mean_beta, self.clip_value)
        for uid in list(self.theta):
            self.theta[uid] = _clip(self.theta[uid] - mean_beta, self.clip_value)


def _collect_interactions(rows: list[dict[str, str]]) -> list[tuple[str, int, int]]:
    interactions: list[tuple[str, int, int]] = []
    for row in rows:
        uid = row["uid"]
        for step in clean_sequence(row):
            interactions.append((uid, int(step["qid"]), int(step["response"])))
    return interactions


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(probability: float) -> float:
    p = min(0.98, max(0.02, probability))
    return math.log(p / (1.0 - p))


def _clip(value: float, bound: float) -> float:
    return min(bound, max(-bound, value))
