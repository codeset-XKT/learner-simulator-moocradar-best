from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from learner_simulator.action import build_statistical_action
from learner_simulator.behavior import (
    apply_behavior_adjustment,
    non_cognitive_factors,
    sample_error_type,
)
from learner_simulator.data import clean_sequence
from learner_simulator.dkt import DKTProficiency
from learner_simulator.dneuralcdm import DNeuralCDMProficiency
from learner_simulator.irt import IRTModel
from learner_simulator.mikt import MIKTProficiency
from learner_simulator.memory import LearnerMemory
from learner_simulator.profile import LearnerProfile, build_learner_profiles, default_profile


@dataclass
class RunningRate:
    correct: int = 0
    total: int = 0

    def update(self, response: int) -> None:
        self.correct += int(response == 1)
        self.total += 1

    def value(self, default: float) -> float:
        if self.total == 0:
            return default
        return self.correct / self.total


@dataclass
class LearnerState:
    uid: str
    mastery: dict[int, float] = field(default_factory=dict)

    def get_mastery(self, cid: int, default: float) -> float:
        return self.mastery.get(cid, default)

    def update(self, cid: int, response: int, learning_rate: float) -> None:
        old = self.mastery.get(cid, 0.5)
        target = 1.0 if response == 1 else 0.0
        self.mastery[cid] = old + learning_rate * (target - old)


class RandomLearnerSimulator:
    """Random/statistical baseline simulator.

    This is the minimal non-LLM simulator. It estimates response probabilities
    from user, item, concept, and dynamic mastery statistics, then samples a
    Bernoulli response.
    """

    def __init__(
        self,
        seed: int = 42,
        user_weight: float = 0.30,
        item_weight: float = 0.25,
        concept_weight: float = 0.25,
        mastery_weight: float = 0.20,
        irt_weight: float = 0.20,
        irt_epochs: int = 8,
        irt_learning_rate: float = 0.04,
        irt_l2: float = 0.001,
        learning_rate: float = 0.12,
        short_window: int = 5,
        long_threshold: int = 3,
        behavior_control: bool = True,
        dneuralcdm_proficiency_path: str | None = None,
        mikt_proficiency_path: str | None = None,
        dkt_proficiency_path: str | None = None,
    ) -> None:
        self.random = random.Random(seed)
        self.user_weight = user_weight
        self.item_weight = item_weight
        self.concept_weight = concept_weight
        self.mastery_weight = mastery_weight
        self.irt_weight = irt_weight
        self.learning_rate = learning_rate
        self.short_window = short_window
        self.long_threshold = long_threshold
        self.behavior_control = behavior_control
        self.dneuralcdm_proficiency = DNeuralCDMProficiency(dneuralcdm_proficiency_path)
        self.mikt_proficiency = MIKTProficiency(mikt_proficiency_path)
        self.dkt_proficiency = DKTProficiency(dkt_proficiency_path)
        self.global_rate = RunningRate()
        self.user_rates: dict[str, RunningRate] = defaultdict(RunningRate)
        self.item_rates: dict[int, RunningRate] = defaultdict(RunningRate)
        self.concept_rates: dict[int, RunningRate] = defaultdict(RunningRate)
        self.profiles: dict[str, LearnerProfile] = {}
        self.concept_labels: list[str] = []
        self.irt_model = IRTModel(
            epochs=irt_epochs,
            learning_rate=irt_learning_rate,
            l2=irt_l2,
            seed=seed,
        )

    def fit(
        self,
        rows: list[dict[str, str]],
        questions: dict[str, dict[str, Any]] | None = None,
        profile_normalization_rows: list[dict[str, str]] | None = None,
    ) -> None:
        for row in rows:
            uid = row["uid"]
            for step in clean_sequence(row):
                response = step["response"]
                qid = step["qid"]
                cid = step["cid"]
                self.global_rate.update(response)
                self.user_rates[uid].update(response)
                self.item_rates[qid].update(response)
                self.concept_rates[cid].update(response)
        self.irt_model.fit(rows)
        self.concept_labels = sorted(
            {
                str(routes[0])
                for question in (questions or {}).values()
                for routes in [question.get("kc_routes") or []]
                if routes
            }
        )
        self.profiles = build_learner_profiles(
            rows,
            questions,
            ability_estimator=self.irt_model,
            normalization_rows=profile_normalization_rows,
        )

    def predict_probability(self, uid: str, qid: int, cid: int, state: LearnerState) -> float:
        default = self.global_rate.value(0.5)
        user_rate = self.user_rates[uid].value(default)
        item_rate = self.item_rates[qid].value(default)
        concept_rate = self.concept_rates[cid].value(default)
        mastery = state.get_mastery(cid, concept_rate)
        irt_probability = self.irt_model.predict(uid, qid)

        weights = [
            self.user_weight,
            self.item_weight,
            self.concept_weight,
            self.mastery_weight,
            self.irt_weight,
        ]
        total_weight = sum(max(0.0, weight) for weight in weights)
        if total_weight <= 0:
            total_weight = 1.0
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]

        probability = (
            max(0.0, weights[0]) * user_rate
            + max(0.0, weights[1]) * item_rate
            + max(0.0, weights[2]) * concept_rate
            + max(0.0, weights[3]) * mastery
            + max(0.0, weights[4]) * irt_probability
        ) / total_weight
        return min(0.98, max(0.02, probability))

    def probability_components(
        self,
        uid: str,
        qid: int,
        cid: int,
        state: LearnerState,
    ) -> dict[str, float]:
        default = self.global_rate.value(0.5)
        concept_rate = self.concept_rates[cid].value(default)
        return {
            "user_rate": round(self.user_rates[uid].value(default), 6),
            "item_rate": round(self.item_rates[qid].value(default), 6),
            "concept_rate": round(concept_rate, 6),
            "mastery": round(state.get_mastery(cid, concept_rate), 6),
            "mastery_source": (
                self._mastery_source(uid, cid)
                if self._mastery_source(uid, cid) is not None
                else "dynamic"
            ),
            "irt_probability": round(self.irt_model.predict(uid, qid), 6),
            "irt_theta": round(self.irt_model.user_theta(uid), 6),
            "irt_beta": round(self.irt_model.item_beta(qid), 6),
        }

    def simulate_sequence(
        self,
        row: dict[str, str],
        questions: dict[str, dict[str, Any]],
        history_row: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        uid = row["uid"]
        state, memory = self.initialize_from_history(
            uid,
            history_row,
            questions,
        )
        profile = self.get_profile(uid)
        simulated_steps = []

        for step in clean_sequence(row):
            qid = step["qid"]
            cid = step["cid"]
            real_response = step["response"]
            components = self.probability_components(uid, qid, cid, state)
            cognitive_probability = self.predict_probability(uid, qid, cid, state)
            mastery_before = components["mastery"]
            factors = non_cognitive_factors(profile.to_context(), int(step.get("position") or 0), self.random)
            probability = (
                apply_behavior_adjustment(cognitive_probability, factors)
                if self.behavior_control
                else cognitive_probability
            )
            simulated_response = int(self.random.random() < probability)
            error_type = "none" if simulated_response == 1 else sample_error_type(components, factors, self.random)
            state.update(cid, simulated_response, learning_rate=self.learning_rate)

            qmeta = questions.get(str(qid), {})
            kc_routes = qmeta.get("kc_routes", [])
            memory_context = memory.snapshot(cid, kc_routes, mastery_before)
            step_result = {
                "uid": uid,
                "source": "simulated",
                "step_index": step.get("position"),
                "timestamp": step.get("timestamp"),
                "qid": qid,
                "cid": cid,
                "p_cognitive": round(cognitive_probability, 6),
                "p_correct": round(probability, 6),
                "probability_components": components,
                "non_cognitive_factors": factors,
                "statistical_sampled_response": simulated_response,
                "statistical_sampled_error_type": error_type,
                "irt_probability": components["irt_probability"],
                "irt_theta": components["irt_theta"],
                "irt_beta": components["irt_beta"],
                "simulated_response": simulated_response,
                "real_response": real_response,
                "question_type": qmeta.get("type"),
                "content": qmeta.get("content"),
                "options": qmeta.get("options"),
                "answer": qmeta.get("answer"),
                "analysis": qmeta.get("analysis"),
                "content_preview": str(qmeta.get("content", ""))[:80],
                "kc_routes": kc_routes,
                "learner_profile": profile.to_context(),
                "memory_context": memory_context,
                "agent_action": build_statistical_action(
                    cid=cid,
                    kc_routes=kc_routes,
                    probability=probability,
                    simulated_response=simulated_response,
                    answer=qmeta.get("answer"),
                    error_type=error_type,
                ),
            }
            memory.observe(step_result)
            simulated_steps.append(step_result)

        summary = build_sequence_summary(uid, simulated_steps)
        summary["history_interactions"] = (
            len(clean_sequence(history_row)) if history_row is not None else 0
        )
        summary["target_interactions"] = len(simulated_steps)
        return summary

    def initialize_from_history(
        self,
        uid: str,
        history_row: dict[str, str] | None,
        questions: dict[str, dict[str, Any]],
    ) -> tuple[LearnerState, LearnerMemory]:
        """Initialize dynamic mastery and memory from observed interactions."""

        state = LearnerState(uid=uid)
        memory = LearnerMemory(
            uid=uid,
            short_window=self.short_window,
            long_threshold=self.long_threshold,
        )
        if history_row is None:
            return state, memory

        history = clean_sequence(history_row)
        offset = -len(history)
        for index, step in enumerate(history):
            qid = step["qid"]
            cid = step["cid"]
            response = int(step["response"])
            state.update(cid, response, learning_rate=self.learning_rate)
            qmeta = questions.get(str(qid), {})
            memory.observe(
                {
                    "uid": uid,
                    "source": "observed_history",
                    "step_index": offset + index,
                    "timestamp": step.get("timestamp"),
                    "qid": qid,
                    "cid": cid,
                    "kc_routes": qmeta.get("kc_routes", []),
                    "question_type": qmeta.get("type"),
                    "content": qmeta.get("content"),
                    "content_preview": str(qmeta.get("content", ""))[:80],
                    "simulated_response": response,
                    "real_response": response,
                    "p_correct": None,
                }
            )
        self._seed_state_from_external_proficiency(uid, state)
        return state, memory

    def get_profile(self, uid: str) -> LearnerProfile:
        return self.profiles.get(uid) or default_profile(uid, self.global_rate.value(0.5))

    def concept_options(self, true_concept: str, seed: int, count: int = 3) -> list[str]:
        candidates = [label for label in self.concept_labels if label != true_concept]
        rng = random.Random(seed)
        distractors = rng.sample(candidates, k=min(max(0, count - 1), len(candidates)))
        options = [true_concept] + distractors
        while len(options) < count:
            options.append("unknown knowledge concept")
        rng.shuffle(options)
        return options

    def summary(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "global_correct_rate": round(self.global_rate.value(0.0), 4),
            "observed_users": len(self.user_rates),
            "observed_items": len(self.item_rates),
            "observed_concepts": len(self.concept_rates),
            "hyperparameters": {
                "user_weight": self.user_weight,
                "item_weight": self.item_weight,
                "concept_weight": self.concept_weight,
                "mastery_weight": self.mastery_weight,
                "irt_weight": self.irt_weight,
                "learning_rate": self.learning_rate,
                "short_window": self.short_window,
                "long_threshold": self.long_threshold,
                "behavior_control": self.behavior_control,
            },
            "irt_summary": self.irt_model.summary(),
            "dneuralcdm_proficiency_available": self.dneuralcdm_proficiency.available(),
            "mikt_proficiency_available": self.mikt_proficiency.available(),
            "dkt_proficiency_available": self.dkt_proficiency.available(),
            "external_proficiency_source": self._external_proficiency_source(),
        }

    def _seed_state_from_external_proficiency(self, uid: str, state: LearnerState) -> None:
        if self.mikt_proficiency.available():
            for cid, value in self.mikt_proficiency.latest_values(uid).items():
                state.mastery[int(cid)] = min(1.0, max(0.0, float(value)))
            return
        if self.dkt_proficiency.available():
            for cid, value in self.dkt_proficiency.latest_values(uid).items():
                state.mastery[int(cid)] = min(1.0, max(0.0, float(value)))
            return
        if not self.dneuralcdm_proficiency.available():
            return
        for cid, value in self.dneuralcdm_proficiency.latest_values(uid).items():
            state.mastery[int(cid)] = min(1.0, max(0.0, float(value)))

    def _external_proficiency_source(self) -> str | None:
        if self.mikt_proficiency.available():
            return "mikt"
        if self.dkt_proficiency.available():
            return "dkt"
        if self.dneuralcdm_proficiency.available():
            return "dneuralcdm"
        return None

    def _mastery_source(self, uid: str, cid: int) -> str | None:
        if self.mikt_proficiency.available() and self.mikt_proficiency.value(uid, cid) is not None:
            return "mikt"
        if self.dkt_proficiency.available() and self.dkt_proficiency.value(uid, cid) is not None:
            return "dkt"
        if (
            self.dneuralcdm_proficiency.available()
            and self.dneuralcdm_proficiency.value(uid, cid) is not None
        ):
            return "dneuralcdm"
        return None


def build_sequence_summary(uid: str, simulated_steps: list[dict[str, Any]]) -> dict[str, Any]:
    if simulated_steps:
        sim_acc = sum(x["simulated_response"] for x in simulated_steps) / len(simulated_steps)
        real_acc = sum(x["real_response"] for x in simulated_steps) / len(simulated_steps)
    else:
        sim_acc = 0.0
        real_acc = 0.0

    return {
        "uid": uid,
        "steps": simulated_steps,
        "simulated_accuracy": round(sim_acc, 4),
        "real_accuracy": round(real_acc, 4),
    }
