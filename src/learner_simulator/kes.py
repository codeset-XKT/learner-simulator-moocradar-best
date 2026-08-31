from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from learner_simulator.dkt import DKT, require_torch, torch


class KnowledgeEvolutionSimulator:
    """DKT-driven Knowledge Evolution-based Simulator (KES).

    KES uses the probability assigned by a DKT state to the next item's
    knowledge concept as the learner's response probability.  In rollout
    mode, sampled simulated feedback evolves the state; in teacher-forcing
    mode, observed prior feedback evolves the state for fair replay studies.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        require_torch()
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.concept_id_map = {
            str(key): int(value)
            for key, value in self.checkpoint["concept_id_map"].items()
        }
        self.model = DKT(
            self.checkpoint["num_concepts"],
            hidden_dim=int(self.checkpoint.get("hidden_dim", 100)),
            dropout=float(self.checkpoint.get("dropout", 0.2)),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    @property
    def training_metadata(self) -> dict[str, Any]:
        return dict(self.checkpoint.get("training_metadata") or {})

    def simulate(
        self,
        history: list[dict[str, Any]],
        target: list[dict[str, Any]],
        feedback_mode: str = "teacher-forcing",
        threshold: float = 0.5,
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], int]:
        if feedback_mode not in {"teacher-forcing", "rollout"}:
            raise ValueError(f"Unsupported KES feedback mode: {feedback_mode}")
        encoded_history = [self._encoded_step(step) for step in history]
        encoded_history = [step for step in encoded_history if step is not None]
        if not encoded_history:
            raise ValueError("KES needs at least one DKT-mapped history interaction")
        generator = random.Random(seed)
        skipped = 0
        with torch.no_grad():
            state, logits = self._initial_state(encoded_history)
            outputs: list[dict[str, Any]] = []
            for step in target:
                concept_index = self.concept_id_map.get(str(step["cid"]))
                if concept_index is None:
                    skipped += 1
                    continue
                probability = float(torch.sigmoid(logits)[0, concept_index].item())
                simulated_response = (
                    int(generator.random() < probability)
                    if feedback_mode == "rollout"
                    else int(probability >= threshold)
                )
                update_response = (
                    int(step["response"])
                    if feedback_mode == "teacher-forcing"
                    else simulated_response
                )
                state, logits = self._advance(state, concept_index, update_response)
                outputs.append({
                    "uid": str(step.get("uid")),
                    "step_index": step.get("position"),
                    "qid": step.get("qid"),
                    "real_response": int(step["response"]),
                    "simulated_response": simulated_response,
                    "kes_probability": probability,
                    "prediction_source": f"kes_dkt_{feedback_mode}",
                })
        return outputs, skipped

    def _encoded_step(self, step: dict[str, Any]) -> tuple[int, int] | None:
        concept_index = self.concept_id_map.get(str(step["cid"]))
        if concept_index is None:
            return None
        return concept_index, int(step["response"])

    def _interaction_tensor(self, concept_index: int, response: int):
        interaction = torch.zeros(
            (1, 1, self.checkpoint["num_concepts"] * 2),
            dtype=torch.float32,
            device=self.device,
        )
        interaction[0, 0, (2 * concept_index) + (0 if response == 1 else 1)] = 1.0
        return interaction

    def _initial_state(self, encoded_history: list[tuple[int, int]]):
        interactions = torch.cat(
            [self._interaction_tensor(concept, response) for concept, response in encoded_history],
            dim=1,
        )
        outputs, state = self.model.lstm(interactions)
        logits = self.model.output(outputs[:, -1, :])
        return state, logits

    def _advance(self, state, concept_index: int, response: int):
        outputs, state = self.model.lstm(self._interaction_tensor(concept_index, response), state)
        logits = self.model.output(outputs[:, -1, :])
        return state, logits
