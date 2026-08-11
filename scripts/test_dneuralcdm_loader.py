from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.dneuralcdm import DNeuralCDMProficiency  # noqa: E402
from learner_simulator.simulators.random_simulator import RandomLearnerSimulator  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stu_know_proficiency.json"
        payload = {
            "_meta": {
                "concept_id_map": {"10": 0, "20": 1},
                "exercise_id_map": {"1": 0, "2": 1},
                "checkpoint_sha256": "abc123",
                "state_alignment": "one_latent_state_after_each_observed_response",
            },
            "students": {
                "u1": [
                    [0.2, 0.8],
                    [0.4, 0.9],
                ]
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        proficiency = DNeuralCDMProficiency(path)
        assert proficiency.available()
        assert proficiency.value("u1", 10) == 0.4
        assert proficiency.value("u1", 20, time_step=0) == 0.8
        assert proficiency.latest_values("u1") == {10: 0.4, 20: 0.9}
        assert proficiency.sequence_length("u1") == 2
        assert proficiency.checkpoint_sha256() == "abc123"

        simulator = RandomLearnerSimulator(dneuralcdm_proficiency_path=str(path))
        state, _ = simulator.initialize_from_history(
            "u1",
            {
                "uid": "u1",
                "questions": "1,2",
                "concepts": "10,20",
                "responses": "1,0",
                "timestamps": "1,2",
            },
            questions={},
        )
        assert round(state.get_mastery(10, 0.5), 4) == 0.4
        assert round(state.get_mastery(20, 0.5), 4) == 0.9
    print("dneuralcdm_loader_test_ok")


if __name__ == "__main__":
    main()
