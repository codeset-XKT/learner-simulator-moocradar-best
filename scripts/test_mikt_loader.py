from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.mikt import MIKTProficiency  # noqa: E402
from learner_simulator.simulators.random_simulator import RandomLearnerSimulator  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mikt_know_proficiency.json"
        payload = {
            "_meta": {
                "concept_id_map": {"10": 0, "20": 1},
            },
            "students": {
                "u1": [0.3, 0.85],
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        proficiency = MIKTProficiency(path)
        assert proficiency.available()
        assert proficiency.value("u1", 10) == 0.3
        assert proficiency.value("u1", 20) == 0.85
        assert proficiency.latest_values("u1") == {10: 0.3, 20: 0.85}

        simulator = RandomLearnerSimulator(mikt_proficiency_path=str(path))
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
        assert round(state.get_mastery(10, 0.5), 4) == 0.3
        assert round(state.get_mastery(20, 0.5), 4) == 0.85
        assert simulator.summary()["external_proficiency_source"] == "mikt"
    print("mikt_loader_test_ok")


if __name__ == "__main__":
    main()
