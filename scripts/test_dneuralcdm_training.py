from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.dneuralcdm import (  # noqa: E402
    DNeuralCDM,
    _step_loss,
    dneuralcdm_model_from_checkpoint,
    nn,
    torch,
)


class FixedOutputModel(nn.Module):
    def forward(self, sequences, exercise_ids, masks):
        outputs = torch.tensor(
            [[0.9, 0.8], [0.2, 0.99]],
            dtype=torch.float32,
            device=sequences.device,
        )
        return outputs, None, None


def main() -> None:
    model = DNeuralCDM(2, 2, 2, 2)
    with torch.no_grad():
        model.exer_disc.weight.fill_(-10.0)
        model.exer_disc.bias.fill_(-10.0)
        raw = model.exer_disc(torch.eye(2))
        assert torch.all(torch.sigmoid(raw) > 0)

    legacy = dneuralcdm_model_from_checkpoint(
        {
            "num_exercises": 2,
            "num_know": 2,
            "embedding_dim": 2,
            "hidden_dim": 2,
        }
    )
    current = dneuralcdm_model_from_checkpoint(
        {
            "num_exercises": 2,
            "num_know": 2,
            "embedding_dim": 2,
            "hidden_dim": 2,
            "architecture_version": 2,
        }
    )
    assert legacy.positive_discrimination is False
    assert current.positive_discrimination is True

    batch = (
        torch.zeros(2, 3, 4),
        torch.zeros(2, 3, 2),
        torch.zeros(2, 3, 2),
        torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        [["u1"] * 3, ["u2"] * 2],
        torch.tensor([3, 2]),
    )
    _, outputs, targets = _step_loss(
        FixedOutputModel(), batch, nn.BCELoss(), torch.device("cpu")
    )
    assert torch.allclose(outputs, torch.tensor([0.9, 0.8, 0.2]))
    assert torch.equal(targets, torch.tensor([1.0, 1.0, 0.0]))
    print("dneuralcdm_training_test_ok")


if __name__ == "__main__":
    main()
