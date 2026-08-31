from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class DAISimConfig:
    item_count: int
    concept_count: int
    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.5


class DAISimGenerator(nn.Module):
    """MAIL-style learner interaction generator adapted from DAISim.

    The learner state is initialized from the correctness-weighted concept
    profile of observed interactions, then updated autoregressively across a
    supplied item sequence.  It intentionally predicts response outcomes only;
    it does not generate textual answers or explanations.
    """

    def __init__(self, config: DAISimConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.item_count, config.embedding_dim)
        self.concept_embedding = nn.Embedding(config.concept_count, config.embedding_dim)
        self.user_projection = nn.Linear(config.concept_count, config.hidden_dim)
        self.state_projection = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.action = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 3, 1),
        )
        self.transition = nn.GRUCell(config.embedding_dim * 2, config.hidden_dim)

    def initial_state(self, history_feature: torch.Tensor) -> torch.Tensor:
        return self.user_projection(history_feature)

    def _step(
        self,
        state: torch.Tensor,
        item: torch.Tensor,
        concept: torch.Tensor,
        response_for_update: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        item_embedding = self.item_embedding(item)
        concept_embedding = self.concept_embedding(concept)
        state_embedding = self.state_projection(state)
        logit = self.action(torch.cat([state_embedding, item_embedding, concept_embedding], dim=-1)).squeeze(-1)
        response = response_for_update.unsqueeze(-1)
        transition_input = torch.cat(
            [concept_embedding * response, concept_embedding * (1.0 - response)], dim=-1
        )
        return logit, self.transition(transition_input, state)

    def teacher_forced_logits(
        self,
        history_feature: torch.Tensor,
        items: torch.Tensor,
        concepts: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.initial_state(history_feature)
        logits: list[torch.Tensor] = []
        for index in range(items.shape[1]):
            logit, next_state = self._step(state, items[:, index], concepts[:, index], responses[:, index])
            if mask is None:
                state = next_state
            else:
                state = torch.where(mask[:, index].unsqueeze(-1), next_state, state)
            logits.append(logit)
        return torch.stack(logits, dim=1)

    def rollout_probabilities(
        self,
        history_feature: torch.Tensor,
        items: torch.Tensor,
        concepts: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.initial_state(history_feature)
        probabilities: list[torch.Tensor] = []
        for index in range(items.shape[1]):
            placeholder = torch.zeros(items.shape[0], device=items.device)
            logit, _ = self._step(state, items[:, index], concepts[:, index], placeholder)
            probability = torch.sigmoid(logit)
            _, next_state = self._step(state, items[:, index], concepts[:, index], probability)
            if mask is None:
                state = next_state
            else:
                state = torch.where(mask[:, index].unsqueeze(-1), next_state, state)
            probabilities.append(probability)
        return torch.stack(probabilities, dim=1)


class DAISimDiscriminator(nn.Module):
    """Pairwise adversarial discriminator over learner-item-response traces."""

    def __init__(self, config: DAISimConfig) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(config.item_count, config.embedding_dim)
        self.concept_embedding = nn.Embedding(config.concept_count, config.embedding_dim)
        self.user_projection = nn.Linear(config.concept_count, config.hidden_dim)
        self.trace_encoder = nn.GRU(config.embedding_dim * 4, config.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        history_feature: torch.Tensor,
        items: torch.Tensor,
        concepts: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        item_embedding = self.item_embedding(items)
        concept_embedding = self.concept_embedding(concepts)
        response = responses.unsqueeze(-1)
        trace = torch.cat(
            [
                item_embedding,
                concept_embedding,
                concept_embedding * response,
                concept_embedding * (1.0 - response),
            ],
            dim=-1,
        )
        initial = self.user_projection(history_feature).unsqueeze(0)
        if mask is None:
            mask = torch.ones(items.shape, dtype=torch.bool, device=items.device)
        lengths = mask.sum(dim=1).to(dtype=torch.long).cpu()
        packed = pack_padded_sequence(trace, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.trace_encoder(packed, initial)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=items.shape[1])
        scores = self.output(self.dropout(encoded)).squeeze(-1)
        return (scores * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def build_index(values: Iterable[int]) -> dict[int, int]:
    return {value: index + 1 for index, value in enumerate(sorted(set(values)))}


def encode_steps(
    steps: list[dict[str, Any]],
    item_index: dict[int, int],
    concept_index: dict[int, int],
) -> tuple[list[int], list[int], list[float]]:
    return (
        [item_index.get(int(step["qid"]), 0) for step in steps],
        [concept_index.get(int(step["cid"]), 0) for step in steps],
        [float(step["response"]) for step in steps],
    )


def history_feature(
    steps: list[dict[str, Any]],
    concept_index: dict[int, int],
) -> torch.Tensor:
    feature = torch.zeros(len(concept_index) + 1, dtype=torch.float32)
    if not steps:
        return feature
    for step in steps:
        feature[concept_index.get(int(step["cid"]), 0)] += float(step["response"])
    return feature / float(len(steps))


def checkpoint_payload(
    generator: DAISimGenerator,
    discriminator: DAISimDiscriminator,
    config: DAISimConfig,
    item_index: dict[int, int],
    concept_index: dict[int, int],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": "DAISim_official_code_derived_reproduction",
        "official_repository_commit": "a59b3d106b6255ee4502d001392a13da41983d0f",
        "config": asdict(config),
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "item_index": {str(key): value for key, value in item_index.items()},
        "concept_index": {str(key): value for key, value in concept_index.items()},
        "training_metadata": metadata,
    }


def load_generator(checkpoint: dict[str, Any], device: torch.device) -> tuple[DAISimGenerator, dict[int, int], dict[int, int]]:
    config = DAISimConfig(**checkpoint["config"])
    generator = DAISimGenerator(config).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    item_index = {int(key): int(value) for key, value in checkpoint["item_index"].items()}
    concept_index = {int(key): int(value) for key, value in checkpoint["concept_index"].items()}
    return generator, item_index, concept_index


def train_epoch(
    generator: DAISimGenerator,
    discriminator: DAISimDiscriminator,
    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    adversarial_weight: float,
    run_adversarial: bool,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    direct_total = 0.0
    discriminator_total = 0.0
    adversarial_total = 0.0
    step_count = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        max_length = max(values[1].numel() for values in batch)
        feature = torch.stack([values[0] for values in batch]).to(device)
        items = torch.zeros((len(batch), max_length), dtype=torch.long, device=device)
        concepts = torch.zeros((len(batch), max_length), dtype=torch.long, device=device)
        responses = torch.zeros((len(batch), max_length), dtype=torch.float32, device=device)
        mask = torch.zeros((len(batch), max_length), dtype=torch.bool, device=device)
        for index, (_, item_ids, concept_ids, response_values) in enumerate(batch):
            length = item_ids.numel()
            items[index, :length] = item_ids.to(device)
            concepts[index, :length] = concept_ids.to(device)
            responses[index, :length] = response_values.to(device)
            mask[index, :length] = True

        generator.train()
        logits = generator.teacher_forced_logits(feature, items, concepts, responses, mask)
        direct_loss = (F.binary_cross_entropy_with_logits(logits, responses, reduction="none") * mask).sum() / mask.sum()
        generator_optimizer.zero_grad()
        direct_loss.backward()
        generator_optimizer.step()

        discriminator_loss = torch.zeros((), device=device)
        adversarial_loss = torch.zeros((), device=device)
        if run_adversarial:
            with torch.no_grad():
                fake = generator.rollout_probabilities(feature, items, concepts, mask)
            discriminator.train()
            real_score = discriminator(feature, items, concepts, responses, mask)
            fake_score = discriminator(feature, items, concepts, fake, mask)
            discriminator_loss = F.softplus(-real_score).mean() + F.softplus(fake_score).mean()
            pairwise_loss = F.softplus(-(real_score - fake_score)).mean()
            discriminator_optimizer.zero_grad()
            (0.5 * discriminator_loss + 0.5 * pairwise_loss).backward()
            discriminator_optimizer.step()

            generator.train()
            fake = generator.rollout_probabilities(feature, items, concepts, mask)
            adversarial_loss = F.softplus(-discriminator(feature, items, concepts, fake, mask)).mean()
            generator_optimizer.zero_grad()
            (adversarial_weight * adversarial_loss).backward()
            generator_optimizer.step()

        direct_total += float(direct_loss.detach().cpu()) * len(batch)
        discriminator_total += float(discriminator_loss.detach().cpu()) * len(batch)
        adversarial_total += float(adversarial_loss.detach().cpu()) * len(batch)
        step_count += len(batch)
    divisor = max(1, step_count)
    return {
        "direct_bce": direct_total / divisor,
        "discriminator_loss": discriminator_total / divisor,
        "generator_adversarial_loss": adversarial_total / divisor,
    }
