"""Reference neural KT models used as non-LLM comparison baselines.

The models in this module follow the *prediction-before-observation* order:
the probability for interaction ``t`` is produced from interactions strictly
before ``t``.  This is deliberately explicit because the project uses these
models both for standard KT evaluation and for autoregressive learner rollout.

DeepIRT is a PyTorch compatibility port of the public TensorFlow-1 reference
implementation by Yeung (2019).  SAKT follows Pandey & Karypis (2019).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("Neural reference KT models require PyTorch.")


@dataclass(frozen=True)
class NeuralKTConfig:
    num_items: int
    embed_dim: int = 64
    memory_size: int = 20
    dropout: float = 0.2
    num_heads: int = 4
    max_seq_len: int = 256


if nn is not None:

    class DeepIRT(nn.Module):
        """Deep-IRT / DKVMN with an IRT output layer.

        ``forward(items, responses)`` returns a probability at every position.
        Crucially, position t is read before the item's response is written to
        the value memory.  Thus it is suitable for a true next-response task.
        Item ids use 0 as padding and 1..num_items as valid identifiers.
        """

        def __init__(self, config: NeuralKTConfig) -> None:
            super().__init__()
            self.config = config
            d = int(config.embed_dim)
            m = int(config.memory_size)
            self.key_memory = nn.Parameter(torch.empty(m, d))
            self.value_memory = nn.Parameter(torch.empty(m, d))
            self.item_embedding = nn.Embedding(config.num_items + 1, d, padding_idx=0)
            self.interaction_embedding = nn.Embedding(2 * (config.num_items + 1), d, padding_idx=0)
            self.erase = nn.Linear(d, d)
            self.add = nn.Linear(d, d)
            self.summary = nn.Linear(2 * d, d)
            self.ability = nn.Linear(d, 1)
            self.difficulty = nn.Linear(d, 1)
            self.dropout = nn.Dropout(config.dropout)
            nn.init.xavier_uniform_(self.key_memory)
            nn.init.xavier_uniform_(self.value_memory)
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.xavier_uniform_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)

        def initial_memory(self, batch_size: int, device=None):
            device = device or self.key_memory.device
            return self.value_memory.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)

        def predict_and_update(self, item, response, memory):
            """Return p(response=1) before applying the given feedback."""
            item_embed = self.item_embedding(item)
            correlation = torch.softmax(
                torch.matmul(item_embed, self.key_memory.t()), dim=-1
            )
            read = torch.bmm(correlation.unsqueeze(1), memory).squeeze(1)
            summary = torch.tanh(self.summary(torch.cat([read, item_embed], dim=-1)))
            ability = self.ability(summary).squeeze(-1)
            difficulty = torch.tanh(self.difficulty(item_embed)).squeeze(-1)
            probability = torch.sigmoid(3.0 * ability - difficulty)

            valid = item.ne(0).float().view(-1, 1, 1)
            interaction_id = item + response.long().clamp(0, 1) * (self.config.num_items + 1)
            interaction_embed = self.interaction_embedding(interaction_id)
            erase = torch.sigmoid(self.erase(interaction_embed)).unsqueeze(1)
            add = torch.tanh(self.add(interaction_embed)).unsqueeze(1)
            weights = correlation.unsqueeze(-1)
            updated = memory * (1.0 - weights * erase) + weights * add
            memory = valid * updated + (1.0 - valid) * memory
            return probability, memory, ability, difficulty

        def forward(self, items, responses):
            memory = self.initial_memory(items.shape[0], items.device)
            probabilities = []
            abilities = []
            difficulties = []
            for index in range(items.shape[1]):
                p, memory, ability, difficulty = self.predict_and_update(
                    items[:, index], responses[:, index], memory
                )
                probabilities.append(p)
                abilities.append(ability)
                difficulties.append(difficulty)
            return (
                torch.stack(probabilities, dim=1),
                torch.stack(abilities, dim=1),
                torch.stack(difficulties, dim=1),
            )


    class SAKT(nn.Module):
        """One-block causal Self-Attentive Knowledge Tracing model.

        Each query item attends only to earlier interaction tokens; diagonal
        masking removes the current response from the prediction path.
        """

        def __init__(self, config: NeuralKTConfig) -> None:
            super().__init__()
            self.config = config
            d = int(config.embed_dim)
            if d % int(config.num_heads):
                raise ValueError("embed_dim must be divisible by num_heads")
            self.item_embedding = nn.Embedding(config.num_items + 1, d, padding_idx=0)
            self.interaction_embedding = nn.Embedding(2 * (config.num_items + 1), d, padding_idx=0)
            self.position_embedding = nn.Embedding(config.max_seq_len, d)
            self.attention = nn.MultiheadAttention(
                d, config.num_heads, dropout=config.dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(2 * d, d)
            )
            self.norm2 = nn.LayerNorm(d)
            self.output = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(d, 1))
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.xavier_uniform_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, items, responses):
            batch, length = items.shape
            if length > self.config.max_seq_len:
                raise ValueError(f"sequence length {length} exceeds max_seq_len={self.config.max_seq_len}")
            positions = torch.arange(length, device=items.device).unsqueeze(0).expand(batch, -1)
            query = self.item_embedding(items) + self.position_embedding(positions)
            interaction_id = items + responses.long().clamp(0, 1) * (self.config.num_items + 1)
            key_value = self.interaction_embedding(interaction_id) + self.position_embedding(positions)
            # True blocks all keys j >= query t.  The first position has no
            # history and is excluded by the training mask below.
            causal = torch.triu(
                torch.ones((length, length), dtype=torch.bool, device=items.device), diagonal=0
            )
            # A fully masked first query would yield NaNs in MultiheadAttention.
            # Permit its own token solely as a numerical placeholder; callers
            # never train/evaluate t=0.
            causal[0, 0] = False
            padding = items.eq(0)
            attended, _ = self.attention(query, key_value, key_value, attn_mask=causal, key_padding_mask=padding, need_weights=False)
            hidden = self.norm1(query + attended)
            hidden = self.norm2(hidden + self.ffn(hidden))
            return torch.sigmoid(self.output(hidden).squeeze(-1))


def encode_interaction(item: int, response: int, num_items: int) -> int:
    """Public helper mirroring the item-response indexing in this module."""
    if item <= 0:
        return 0
    return int(item) + int(bool(response)) * (num_items + 1)


def checkpoint_payload(
    method: str,
    state_dict: dict[str, Any],
    config: NeuralKTConfig,
    item_id_map: dict[str, int],
    training_metadata: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "format": "learner_simulator_reference_kt_v1",
        "method": method,
        "state_dict": state_dict,
        "config": {
            "num_items": config.num_items,
            "embed_dim": config.embed_dim,
            "memory_size": config.memory_size,
            "dropout": config.dropout,
            "num_heads": config.num_heads,
            "max_seq_len": config.max_seq_len,
        },
        "item_id_map": item_id_map,
        "training_metadata": training_metadata,
        **extra,
    }
