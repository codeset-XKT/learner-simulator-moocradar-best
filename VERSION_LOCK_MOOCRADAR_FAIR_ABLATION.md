# Version Lock: MoocRadar Leakage-Safe Fair Ablation

## Frozen Version ID

`baseline-2026-08-11-moocradar-fair-ablation-v1`

This lock identifies the active paper-oriented implementation. Historical
locks remain available for reproducing earlier prompt and profile variants.

## Method Boundary

The executable method is the `multi-role` simulator with:

1. history-derived learner-state profile;
2. one leakage-safe DNeuralCDM checkpoint for initial state, current-item
   probability diagnostics, and target-stage latent-state evolution;
3. Rasch/1PL IRT ability-difficulty evidence;
4. qualitative item-conditioned evidence integration;
5. Four-tier learner response generation;
6. sequential dynamic state evolution.

`LearnerCorrect` is always produced by the LLM. NCDM, IRT, DKT, sampled
probabilities, and ground-truth target labels never override that decision.
The deprecated `learning_tool_state`, KT Decision, KT State Guidance, and
Pre-response Tendency State paths are absent.

## Formal Protocol

- Dataset: MoocRadar.
- Per learner: 90 observed interactions and 10 target interactions.
- Feedback: teacher forcing unless a run explicitly studies rollout.
- Cohorts: ten disjoint 50-learner batches generated with seed `20260803`.
- Planned evaluation population: 500 unique learners and 5,000 target steps.
- LLM model: `deepseek-v4-flash-0731`, temperature `0.0`, thinking enabled.
- Failed LLM step policy: exclude the complete learner sequence from aggregate
  LLM metrics; preserve the raw trace and repair only complete sequences.

The formal cohort assets are under:

`experiments/cohorts/moocradar_500x10_batches/`

The aggregate 500-user cohort file is the exclusion set used when training the
formal NCDM checkpoint. It must not be replaced with a single-batch holdout.

## NCDM Grounding

Formal runs use architecture version 2 with positive exercise discrimination,
masked padded positions, 32-dimensional embeddings, a 64-dimensional hidden
layer, and best-validation checkpoint selection across 30 epochs.

Expected local checkpoint directory:

`outputs/dneuralcdm/moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64/`

The model weights are intentionally excluded from Git. Checkpoint metadata must
prove that all 500 planned evaluation UIDs were held out from training.

## Core Ablations

- `no-learner-state-profile`
- `no-item-conditioned-integration`
- `no-four-tier`
- `no-dynamic-state-evolution`

All variants keep the same learners, targets, order, LLM configuration,
teacher-forced feedback prefix, and scoring path. `no-four-tier` uses the
reduced-response contract rather than answer-only generation.
`no-dynamic-state-evolution` freezes only the prompt-facing NCDM latent state;
it retains the same accepted feedback prefix and target-memory updates as Full.

## Verified Results

The following are complete 500-step runs using the all-500-user exclusion
checkpoint. Values are descriptive evidence from two batches, not an aggregate
claim over the planned 5,000 targets.

### Batch 02

| Variant | ACC | F1 | Balanced ACC | Specificity | MCC | LDE | CDE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 89.60% | 94.13% | 75.04% | 56.36% | 0.486 | 0.064 | 0.115 |
| w/o learner-state profile | 89.60% | 94.12% | 75.83% | 58.18% | 0.494 | 0.068 | 0.099 |
| w/o item-conditioned integration | 90.00% | 94.37% | 75.26% | 56.36% | 0.497 | 0.060 | 0.109 |
| w/o Four-tier | 88.60% | 93.59% | 71.29% | 49.09% | 0.422 | 0.066 | 0.121 |
| w/o dynamic state evolution | 88.60% | 93.56% | 72.88% | 52.73% | 0.441 | 0.074 | 0.120 |

### Batch 03

| Variant | ACC | F1 | Balanced ACC | Specificity | MCC | LDE | CDE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 85.00% | 91.02% | 72.67% | 54.22% | 0.456 | 0.070 | 0.110 |
| w/o learner-state profile | 86.40% | 91.85% | 75.44% | 59.04% | 0.509 | 0.064 | 0.121 |
| w/o item-conditioned integration | 85.00% | 90.97% | 73.64% | 56.63% | 0.466 | 0.070 | 0.117 |
| w/o Four-tier | 85.40% | 91.28% | 72.91% | 54.22% | 0.465 | 0.074 | 0.112 |
| w/o dynamic state evolution | 85.40% | 91.19% | 74.84% | 59.04% | 0.485 | 0.066 | 0.119 |

These batches show mixed effects outside the Four-tier degradation on Batch 02.
Do not claim every module is effective from these two batches alone; complete
the remaining batches and use paired repeated-cohort statistics.

## Verification

Run all `scripts/test_*.py` files and compile `src`, `scripts`, and
`experiments` before publishing. Secrets, datasets, outputs, checkpoints,
temporary repair cohorts, and Python caches are not part of this lock.
