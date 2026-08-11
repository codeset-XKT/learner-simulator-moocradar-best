# Ablation Experiments

The paper-facing ablation suite targets the current `multi-role` simulator. All
variants must use the same cohort, target order, LLM configuration, feedback
protocol, and external answer scorer.

## Core Ablations

- `no-learner-state-profile`: remove the complete learner-state profile path,
  including cognitive profile and ability profile.
- `no-item-conditioned-integration`: retain raw NCDM and IRT evidence, but
  remove their qualitative item-conditioned integration.
- `no-four-tier`: retain the same evidence, reference answer, concept choice,
  correctness decision, and submitted answer, while removing confidence and
  reasoning tiers.
- `no-dynamic-state-evolution`: freeze the state initialized from observed
  history. It retains the same accepted feedback prefix and target-memory
  updates as Full, but does not recompute the prompt-facing NCDM latent state.

`no-dynamic-state-evolution` is not a teacher-forcing versus rollout comparison.
The feedback mode must remain fixed across Full and this ablation; the ablation
tests whether target-stage state evolution contributes useful information.

## Supplementary Ablations

- `no-ncdm`: remove NCDM state and prediction evidence while retaining
  IRT and history-derived state.
- `no-irt`: remove IRT ability-difficulty evidence while retaining NCDM.
- `no-ncdm-irt`: remove both external measurement-model evidence paths.

Removed historical aliases are intentionally unsupported so archived names
cannot silently run a different experiment.

## Run

```powershell
python experiments/ablation/run_ablation.py `
  --simulator multi-role `
  --dataset-root data/moocradar `
  --cohort-file experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch01_50x10_seed20260803.json `
  --dneuralcdm-checkpoint outputs/dneuralcdm/moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64/best_model.pt `
  --variants full,no-learner-state-profile,no-item-conditioned-integration,no-four-tier,no-dynamic-state-evolution `
  --parallel-variants 2 `
  --progress
```

Run the supplementary measurement-model ablations:

```powershell
python experiments/ablation/run_ablation.py `
  --simulator multi-role `
  --dataset-root data/moocradar `
  --cohort-file experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch01_50x10_seed20260803.json `
  --dneuralcdm-checkpoint outputs/dneuralcdm/moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64/best_model.pt `
  --variants full,no-ncdm,no-irt,no-ncdm-irt `
  --parallel-variants 2 `
  --progress
```

Every completed variant immediately writes a checkpoint under
`outputs/ablation/checkpoints/`. Reports contain all step traces, prompts, raw
LLM responses, and structured intermediate states by default. Existing files
are preserved; reruns create timestamped sibling files.

Full uses one trained NCDM checkpoint as the canonical knowledge-state source.
It computes the initial state from the observed history and recomputes that
state after each accepted feedback response. Exported proficiency JSON, DKT,
and MIKT artifacts are not injected into Full. The checkpoint must contain
training metadata proving that every evaluated cohort UID was held out; retrain
old checkpoints with the exact `--cohort-file` before running formal ablations.

Because LLM output is stochastic, formal experiments should use repeated runs
and report mean plus standard deviation. A paired analysis should use the same
target interactions across all variants.

Run `python scripts/test_ablation_variants.py` after changing module wiring. The
test checks that each ablated information path is absent from both prompts and
recorded intermediate states.
