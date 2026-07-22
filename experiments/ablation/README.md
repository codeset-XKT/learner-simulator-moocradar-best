# Ablation Experiments

The ablation suite removes four central modules without changing learners, targets, sequence order, LLM configuration, or external answer scoring:

- `no-profile`: remove the static learner profile from the system prompt.
- `no-memory`: remove short-term and long-term memory from the action prompt.
- `no-proficiency`: remove current concept mastery from the action prompt.
- `no-four-tier`: retain only `StudentAnswer`; remove reasoning and confidence elicitation.
- `no-cognitive-selection`: remove the cognitive route/selection module.
- `no-cognitive-profile`: remove the cognitive profile evidence.
- `no-ability-profile`: remove the compact ability profile evidence.

Run selected variants on a fixed cohort:

```powershell
python experiments/ablation/run_ablation.py `
  --cohort-file experiments/cohorts/moocradar_90_10_10x10.json `
  --variants full,no-cognitive-selection,no-cognitive-profile,no-ability-profile `
  --parallel-variants 2 `
  --progress `
  --save-steps
```

Save the exact cohort on the first run:

```powershell
python experiments/ablation/run_ablation.py --source-rows 1000 --max-users 4 --save-cohort experiments/cohorts/ablation_4x10.json --progress
```

Run independent variants concurrently while preserving sequential state updates
inside each learner:

```powershell
python experiments/ablation/run_ablation.py --cohort-file experiments/cohorts/ablation_4x10.json --parallel-variants 5 --progress
```

Reuse the first three learners from the locked cohort:

```powershell
python experiments/ablation/run_ablation.py --cohort-file experiments/cohorts/ablation_4x10.json --cohort-user-limit 3 --variants full,no-four-tier --parallel-variants 2 --progress
```

Each completed variant is immediately saved under
`outputs/ablation/checkpoints/`, so a long experiment does not lose completed
conditions if a later API call is interrupted.

Run a subset:

```powershell
python experiments/ablation/run_ablation.py --variants full,no-memory --source-rows 1000 --max-users 3 --progress
```

Because LLM output is stochastic, formal experiments should use multiple seeds or repeated runs and report mean plus standard deviation.

Current note: the active research branch is named `multi-role` in comparison
experiments, but the ablation runner still targets the older `full` branch.
Check `README.md` and `RESULTS_SUMMARY.md` before interpreting ablation results
as evidence for the current multi-role simulator.
