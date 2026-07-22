# Version Lock: 2026-07-04 MoocRadar Ability Summary Baseline

This file fixes the historical MoocRadar run that produced the strongest
observed result before later FoundationalAssist prompt-calibration changes.

## Frozen Version ID

`baseline-2026-07-04-moocradar-ability-summary-no-irt-v1`

## Why This Lock Exists

This lock preserves the MoocRadar result where the new ability-summary profile
was useful after removing the earlier IRT-style ability profile.

It should be treated as a historical experimental baseline, separate from the
later `baseline-2026-07-04-foundationalassist-calibrated-v1` lock.

## Important Reproducibility Note

The exact source state used by this MoocRadar run was not committed to git before
later edits. The run result shows that the active simulator did **not** include
the later `tendency_calibration` field. After this run, `llm_simulator.py` and
`agent4edu_prompt.py` were modified to add response tendency calibration for the
FoundationalAssist experiment.

Therefore this lock preserves:

- the original result file;
- the fixed MoocRadar cohort;
- the DKT proficiency file;
- the module design and metrics recorded in the result;
- a source/evidence manifest at lock time.

It should not be confused with an exact executable source snapshot unless the
pre-calibration prompt code is reconstructed or recovered separately.

## Experiment Context

- dataset: `MoocRadar`
- prepared data root: `data/moocradar`
- cohort file: `experiments/cohorts/moocradar_90_10_10x10.json`
- protocol: 10 learners, 90 observed history steps, 10 target simulation steps
- DKT proficiency:
  `outputs/dkt/moocradar_full_e50_h100/moocradar_90_10_10x10_proficiency.json`
- result file:
  `outputs/ablation/moocradar_10x10_ability_summary_no_irt_full_no_ability_profile.json`
- output validity: 100/100 valid LLM outputs for both variants
- model used by the surrounding project configuration at this stage:
  `deepseek-v4-flash`

## Current Method In This Historical Run

The historical Full simulator used:

- statistical cognitive profile;
- compact ability summary profile;
- short-term and reinforced memory;
- DKT proficiency;
- non-cognitive behavioral factors;
- four-tier LLM output;
- external answer scoring.

The ability summary profile contained:

- knowledge breadth;
- practice depth;
- challenge adaptation;
- cross-domain generalization;
- profile confidence and evidence counts.

The old IRT ability estimate was not used as part of the learner profile in this
locked result.

## Locked Result

| Method | ACC | F1 | Balanced ACC | Specificity | MCC | LDE | CDE | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full + ability summary | 75.00% | 84.28% | 74.00% | 72.73% | 0.328 | 0.190 | 0.398 | 15m35s |
| w/o ability profile | 70.00% | 81.01% | 63.23% | 54.55% | 0.179 | 0.240 | 0.429 | 14m25s |

True correct rate of the fixed target batch: 89.00%.

Observed effect of adding the ability summary profile:

- ACC: +5.00 percentage points;
- Balanced ACC: +10.78 percentage points;
- Specificity: +18.18 percentage points;
- MCC: 0.179 -> 0.328;
- LDE and CDE both decreased.

## Result Interpretation At Lock Time

This is the best evidence so far that the compact ability-summary profile can
help the simulator distinguish negative samples while keeping ACC and F1 high on
MoocRadar.

However, this batch is highly positive-skewed: the real correct rate is 89.00%.
The result is useful as a positive baseline, but it should be validated on larger
and less skewed MoocRadar cohorts before being treated as a stable paper result.

## Reproduction Command Used Conceptually

```powershell
$env:DASHSCOPE_API_KEY=(Get-Content -Raw -LiteralPath 'E:\yyx\8 Learner Simulator\key.txt').Trim()
python experiments\ablation\run_ablation.py `
  --dataset-root data\moocradar `
  --cohort-file experiments\cohorts\moocradar_90_10_10x10.json `
  --history-steps 90 `
  --target-steps 10 `
  --dkt-proficiency outputs\dkt\moocradar_full_e50_h100\moocradar_90_10_10x10_proficiency.json `
  --variants full,no-ability-profile `
  --parallel-variants 2 `
  --checkpoint-dir outputs\ablation\moocradar_10x10_ability_summary_no_irt_checkpoints `
  --output outputs\ablation\moocradar_10x10_ability_summary_no_irt_full_no_ability_profile.json `
  --progress `
  --save-steps
```

Because the exact pre-calibration source state was not committed, rerunning this
command on the later FoundationalAssist-calibrated source may not reproduce the
same numbers exactly.

## Lock Artifacts

Evidence bundle:

`outputs/version_locks/baseline-2026-07-04-moocradar-ability-summary-no-irt-v1_evidence.zip`

Evidence/source manifest:

`outputs/version_locks/baseline-2026-07-04-moocradar-ability-summary-no-irt-v1_manifest.csv`
