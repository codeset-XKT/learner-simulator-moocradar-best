# Results Summary

This file records the main fixed-cohort results that should be used for project
handoff and method comparison. Snapshot date: 2026-07-26.

## Reading The Metrics

- ACC and F1 reward overall answer agreement but can look strong on
  positive-skewed cohorts.
- Balanced Accuracy, Specificity, and MCC show whether the simulator separates
  correct and incorrect responses.
- LDE and CDE measure distribution consistency; lower is better.

## MoocRadar Medium70 10x10

Fixed cohort: `experiments/cohorts/moocradar_90_10_10x10_medium70.json`.
Protocol: 90 observed history interactions and 10 teacher-forcing target
simulations for each of 10 learners. True target correct rate: 70.00%.

Main note: `docs/moocradar_medium70_balanced_state.md`.

| Method | ACC | F1 | Balanced Acc | Specificity | MCC | LDE | CDE | Confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Balanced-state Full (current) | 69.00% | 76.34% | 67.38% | 63.33% | 0.327 | 0.110 | 0.349 | TN19 / FP11 / FN20 / TP50 |
| Agent4Edu-style | 70.00% | 82.35% | 50.00% | 0.00% | null | 0.300 | 0.313 | TN0 / FP30 / FN0 / TP70 |
| Random simulator | 53.00% | 62.99% | 50.24% | 43.33% | 0.004 | 0.190 | 0.426 | not recorded in summary |
| DKT predictor | 74.00% | 83.95% | 58.57% | 20.00% | 0.290 | 0.220 | 0.221 | TN6 / FP24 / FN2 / TP68 |
| Previous multi-task Full | 64.00% | 72.31% | 61.90% | 56.67% | 0.223 | 0.160 | 0.364 | not recorded in summary |
| Historical dedup-profile Full | 66.00% | 73.44% | 65.24% | 63.33% | 0.283 | 0.200 | 0.349 | not recorded in summary |
| Simplified multi-role Full | 65.00% | 72.44% | 64.52% | 63.33% | 0.269 | 0.190 | 0.370 | not recorded in summary |

Interpretation: the current Balanced-state Full version is the strongest
recorded LLM simulator on this medium-correct-rate MoocRadar cohort. It does
not beat DKT on ACC/F1, but it has better Balanced Accuracy, Specificity, MCC,
and LDE, mainly because it identifies many more true negatives than DKT.

## MoocRadar 10x10

Fixed cohort: `experiments/cohorts/moocradar_90_10_10x10.json`.
Protocol: 90 observed history interactions and 10 target interactions.
True target correct rate: 89.00%.

| Version | ACC | F1 | Balanced Acc | Specificity | MCC | LDE | CDE | Predicted Correct Rate | Confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Historical Full + ability summary | 75.00% | 84.28% | 74.00% | 72.73% | 0.328 | 0.190 | 0.398 | 70.00% | TP67 / TN8 / FP3 / FN22 |
| Balanced-state light Full (locked 2026-07-26) | 75.00% | 84.47% | 70.02% | 63.64% | 0.279 | 0.210 | 0.418 | 72.00% | TP68 / TN7 / FP4 / FN21 |
| DKT reference | 93.00% | 96.17% | 72.17% | 45.45% | 0.584 | — | — | 94.00% | TP88 / TN5 / FP6 / FN1 |
| p_correct >= 0.5 | 92.00% | 95.65% | 67.62% | 36.36% | 0.506 | — | — | 95.00% | TP88 / TN4 / FP7 / FN1 |
| Agent4Edu-style | 89.00% | 94.18% | 50.00% | 0.00% | null | 0.110 | 0.220 | 100.00% | TP89 / TN0 / FP11 / FN0 |
| Random simulator | 75.00% | 85.03% | 58.07% | 36.36% | 0.122 | 0.190 | 0.228 | 78.00% | not recorded here |
| Current multi-role Ability Evidence | 78.00% | 86.59% | 71.71% | 63.64% | 0.314 | 0.160 | 0.398 | 75.00% | TP71 / TN7 / FP4 / FN18 |
| Route Evidence transition | 74.00% | 83.95% | 65.48% | 54.55% | 0.218 | 0.220 | 0.424 | 73.00% | not recorded here |
| Earlier educational multi-agent | 70.00% | 80.77% | 67.21% | 63.64% | 0.229 | 0.260 | 0.429 | 67.00% | not recorded here |
| Old multi-role + LPR + teacher forcing | 71.00% | 81.53% | 67.77% | 63.64% | 0.238 | 0.230 | 0.449 | not recorded here | not recorded here |

Main files:

- `outputs/ablation/moocradar_10x10_ability_summary_no_irt_full_no_ability_profile.json`
- `locked_results/moocradar_10x10_normal_balanced_state_light_full.json`
- `locked_results/moocradar_10x10_normal_agent4edu.json`
- `locked_results/moocradar_10x10_normal_random.json`
- `outputs/comparison/moocradar_10x10_educational_multi_agent_ability_evidence_teacher_forcing.json`
- `outputs/comparison/moocradar_10x10_educational_multi_agent_route_evidence_teacher_forcing.json`
- `outputs/comparison/moocradar_10x10_educational_multi_agent_teacher_forcing.json`
- `outputs/comparison/moocradar_10x10_multi_role_lpr_teacher_forcing.json`

Interpretation: the historical ability-summary run remains the best MoocRadar
result for Balanced Accuracy, Specificity, and MCC among the locked LLM
simulators. The Balanced-state light lock confirms that the active Full branch
still separates negative samples far better than Agent4Edu-style on this
positive-skewed batch: Agent4Edu-style matches the all-correct baseline with
Specificity 0.00%, while Balanced-state light Full reaches Specificity 63.64%.
DKT remains the strongest pure predictor and should be reported as a reference,
not as a simulator replacement.

## XES3G5M 10x10

Result file:
`outputs/comparison/xes_10x10_multi_role_ability_evidence_vs_agent4edu_teacher_forcing.json`.

| Method | ACC | F1 | Balanced Acc | Specificity | MCC | LDE | CDE | Predicted Correct Rate | Confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Current multi-role Ability Evidence | 64.00% | 73.13% | 66.73% | 71.43% | 0.274 | 0.260 | 0.372 | 55.00% | TP49 / TN15 / FP6 / FN30 |
| Agent4Edu | 79.00% | 88.27% | 50.00% | 0.00% | null | 0.210 | 0.197 | 100.00% | TP79 / TN0 / FP21 / FN0 |

Interpretation: Agent4Edu has higher ACC/F1 because it predicts every target
interaction as correct on a positive-skewed batch. The current multi-role model
is weaker on ACC/F1 but clearly stronger on negative-sample recognition.

## FoundationalAssist 10x10

Fixed cohort: `experiments/cohorts/foundationalassist_90_10_10x10.json`.
True target correct rate: 63.00%.

| Method | ACC | F1 | Balanced Acc | Specificity | MCC | LDE | CDE | Predicted Correct Rate | Confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Current multi-role Ability Evidence | 43.00% | 34.48% | 49.74% | 75.68% | -0.006 | 0.430 | 0.416 | 24.00% | TP15 / TN28 / FP9 / FN48 |
| Agent4Edu | 65.00% | 77.42% | 54.38% | 13.51% | 0.156 | 0.310 | 0.337 | 92.00% | TP60 / TN5 / FP32 / FN3 |
| Random | 50.00% | 50.98% | 53.07% | 64.86% | 0.061 | 0.280 | 0.318 | 39.00% | TP26 / TN24 / FP13 / FN37 |
| Locked calibrated Full | 45.00% | 40.86% | 50.21% | 70.27% | 0.0045 | 0.470 | 0.487 | 30.00% | TP19 / TN26 / FP11 / FN44 |
| Locked Full w/o ability profile | 45.00% | 48.60% | 46.31% | 51.35% | -0.0718 | 0.390 | 0.439 | 44.00% | not recorded here |

Main files:

- `outputs/comparison/foundationalassist_10x10_multi_role_ability_evidence_teacher_forcing.json`
- `outputs/comparison/foundationalassist_10x10_agent4edu_random.json`
- `outputs/ablation/foundationalassist_10x10_calibrated_full_no_ability.json`

Interpretation: FoundationalAssist exposes a current failure mode. The current
multi-role version is too conservative: it predicts correct only 24.00% of the
time, causing 48 false negatives. It preserves high specificity but does not
beat Agent4Edu or random on the balanced metrics.

## Practical Baseline Choice

- Use the current `multi-role` simulator as the active method branch.
- Use the historical MoocRadar ability-summary run as the best locked positive
  evidence for the compact ability profile.
- Treat FoundationalAssist as the main stress test for over-conservative
  response behavior.
- Do not use ACC/F1 alone as the main claim. Always include Balanced Accuracy,
  Specificity, MCC, LDE, and CDE.
