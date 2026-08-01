# Version Lock: MoocRadar Anchor-Consistency Full Simulator

This lock fixes the current MoocRadar learner-simulation source state after
replacing success-preservation prompts with a symmetric KT/CDM decision-anchor
protocol.

## Frozen Version ID

`baseline-2026-08-01-moocradar-anchor-consistency-v1`

## Core Method

The Full simulator uses an Agent4Edu-style fixed-history protocol:

- 90 observed history interactions build the learner state;
- 10 target interactions are simulated with teacher forcing for state updates;
- NCDM current-item response probability is used as the primary KT/CDM anchor;
- IRT is retained as secondary ability-difficulty evidence;
- cognitive profile, ability profile, short-term memory, and historical replay
  calibration shape response style and confidence;
- the LLM explicitly outputs `LearnerCorrect` for Task4;
- Four-tier fields are retained as behavioral diagnostics, not as the primary
  Task4 correctness source.

## Main Change From The Previous Task4 Version

The earlier Task4 version made the LLM overly optimistic because NCDM evidence
was repeatedly verbalized as success preservation, for example
`preserve_ncdm_supported_success`.

This version changes the KT/CDM evidence into symmetric decision anchors:

- `kt_strong_correct_anchor`
- `kt_lean_correct_anchor`
- `kt_boundary_anchor`
- `kt_lean_incorrect_anchor`
- `kt_strong_incorrect_anchor`

The prompt now asks the LLM to normally follow strong/moderate KT anchor
directions and deviate only when concrete item-specific memory, item demand, or
profile evidence contradicts the anchor. This keeps NCDM as the main signal
without forcing the LLM to copy it.

## Locked MoocRadar Normal Batch

- cohort: `experiments/cohorts/moocradar_90_10_30x10_single_answer_targets.json`
- target items: 300
- target answer cardinality: 300/300 single-answer items
- real correct rate: 92.33%
- result:
  `locked_results/moocradar_30x10_single_answer_targets_anchor_consistency_full_llm.json`

| Method | ACC | F1 | Balanced ACC | Specificity | MCC | LDE | CDE | Predicted Correct Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours Full anchor-consistency | 94.65% | 97.14% | 75.18% | 52.17% | 0.580 | 0.040 | 0.112 | 94.33% |
| Ours Full previous Task4 | 95.00% | 97.36% | 67.39% | 34.78% | 0.574 | 0.050 | 0.134 | 97.33% |
| Probability-sampling baseline | 76.67% | 86.38% | 57.46% | 34.78% | 0.098 | 0.167 | 0.219 | 79.00% |
| DKT direct | 95.33% | 97.52% | 73.55% | 47.83% | 0.616 | - | - | 95.67% |
| NCDM direct | 94.33% | 96.97% | 73.01% | 47.83% | 0.545 | - | - | 94.67% |

Full anchor-consistency confusion matrix:

`TN=12, FP=11, FN=5, TP=272`

Validity note: 299/300 LLM outputs were parsed successfully. One LLM call
failed to parse.

## Locked MoocRadar Medium-Difficulty Batch

- cohort:
  `experiments/cohorts/moocradar_90_10_30x10_single_answer_medium60_70.json`
- target items: 300
- target answer cardinality: 300/300 single-answer items
- target correct-rate range used for sampling: 60%-70%
- real correct rate: 66.00%
- result:
  `locked_results/moocradar_30x10_single_answer_medium60_70_anchor_consistency_full_llm.json`

| Method | ACC | F1 | Balanced ACC | Specificity | MCC | LDE | CDE | Predicted Correct Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours Full anchor-consistency | 69.57% | 77.97% | 64.19% | 47.06% | 0.298 | 0.157 | 0.301 | 71.67% |
| Ours Full previous Task4 | 68.00% | 78.76% | 57.69% | 25.49% | 0.202 | 0.247 | 0.331 | 84.67% |
| Probability-sampling baseline | 52.67% | 58.72% | 53.45% | 55.88% | 0.065 | 0.233 | 0.378 | 48.67% |
| DKT direct | 71.33% | 79.52% | 65.21% | 46.08% | 0.329 | - | - | 74.00% |
| NCDM direct | 70.67% | 78.43% | 65.89% | 50.98% | 0.329 | - | - | 70.00% |

Full anchor-consistency confusion matrix:

`TN=48, FP=54, FN=37, TP=161`

Validity note: 299/300 LLM outputs were parsed successfully. One LLM call
failed to parse.

## Interpretation

This version fixes the main failure mode of the prior Task4 version. The prior
version improved ACC/F1 but over-predicted correct answers. The new anchor
protocol restores negative-sample discrimination while keeping the LLM's final
Task4 decision autonomous.

On the normal single-answer MoocRadar batch, Full anchor-consistency exceeds the
direct NCDM baseline on Balanced Accuracy and Specificity and is close to DKT in
ACC/F1. On the medium-difficulty batch, it approaches DKT/NCDM balanced
performance and strongly improves over the previous Task4 prompt.

## Reproduction Commands

Normal batch:

```powershell
$env:DASHSCOPE_API_KEY=(Get-Content -Raw -LiteralPath 'E:\yyx\8 Learner Simulator\key.txt').Trim()
conda run --no-capture-output -n pytorch_macos_env python experiments\ablation\run_ablation.py `
  --simulator multi-role `
  --variants full `
  --parallel-variants 1 `
  --cohort-file experiments\cohorts\moocradar_90_10_30x10_single_answer_targets.json `
  --feedback-mode teacher-forcing `
  --dneuralcdm-proficiency outputs\dneuralcdm\moocradar_90_10_30x10_single_answer_targets\stu_know_proficiency.json `
  --dneuralcdm-checkpoint outputs\dneuralcdm\moocradar_90_10_10x10_medium70_e30_d32_h64\best_model.pt `
  --dkt-proficiency outputs\dkt\moocradar_full_e50_h100\moocradar_90_10_50x10_proficiency.json `
  --llm-config configs\llm.example.json `
  --progress `
  --output outputs\ablation\moocradar_30x10_single_answer_targets_anchor_consistency_full_llm.json
```

Medium-difficulty batch:

```powershell
$env:DASHSCOPE_API_KEY=(Get-Content -Raw -LiteralPath 'E:\yyx\8 Learner Simulator\key.txt').Trim()
conda run --no-capture-output -n pytorch_macos_env python experiments\ablation\run_ablation.py `
  --simulator multi-role `
  --variants full `
  --parallel-variants 1 `
  --cohort-file experiments\cohorts\moocradar_90_10_30x10_single_answer_medium60_70.json `
  --feedback-mode teacher-forcing `
  --dneuralcdm-proficiency outputs\dneuralcdm\moocradar_90_10_30x10_single_answer_medium60_70\stu_know_proficiency.json `
  --dneuralcdm-checkpoint outputs\dneuralcdm\moocradar_90_10_10x10_medium70_e30_d32_h64\best_model.pt `
  --dkt-proficiency outputs\dkt\moocradar_full_e50_h100\moocradar_90_10_50x10_proficiency.json `
  --llm-config configs\llm.example.json `
  --progress `
  --output outputs\ablation\moocradar_30x10_single_answer_medium60_70_anchor_consistency_full_llm.json
```
