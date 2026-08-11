# Version Lock: 2026-07-28 MoocRadar Medium Agent-Style Four-tier Full

This file fixes the current MoocRadar medium-difficulty run after the simulator
was refactored toward an Agent4Edu-style tool-conditioned Four-tier protocol.

## Frozen Version ID

`baseline-2026-07-28-moocradar-medium-agent-style-four-tier-full-v1`

## Why This Lock Exists

This lock preserves the current executable project state and the latest
MoocRadar medium-difficulty 10x10 result. The method is not a direct clone of
Agent4Edu. It keeps the Agent4Edu-style Profile/Memory/Tool/Action protocol, but
uses the tool state to condition Four-tier learner response generation rather
than only predicting a Task4 Yes/No correctness label.

## Experiment Context

- dataset: `MoocRadar`
- prepared data root: `data/moocradar`
- cohort file: `experiments/cohorts/moocradar_90_10_10x10_medium70.json`
- protocol: 10 learners, 90 observed history steps, 10 target simulation steps
- feedback mode: teacher-forcing
- NCDM proficiency:
  `outputs/dneuralcdm/moocradar_90_10_10x10_medium70_e30_d32_h64/stu_know_proficiency.json`
- locked result:
  `locked_results/moocradar_medium_agent_style_tool_four_tier_full.json`
- output validity: 100/100 valid LLM outputs
- model configuration at lock time: `deepseek-v4-flash`

## Current Method

The locked Full simulator uses:

- cognitive learner profile from historical response statistics;
- compact learner memory from historical and recent interactions;
- NCDM-first learning tool state with IRT as secondary difficulty evidence;
- Agent4Edu-style internal task protocol;
- Four-tier output: answer, answer confidence, reasoning, and reasoning confidence;
- external response scoring and unified evaluation.

## Locked Result

True correct rate of the fixed target batch: 70.00%.

| Method | ACC | F1 | Balanced ACC | Specificity | MCC | LDE | CDE | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current Full | 65.00% | 72.00% | 65.48% | 66.67% | 0.285 | 0.150 | 0.407 | 23m27s |

Confusion matrix:

- TN: 20
- FP: 10
- FN: 25
- TP: 45

## Same-Batch Reference Results

| Method | ACC | F1 | Balanced ACC | Specificity | MCC | LDE | CDE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced-state Full historical version | 68.00% | 75.38% | 66.67% | 63.33% | 0.312 | 0.160 | 0.332 |
| Historical best Full | 61.00% | 69.29% | 59.76% | 56.67% | 0.181 | 0.170 | 0.404 |
| Agent4Edu-style baseline | 70.00% | 82.35% | 50.00% | 0.00% | None | 0.300 | 0.313 |
| Probability-sampling baseline | 53.00% | 62.99% | 50.24% | 43.33% | 0.004 | 0.190 | 0.426 |
| DKT target prediction | 74.00% | 83.95% | 58.57% | 20.00% | 0.290 | 0.220 | 0.222 |

## Reproduction Command

```powershell
$env:DASHSCOPE_API_KEY=(Get-Content -Raw -LiteralPath 'key.txt').Trim()
python experiments\ablation\run_ablation.py `
  --simulator multi-role `
  --dataset-root data\moocradar `
  --cohort-file experiments\cohorts\moocradar_90_10_10x10_medium70.json `
  --dneuralcdm-proficiency outputs\dneuralcdm\moocradar_90_10_10x10_medium70_e30_d32_h64\stu_know_proficiency.json `
  --variants full `
  --parallel-variants 1 `
  --feedback-mode teacher-forcing `
  --progress `
  --save-steps `
  --checkpoint-dir outputs\ablation\checkpoints\moocradar_medium_agent_style_tool_four_tier `
  --output outputs\ablation\moocradar_medium_agent_style_tool_four_tier_full.json
```

No API key, raw dataset, checkpoint, or large output cache is included in this
repository snapshot.
