# Version Lock: 2026-07-26 MoocRadar Balanced-State Light

This file fixes the current Balanced-state light Full version after the
negative-proxy prompt was softened.

## Frozen Version ID

`baseline-2026-07-26-moocradar-balanced-state-light-v1`

## What Changed

The executable code is intentionally close to the historical Balanced-state
Full version. The only prompt change in this lock is in
`src/learner_simulator/agent4edu_prompt.py`: cognitive-affective and error
fields are framed as broad tendencies rather than automatic error triggers.

This avoids the failed DKT-weighted variant where explicit KT predicted labels
were exposed to the response prompt. The active prompt still keeps KT as
qualitative learner-state evidence rather than a post-hoc override or direct
answer label.

## Experiment Context

- dataset: `MoocRadar`
- cohort file: `experiments/cohorts/moocradar_90_10_10x10.json`
- protocol: 10 learners, 90 observed history steps, 10 target simulation steps
- target correct rate: 89.00%
- DKT proficiency:
  `outputs/dkt/moocradar_full_e50_h100/moocradar_90_10_10x10_proficiency.json`
- LLM model in `configs/llm.example.json`: `deepseek-v4-flash`
- feedback mode: `teacher-forcing`

## Locked Results

| Method | ACC | F1 | Balanced Acc | Specificity | MCC | LDE | CDE | Predicted Correct Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DKT reference | 93.00% | 96.17% | 72.17% | 45.45% | 0.584 | — | — | 94.00% |
| p_correct>=0.5 | 92.00% | 95.65% | 67.62% | 36.36% | 0.506 | — | — | 95.00% |
| Agent4Edu-style | 89.00% | 94.18% | 50.00% | 0.00% | null | 0.110 | 0.220 | 100.00% |
| All-correct baseline | 89.00% | 94.18% | 50.00% | 0.00% | null | — | — | 100.00% |
| Ours Full | 75.00% | 84.47% | 70.02% | 63.64% | 0.279 | 0.210 | 0.418 | 72.00% |
| Random simulator | 75.00% | 85.03% | 58.07% | 36.36% | 0.122 | 0.190 | 0.228 | 78.00% |

## Interpretation

This high-correct-rate batch rewards all-correct behavior on ACC and F1. The
important result is that Ours Full keeps substantially better negative-sample
recognition than Agent4Edu-style: Specificity is 63.64% versus 0.00%, and
Balanced Accuracy is 70.02% versus 50.00%.

DKT remains the strongest pure predictor on this batch. The simulator should
therefore not be claimed as a better KT predictor; its current strength is
behavioral simulation with explicit negative-sample recognition under the
four-tier learner response protocol.

## Locked Artifacts

- `locked_results/moocradar_10x10_normal_balanced_state_light_full.json`
- `locked_results/moocradar_10x10_normal_agent4edu.json`
- `locked_results/moocradar_10x10_normal_random.json`
- `locked_results/moocradar_10x10_normal_balanced_state_light_summary.json`
