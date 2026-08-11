# Version Lock: 2026-07-04 FoundationalAssist Calibrated Baseline

This file fixes the current learner-simulator code state as a reproducible
baseline before further method redesign.

## Frozen Version ID

`baseline-2026-07-04-foundationalassist-calibrated-v1`

## Scope

This lock covers the current code under:

- `src/learner_simulator/`
- `scripts/`
- `experiments/`
- `configs/`
- `requirements.txt`
- `README.md`
- `AGENTS.md`

Generated caches, trained checkpoints, raw datasets, and large experiment output
files are not part of the source snapshot. Key result files are referenced below.

## Current Full Simulator

The current Full simulator is an LLM-based first-attempt learner simulator using:

- fixed Agent4Edu-style protocol: 90 observed history interactions and 10 target
  simulation interactions per learner;
- statistical cognitive profile from observed history;
- compact ability profile from observed history and global item statistics;
- short-term and reinforced memory from observed and simulated records;
- DKT external proficiency when a proficiency file is provided;
- non-cognitive behavioral factors;
- prompt-level response tendency calibration;
- four-tier output contract: `StudentAnswer`, `AnswerConfidence`,
  `StudentReasoning`, and `ReasoningConfidence`;
- external answer scoring against question metadata.

The Full simulator must not feed `p_correct`, sampled correctness labels,
reference answers, or reference analyses into the LLM prompt. `p_correct` remains
only a statistical/random baseline signal.

## Current Known State

- `LLMLearnerSimulator` currently sets `cognitive_strategy = None` at runtime.
  The current Full model therefore does not actively use the older cognitive
  strategy controller, even though the ablation runner still exposes a
  `no-cognitive-selection` switch.
- The active prompt-control mechanism is response tendency calibration, not
  post-hoc correctness flipping.
- Correctness is always computed externally from the submitted answer.
- The current baseline is not considered the final publishable method. It is a
  fixed comparison point before adding a calibrated response decoder or other
  stronger modeling layer.

## LLM Configuration At Lock Time

`configs/llm.example.json`:

- provider: OpenAI-compatible
- model: `deepseek-v4-flash`
- base URL: Aliyun Bailian compatible endpoint
- temperature: `0.2`
- max tokens: `4096`
- streaming: enabled
- thinking: enabled

The API key is intentionally not stored in source files.

## Fixed Dataset/Experiment Context

Recent fixed experiment context:

- dataset: `FOUNDATIONALASSIST`
- prepared data root: `data/foundationalassist`
- cohort file:
  `experiments/cohorts/foundationalassist_90_10_10x10.json`
- protocol: 10 learners, 90 observed history steps, 10 simulated target steps
- DKT proficiency:
  `outputs/dkt/foundationalassist_e50_h100/foundationalassist_90_10_10x10_proficiency.json`
- latest Full vs ability-profile ablation result:
  `outputs/ablation/foundationalassist_10x10_calibrated_full_no_ability.json`

## Latest Locked Result

The latest calibrated FOUNDATIONALASSIST 10x10 run produced:

| Method | ACC | F1 | Balanced ACC | Specificity | MCC | LDE | CDE | Predicted Correct Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 45.00% | 40.86% | 50.21% | 70.27% | 0.0045 | 0.470 | 0.487 | 30.00% |
| w/o ability profile | 45.00% | 48.60% | 46.31% | 51.35% | -0.0718 | 0.390 | 0.439 | 44.00% |

True correct rate of this fixed target batch: 63.00%.

Interpretation at lock time:

- response tendency calibration reduced the previous over-pessimism;
- the ability profile preserves negative-sample discrimination but still makes
  the model conservative;
- prompt-only simulation remains insufficient for the ideal goal of beating
  Agent4Edu simultaneously on ACC, F1, and Balanced ACC.

## Reproduction Commands

Run regression tests:

```powershell
python scripts/test_agent4edu_protocol.py
python scripts/test_four_tier.py
python scripts/test_experiment_scaffold.py
python scripts/test_cognitive_strategy.py
```

Run the locked Full vs no-ability-profile experiment:

```powershell
$env:DASHSCOPE_API_KEY=(Get-Content -Raw -LiteralPath 'E:\yyx\8 Learner Simulator\key.txt').Trim()
python experiments\ablation\run_ablation.py `
  --dataset-root data\foundationalassist `
  --cohort-file experiments\cohorts\foundationalassist_90_10_10x10.json `
  --history-steps 90 `
  --target-steps 10 `
  --dkt-proficiency outputs\dkt\foundationalassist_e50_h100\foundationalassist_90_10_10x10_proficiency.json `
  --variants full,no-ability-profile `
  --parallel-variants 2 `
  --checkpoint-dir outputs\ablation\foundationalassist_10x10_calibrated_full_no_ability_checkpoints `
  --output outputs\ablation\foundationalassist_10x10_calibrated_full_no_ability.json `
  --progress `
  --save-steps
```

## Source Integrity

The source hash manifest for this lock is:

`outputs/version_locks/baseline-2026-07-04-foundationalassist-calibrated-v1_manifest.csv`

The source zip snapshot for this lock is:

`outputs/version_locks/baseline-2026-07-04-foundationalassist-calibrated-v1_source.zip`
