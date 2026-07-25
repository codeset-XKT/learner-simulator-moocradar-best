# Learner Simulator

This project studies LLM-based learner simulation under a fixed knowledge
tracing style protocol. The active experiments compare this project's simulator
against Agent4Edu-style and random baselines on fixed cohorts.

## Current Method

The current research method is exposed as the `multi-role` baseline name for
backward compatibility. Internally it is now a compact three-stage educational
simulation pipeline:

```text
Learner Profile Encoder
Item-conditioned Evidence Encoder
Four-tier Response Simulator
```

- Learner Profile Encoder: stable learner-level traits derived from observed
  history, including general performance, stability, error tendency,
  affective-state proxies, transfer traits, and broad ability traits.
- Item-conditioned Evidence Encoder: interaction-level evidence derived from
  the current item, DKT state, related practice memory, item demand, and
  transfer burden.
- Four-tier Response Simulator: LLM generation of `StudentAnswer`,
  `AnswerConfidence`, `StudentReasoning`, and `ReasoningConfidence`; correctness
  is scored externally against question metadata for evaluation only.

The Full/LLM simulator still exists as a historical baseline. The current
paper-oriented method should use `multi-role` unless a locked baseline document
explicitly says otherwise.

## Fixed Protocol

All formal experiments use the Agent4Edu-style fixed-history protocol:

```text
90 observed interactions -> profile / memory / mastery initialization
10 target interactions   -> sequential learner simulation and evaluation
```

Rules:

- Aggregate rows by unique UID and sort interactions chronologically.
- Use interactions 1-90 as observed history and 91-100 as simulation targets.
- Use the same learners and target interactions for every baseline and ablation.
- Score `StudentAnswer` externally; do not let the LLM self-label correctness.
- Do not feed `p_correct`, sampled labels, reference answers, or reference
  analyses into this project's full or multi-role prompts.
- Agent4Edu reproduction may expose reference answers and analyses because its
  official action prompt does so; reports mark that leakage explicitly.

Target-step feedback is controlled by:

```powershell
--feedback-mode rollout
--feedback-mode teacher-forcing
```

`rollout` feeds simulated responses back into state and memory. `teacher-forcing`
scores the prediction but feeds the ground-truth target response into state and
memory before the next target step.

## Locked Baselines

Two historical baselines are fixed and should be read before changing prompt
logic, profile modules, simulator decision logic, or ablation wiring:

- `VERSION_LOCK.md`: FoundationalAssist calibrated Full baseline,
  `baseline-2026-07-04-foundationalassist-calibrated-v1`.
- `VERSION_LOCK_MOOCRADAR.md`: historical MoocRadar ability-summary baseline,
  `baseline-2026-07-04-moocradar-ability-summary-no-irt-v1`.

The current result ledger is `RESULTS_SUMMARY.md`. The latest medium-correct-rate
MoocRadar fixed-cohort result is documented in
`docs/moocradar_medium70_balanced_state.md`.

## Datasets

Prepared in-project data roots:

```text
data/foundationalassist
data/junyi
data/moocradar
```

External dataset roots used during development include:

```text
E:/yyx/KT数据集/XES3G5M/XES3G5M
E:/yyx/KT数据集/JunYi/JunYi
E:/yyx/KT数据集/MoocRadar/MOOCRadar
E:/yyx/KT数据集/FOUNDATIONALASSIST
```

Each prepared dataset should provide:

```text
kc_level/train_valid_sequences.csv
metadata/questions.json
```

Optional metadata such as mappings, images, or route maps can be used when
available.

## LLM Configuration

`configs/llm.example.json` uses an OpenAI-compatible chat endpoint. The API key
must be supplied through `DASHSCOPE_API_KEY` or loaded from the local `key.txt`;
do not store real keys in source-controlled config files.

```powershell
$env:DASHSCOPE_API_KEY=(Get-Content -Raw -LiteralPath 'E:\yyx\8 Learner Simulator\key.txt').Trim()
```

Current development config at the time of this cleanup:

- provider: OpenAI-compatible
- model: `deepseek-v4-flash`
- base URL: Aliyun Bailian compatible endpoint
- temperature: `0.2`
- streaming: enabled
- thinking: enabled

## Running Comparisons

Run current multi-role, Agent4Edu, and random baselines on a fixed cohort:

```powershell
python experiments\comparison\run_comparison.py `
  --baselines multi-role,agent4edu,random `
  --dataset-root data\moocradar `
  --cohort-file experiments\cohorts\moocradar_90_10_10x10.json `
  --dkt-proficiency outputs\dkt\moocradar_full_e50_h100\moocradar_90_10_10x10_proficiency.json `
  --feedback-mode teacher-forcing `
  --progress `
  --save-steps
```

Run only the current multi-role simulator on FoundationalAssist:

```powershell
python experiments\comparison\run_comparison.py `
  --baselines multi-role `
  --dataset-root data\foundationalassist `
  --cohort-file experiments\cohorts\foundationalassist_90_10_10x10.json `
  --dkt-proficiency outputs\dkt\foundationalassist_e50_h100\foundationalassist_90_10_10x10_proficiency.json `
  --feedback-mode teacher-forcing `
  --progress `
  --output outputs\comparison\foundationalassist_10x10_multi_role_ability_evidence_teacher_forcing.json
```

## Running Ablations

The ablation runner keeps the same learners, targets, sequence order, LLM
configuration, and external scoring:

```powershell
python experiments\ablation\run_ablation.py `
  --cohort-file experiments\cohorts\moocradar_90_10_10x10.json `
  --variants full,no-profile,no-memory,no-proficiency,no-four-tier,no-cognitive-selection,no-cognitive-profile,no-ability-profile `
  --parallel-variants 2 `
  --progress `
  --save-steps
```

Long ablation runs save per-variant checkpoints under the configured checkpoint
directory as each variant finishes.

## Project Layout

```text
configs/llm.example.json
experiments/common.py
experiments/comparison/run_comparison.py
experiments/ablation/run_ablation.py
scripts/prepare_foundationalassist.py
scripts/prepare_junyi.py
scripts/prepare_moocradar.py
scripts/train_dkt.py
scripts/export_dkt_proficiency.py
scripts/test_multi_role_simulator.py
src/learner_simulator/
```

Key source modules:

- `agent4edu_baseline.py`: Agent4Edu-style baseline prompt and parsing.
- `cognitive_profile.py`: statistical cognitive profile.
- `ability_profile.py`: compact ability summary profile.
- `dkt.py`: DKT model and proficiency export support.
- `educational_multi_agent_prompt.py`: current multi-role prompt builders.
- `four_tier.py`: four-tier parsing and external scoring helpers.
- `evaluation.py`: response, distribution, and cognitive-consistency metrics.
- `simulators/multi_role_simulator.py`: current three-stage simulator.

## Metrics

Response consistency:

- `sample_match_acc`
- `sample_f1`
- `response_balanced_accuracy`
- `response_specificity`
- `response_mcc`

Distribution consistency:

- `learner_distribution_error` (LDE, lower is better)
- `concept_distribution_error` (CDE, lower is better)

Probability/confidence diagnostics, when available:

- `llm_auc`
- `answer_confidence_auc`
- `llm_brier`
- `four_tier_mean_answer_confidence`
- `four_tier_mean_reasoning_confidence`

Because many fixed cohorts are positive-skewed, ACC and F1 are not sufficient.
Always report Balanced Accuracy, Specificity, MCC, LDE, and CDE with ACC/F1.

## Verification

Run offline regression tests before and after prompt or simulator changes:

```powershell
python scripts/test_agent4edu_protocol.py
python scripts/test_four_tier.py
python scripts/test_experiment_scaffold.py
python scripts/test_cognitive_strategy.py
python scripts/test_multi_role_simulator.py
```

For syntax checks:

```powershell
python -m compileall src scripts experiments
```
