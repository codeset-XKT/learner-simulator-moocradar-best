# Learner Simulator

This project studies LLM-based learner simulation under a fixed knowledge
tracing style protocol. The active experiments compare this project's simulator
against Agent4Edu-style and probability-sampling baselines on fixed cohorts.

## Current Method

The current research method is exposed as the `multi-role` baseline name for
backward compatibility. It is a compact process-oriented learner simulator:

```text
Learner Profile Encoder
NCDM State Evidence + IRT Evidence + Observed-history Replay
Item-conditioned Evidence Integration
Four-tier Learner Response Generation
Dynamic State Evolution
```

- Learner Profile Encoder: stable learner-level traits derived from observed
  history, represented as non-overlapping cognitive and ability summaries.
- NCDM State Evidence: one trained DNeuralCDM checkpoint is the canonical
  knowledge-state source. Full infers the initial concept state directly from
  the observed sequence and uses the same checkpoint for current-item response
  probability and target-stage state evolution. It does not consume DKT, MIKT,
  or an exported proficiency JSON.
- IRT Ability-Difficulty Evidence: a Rasch/1PL module estimates learner ability
  `theta`, item difficulty `beta`, and their relative challenge from observed
  history. It is exposed to the LLM as ability-versus-difficulty evidence, not
  as concept mastery, not as `p_correct`, and not as a sampled correctness
  label.
- Item-conditioned Evidence Integration: a qualitative summary of profile,
  related memory, current concept readiness, IRT challenge, and item demand.
  It does not duplicate the raw NCDM probability or prescribe a label.
- Historical Reflective Calibration: the observed history is internally split
  into 80 prefix interactions and 10 replay interactions under the default
  90-history protocol. Replay uses the NCDM state after the prefix and never
  uses target labels.
- Four-tier Response Generation: the LLM first outputs `LearnerCorrect`, then
  renders `StudentAnswer`, confidence, and reasoning tiers. The reference answer
  is exposed only for answer rendering after that decision; external answer
  matching is retained as a consistency diagnostic.
- Dynamic State Evolution: after each target interaction, the selected rollout
  or teacher-forcing feedback updates memory and recomputes the latent NCDM
  state from the complete accepted prefix.

The Full/LLM simulator still exists as a historical baseline. The current
paper-oriented method should use `multi-role` unless a locked baseline document
explicitly says otherwise.

## Fixed Protocol

All formal experiments use the Agent4Edu-style fixed-history protocol:

```text
90 observed interactions -> profile / memory / NCDM state initialization
10 target interactions   -> sequential learner simulation and evaluation
```

Rules:

- Aggregate rows by unique UID and sort interactions chronologically.
- Use interactions 1-90 as observed history and 91-100 as simulation targets.
- Use the same learners and target interactions for every baseline and ablation.
- Use the LLM's structured `LearnerCorrect` field as the simulated response;
  never replace a disagreement with an NCDM/DKT threshold decision.
- Do not feed random-baseline `p_correct`, sampled labels, target labels, or
  reference analyses into the `multi-role` prompt. A reference answer may be
  supplied only after the correctness-decision instruction for rendering a
  valid submitted answer.
- The paper-oriented `multi-role` Full run requires
  `--dneuralcdm-checkpoint`. The exported `--dneuralcdm-proficiency`, DKT, and
  MIKT paths are ignored by this method. The checkpoint must be trained with
  the same `--cohort-file`; Full rejects old checkpoints that cannot prove each
  evaluated learner was excluded from training.
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

## Version Locks

The active paper-oriented implementation is frozen in
`VERSION_LOCK_MOOCRADAR_FAIR_ABLATION.md`. It uses the ten disjoint MoocRadar
50x10 cohorts and one leakage-safe NCDM checkpoint trained while excluding all
500 planned evaluation learners.

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
- model: `deepseek-v4-flash-0731`
- base URL: Aliyun Bailian compatible endpoint
- temperature: `0.0`
- streaming: enabled
- thinking: enabled

## Running Comparisons

Train a leakage-safe NCDM checkpoint for the fixed cohort before the first
formal run (the trainer preserves the true best-validation epoch):

Newly trained checkpoints use the corrected `architecture_version=2`: exercise
discrimination is constrained positive, padded sequence positions are excluded
from both loss and validation metrics, and the best epoch is selected by
valid-position validation BCE. Legacy checkpoints remain loadable with their
original forward semantics for historical-result reproduction.

```powershell
python scripts\train_dneuralcdm.py `
  --dataset-root data\moocradar `
  --cohort-file experiments\cohorts\moocradar_500x10_batches\moocradar_500x10_all_500users_seed20260803.json `
  --output-dir outputs\dneuralcdm\moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64 `
  --epochs 30 `
  --embedding-dim 32 `
  --hidden-dim 64
```

If an otherwise complete LLM run contains a few transient API/format failures,
rerun the affected learners with `--simulate-uids`, then merge only complete
learner sequences:

```powershell
python scripts\merge_sequence_repairs.py `
  --base outputs\ablation\main\combined.json `
  --repair outputs\ablation\main\repairs\combined.json `
  --variant full `
  --output outputs\ablation\main\combined_repaired.json
```

Run current multi-role, Agent4Edu, and probability-sampling baselines on a fixed
cohort. The command-line name is still `random` for backward compatibility, but
paper tables should call it `Probability-sampling Baseline` because it samples
from an estimated `p_correct`, not from a uniform 0.5 coin flip:

```powershell
python experiments\comparison\run_comparison.py `
  --baselines multi-role,agent4edu,random `
  --dataset-root data\moocradar `
  --cohort-file experiments\cohorts\moocradar_500x10_batches\moocradar_500x10_batch01_50x10_seed20260803.json `
  --dneuralcdm-checkpoint outputs\dneuralcdm\moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64\best_model.pt `
  --feedback-mode teacher-forcing `
  --progress
```

Run only the current multi-role simulator on FoundationalAssist:

```powershell
python experiments\comparison\run_comparison.py `
  --baselines multi-role `
  --dataset-root data\foundationalassist `
  --cohort-file experiments\cohorts\foundationalassist_90_10_10x10.json `
  --dneuralcdm-checkpoint outputs\dneuralcdm\foundationalassist_90_10_10x10_medium65_e30_d32_h64\best_model.pt `
  --feedback-mode teacher-forcing `
  --progress `
  --output outputs\comparison\foundationalassist_10x10_multi_role_ability_evidence_teacher_forcing.json
```

## Running Ablations

The ablation runner keeps the same learners, targets, sequence order, LLM
configuration, and external scoring:

```powershell
python experiments\ablation\run_ablation.py `
  --simulator multi-role `
  --dataset-root data\moocradar `
  --cohort-file experiments\cohorts\moocradar_500x10_batches\moocradar_500x10_batch01_50x10_seed20260803.json `
  --dneuralcdm-checkpoint outputs\dneuralcdm\moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64\best_model.pt `
  --variants full,no-learner-state-profile,no-item-conditioned-integration,no-four-tier,no-dynamic-state-evolution `
  --parallel-variants 2 `
  --progress
```

The supplementary knowledge-state-source variants are `no-ncdm`, `no-irt`,
and `no-ncdm-irt`. The paper-facing ablations preserve a common response task:

- `no-four-tier` uses the reduced-response contract, retaining `LearnerCorrect`,
  reference-answer rendering, and the same primary response label while removing
  reasoning, confidence, and Four-tier diagnosis.
- `no-item-conditioned-integration` removes only the qualitative integration
  section. Item-conditioned evidence never consumes the NCDM current-item
  response probability; that probability is retained only for external metrics.
- `no-dynamic-state-evolution` freezes the prompt-facing latent knowledge state
  while preserving the same teacher-forced response prefix and target-memory
  updates as Full. It does not change teacher forcing into rollout.

Long ablation runs save per-variant checkpoints under the configured checkpoint
directory as each variant finishes.

All experiment entry points save every per-step trace and complete prompt by
default. Reports include a secret-redacted run manifest. Existing report and
checkpoint files are never overwritten; reruns with the same output name create
a timestamped sibling file. Use `--no-save-steps` or `--no-include-prompt` only
for temporary debugging runs where a compact report is intentional.

## Project Layout

```text
configs/llm.example.json
experiments/common.py
experiments/comparison/run_comparison.py
experiments/ablation/run_ablation.py
experiments/evaluation/summarize_results.py
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
- `dneuralcdm.py`: canonical Full knowledge-state and response predictor.
- `dkt.py`: DKT predictive-reference support; not injected into Full.
- `educational_multi_agent_prompt.py`: current multi-role prompt builders.
- `four_tier.py`: four-tier parsing and external scoring helpers.
- `evaluation.py`: response, distribution, and cognitive-consistency metrics.
- `evaluation_views.py`: paper-oriented metric layers and result-table views.
- `simulators/multi_role_simulator.py`: current multi-role simulator.

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

For paper tables, summarize existing outputs with:

```powershell
python experiments\evaluation\summarize_results.py `
  --inputs <ours.json> <agent4edu.json> <random.json> <dkt_target_eval.json> `
  --names "Ours Full" "Agent4Edu-style" "Probability-sampling Baseline" "DKT" `
  --output outputs\evaluation\summary.json `
  --markdown-output outputs\evaluation\summary.md
```

This script splits results into response consistency, distribution consistency,
task consistency, and diagnostic consistency. Agent4Edu and DKT naturally have
`None` for diagnostic fields they do not expose.

## Verification

Run offline regression tests before and after prompt or simulator changes:

```powershell
python scripts/test_agent4edu_protocol.py
python scripts/test_four_tier.py
python scripts/test_experiment_scaffold.py
python scripts/test_cognitive_strategy.py
python scripts/test_multi_role_simulator.py
python scripts/test_evaluation_views.py
```

For syntax checks:

```powershell
python -m compileall src scripts experiments
```
