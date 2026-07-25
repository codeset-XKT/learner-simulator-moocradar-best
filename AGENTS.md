# Project Rules

## Locked Baselines

- Previous fixed baseline: `baseline-2026-07-04-foundationalassist-calibrated-v1`.
- Read `VERSION_LOCK.md` before changing prompts, simulator decision logic, profile modules, or ablation wiring.
- Treat this lock as the reproducible reference for the prompt-calibrated Full simulator at that lock point. The working tree may contain later experimental code.
- Historical MoocRadar good-result baseline:
  `baseline-2026-07-04-moocradar-ability-summary-no-irt-v1`.
- Read `VERSION_LOCK_MOOCRADAR.md` before comparing against the MoocRadar ability-summary result.
- Current fixed-cohort result ledger: `RESULTS_SUMMARY.md`. Update it when a
  new run changes the best-known result or exposes a new failure mode.

## Experiment Protocol

- Use only the fixed Agent4Edu-style protocol.
- Aggregate sequence rows by unique UID and sort interactions chronologically.
- Use interactions 1-90 as observed history and interactions 91-100 as simulation targets.
- Observed history must initialize Profile, IRT, mastery, short-term memory, and long-term memory.
- Simulate all 10 target interactions sequentially. Do not add ratio-based, original-split, or variable-step experiment modes.
- Random, probability, and LLM baselines must use exactly the same learners and target interactions.

## LLM Contract

- The supported output contract is the four-tier response defined in `four_tier.py`.
- The explicit `no-four-tier` ablation may use the answer-only contract; it must retain the same 90+10 data, state, and external-scoring pipeline.
- The LLM generates StudentAnswer, AnswerConfidence, StudentReasoning, and ReasoningConfidence.
- Correctness is determined externally from StudentAnswer and question metadata.
- Do not restore Task4 self-evaluation or feed `p_correct`, sampled labels, reference answers, or reference analyses into the prompt.
- The item-conditioned ability evidence may shape reasoning depth, confidence, and learner-level expression. It must not prescribe a correct/incorrect label.
- The `multi-role` experiment name now maps to a compact three-stage educational simulation pipeline: Learner Profile Encoder, Item-conditioned Evidence Encoder, and Four-tier Response Simulator. External four-tier scoring is an evaluation step, not an internal simulation module. Downstream modules receive structured summaries, not raw nested prompts.
- Treat `multi-role` as the active method branch. Treat `full` as the older
  LLMLearnerSimulator branch unless a version lock asks for it explicitly.

## Verification

```powershell
python scripts/test_agent4edu_protocol.py
python scripts/test_four_tier.py
python scripts/test_experiment_scaffold.py
python scripts/test_cognitive_strategy.py
python scripts/evaluate.py --simulator random --source-rows 1000 --max-users 3
```

## Experiment Layout

- Put model/baseline comparisons in `experiments/comparison/`.
- Put module ablations in `experiments/ablation/`.
- Every comparison and ablation must reuse the same selected UIDs and target interactions.
- The isolated Agent4Edu reproduction may expose reference answers and analyses because the official action prompt does so. Reports must mark this leakage explicitly.
- Do not import Agent4Edu's Task4 self-evaluation or answer leakage into the project's full simulator.
- The Agent4Edu reproduction uses this project's dynamic mastery estimator in place of DNeuralCDM, as an explicit adapter.
- Do not report ACC/F1 alone on positive-skewed cohorts. Always include
  Balanced Accuracy, Specificity, MCC, LDE, and CDE.
