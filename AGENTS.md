# Project Rules

## Method Boundary

- `multi-role` is the active paper method; `full` is the legacy
  `LLMLearnerSimulator` baseline.
- Multi-role Full uses one DNeuralCDM checkpoint for initial history-conditioned
  state, current-item probability, and dynamic state evolution.
- Never inject DKT, MIKT, exported NCDM proficiency JSON, random-baseline
  `p_correct`, or sampled labels into multi-role Full.
- The response remains the LLM's `LearnerCorrect` decision. Never override an
  LLM disagreement with an NCDM/DKT threshold label.
- The reference answer may be shown only for rendering `StudentAnswer` after
  the correctness decision. Reference analysis and target labels stay hidden.

## Experiment Protocol

- Aggregate sequence rows by unique UID and sort chronologically.
- Default formal protocol: interactions 1-90 are observed history and 91-100
  are sequential simulation targets.
- Use identical learners, targets, ordering, LLM configuration, feedback mode,
  and scoring for every comparison and ablation.
- Fit NCDM and profile-normalization statistics outside all cohort UIDs. Cohort
  history is inference context, never training or normalization data.
- Full must reject checkpoints without per-user holdout metadata. Train with
  the exact fixed `--cohort-file`; do not grandfather old checkpoints.
- `rollout` feeds simulated responses back; `teacher-forcing` feeds target
  labels back only after scoring the current prediction.
- Exclude an entire learner sequence from LLM aggregate metrics when any step
  has an API or parse failure; preserve every raw step in the archived report.

## Official Ablations

- `no-learner-state-profile`
- `no-item-conditioned-integration`
- `no-four-tier`
- `no-dynamic-state-evolution`
- `no-ncdm`, `no-irt`, and `no-ncdm-irt`

Do not restore historical aliases. `no-four-tier` removes only confidence and
reasoning tiers; it retains the evidence, reference answer, concept decision,
`LearnerCorrect`, submitted answer, and scoring path.

## Evaluation

- Report ACC, F1, Balanced Accuracy, Specificity, MCC, LDE, and CDE.
- Report NCDM/DKT probability count and coverage whenever their direct
  predictive metrics are shown.
- Four-tier answer matching, confidence calibration, task consistency, and
  mastery monotonicity are diagnostics, not replacements for response metrics.
- The isolated Agent4Edu reproduction may expose answer metadata because its
  official prompt does; reports must state this explicitly.

## Locked References

- `VERSION_LOCK_MOOCRADAR_FAIR_ABLATION.md`: active leakage-safe Full and
  fair-ablation implementation over ten disjoint 50x10 cohorts.
- `VERSION_LOCK.md`: FoundationalAssist historical calibrated baseline.
- `VERSION_LOCK_MOOCRADAR.md`: MoocRadar historical ability-summary baseline.
- `RESULTS_SUMMARY.md`: historical result ledger, not the active method spec.

## Verification

Run `scripts/test_ablation_variants.py`, `scripts/test_multi_role_simulator.py`,
`scripts/test_profile_ablation_prompt.py`, `scripts/test_four_tier.py`,
`scripts/test_evaluation_views.py`, `scripts/test_experiment_scaffold.py`, and
`scripts/test_dneuralcdm_loader.py` after method or ablation changes.
