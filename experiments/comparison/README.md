# Comparison Experiments

All baselines use the same unique learners and the fixed 90-history + 10-target protocol.

The main entry point is `run_comparison.py`. Use `multi-role` for the current
NCDM-grounded process-oriented simulator.

Run a fixed-cohort comparison:

```powershell
python experiments/comparison/run_comparison.py `
  --baselines multi-role,agent4edu,random `
  --dataset-root data/moocradar `
  --cohort-file experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch01_50x10_seed20260803.json `
  --dneuralcdm-checkpoint outputs/dneuralcdm/moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64/best_model.pt `
  --feedback-mode teacher-forcing `
  --progress
```

All methods save complete per-step traces and prompts by default. The report also
stores a secret-redacted run manifest. If the requested output already exists,
the new run is written to a timestamped sibling file and the earlier result is
left untouched.

The comparison contains:

- `multi-role`: current learner-state profile, NCDM/IRT/history evidence,
  item-conditioned integration, Four-tier response, and state evolution method.
- `full`: older LLMLearnerSimulator branch, mainly for locked historical
  baselines.
- `agent4edu`: isolated reproduction of the official Task1-Task4 action prompt
  and reflection flow. It exposes the reference answer and analysis and uses
  Task4 as the response prediction.
- `random`: command-line name for the `Probability-sampling Baseline`, which
  samples Bernoulli responses from the project's structured `p_correct` estimate.
  It is not a uniform random 0.5 baseline.

The Agent4Edu result is not directly task-identical to `full`: its official prompt exposes answer metadata and asks for a Task4 correctness judgment. Reports mark this explicitly.

For result interpretation, do not compare ACC/F1 alone. Positive-skewed cohorts
can make all-correct behavior look strong, so reports should include Balanced
Accuracy, Specificity, MCC, LDE, and CDE.

After running several methods on the same cohort, use the paper-oriented
evaluation summary:

```powershell
python experiments/evaluation/summarize_results.py `
  --inputs outputs/comparison/<ours>.json outputs/comparison/<agent4edu>.json outputs/comparison/<random>.json outputs/dkt/<target_eval>.json `
  --names "Ours Full" "Agent4Edu-style" "Probability-sampling Baseline" "DKT" `
  --output outputs/evaluation/summary.json `
  --markdown-output outputs/evaluation/summary.md
```

Agent4Edu's Task1-Task4 action prompt and this project's diagnostic task
decomposition are not task-identical. Compare them through the unified response,
distribution, task, and diagnostic metric layers.
