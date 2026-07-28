# Comparison Experiments

All baselines use the same unique learners and the fixed 90-history + 10-target protocol.

The main entry point is `run_comparison.py`. Use `multi-role` for the current
three-stage educational simulator: Learner Profile Encoder, Item-conditioned
Evidence Encoder, and Four-tier Response Simulator.

Run a fixed-cohort comparison:

```powershell
python experiments/comparison/run_comparison.py `
  --baselines multi-role,agent4edu,random `
  --cohort-file experiments/cohorts/moocradar_90_10_10x10.json `
  --dkt-proficiency outputs/dkt/moocradar_full_e50_h100/moocradar_90_10_10x10_proficiency.json `
  --feedback-mode teacher-forcing `
  --progress `
  --save-steps
```

The comparison contains:

- `multi-role`: current three-stage educational learner simulator, with
  external answer scoring.
- `full`: older LLMLearnerSimulator branch, mainly for locked historical
  baselines.
- `agent4edu`: isolated reproduction of the official Task1-Task4 action prompt and reflection flow. It exposes the reference answer and analysis, uses Task4 as the response prediction, and substitutes this project's dynamic mastery for DNeuralCDM.
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
