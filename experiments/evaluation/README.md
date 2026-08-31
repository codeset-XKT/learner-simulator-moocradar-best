# Paper-Oriented Evaluation

This directory contains result summarization utilities. It does not run LLM
simulation by itself; it reads existing comparison, ablation, and DKT JSON
outputs and converts them into a unified paper-facing view.

## Metric Layers

The project reports four layers:

- Response consistency: ACC, F1, Balanced Accuracy, Specificity, MCC, and AUC
  when a probability score is available.
- Distribution consistency: Agent4Edu-compatible ROUGE-3, Learner Distribution
  Error (LDE), and Concept Distribution Error (CDE). ROUGE-3 compares each
  learner's ordered simulated and real binary response sequences using
  multiset trigram overlap. The paper-facing `rouge_3` value is macro F1 over
  learners; precision, recall, and evaluated-user count are also archived.
- Task consistency: learner-state/task availability, concept selection,
  response-generation consistency, and state-evolution diagnostics when the
  simulator exposes them.
- Diagnostic consistency: mastery-bucket monotonicity, confidence monotonicity,
  and Four-tier confidence calibration.

This separates the paper story from a single answer-correctness metric. DKT is
still reported as a strong predictive reference, while LLM simulators are also
evaluated for diagnostic response behavior.

## Summarize Existing Results

Example for the MoocRadar medium-correct-rate cohort:

```powershell
python experiments\evaluation\summarize_results.py `
  --inputs `
    outputs\ablation\moocradar_10x10_medium70_dkt_first_rule_full_rerun.json `
    outputs\comparison\moocradar_10x10_medium70_agent4edu.json `
    outputs\comparison\moocradar_10x10_medium70_random.json `
    outputs\dkt\moocradar_full_e50_h100\moocradar_90_10_10x10_medium70_target_eval.json `
  --names "Ours Full" "Agent4Edu-style" "Probability-sampling Baseline" "DKT" `
  --output outputs\evaluation\moocradar_medium70_summary.json `
  --markdown-output outputs\evaluation\moocradar_medium70_summary.md
```

The Markdown output marks the best value in each metric column. Arrows in the
headers indicate the preferred direction.

## Task Interpretation

Agent4Edu's tasks are external educational-agent tasks:

- attempt decision
- knowledge concept identification
- problem-solving process and final answer generation
- correctness prediction

This project's current simulator uses a process-oriented decomposition:

- learner-state/profile inference
- item-conditioned knowledge and ability activation
- Four-tier response generation
- state-evolution diagnostics

Direct NCDM or DKT predictive metrics are computed on every step for which the
model exposes a valid probability. Always report `*_probability_count` and
`*_probability_coverage` beside those metrics when coverage is below 100%.

The two designs should be compared at the metric level, not treated as identical
task definitions.
