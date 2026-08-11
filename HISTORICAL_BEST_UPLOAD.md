# Learner Simulator GitHub Snapshot

This repository preserves several fixed MoocRadar baselines alongside the
active fair-ablation implementation.

## Active Version

The active executable version is:

`baseline-2026-08-11-moocradar-fair-ablation-v1`

Its method boundary, formal 500-user exclusion plan, and verified Batch 02/03
results are documented in `VERSION_LOCK_MOOCRADAR_FAIR_ABLATION.md`.

## Historical Fixed Version

The later historical prompt variant is:

`baseline-2026-07-28-moocradar-medium-agent-style-four-tier-full-v1`

It is documented in:

`VERSION_LOCK_MOOCRADAR_MEDIUM_AGENT_STYLE.md`

Its locked result file is:

`locked_results/moocradar_medium_agent_style_tool_four_tier_full.json`

On the fixed MoocRadar medium-difficulty 10x10 batch, that Full simulator
obtained:

- ACC: 65.00%
- F1: 72.00%
- Balanced ACC: 65.48%
- Specificity: 66.67%
- MCC: 0.285
- LDE: 0.150
- CDE: 0.407

## Historical Best Baseline

The repository also preserves the older locked MoocRadar historical best
baseline:

`baseline-2026-07-04-moocradar-ability-summary-no-irt-v1`

That locked result is documented in `VERSION_LOCK_MOOCRADAR.md`.

The `locked_results/` directory contains historical result artifacts. The
active fair-ablation lock records compact result tables rather than committing
the very large raw per-step reports.

This repository includes:

- historical MoocRadar medium-difficulty result JSON files;
- the historical MoocRadar 10x10 result JSON;
- the source/evidence manifest recorded for the historical lock;
- the ten formal disjoint 50x10 cohort files and their aggregate 500-user
  exclusion plan.

No API key, raw dataset, checkpoint, or large output cache is included.
