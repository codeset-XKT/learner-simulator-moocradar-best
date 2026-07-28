# Learner Simulator GitHub Snapshot

This repository snapshot contains two fixed MoocRadar baselines.

## Current Fixed Version

The current executable version is:

`baseline-2026-07-28-moocradar-medium-agent-style-four-tier-full-v1`

It is documented in:

`VERSION_LOCK_MOOCRADAR_MEDIUM_AGENT_STYLE.md`

The locked result file is:

`locked_results/moocradar_medium_agent_style_tool_four_tier_full.json`

On the fixed MoocRadar medium-difficulty 10x10 batch, the current Full simulator
obtained:

- ACC: 65.00%
- F1: 72.00%
- Balanced ACC: 65.48%
- Specificity: 66.67%
- MCC: 0.285
- LDE: 0.150
- CDE: 0.407

## Historical Best Baseline

This repository also preserves the older locked MoocRadar historical best
baseline:

`baseline-2026-07-04-moocradar-ability-summary-no-irt-v1`

That locked result is documented in `VERSION_LOCK_MOOCRADAR.md`.

The `locked_results/` directory contains:

- the current MoocRadar medium-difficulty 10x10 result JSON;
- the historical MoocRadar 10x10 result JSON;
- the source/evidence manifest recorded for the lock.

No API key, raw dataset, checkpoint, or large output cache is included.
