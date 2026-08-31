from __future__ import annotations

"""Create stable, cohort-first aliases for formal experiment outputs.

The original directories are deliberately left in place because report JSON files
contain their historical paths.  This catalog is the single canonical browsing
surface: ``outputs/canonical/<dataset>/<cohort>/<method>``.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "outputs" / "canonical"


RUNS = {
    "moocradar/cohort_batch09/deepseek_v6_primaryroute": (
        "outputs/formal/v6_primaryroute/moocradar/"
        "batch02_50x10_cohort_batch09_20260826_p40"
    ),
    "moocradar/cohort_batch09/glm53_flash_v6_primaryroute": (
        "outputs/formal/v6_primaryroute/moocradar/"
        "glm53_flash_batch09_50x10_20260828_p40"
    ),
    "moocradar/cohort_batch09/agent4edu_officialalign_ncdmirt": (
        "outputs/comparison/"
        "moocradar_v6primary_batch09_50x10_agent4edu_"
        "officialalign_ncdmirt_20260827_p40"
    ),
    "moocradar/cohort_batch09/agent4edu_pre_ncdmirt_archive": (
        "outputs/comparison/"
        "moocradar_v6primary_batch09_50x10_agent4edu_"
        "officialalign_20260827_p40"
    ),
    "moocradar/cohort_batch10/deepseek_v6_primaryroute": (
        "outputs/formal/v6_primaryroute/moocradar/"
        "batch01_50x10_samecohort_20260826_p40"
    ),
    "moocradar/cohort_batch10/glm53_flash_v6_primaryroute": (
        "outputs/formal/v6_primaryroute/moocradar/"
        "glm53_flash_batch10_50x10_20260828_p40"
    ),
    "moocradar/cohort_batch10/deepseek_v6_preprimaryroute_archive": (
        "outputs/formal/v6/moocradar/batch02_50x10_20260825_full_p40"
    ),
    "xes3g5m/cohort_batch01/deepseek_v6_primaryroute_base": (
        "outputs/formal/v6_primaryroute/xes3g5m/"
        "batch01_50x10_cohort_20260803_20260827_p40"
    ),
    "xes3g5m/cohort_batch01/deepseek_v6_primaryroute_threads1_archive": (
        "outputs/formal/v6_primaryroute/xes3g5m/"
        "batch01_50x10_cohort_20260803_20260827_p40_threads1"
    ),
    "xes3g5m/cohort_batch01/deepseek_v6_primaryroute_profilecache": (
        "outputs/formal/v6_primaryroute/xes3g5m/"
        "batch01_50x10_cohort_20260803_20260827_profilecache_p40"
    ),
    "xes3g5m/cohort_batch02/deepseek_v6_primaryroute_profilecache": (
        "outputs/formal/v6_primaryroute/xes3g5m/"
        "batch02_50x10_cohort_20260804_20260827_profilecache_p40"
    ),
}

LEGACY_ROOTS = {
    "legacy/v5/moocradar": "outputs/formal/v5/moocradar",
    "legacy/v5/xes3g5m": "outputs/formal/v5/xes3g5m",
    "legacy/v5/foundationalassist": "outputs/formal/v5/foundationalassist",
    "legacy/v6/moocradar": "outputs/formal/v6/moocradar",
}

COHORT_METADATA = {
    "moocradar/cohort_batch09": {
        "cohort_file": "experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch09_50x10_seed20260803.json",
        "sample_size": "50x10",
    },
    "moocradar/cohort_batch10": {
        "cohort_file": "experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch10_50x10_seed20260803.json",
        "sample_size": "50x10",
    },
    "xes3g5m/cohort_batch01": {
        "source_cohort_date": "20260803",
        "sample_size": "50x10",
    },
    "xes3g5m/cohort_batch02": {
        "source_cohort_date": "20260804",
        "sample_size": "50x10",
    },
}


def install_alias(relative_name: str, target_relative_path: str) -> None:
    alias = CANONICAL / relative_name
    target = (ROOT / target_relative_path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Cannot catalog missing run: {target}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        if alias.resolve() == target:
            return
        raise RuntimeError(f"Alias points elsewhere: {alias} -> {alias.resolve()}")
    if alias.exists():
        raise RuntimeError(f"Alias path is not a symlink: {alias}")
    alias.symlink_to(target, target_is_directory=True)


def main() -> None:
    CANONICAL.mkdir(parents=True, exist_ok=True)
    for alias, target in {**RUNS, **LEGACY_ROOTS}.items():
        install_alias(alias, target)
    index = {
        "schema_version": 1,
        "layout": "outputs/canonical/<dataset>/<cohort>/<method>",
        "note": (
            "Canonical entries are symlinks. Historical source paths remain "
            "available so embedded report references never break."
        ),
        "runs": RUNS,
        "cohorts": COHORT_METADATA,
        "legacy_roots": LEGACY_ROOTS,
    }
    (CANONICAL / "INDEX.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (CANONICAL / "README.md").write_text(
        "# Canonical experiment-output catalog\n\n"
        "Browse results by actual fixed sample cohort, not by the historical "
        "run-directory name. Each entry is a symlink to the original run; "
        "no result data is copied or moved.\n\n"
        "- `moocradar/cohort_batch09/` and `cohort_batch10/` group Full, "
        "ablations, KT baselines, and comparison baselines for the same sample.\n"
        "- Every dataset uses `cohort_batchNN/`; source dates and seeds live in "
        "`INDEX.json` and run manifests rather than in directory names.\n"
        "- `legacy/` preserves older outputs whose fixed-cohort metadata is not "
        "reliably recoverable.\n",
        encoding="utf-8",
    )
    print(f"catalog_ready={CANONICAL}")
    print(f"canonical_runs={len(RUNS)} legacy_roots={len(LEGACY_ROOTS)}")


if __name__ == "__main__":
    main()
