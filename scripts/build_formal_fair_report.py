from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from learner_simulator.evaluation import evaluate_steps  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_uids(report: dict) -> set[str]:
    grouped: dict[str, list[dict]] = {}
    for step in report["all_steps"]:
        grouped.setdefault(str(step["uid"]), []).append(step)
    expected = int(report["protocol"]["target_steps"])
    return {
        uid
        for uid, steps in grouped.items()
        if len(steps) == expected
        and all(
            step.get("prediction_valid")
            and not step.get("llm_error")
            and isinstance(step.get("llm_parsed_action"), dict)
            for step in steps
        )
    }


def exact_mcnemar_p(full_only: int, ablation_only: int) -> float:
    discordant = full_only + ablation_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(full_only, ablation_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--version-id", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    cohort_path = Path(args.cohort).resolve()
    output_dir = Path(args.output_dir).resolve()
    all_dir = output_dir / "offline_all_steps"
    fair_dir = output_dir / "offline_common_valid_steps"
    all_dir.mkdir(parents=True, exist_ok=True)
    fair_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    variants = list(payload["variants"])
    per_variant_valid = {
        variant: valid_uids(payload["reports"][variant]) for variant in variants
    }
    common_uids = set.intersection(*(per_variant_valid[v] for v in variants))

    report = {
        "study": "formal_fixed_cohort_fair_comparison",
        "version_id": args.version_id,
        "source_combined": str(input_path),
        "source_combined_sha256": sha256(input_path),
        "cohort_file": str(cohort_path),
        "cohort_sha256": sha256(cohort_path),
        "variants": variants,
        "fairness_policy": "same users and all ten ordered target steps valid in every variant",
        "common_valid_user_count": len(common_uids),
        "common_valid_step_count": len(common_uids) * 10,
        "common_valid_uids": sorted(common_uids),
        "excluded_uids_by_variant": {
            variant: sorted(set().union(*per_variant_valid.values()) - uids)
            for variant, uids in per_variant_valid.items()
        },
        "summary": {},
        "paired_comparisons_vs_full": {},
    }

    csv_fields = [
        "variant", "uid", "step_index", "timestamp", "qid", "cid",
        "real_response", "simulated_response", "prediction_valid",
        "prediction_source", "response_learner_correct", "rendered_answer_correct",
        "ncdm_correct_probability", "ncdm_predicted_response",
        "ncdm_concept_mastery", "feedback_response", "llm_elapsed_seconds",
        "llm_error", "llm_result",
    ]
    for variant in variants:
        source_steps = payload["reports"][variant]["all_steps"]
        fair_steps = [s for s in source_steps if str(s["uid"]) in common_uids]
        metrics = evaluate_steps(fair_steps, threshold=0.5)
        report["summary"][variant] = metrics

        with (all_dir / f"{variant}.jsonl").open("w", encoding="utf-8") as handle:
            for step in source_steps:
                handle.write(json.dumps(step, ensure_ascii=False) + "\n")
        with (fair_dir / f"{variant}.jsonl").open("w", encoding="utf-8") as handle:
            for step in fair_steps:
                handle.write(json.dumps(step, ensure_ascii=False) + "\n")
        with (fair_dir / f"{variant}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields)
            writer.writeheader()
            for step in fair_steps:
                writer.writerow({"variant": variant, **{k: step.get(k) for k in csv_fields if k != "variant"}})

    full_steps = {
        (str(step["uid"]), int(step["step_index"])): step
        for step in payload["reports"]["full"]["all_steps"]
        if str(step["uid"]) in common_uids
    }
    for variant in variants:
        if variant == "full":
            continue
        variant_steps = {
            (str(step["uid"]), int(step["step_index"])): step
            for step in payload["reports"][variant]["all_steps"]
            if str(step["uid"]) in common_uids
        }
        full_only = 0
        ablation_only = 0
        for key, full_step in full_steps.items():
            other = variant_steps[key]
            full_correct = int(full_step["simulated_response"]) == int(full_step["real_response"])
            other_correct = int(other["simulated_response"]) == int(other["real_response"])
            full_only += full_correct and not other_correct
            ablation_only += other_correct and not full_correct
        report["paired_comparisons_vs_full"][variant] = {
            "full_only_correct": full_only,
            "ablation_only_correct": ablation_only,
            "discordant": full_only + ablation_only,
            "exact_mcnemar_p_two_sided": exact_mcnemar_p(full_only, ablation_only),
            "accuracy_delta_full_minus_ablation": round(
                report["summary"]["full"]["sample_match_acc"]
                - report["summary"][variant]["sample_match_acc"],
                6,
            ),
        }

    report_path = output_dir / "formal_fair_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    artifacts = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "artifact_manifest_sha256.txt"
    )
    with (output_dir / "artifact_manifest_sha256.txt").open("w", encoding="utf-8") as handle:
        for path in artifacts:
            handle.write(f"{sha256(path)}  {path.relative_to(output_dir)}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
