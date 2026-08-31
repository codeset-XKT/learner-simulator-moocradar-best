#!/usr/bin/env python3
"""Crash- and quota-resumable DBE-KT22 formal-500 LLM experiment queue.

Each durable unit is one variant on <=8 learners (<=80 LLM calls). Five
variant units run together, so total LLM concurrency never exceeds 40 while
every successfully completed unit remains reusable after interruption.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path('/ai/KT/LeanerSim')
sys.path.insert(0, str(ROOT / 'src'))
from learner_simulator.formal_metrics import evaluate_formal_response_metrics

VARIANTS = [
    'full',
    'no-evidence-representation',
    'no-state-item-alignment',
    'no-structured-response-process',
    'no-dynamic-state-evolution',
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--batch-start', type=int, default=3)
    p.add_argument('--batch-end', type=int, default=10)
    p.add_argument('--max-learners-per-unit', type=int, default=8)
    p.add_argument('--max-repair-attempts', type=int, default=5)
    return p.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def chunk_paths(unit_dir: Path, variant: str, chunk_index: int) -> list[Path]:
    base = unit_dir / 'chunks' / variant / f'chunk{chunk_index:02d}.json'
    repaired = sorted(base.parent.glob(f'{base.stem}_repaired_attempt*.json'))
    return list(reversed(repaired)) + [base]


def report_for(path: Path, variant: str) -> dict:
    payload = load(path)
    return payload['reports'][variant]


def matching_complete(path: Path, variant: str, expected_uids: list[str]) -> bool:
    try:
        report = report_for(path, variant)
        steps = report.get('all_steps') or []
        return (
            len(steps) == len(expected_uids) * 10
            and {str(s.get('uid')) for s in steps} == set(expected_uids)
            and not any(bool(s.get('llm_error')) for s in steps)
        )
    except Exception:
        return False


def repair_if_needed(path: Path, variant: str, attempts: int, log_dir: Path) -> Path:
    current = path
    for attempt in range(1, attempts + 1):
        report = report_for(current, variant)
        failures = [s for s in report.get('all_steps') or [] if bool(s.get('llm_error'))]
        if not failures:
            return current
        failure_text = ' '.join(json.dumps(s, ensure_ascii=False) for s in failures).lower()
        if re.search(r'insufficient|balance|quota|credit|billing|余额|欠费|额度', failure_text):
            raise RuntimeError(
                f'suspected API credit/quota failure in {current}; preserving completed chunks and stopping queue'
            )
        next_path = current.parent / f'{current.stem}_repaired_attempt{attempt}.json'
        with (log_dir / f'{variant}_{current.stem}_repair{attempt}.log').open('w') as log:
            status = subprocess.run([
                sys.executable, 'scripts/repair_failed_steps.py',
                '--base', str(current), '--config', 'configs/llm.glm-5.3-flash_v6_repair_lowthinking.json',
                '--output', str(next_path), '--attempt', str(attempt),
                '--max-workers', '40', '--commit-batch-size', '40',
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if not next_path.exists():
            raise RuntimeError(f'repair did not write {next_path}; exit={status}')
        current = next_path
    if any(bool(s.get('llm_error')) for s in report_for(current, variant).get('all_steps') or []):
        raise RuntimeError(f'{variant} still has llm_error after {attempts} targeted repairs: {current}')
    return current


def run_unit(
    unit_dir: Path, cohort: Path, variant: str, chunk_index: int, uids: list[str], max_repairs: int,
) -> tuple[str, int, str]:
    for candidate in chunk_paths(unit_dir, variant, chunk_index):
        if candidate.exists():
            candidate = repair_if_needed(candidate, variant, max_repairs, unit_dir / 'logs')
            if matching_complete(candidate, variant, uids):
                return variant, chunk_index, str(candidate)

    out = unit_dir / 'chunks' / variant / f'chunk{chunk_index:02d}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = unit_dir / 'checkpoints' / variant / f'chunk{chunk_index:02d}'
    log_path = unit_dir / 'logs' / f'{variant}_chunk{chunk_index:02d}.log'
    cmd = [
        sys.executable, 'experiments/ablation/run_ablation.py',
        '--dataset-root', 'data/DBE-KT22', '--cohort-file', str(cohort),
        '--variants', variant, '--parallel-variants', '1', '--simulator', 'multi-role',
        '--feedback-mode', 'teacher-forcing', '--parallel-learners', str(len(uids)),
        '--llm-config', 'configs/llm.glm-5.3-flash_v6_single_call_thinking.json',
        '--dneuralcdm-checkpoint', 'outputs/dneuralcdm/dbe_kt22_formal500_leakage_safe_e30_d32_h64/best_model.pt',
        '--checkpoint-dir', str(checkpoint_dir), '--output', str(out), '--simulate-uids', ','.join(uids),
        '--progress',
    ]
    env = os.environ.copy()
    env.update({'PYTHONPATH': 'src', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    with log_path.open('w') as log:
        status = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if not out.exists():
        raise RuntimeError(f'{variant} chunk{chunk_index:02d} did not produce output; exit={status}; see {log_path}')
    final = repair_if_needed(out, variant, max_repairs, unit_dir / 'logs')
    if not matching_complete(final, variant, uids):
        raise RuntimeError(f'incomplete output after repair: {final}')
    return variant, chunk_index, str(final)


def finalize_batch(unit_dir: Path, cohort: Path, archive: Path, artifact: Path) -> None:
    cohort_payload = load(cohort)
    all_uids = [str(x) for x in cohort_payload['uids']]
    reference = ROOT / 'outputs/baselines/dbe_kt22/formal500_seed20260830_teacher_forcing/fixed_full_irt_reference.json'
    partition = load(reference)['fixed_full_irt_2x2_partition']
    reports, metrics, audit = {}, {}, {}
    for variant in VARIANTS:
        steps = []
        source_report = None
        for index in range(1, ((len(all_uids) - 1) // 8) + 2):
            paths = chunk_paths(unit_dir, variant, index)
            found = next((p for p in paths if p.exists() and matching_complete(p, variant, all_uids[(index-1)*8:index*8])), None)
            if found is None:
                continue
            report = report_for(found, variant)
            source_report = source_report or report
            steps.extend(report['all_steps'])
        if len(steps) != 500 or {str(s.get('uid')) for s in steps} != set(all_uids):
            raise RuntimeError(f'cannot finalize {variant}: {len(steps)} steps')
        reports[variant] = {**(source_report or {}), 'all_steps': steps, 'ablation_variant': variant}
        metrics[variant] = evaluate_formal_response_metrics(steps, partition)
        audit[variant] = {'users': 50, 'steps': 500, 'llm_error': 0}
    merged = {
        'study': f'dbe_kt22_glm53_flash_{unit_dir.name}',
        'protocol': {'feedback_mode': 'teacher-forcing', 'history_steps': 90, 'target_steps': 10,
                     'durable_unit': 'variant_on_at_most_8_learners', 'total_llm_concurrency': 40},
        'cohort_file': str(cohort), 'cohort_sha256': hashlib.sha256(cohort.read_bytes()).hexdigest(),
        'partition_reference': str(reference),
        'formal_metrics_v6': {'fixed_full_irt_2x2_partition': partition, 'methods': metrics},
        'audit': audit, 'reports': reports,
    }
    save(unit_dir / 'merged_final_report.json', merged)
    destination = artifact / 'results' / unit_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(unit_dir, destination)
    with (artifact / 'README.md').open('a', encoding='utf-8') as f:
        f.write(f'\n- Added {unit_dir.name}: Full + four ablations; all five variants use the same 50-user DBE cohort and durable <=8-user units.\n')
    tmp = archive.with_suffix(archive.suffix + '.tmp')
    with tarfile.open(tmp, 'w:gz') as tar:
        tar.add(artifact, arcname=artifact.name)
    tmp.replace(archive)


def main() -> None:
    cfg = args()
    artifact = ROOT / 'outputs/external_storage/dbe_kt22_formal500_shared_cohort_artifacts_glm53_batches01_02'
    archive = artifact.with_suffix('.tar.gz')
    if not artifact.exists() or not archive.exists():
        raise RuntimeError('expected external artifact is missing')
    for batch in range(cfg.batch_start, cfg.batch_end + 1):
        cohort = ROOT / f'experiments/cohorts/dbe_kt22_formal500_seed20260830/dbe_kt22_500x10_batch{batch:02d}_50x10_seed20260830.json'
        if not cohort.exists():
            raise RuntimeError(f'missing cohort: {cohort}')
        unit_dir = ROOT / f'outputs/formal/v6_primaryroute/dbe_kt22/glm53_flash_batch{batch:02d}_50x10_seed20260830_full_ablations_resumable_p40'
        (unit_dir / 'logs').mkdir(parents=True, exist_ok=True)
        uids = [str(x) for x in load(cohort)['uids']]
        state_path = unit_dir / 'queue_state.json'
        save(state_path, {'batch': batch, 'cohort': str(cohort), 'uids': uids, 'variants': VARIANTS,
                          'unit_size': cfg.max_learners_per_unit, 'status': 'running'})
        chunks = [uids[i:i+cfg.max_learners_per_unit] for i in range(0, len(uids), cfg.max_learners_per_unit)]
        for chunk_index, chunk_uids in enumerate(chunks, start=1):
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                jobs = [executor.submit(run_unit, unit_dir, cohort, variant, chunk_index, chunk_uids, cfg.max_repair_attempts) for variant in VARIANTS]
                for job in concurrent.futures.as_completed(jobs):
                    variant, index, path = job.result()
                    print(json.dumps({'batch': batch, 'chunk': index, 'variant': variant, 'saved': path}), flush=True)
            save(state_path, {'batch': batch, 'cohort': str(cohort), 'uids': uids, 'variants': VARIANTS,
                              'unit_size': cfg.max_learners_per_unit, 'status': 'running', 'completed_chunks_through': chunk_index})
        finalize_batch(unit_dir, cohort, archive, artifact)
        save(state_path, {'batch': batch, 'cohort': str(cohort), 'uids': uids, 'variants': VARIANTS,
                          'unit_size': cfg.max_learners_per_unit, 'status': 'complete'})
        print(json.dumps({'batch': batch, 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
