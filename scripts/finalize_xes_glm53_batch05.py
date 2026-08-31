#!/usr/bin/env python3
"""Finalize the repaired XES GLM batch with reproducible KT formal metrics."""
import importlib.util
import json
from pathlib import Path

from learner_simulator.formal_metrics import (
    build_ability_difficulty_partition,
    evaluate_formal_response_metrics,
)

ROOT = Path('/ai/KT/LeanerSim')
OUT = ROOT / ('outputs/formal/v6_primaryroute/xes3g5m/'
              'glm53_flash_cohort_remaining300_batch05_50x10_seed20260824_p40')
COHORT = ROOT / ('experiments/cohorts/xes3g5m_formal_v5_remaining300_seed20260824/'
                 'xes3g5m_500x10_batch05_50x10_seed20260824.json')
COMBINED = OUT / 'combined_repaired_attempt7.json'
KT = OUT / 'kt_baselines.json'
FINAL = OUT / 'final_report.json'
BASELINE_STEPS = OUT / 'kt_baseline_steps.json'
DKT = ROOT / 'outputs/dkt/xes3g5m_remaining300_leakage_safe_fullsource_e10_h128/best_model.pt'
NCDM = ROOT / 'outputs/dneuralcdm/xes3g5m_remaining300_leakage_safe_s20000_e30_d32_h64/best_model.pt'


def load_evaluator():
    path = ROOT / 'scripts/evaluate_cohort_kt_models.py'
    spec = importlib.util.spec_from_file_location('cohort_eval', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    report = json.loads(COMBINED.read_text(encoding='utf-8-sig'))
    for name, variant in report['reports'].items():
        steps = variant['all_steps']
        assert len(steps) == 500 and not any(s.get('llm_error') for s in steps), name
    partition = build_ability_difficulty_partition(report['reports']['full']['all_steps'])
    evaluator = load_evaluator()
    _, history, target = evaluator._load_cohort(COHORT)
    dkt = evaluator._eval_dkt(history, target, DKT, 0.5)
    ncdm = evaluator._eval_ncdm(history, target, NCDM, 0.5)
    assert dkt[5] == 0 and ncdm[5] == 0
    report['kt_baselines'] = json.loads(KT.read_text(encoding='utf-8-sig'))
    report['formal_metrics_v6']['fixed_full_irt_2x2_partition'] = partition
    report['formal_metrics_v6']['baselines'] = {
        'dkt': evaluate_formal_response_metrics(dkt[4], partition),
        'ncdm': evaluate_formal_response_metrics(ncdm[4], partition),
    }
    BASELINE_STEPS.write_text(json.dumps({
        'dkt_direct_teacher_forcing_steps': dkt[4],
        'ncdm_direct_teacher_forcing_steps': ncdm[4],
        'fixed_full_irt_2x2_partition': partition,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    report['kt_baseline_step_archive'] = str(BASELINE_STEPS)
    FINAL.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, result in report['formal_metrics_v6']['methods'].items():
        print(name, result['baa'], result['balanced_accuracy'], result['f1'], result['adcde'])
    for name, result in report['formal_metrics_v6']['baselines'].items():
        print(name, result['baa'], result['balanced_accuracy'], result['f1'], result['adcde'])


if __name__ == '__main__':
    main()
