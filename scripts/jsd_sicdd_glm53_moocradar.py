#!/usr/bin/env python3
"""Offline distribution diagnostics for the five pooled GLM MOOCradar cohorts.

JSD is the Jensen-Shannon divergence (base 2) between the marginal binary
response distributions.  SICDD is kept only as a transparent diagnostic name:
it is the unweighted macro mean of the same JSD computed inside the fixed,
external IRT ability x item-difficulty 2x2 cells.  Both are lower-is-better.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path('/ai/KT/LeanerSim')
BASE = ROOT / 'outputs/formal/v6_primaryroute/moocradar'
BATCHES = ['01', '02', '03', '09', '10']
METHODS = ['full', 'no-evidence-representation', 'no-state-item-alignment',
           'no-structured-response-process', 'no-dynamic-state-evolution']
KT = BASE / 'glm53_flash_pooled_c2st_kt_records_batch01_02_03_09_10_250x10_20260828.json'
OUT = BASE / 'glm53_flash_pooled_jsd_sicdd_batch01_02_03_09_10_250x10_20260828.json'


def jsd_binary(real, simulated):
    """Base-2 JSD of empirical Bernoulli distributions; range [0, 1]."""
    n = len(real)
    assert n == len(simulated) and n
    p = [1 - sum(real) / n, sum(real) / n]
    q = [1 - sum(simulated) / n, sum(simulated) / n]
    m = [(a + b) / 2 for a, b in zip(p, q)]
    def kl(a, b):
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x)
    return (kl(p, m) + kl(q, m)) / 2


def add_step(target, step, group_map):
    key = '%s|%s|%s' % (step['uid'], step['step_index'], step['qid'])
    group = group_map[key]
    target['global_real'].append(int(step['real_response']))
    target['global_sim'].append(int(step['simulated_response']))
    target['by_group'][group]['real'].append(int(step['real_response']))
    target['by_group'][group]['sim'].append(int(step['simulated_response']))


def summarize(rec):
    per_group = {}
    for group, values in sorted(rec['by_group'].items()):
        per_group[group] = {
            'count': len(values['real']),
            'jsd': jsd_binary(values['real'], values['sim']),
            'real_correct_rate': sum(values['real']) / len(values['real']),
            'simulated_correct_rate': sum(values['sim']) / len(values['sim']),
        }
    return {
        'count': len(rec['global_real']),
        'jsd': jsd_binary(rec['global_real'], rec['global_sim']),
        'sicdd': sum(x['jsd'] for x in per_group.values()) / len(per_group),
        'sicdd_definition': 'unweighted_macro_mean_of_base2_binary_JSD_within_fixed_external_IRT_2x2_ability_difficulty_cells',
        'groups': per_group,
    }


def empty():
    return {'global_real': [], 'global_sim': [],
            'by_group': defaultdict(lambda: {'real': [], 'sim': []})}


def main():
    records = {m: empty() for m in METHODS + ['dkt', 'ncdm']}
    for batch in BATCHES:
        d = json.loads((BASE / ('glm53_flash_batch%s_50x10_20260828_p40' % batch) / 'final_report.json').read_text())
        groups = d['formal_metrics_v6']['fixed_full_irt_2x2_partition']['groups']
        for method in METHODS:
            for step in d['reports'][method]['all_steps']:
                add_step(records[method], step, groups)
    kt = json.loads(KT.read_text())
    # All batch-level group keys are unique by uid; reuse them to give DKT/NCDM
    # exactly the Full-derived, method-independent partition used above.
    group_map = {}
    for batch in BATCHES:
        d = json.loads((BASE / ('glm53_flash_batch%s_50x10_20260828_p40' % batch) / 'final_report.json').read_text())
        group_map.update(d['formal_metrics_v6']['fixed_full_irt_2x2_partition']['groups'])
    for method in ['dkt', 'ncdm']:
        for step in kt[method]:
            add_step(records[method], step, group_map)
    output = {
        'dataset': 'moocradar',
        'model': 'glm-5.3-flash',
        'aggregation': 'raw_step_pooling_no_batch_average',
        'batches': BATCHES,
        'learners_per_method': 250,
        'steps_per_method': 2500,
        'jsd_definition': 'base2_Jensen_Shannon_divergence_between_empirical_global_binary_real_and_simulated_response_distributions; lower_is_better',
        'sicdd_caveat': 'SICDD is not a standard published metric name; this report records its exact diagnostic formula rather than presenting it as a novel evaluation metric.',
        'methods': {m: summarize(records[m]) for m in records},
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    for method, result in output['methods'].items():
        print('%-34s JSD=%.6f SICDD=%.6f' % (method, result['jsd'], result['sicdd']))
    print(OUT)


if __name__ == '__main__':
    main()
