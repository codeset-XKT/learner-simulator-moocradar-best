#!/usr/bin/env python3
"""Convert the official DBE-KT22 archives to LearnerSim's canonical format.

The official release provides relational CSV tables and an official sequencer.
This adapter preserves every transaction, orders each learner's interactions by
the official ``start_time`` rule, and serializes one chronological sequence per
learner.  DBE-KT22 permits multiple KCs per item; canonical KT rows require a
single integer concept ID, so the primary concept is the lowest relation ID
from the official Question_KC_Relationships table.  The full KC list remains
in questions.json for later LLM prompting and audit.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='data/DBE-KT22')
    return parser.parse_args()


def read_csv(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as raw:
        return list(csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline='')))


def truth(value: str | None) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes'}


def timestamp_ms(value: str, fallback: int) -> int:
    value = (value or '').strip()
    if not value:
        return fallback
    for fmt in ('%Y-%m-%d %H:%M:%S.%f %z', '%Y-%m-%d %H:%M:%S %z'):
        try:
            return int(datetime.strptime(value, fmt).timestamp() * 1000)
        except ValueError:
            pass
    return fallback


def label(index: int) -> str:
    """A, B, ..., Z, AA, ... for the official ordered choice list."""
    text = ''
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        text = chr(ord('A') + remainder) + text
    return text


def main() -> None:
    raw_args = args()
    dataset_root = Path(raw_args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    archive_path = dataset_root / '2_DBE_KT22_datafiles_100102_csv.zip'
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        kcs = read_csv(archive, 'KCs.csv')
        questions_raw = read_csv(archive, 'Questions.csv')
        choices_raw = read_csv(archive, 'Question_Choices.csv')
        relations_raw = read_csv(archive, 'Question_KC_Relationships.csv')
        transactions = read_csv(archive, 'Transaction.csv')

    kc_by_id = {str(row['id']): row for row in kcs}
    choices_by_question: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in choices_raw:
        choices_by_question[str(row['question_id'])].append(row)
    for rows in choices_by_question.values():
        rows.sort(key=lambda row: int(row['id']))

    kcs_by_question: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in relations_raw:
        kcs_by_question[str(row['question_id'])].append(row)
    for rows in kcs_by_question.values():
        rows.sort(key=lambda row: int(row['id']))

    questions: dict[str, dict[str, Any]] = {}
    primary_concept: dict[str, int] = {}
    for row in sorted(questions_raw, key=lambda item: int(item['id'])):
        qid = str(row['id'])
        relation_rows = kcs_by_question.get(qid, [])
        if not relation_rows:
            raise ValueError(f'Question {qid} has no KC relationship')
        concept_ids = [int(rel['knowledgecomponent_id']) for rel in relation_rows]
        primary_concept[qid] = concept_ids[0]
        option_rows = choices_by_question.get(qid, [])
        options = [
            {'label': label(index), 'text': option['choice_text']}
            for index, option in enumerate(option_rows)
        ]
        answer = [label(index) for index, option in enumerate(option_rows) if truth(option['is_correct'])]
        if len(answer) != 1:
            raise ValueError(f'Question {qid} has {len(answer)} correct choices; fixed 90+10 protocol requires one')
        routes = [
            f"KC {cid}: {kc_by_id[str(cid)]['name']}"
            for cid in concept_ids
        ]
        questions[qid] = {
            'id': int(qid),
            'source_id': qid,
            'type': 'single_choice',
            'question_type': 'single_choice',
            'content': row.get('question_rich_text') or row.get('question_text') or '',
            'title': row.get('question_title') or '',
            'options': options,
            'answer': answer,
            'analysis': row.get('explanation') or '',
            'hint': row.get('hint_text') or '',
            'difficulty': row.get('difficulty') or '',
            'kc_routes': routes,
            'skill_id': primary_concept[qid],
            'skill_name': kc_by_id[str(primary_concept[qid])]['name'],
            'all_skill_ids': concept_ids,
            'all_skill_names': [kc_by_id[str(cid)]['name'] for cid in concept_ids],
            'primary_kc_rule': 'lowest Question_KC_Relationships.id',
            'language': 'English',
        }

    interactions: dict[str, list[tuple[int, int, str, int]]] = defaultdict(list)
    skipped_unknown_questions = 0
    for order, row in enumerate(transactions):
        qid = str(row['question_id'])
        if qid not in primary_concept:
            skipped_unknown_questions += 1
            continue
        uid = str(row['student_id'])
        interactions[uid].append((
            timestamp_ms(row.get('start_time', ''), order),
            order,
            qid,
            int(truth(row.get('answer_state'))),
        ))

    sequence_rows: list[dict[str, str]] = []
    interaction_count = 0
    for uid in sorted(interactions, key=lambda value: int(value)):
        entries = sorted(interactions[uid], key=lambda value: (value[0], value[1]))
        interaction_count += len(entries)
        sequence_rows.append({
            'uid': uid,
            'questions': ','.join(entry[2] for entry in entries),
            'concepts': ','.join(str(primary_concept[entry[2]]) for entry in entries),
            'responses': ','.join(str(entry[3]) for entry in entries),
            'timestamps': ','.join(str(entry[0]) for entry in entries),
        })

    metadata_dir = dataset_root / 'metadata'
    kc_dir = dataset_root / 'kc_level'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    kc_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / 'questions.json').write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    with (kc_dir / 'train_valid_sequences.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['uid', 'questions', 'concepts', 'responses', 'timestamps'])
        writer.writeheader()
        writer.writerows(sequence_rows)

    manifest = {
        'dataset': 'DBE-KT22',
        'source_archive': str(archive_path),
        'source_transaction_rows': len(transactions),
        'canonical_interactions': interaction_count,
        'canonical_users': len(sequence_rows),
        'questions': len(questions),
        'knowledge_components': len(kcs),
        'sequence_order': 'student_id, ascending start_time, stable raw-row tie-break',
        'interaction_filter': 'none; hidden and non-hidden official transaction rows are retained',
        'response_rule': 'answer_state true -> 1; otherwise -> 0',
        'primary_kc_rule': 'lowest Question_KC_Relationships.id; all linked KCs retained in metadata',
        'skipped_unknown_questions': skipped_unknown_questions,
    }
    (dataset_root / 'PREPROCESSING_MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
