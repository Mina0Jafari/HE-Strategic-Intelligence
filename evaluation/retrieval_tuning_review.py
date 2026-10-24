"""Audit a frozen ChromaDB retrieval index and export evidence for human review.

Run from project root: python evaluation/retrieval_tuning_review.py
Do not assign ground-truth labels based only on these retrieved candidates.
"""
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_retriever import search_documents  # noqa: E402

BENCHMARK = Path(__file__).with_name('benchmark_questions.csv')
OUTPUT_DIR = Path(__file__).parent / 'results'
TOP_K_VALUES = (3, 5, 8)


def matches_expected_source(hit, benchmark_row):
    if hit.get('source') != benchmark_row['expected_source'].strip():
        return False
    expected_section = benchmark_row['expected_section'].strip()
    return not expected_section or hit.get('section') == expected_section


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK.open(newline='', encoding='utf-8-sig') as f:
        cases = list(csv.DictReader(f))
    if not cases:
        raise ValueError('Benchmark contains no questions')

    predictions, review_rows, score_rows = [], [], []
    for case in cases:
        start = time.perf_counter()
        hits = search_documents(case['question'], n_results=max(TOP_K_VALUES))
        latency = round(time.perf_counter() - start, 4)
        verified = case['label_status'].strip().lower() == 'verified'
        entries = []
        for rank, hit in enumerate(hits, 1):
            passage = hit.get('text', '') or ''
            matched = matches_expected_source(hit, case)
            entries.append({
                'rank': rank, 'source': hit.get('source', ''),
                'section': hit.get('section', ''), 'page_number': hit.get('page_number', ''),
                'row_number': hit.get('row_number', ''), 'distance': hit.get('distance'),
                'text': passage,
            })
            review_rows.append({
                'question_id': case['question_id'], 'split': case['split'],
                'question': case['question'], 'rank': rank,
                'source': hit.get('source', ''), 'section': hit.get('section', ''),
                'page_number': hit.get('page_number', ''),
                'row_number': hit.get('row_number', ''),
                'distance': hit.get('distance', ''),
                'matches_provisional_source_and_section': int(matched),
                'passage_text': passage,
                'relevant_0_or_1': '',  # fill manually after comparing with original source
                'reviewer_notes': '',
            })
        predictions.append({
            'question_id': case['question_id'], 'question': case['question'],
            'expected_source_provisional': case['expected_source'],
            'expected_section_provisional': case['expected_section'],
            'label_status': case['label_status'], 'latency_seconds': latency,
            'retrieved': entries,
        })
        for k in TOP_K_VALUES:
            ranks = [i for i, hit in enumerate(hits[:k], 1)
                     if matches_expected_source(hit, case)]
            score_rows.append({
                'question_id': case['question_id'], 'split': case['split'],
                'k': k, 'label_status': case['label_status'],
                'source_hit_at_k': int(bool(ranks)) if verified else '',
                'reciprocal_rank': round(1/ranks[0], 5) if (verified and ranks) else (0.0 if verified else ''),
                'latency_seconds': latency,
                'first_rank': ranks[0] if verified and ranks else '',
            })
        print(f"{case['question_id']}: {len(hits)} passages, {latency:.3f} s")

    with (OUTPUT_DIR / 'retrieval_predictions_with_text.json').open('w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    for filename, rows in [('passage_relevance_review.csv', review_rows),
                           ('retrieval_scores_review.csv', score_rows)]:
        with (OUTPUT_DIR / filename).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print('\nSaved evidence-rich predictions and passage review sheet to evaluation/results/')
    for split in ('dev', 'test'):
        for k in TOP_K_VALUES:
            subset = [r for r in score_rows if r['split'] == split and r['k'] == k
                      and r['label_status'].strip().lower() == 'verified']
            if subset:
                print(f"{split} top-{k}: N={len(subset)}, "
                      f"source Hit@K={mean(float(r['source_hit_at_k']) for r in subset):.3f}, "
                      f"source MRR={mean(float(r['reciprocal_rank']) for r in subset):.3f}")
            else:
                print(f'{split} top-{k}: no verified source labels, accuracy withheld')
    print('Review evidence and original sources before approving any labels.')


if __name__ == '__main__':
    main()
