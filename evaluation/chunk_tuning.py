"""Controlled RAG chunk-size/overlap experiment.
"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import chromadb
from data_loader import load_excel_workbook, prepare_records
from document_retriever import load_text_sections, load_external_records, split_text_into_chunks
from pypdf import PdfReader

EVAL = PROJECT / 'evaluation'
BENCHMARK = EVAL / 'benchmark_questions.csv'
RESULTS = EVAL / 'results'
EXPERIMENT_DB = EVAL / 'chunk_tuning_chroma'  # NOT the application's chroma_db/
CHUNK_SIZES = (500, 1000, 1500)
OVERLAPS = (100, 200)
TOP_K = (3, 5, 8)
BASELINE = (1200, 200)


def source_records(size: int, overlap: int):
    records = []
    text_file = PROJECT / 'data' / 'institutional_performance.txt'
    for section in load_text_sections():
        records.append((section['text'], {'source': text_file.name, 'section': section['heading'], 'source_type': 'text'}))

    for row in prepare_records(load_excel_workbook()):
        records.append((row['content'], {'source': row['source'], 'section': row['worksheet'], 'source_type': 'excel', 'row_number': int(row['row_number'])}))

    for pdf in sorted((PROJECT / 'data').glob('*.pdf')):
        reader = PdfReader(str(pdf))
        for page_num, page in enumerate(reader.pages, 1):
            for chunk_no, chunk in enumerate(split_text_into_chunks(page.extract_text() or '', size, overlap), 1):
                records.append((chunk, {'source': pdf.name, 'section': f'Page {page_num}', 'page_number': page_num, 'chunk_number': chunk_no, 'source_type': 'pdf'}))

    for entry in load_external_records(chunk_size=size, chunk_overlap=overlap):
        records.append((entry['content'], {'source': entry['source'], 'section': entry['section'], 'source_type': 'external', 'source_url': entry.get('source_url') or '', 'chunk_number': int(entry['chunk_number'])}))
    return records


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    with BENCHMARK.open(encoding='utf-8-sig', newline='') as f:
        questions = list(csv.DictReader(f))
    if not questions or any(q.get('label_status') != 'provisional' for q in questions):
        print('Note: review benchmark label_status before using the results as final ground truth.')
    if not questions:
        raise ValueError('Benchmark is empty')

    client = chromadb.PersistentClient(path=str(EXPERIMENT_DB))
    settings = [BASELINE] + [(s, o) for s in CHUNK_SIZES for o in OVERLAPS if (s, o) != BASELINE]
    rows = []
    manifest = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'benchmark_sha256': hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
                'notes': 'Source-filename proxy only; benchmark labels provisional. Text and Excel records are not re-chunked.',
                'settings': []}

    for size, overlap in settings:
        name = f'chunk_eval_c{size}_o{overlap}'
        print(f'Indexing {name}...', flush=True)
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)
        collection = client.create_collection(name=name)
        records = source_records(size, overlap)
        if not records:
            raise RuntimeError('No records were extracted. Check data/ contents.')
        for offset in range(0, len(records), 100):
            batch = records[offset:offset+100]
            collection.add(ids=[f'rec_{offset+i}' for i in range(len(batch))],
                           documents=[doc for doc, _ in batch],
                           metadatas=[meta for _, meta in batch])
        manifest['settings'].append({'collection': name, 'chunk_size_characters': size, 'overlap_characters': overlap, 'indexed_records': len(records)})
        print(f'  {len(records)} records indexed', flush=True)

        for q in questions:
            started = time.perf_counter()
            query = collection.query(query_texts=[q['question']], n_results=min(max(TOP_K), collection.count()), include=['metadatas', 'documents', 'distances'])
            elapsed = time.perf_counter() - started
            metadata = query['metadatas'][0]
            docs = query['documents'][0]
            distances = query['distances'][0]
            for k in TOP_K:
                subset = metadata[:k]
                expected = q['expected_source']
                matches = [i+1 for i, m in enumerate(subset) if m.get('source') == expected]
                rows.append({'question_id': q['question_id'], 'split': q['split'], 'category': q.get('category',''),
                             'chunk_size_chars': size, 'overlap_chars': overlap, 'top_k': k,
                             'expected_source_provisional': expected, 'label_status': q['label_status'],
                             'expected_source_in_top_k': int(bool(matches)),
                             'expected_source_first_rank': matches[0] if matches else '',
                             'reciprocal_rank_source_proxy': round(1/matches[0], 6) if matches else 0.0,
                             'query_seconds_at_k8': round(elapsed, 6),
                             'ranked_sources': json.dumps([m['source'] for m in subset]),
                             'ranked_sections': json.dumps([m.get('section','') for m in subset]),
                             'ranked_distances': json.dumps([round(float(x),5) for x in distances[:k]]),
                             'ranked_passages': json.dumps(docs[:k], ensure_ascii=False)})
            print(f"  {q['question_id']} queried ({elapsed:.3f}s)", flush=True)

    output = RESULTS / 'chunk_tuning_source_proxy.csv'
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Select a configuration using DEVELOPMENT data only, then report on held-out data.
    summaries = []
    for size, overlap in settings:
        for k in TOP_K:
            for split in ('dev', 'test'):
                subset = [r for r in rows if r['chunk_size_chars'] == size and r['overlap_chars'] == overlap and r['top_k'] == k and r['split'] == split]
                if subset:
                    summaries.append({'split': split, 'chunk_size_chars': size, 'overlap_chars': overlap,
                                      'top_k': k, 'n_questions': len(subset),
                                      'source_hit_rate_proxy': round(sum(r['expected_source_in_top_k'] for r in subset)/len(subset), 4),
                                      'source_mrr_proxy': round(sum(r['reciprocal_rank_source_proxy'] for r in subset)/len(subset), 4),
                                      'mean_query_seconds': round(sum(r['query_seconds_at_k8'] for r in subset)/len(subset), 4)})
    with (RESULTS / 'chunk_tuning_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    # Deterministic dev-only selection; prefer smaller Top-K if equal.
    dev = [r for r in summaries if r['split'] == 'dev']
    best = sorted(dev, key=lambda r: (-r['source_hit_rate_proxy'], -r['source_mrr_proxy'], r['top_k'], r['chunk_size_chars'], r['overlap_chars']))[0]
    held_out = next((r for r in summaries if r['split']=='test' and all(r[x]==best[x] for x in ('chunk_size_chars','overlap_chars','top_k'))), None)
    manifest['selected_by_dev_source_proxy'] = best
    manifest['held_out_source_proxy_at_selected_configuration'] = held_out
    (RESULTS / 'chunk_tuning_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\nExperiment finished. Application index unchanged.')
    print('Development-selected configuration (PROVISIONAL SOURCE PROXY):', best)
    print('Held-out test (PROVISIONAL SOURCE PROXY):', held_out)
    print('Outputs in:', RESULTS)
    print('Do not report these metrics as validated passage relevance.')

if __name__ == '__main__':
    main()
