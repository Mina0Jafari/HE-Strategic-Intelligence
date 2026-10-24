"""Controlled comparison of three strategic question-answering architectures.

Run from the HE_Strategic_AI project root with:
    python compare_systems.py

Uses the *same* retrieved evidence for single-agent and multi-agent RAG.
Writes experimental outputs under evaluation/results/comparison/.
Review factual quality separately against reference answers; this runner does not
assign subjective accuracy scores automatically.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / '.env')
from agents.retrieval_agent import RetrievalAgent
from agents.analysis_agent import AnalysisAgent
from agents.reporting_agent import ReportingAgent

MODEL = os.getenv('EXPERIMENT_MODEL', 'gpt-4.1-mini')
TOP_K = 3
QUESTION_FILE = ROOT / 'evaluation' / 'benchmark_questions.csv'
OUT = ROOT / 'evaluation' / 'results' / 'comparison'
OUT.mkdir(parents=True, exist_ok=True)


def read_questions(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Benchmark file has no questions')
    for row in rows:
        if 'question' not in row or not row['question'].strip():
            raise ValueError('Benchmark CSV requires a non-empty question column')
    return rows


def call_model(client, question, evidence=None):
    evidence_block = ''
    if evidence is not None:
        evidence_block = '\n\nEVIDENCE:\n' + '\n\n'.join(
            f"[{e['source_id']}] {e['document']} / {e['section']}\n{e['content']}" for e in evidence
        )
    instructions = (
        'You are a higher education strategic decision-support analyst. '
        'Give concise findings, limitations, and recommended actions. '
        'Distinguish evidence from assumptions. Do not invent facts or citations. '
        + ('Use only supplied evidence for institution-specific facts and cite source IDs.' if evidence is not None
           else 'No institutional evidence is supplied; explicitly flag unsupported institution-specific claims.')
    )
    res = client.responses.create(model=MODEL, instructions=instructions,
                                  input='QUESTION:\n' + question + evidence_block)
    usage = getattr(res, 'usage', None)
    return res.output_text, {
        'input_tokens': getattr(usage, 'input_tokens', None),
        'output_tokens': getattr(usage, 'output_tokens', None),
        'total_tokens': getattr(usage, 'total_tokens', None),
    }


def run():
    questions = read_questions(QUESTION_FILE)
    client = OpenAI()
    retriever = RetrievalAgent(top_k=TOP_K)
    analyser = AnalysisAgent(model=MODEL)
    reporter = ReportingAgent(model=MODEL)
    records = []
    for index, q in enumerate(questions, 1):
        qid = q.get('question_id') or q.get('id') or f'Q{index:02d}'
        question = q['question'].strip()
        print(f'{qid} ({index}/{len(questions)}): retrieving once...', flush=True)
        retrieval_start = time.perf_counter()
        result = retriever.run(question)
        evidence = result['evidence']
        retrieval_secs = time.perf_counter() - retrieval_start
        for name in ('standalone', 'single_rag', 'multi_agent'):
            print(f'  {name}...', flush=True)
            started = time.perf_counter()
            usage = {'input_tokens': None, 'output_tokens': None, 'total_tokens': None}
            try:
                if name == 'standalone':
                    answer, usage = call_model(client, question)
                elif name == 'single_rag':
                    answer, usage = call_model(client, question, evidence)
                else:
                    if not evidence:
                        answer = 'Insufficient retrieved evidence.'
                    else:
                        analysis = analyser.run(question, evidence)
                        reported = reporter.run(question, analysis['analysis'], evidence)
                        answer = reported['report']
                status, error = 'ok', ''
            except Exception as exc:
                answer, status, error = '', 'error', str(exc)
            seconds = time.perf_counter() - started
            records.append({
                'question_id': qid, 'question': question, 'system': name,
                'model': MODEL, 'top_k': TOP_K, 'status': status, 'error': error,
                'retrieval_seconds': round(retrieval_secs, 4) if name != 'standalone' else 0,
                'generation_seconds': round(seconds, 4),
                'total_seconds': round(seconds + (retrieval_secs if name != 'standalone' else 0), 4),
                'evidence': evidence if name != 'standalone' else [],
                'answer': answer, **usage,
            })
    manifest = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model': MODEL, 'top_k': TOP_K,
        'question_count': len(questions),
        'note': 'Comparisons share a single retrieval result per question. Multi-agent tokens are not yet instrumented. Outputs are not automatically graded.'
    }
    (OUT / 'comparison_outputs.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    with (OUT / 'comparison_timing.csv').open('w', encoding='utf-8', newline='') as f:
        columns = ['question_id','system','status','retrieval_seconds','generation_seconds','total_seconds',
                   'input_tokens','output_tokens','total_tokens','error']
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    (OUT / 'comparison_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nSaved results to {OUT}')


if __name__ == '__main__':
    run()
