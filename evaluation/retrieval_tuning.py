"""Evaluate retrieval Top-K on a frozen ChromaDB index.

Run from project root:
    python evaluation/retrieval_tuning.py

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

BENCHMARK = Path(__file__).with_name("benchmark_questions.csv")
OUTPUT_DIR = Path(__file__).parent / "results"
TOP_K_VALUES = (3, 5, 8)


def is_relevant(hit, row):
    """Match a manually adjudicated source, optionally a source section."""
    if hit.get("source", "") != row["expected_source"].strip():
        return False
    expected_section = row["expected_section"].strip()
    return not expected_section or hit.get("section", "") == expected_section


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK.open(newline="", encoding="utf-8-sig") as f:
        cases = list(csv.DictReader(f))
    if not cases:
        raise ValueError("The benchmark contains no questions")

    query_rows = []
    score_rows = []
    for row in cases:
        started = time.perf_counter()
        hits = search_documents(row["question"], n_results=max(TOP_K_VALUES))
        duration = time.perf_counter() - started
        verified = row["label_status"].strip().lower() == "verified"
        for k in TOP_K_VALUES:
            selected = hits[:k]
            relevant_ranks = [i for i, hit in enumerate(selected, 1) if is_relevant(hit, row)]
            rr = 1 / relevant_ranks[0] if relevant_ranks else 0.0
            # Source Hit@K is NOT passage-level Recall@K: the annotations
            # identify one expected source, not every relevant passage.
            score_rows.append({
                "question_id": row["question_id"],
                "split": row["split"],
                "k": k,
                "label_status": row["label_status"],
                "source_hit_at_k": int(bool(relevant_ranks)) if verified else "",
                "reciprocal_rank": round(rr, 5) if verified else "",
                "latency_seconds": round(duration, 4),
                "first_rank": relevant_ranks[0] if verified and relevant_ranks else "",
            })
        query_rows.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "expected_source_provisional": row["expected_source"],
            "label_status": row["label_status"],
            "latency_seconds": round(duration, 4),
            "retrieved": [{
                "rank": i, "source": hit.get("source"),
                "section": hit.get("section"),
                "distance": hit.get("distance"),
            } for i, hit in enumerate(hits, 1)],
        })
        print(f"{row['question_id']}: {len(hits)} hits, {duration:.3f} s")

    with (OUTPUT_DIR / "retrieval_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(query_rows, f, indent=2, ensure_ascii=False)
    with (OUTPUT_DIR / "retrieval_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    print("\nResults saved to evaluation/results/")
    for split in ("dev", "test"):
        for k in TOP_K_VALUES:
            group = [r for r in score_rows if r["split"] == split and r["k"] == k
                     and r["label_status"].strip().lower() == "verified"]
            if not group:
                print(f"{split} top-{k}: no verified labels; accuracy metrics withheld")
                continue
            print(f"{split} top-{k}: N={len(group)} "
                  f"Source Hit@K={mean(float(r['source_hit_at_k']) for r in group):.3f}, "
                  f"MRR={mean(float(r['reciprocal_rank']) for r in group):.3f}")
    print("\nReview retrieved passages against original sources, then change "
          "label_status from provisional to verified for justified source labels.")


if __name__ == "__main__":
    main()
