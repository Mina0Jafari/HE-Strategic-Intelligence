
"""
compare_systems.py

FINAL DISSERTATION EVALUATION RUNNER

Compares three higher education decision-support architectures:

1. Standalone LLM
2. Single-agent RAG
3. Multi-agent RAG

Experimental principles:
- Identical questions and language model.
- The two RAG architectures share exactly the same evidence.
- Structured Pandas evidence is available to both RAG systems.
- Retrieval time is included in both RAG total times.
- Token usage is recorded for all model calls where available.
- Runs are stored separately and never overwrite previous results.
- Code and benchmark versions are recorded for reproducibility.
- Outputs are retained for independent quality assessment.

Run:
    python compare_systems.py
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from agents.retrieval_agent import RetrievalAgent
from agents.analysis_agent import AnalysisAgent
from agents.reporting_agent import ReportingAgent


MODEL = os.getenv(
    "EXPERIMENT_MODEL",
    "gpt-4.1-mini"
)

TOP_K = 3

QUESTION_FILE = (
    ROOT
    / "evaluation"
    / "benchmark_questions.csv"
)

RESULTS_DIR = (
    ROOT
    / "evaluation"
    / "results"
    / "comparison"
)

SYSTEMS = (
    "standalone",
    "single_rag",
    "multi_agent",
)


# --------------------------------------------------
# 2. GENERAL UTILITIES
# --------------------------------------------------

def utc_now():
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def file_sha256(path):
    """Calculate the SHA-256 hash of a file."""

    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_information():
    """Record Git commit and working-tree state."""

    info = {
        "commit": "unknown",
        "working_tree_clean": None,
    }

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        info["commit"] = commit.stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        info["working_tree_clean"] = (
            not bool(status.stdout.strip())
        )

    except (
        subprocess.CalledProcessError,
        OSError,
    ):
        pass

    return info


def new_run_directory():
    """Create a unique output directory."""

    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    output_dir = RESULTS_DIR / run_id

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    return run_id, output_dir


def implementation_hashes():
    """Hash key code files; an uncommitted Git tree alone is not reproducible."""
    paths = [
        "compare_systems.py", "agents/retrieval_agent.py",
        "agents/analysis_agent.py", "agents/reporting_agent.py",
        "ingestion/query_planner.py", "ingestion/source_router.py",
        "analysis/nss_analysis.py", "analysis/ofs_outcomes_analysis.py",
        "analysis/hesa_analysis.py",
    ]
    return {relative: file_sha256(ROOT / relative) for relative in paths}


def source_file_hashes():
    """Hash fixed official/simulated data inputs where locally available."""
    paths = [
        "data/hesa_provider_enrolments.csv",
        "data/external/nss/NSS26_Summary_Taught_Full-time.xlsx",
        "data/external/source_hub/ofs/student_outcomes_each_provider_Feb2026.zip",
    ]
    return {relative: file_sha256(ROOT / relative) for relative in paths}


# --------------------------------------------------
# 3. LOAD BENCHMARK
# --------------------------------------------------

def read_questions(path):
    """
    Load a fixed benchmark CSV.

    Required:
        question

    Optional:
        question_id
        id
        category
        difficulty
        expected_source
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {path}"
        )

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:

        questions = list(
            csv.DictReader(file)
        )

    if not questions:
        raise ValueError(
            "Benchmark contains no questions."
        )

    seen_ids = set()

    for index, row in enumerate(
        questions,
        start=1
    ):

        question = (
            row.get("question") or ""
        ).strip()

        if not question:
            raise ValueError(
                f"Question {index} is empty."
            )

        question_id = (
            row.get("question_id")
            or row.get("id")
            or f"Q{index:02d}"
        ).strip()

        if question_id in seen_ids:
            raise ValueError(
                f"Duplicate question ID: {question_id}"
            )

        seen_ids.add(question_id)

        row["question_id"] = question_id
        row["question"] = question

    return questions


# --------------------------------------------------
# 4. TOKEN ACCOUNTING
# --------------------------------------------------

def extract_usage(response):
    """Read token usage from an OpenAI API response."""

    usage = getattr(
        response,
        "usage",
        None
    )

    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens}


def normalise_usage(value):
    """Read Responses/agent/planner tokens without inventing missing usage."""
    value = value or {}
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    total_tokens = value.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens}


def combine_usage(*items):
    """
    Sum usage across multiple agent calls.

    Never treat missing usage as zero.
    """

    combined = {}

    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ):

        values = [
            item.get(key)
            for item in items
        ]

        combined[key] = (
            sum(values)
            if all(v is not None for v in values)
            else None
        )

    return combined


# --------------------------------------------------
# 5. SINGLE-MODEL BASELINES
# --------------------------------------------------

BASE_INSTRUCTIONS = """
You are a strategic decision-support analyst
specialising in higher education.

Answer the user's specific question.

Present:
1. Evidence-based findings
2. Relevant numerical comparisons
3. Strategic implications and risks
4. Recommended actions
5. Evidence gaps and limitations

Requirements:
- Do not invent statistics or sources.
- Distinguish observations from recommendations.
- Never infer causation from descriptive differences.
- Preserve years, populations, measures and units.
- Distinguish percentages from percentage points.
- Do not claim statistical significance without testing.
- Do not infer a longitudinal trend from one survey year.
- Distinguish simulated institutional data from
  published university and official sector data.
- Do not present synthetic institutional figures
  as actual University of East London figures.
"""


def format_evidence(evidence):
    """Prepare evidence for the single-agent RAG model."""

    ordered = sorted(
        evidence,
        key=lambda item: (
            0 if item.get("is_calculated") else 1
        )
    )

    return "\n\n".join(
        f"[{item['source_id']}]\n"
        f"Document: {item['document']}\n"
        f"Section: {item['section']}\n"
        f"Data provenance: "
        f"{item.get('data_type', 'unknown')}\n"
        f"Calculated: "
        f"{item.get('is_calculated', False)}\n"
        f"Source URL: "
        f"{item.get('source_url', '')}\n"
        f"Content:\n{item['content']}"
        for item in ordered
    )


def call_model(
    client,
    question,
    evidence=None
):
    """
    Run the standalone or single-agent RAG baseline.
    """

    instructions = BASE_INSTRUCTIONS

    prompt = (
        f"STRATEGIC QUESTION:\n"
        f"{question}"
    )

    if evidence is None:

        instructions += """
No institutional source evidence is supplied.

Do not claim to have inspected local data,
unpublished institutional records or downloaded
datasets.

Where a factual comparison cannot be established,
identify the missing evidence explicitly.

Do not fabricate source IDs or citations.
"""

    else:

        instructions += """
Use only the supplied evidence for
institution-specific numerical claims.

Cite the original source IDs.

Prioritise relevant calculated structured evidence
when the question requires a numerical comparison.

Do not substitute historical report narratives
for newer numerical evidence addressing the question.

Distinguish published UK aggregate NSS figures
from OfS provider benchmarks.

Treat a displayed subset as a subset.

Do not claim that supplied numerical data is missing.

Preserve the source provenance labels.
"""

        prompt += (
            "\n\nSUPPORTING EVIDENCE:\n"
            + format_evidence(evidence)
        )

    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=prompt
    )

    return (
        response.output_text,
        extract_usage(response)
    )


# --------------------------------------------------
# 6. RUN ONE ARCHITECTURE
# --------------------------------------------------

def evaluate_system(
    system_name,
    question,
    evidence,
    client,
    analyser,
    reporter
):
    """
    Generate one response and return answer,
    token usage and optional intermediate analysis.
    """

    if system_name == "standalone":

        answer, usage = call_model(
            client=client,
            question=question,
            evidence=None
        )

        return answer, usage, None

    if system_name == "single_rag":

        if not evidence:
            return (
                "Insufficient retrieved evidence.",
                {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                },
                None
            )

        answer, usage = call_model(
            client=client,
            question=question,
            evidence=evidence
        )

        return answer, usage, None

    if system_name == "multi_agent":

        if not evidence:
            return (
                "Insufficient retrieved evidence.",
                {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                },
                None
            )

        analysis_result = analyser.run(
            question=question,
            evidence=evidence
        )

        reporting_result = reporter.run(
            question=question,
            analysis=analysis_result["analysis"],
            evidence=evidence
        )

        analysis_usage = normalise_usage(
            analysis_result.get("usage")
        )

        reporting_usage = normalise_usage(
            reporting_result.get("usage")
        )

        usage = combine_usage(
            analysis_usage,
            reporting_usage
        )

        return (
            reporting_result["report"],
            usage,
            analysis_result["analysis"]
        )

    raise ValueError(
        f"Unknown architecture: {system_name}"
    )


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

def save_results(
    output_dir,
    records,
    manifest
):
    """Write complete responses and timing data."""

    json_path = (
        output_dir
        / "comparison_outputs.json"
    )

    json_path.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    timing_path = (
        output_dir
        / "comparison_timing.csv"
    )

    columns = [
        "question_id",
        "category",
        "system",
        "model",
        "status",
        "retrieval_seconds",
        "generation_seconds",
        "total_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "planning_input_tokens", "planning_output_tokens",
        "generation_input_tokens", "generation_output_tokens",
        "generation_total_tokens", "planning_method", "selected_sources",
        "retrieval_warnings", "evidence_count", "error",
    ]

    with timing_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(records)

    manifest_path = (
        output_dir
        / "comparison_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nRESULTS SAVED")
    print("=" * 65)

    print("Responses:", json_path)
    print("Timing:", timing_path)
    print("Manifest:", manifest_path)


# --------------------------------------------------
# 8. FINAL EVALUATION
# --------------------------------------------------

def run(limit=None, question_file=None):

    benchmark_path = Path(question_file).resolve() if question_file else QUESTION_FILE
    questions = read_questions(
        benchmark_path
    )
    full_question_count = len(questions)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        questions = questions[:limit]

    run_id, output_dir = (
        new_run_directory()
    )

    git_info = get_git_information()

    client = OpenAI()

    retriever = RetrievalAgent(
        top_k=TOP_K
    )

    analyser = AnalysisAgent(
        model=MODEL
    )

    reporter = ReportingAgent(
        model=MODEL
    )

    records = []

    print("\nFINAL ARCHITECTURE COMPARISON")
    print("=" * 65)

    print("Run ID:", run_id)
    print("Model:", MODEL)
    print("Top-K:", TOP_K)
    print("Questions:", len(questions))
    print("Architectures:", len(SYSTEMS))
    print("Git commit:", git_info["commit"])

    print("=" * 65)

    for index, row in enumerate(
        questions,
        start=1
    ):

        question_id = row["question_id"]
        question = row["question"]

        print(
            f"\n[{index}/{len(questions)}] "
            f"{question_id}"
        )

        print("Question:", question)

        # --------------------------------------
        # RETRIEVE ONCE FOR BOTH RAG SYSTEMS
        # --------------------------------------

        retrieval_start = (
            time.perf_counter()
        )

        evidence = []

        retrieval_error = ""
        retrieval_warnings = []
        planning_usage = normalise_usage(None)
        planning_method = None
        selected_sources = []

        try:

            retrieval_result = (
                retriever.run(question)
            )

            evidence = retrieval_result["evidence"]
            retrieval_warnings = retrieval_result.get("warnings", [])
            planning_usage = normalise_usage(retrieval_result.get("planning_usage"))
            retrieval_plan = retrieval_result.get("retrieval_plan", {})
            planning_method = retrieval_plan.get("planning_method")
            selected_sources = retrieval_plan.get("sources", [])

        except Exception as error:

            retrieval_error = str(error)

            print(
                "  Retrieval failed:",
                retrieval_error
            )

        retrieval_seconds = (
            time.perf_counter()
            - retrieval_start
        )

        # --------------------------------------
        # RUN ALL THREE ARCHITECTURES
        # --------------------------------------

        for system_name in SYSTEMS:

            print(
                f"  Running {system_name}...",
                flush=True
            )

            generation_start = (
                time.perf_counter()
            )

            answer = ""
            intermediate_analysis = None

            usage = {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }

            status = "ok"
            error_message = ""

            try:

                if (
                    system_name != "standalone"
                    and retrieval_error
                ):
                    raise RuntimeError(
                        "Retrieval failed: "
                        + retrieval_error
                    )

                (
                    answer,
                    usage,
                    intermediate_analysis
                ) = evaluate_system(
                    system_name=system_name,
                    question=question,
                    evidence=evidence,
                    client=client,
                    analyser=analyser,
                    reporter=reporter
                )

                if (
                    system_name != "standalone"
                    and not evidence
                ):
                    status = "insufficient_evidence"

            except Exception as error:

                status = "error"
                error_message = str(error)

                print(
                    "  ERROR:",
                    error_message
                )

            generation_seconds = (
                time.perf_counter()
                - generation_start
            )

            system_retrieval_time = (
                0.0
                if system_name == "standalone"
                else retrieval_seconds
            )

            total_seconds = (
                generation_seconds
                + system_retrieval_time
            )

            # Planning is part of retrieval and must be charged to BOTH
            # RAG architectures, even though it runs only once in this
            # paired-evidence experiment. Do not charge standalone LLM.
            generation_usage = normalise_usage(usage)
            billed_usage = (generation_usage if system_name == "standalone"
                            else combine_usage(planning_usage, generation_usage))

            record = {
                "run_id": run_id,
                "question_id": question_id,
                "question": question,
                "category": row.get(
                    "category", ""
                ),
                "system": system_name,
                "model": MODEL,
                "top_k": TOP_K,
                "status": status,
                "error": error_message,
                "retrieval_seconds": round(
                    system_retrieval_time, 4
                ),
                "generation_seconds": round(
                    generation_seconds, 4
                ),
                "total_seconds": round(
                    total_seconds, 4
                ),
                "retrieval_warnings": ([] if system_name == "standalone"
                                       else retrieval_warnings),
                "planning_method": (None if system_name == "standalone"
                                    else planning_method),
                "selected_sources": ([] if system_name == "standalone"
                                     else selected_sources),
                "planning_input_tokens": (None if system_name == "standalone"
                                          else planning_usage["input_tokens"]),
                "planning_output_tokens": (None if system_name == "standalone"
                                           else planning_usage["output_tokens"]),
                "generation_input_tokens": generation_usage["input_tokens"],
                "generation_output_tokens": generation_usage["output_tokens"],
                "generation_total_tokens": generation_usage["total_tokens"],
                "evidence_count": (
                    0
                    if system_name == "standalone"
                    else len(evidence)
                ),
                "evidence": (
                    []
                    if system_name == "standalone"
                    else evidence
                ),
                "answer": answer,
                "intermediate_analysis": (
                    intermediate_analysis
                ),
                **billed_usage
            }

            records.append(record)

            print(
                f"  Status: {status}; "
                f"Time: {total_seconds:.2f}s; "
                f"Tokens: {billed_usage['total_tokens']}"
            )

    # ------------------------------------------
    # EXPERIMENT MANIFEST
    # ------------------------------------------

    manifest = {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "experiment": (
            "Standalone vs single-agent RAG "
            "vs multi-agent RAG"
        ),
        "model": MODEL,
        "top_k": TOP_K,
        "question_file": str(
            benchmark_path.relative_to(ROOT) if benchmark_path.is_relative_to(ROOT)
            else benchmark_path
        ),
        "question_file_sha256": (
            file_sha256(benchmark_path)
        ),
        "question_count": len(questions),
        "full_benchmark_question_count": full_question_count,
        "run_mode": ("full" if limit is None else "smoke_test"),
        "implementation_sha256": implementation_hashes(),
        "source_data_sha256": source_file_hashes(),
        "systems": list(SYSTEMS),
        "git_commit": git_info["commit"],
        "git_working_tree_clean": (
            git_info["working_tree_clean"]
        ),
        "total_response_records": len(records),
        "successful_responses": sum(
            record["status"] == "ok"
            for record in records
        ),
        "failed_responses": sum(
            record["status"] == "error"
            for record in records
        ),
        "methodology_notes": [
            "All systems use the same configured model.",
            "Both RAG systems share one retrieval result per question.",
            "Both RAG systems receive identical source evidence.",
            "Retrieval time is included in RAG total time.",
            "Standalone LLM receives no private institutional evidence.",
            "Structured analysis tools are shared through RetrievalAgent.",
            "Multi-agent generation tokens combine analysis and reporting calls.",
            "Both RAG total token counts also include one shared planning call per question.",
            "Unknown token usage remains null, not zero.",
            "Retrieval warnings and planner methods are retained per question.",
            "Timing is paired per question, with the same retrieval duration assigned to both RAG variants.",
            "Question-level retrieval is executed before generation.",
            "Response quality requires separate reference-based evaluation.",
            "Results are from a fixed dataset snapshot at execution time.",
            "No separate response-quality or correctness scores are implied by the runner."
        ]
    }

    save_results(
        output_dir=output_dir,
        records=records,
        manifest=manifest
    )

    print("\nEVALUATION FINISHED")
    print("=" * 65)

    print(
        "Responses generated:",
        len(records)
    )

    print(
        "Successful:",
        manifest["successful_responses"]
    )

    print(
        "Errors:",
        manifest["failed_responses"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dissertation architecture comparison")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run first N questions as a smoke test; omit for full benchmark")
    parser.add_argument("--questions", type=str, default=None,
                        help="Optional development-test CSV; omit for the fixed benchmark")
    args = parser.parse_args()
    run(limit=args.limit, question_file=args.questions)
