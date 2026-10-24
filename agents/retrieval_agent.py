
"""
retrieval_agent.py

Retrieves relevant evidence from:
1. The ChromaDB institutional knowledge base.
2. Structured simulated undergraduate retention data.
3. Official OfS National Student Survey (NSS) data.

Structured calculations supplement semantic retrieval.
Source provenance is preserved for downstream agents.
"""

import re
import pandas as pd
from document_retriever import search_documents
from data_analysis import analyse_retention
from nss_analysis import analyse_nss


# --------------------------------------------------
# QUESTION INTERPRETATION HELPERS
# --------------------------------------------------

def extract_academic_years(question):
    """
    Extract academic years explicitly mentioned in
    a question, in chronological order.
    """

    years = re.findall(
        r"\b(?:19|20)\d{2}/\d{2,4}\b",
        question
    )

    return sorted(
        set(years),
        key=lambda year: int(year.split("/")[0])
    )


def wants_retention_analysis(question):
    """
    Identify questions requiring structured
    school-level retention calculations.
    """

    text = question.casefold()

    retention_terms = (
        "retention",
        "continuation",
    )

    comparison_terms = (
        "school",
        "schools",
        "compare",
        "comparison",
        "changed",
        "change",
        "trend",
        "decline",
        "increase",
        "highest",
        "lowest",
        "target",
        "over time",
    )

    return (
        any(term in text for term in retention_terms)
        and any(term in text for term in comparison_terms)
    )


def wants_nss_analysis(question):
    """
    Identify questions requiring published
    National Student Survey calculations.
    """

    text = question.casefold()

    return (
        re.search(r"\bnss\b", text) is not None
        or "national student survey" in text
    )


def get_nss_parameters(question):
    """
    Extract supported NSS analysis options.

    The default is a comparison of UEL against
    the published UK aggregate for all undergraduates
    in the full-time teaching-provider dataset.
    """

    text = question.casefold()

    level = "All undergraduates"

    if "first degree" in text:
        level = "First degree"

    elif "other undergraduate" in text:
        level = "Other undergraduate"

    question_filter = None

    theme_match = re.search(
        r"\btheme\s+(\d+)\b",
        text
    )

    if theme_match:
        question_filter = (
            f"Theme {theme_match.group(1)}:"
        )

    question_match = re.search(
        r"\bq(\d{1,2})\b",
        text
    )

    if question_match:
        question_filter = (
            f"Q{int(question_match.group(1)):02d}:"
        )

    return {
        "level": level,
        "question_filter": question_filter,
        "comparison_provider": None,
    }


# --------------------------------------------------
# RETRIEVAL AGENT
# --------------------------------------------------

class RetrievalAgent:
    """
    Retrieves semantic evidence and supplements it
    with deterministic structured calculations
    when relevant to the user's question.
    """

    def __init__(self, top_k=3):
        self.top_k = top_k
        self.name = "Retrieval Agent"

    def run(self, question):

        if not question or not question.strip():
            raise ValueError(
                "A strategic question is required."
            )

        # ------------------------------------------
        # 1. SEMANTIC DOCUMENT RETRIEVAL
        # ------------------------------------------

        documents = search_documents(
            question=question,
            n_results=self.top_k
        )

        evidence = []

        for index, doc in enumerate(
            documents,
            start=1
        ):

            evidence.append({
                "source_id": f"Source {index}",
                "document": doc["source"],
                "section": doc["section"],
                "content": doc["text"],
                "distance": doc["distance"],
                "source_type": doc.get(
                    "source_type", "unknown"
                ),
                "data_type": doc.get(
                    "data_type", "unknown"
                ),
                "source_url": doc.get(
                    "source_url", ""
                ),
                "page_number": doc.get(
                    "page_number", 0
                ),
                "row_number": doc.get(
                    "row_number", 0
                ),
            })

        # ------------------------------------------
        # 2. SIMULATED RETENTION ANALYSIS
        # ------------------------------------------

        if wants_retention_analysis(question):

            years = extract_academic_years(
                question
            )

            start_year = None
            end_year = None

            if len(years) >= 2:
                start_year = years[0]
                end_year = years[-1]

            elif len(years) == 1:
                # A question naming exactly one year
                # produces a single-year snapshot.
                start_year = years[0]
                end_year = years[0]

            # When no years are provided, analyse
            # every available year automatically.
            comparison = analyse_retention(
                start_year=start_year,
                end_year=end_year
            )

            year_columns = [
                column
                for column in comparison.columns
                if re.fullmatch(
                    r"(?:19|20)\d{2}/\d{2,4}",
                    str(column)
                )
            ]

            period = (
                f"{year_columns[0]} to {year_columns[-1]}"
                if len(year_columns) > 1
                else year_columns[0]
            )

            table = comparison.copy()

            # A change is not defined for a
            # single-year snapshot.
            if (
                len(year_columns) == 1
                and "Change (pp)" in table.columns
            ):
                table = table.drop(
                    columns=["Change (pp)"]
                )

            table_text = table.to_csv(
                index=False
            )

            explanation = (
                "SIMULATED INSTITUTIONAL DATA.\n"
                "Institution: Example Metropolitan University "
                "(fictional dataset).\n"
                "Source: Student Retention worksheet.\n"
                "Population: Undergraduate students.\n"
                f"Analysis period: {period}.\n"
                "School-level retention percentages are "
                "taken from the source workbook.\n"
            )

            if len(year_columns) > 1:
                explanation += (
                    "Change (pp) is calculated as "
                    f"{year_columns[-1]} minus "
                    f"{year_columns[0]}.\n"
                    "Negative values indicate a decline.\n"
                )
            else:
                explanation += (
                    "This is a single-year snapshot; "
                    "no change has been calculated.\n"
                )

            explanation += (
                "The source workbook records a "
                "90% retention target.\n"
                "Do not describe these synthetic "
                "figures as official UEL results.\n\n"
                + table_text
            )

            evidence.append({
                "source_id": (
                    f"Source {len(evidence) + 1}"
                ),
                "document": (
                    "institutional_performance.xlsx"
                ),
                "section": (
                    "Student Retention - "
                    "calculated school comparison"
                ),
                "content": explanation,
                "distance": None,
                "source_type": "structured_analysis",
                "data_type": "simulated",
                "source_url": "",
                "page_number": 0,
                "row_number": 0,
                "is_calculated": True,
            })

        # ------------------------------------------
        # 3. OFFICIAL NSS DATA ANALYSIS
        # ------------------------------------------

        if wants_nss_analysis(question):

            parameters = get_nss_parameters(
                question
            )

            result = analyse_nss(
                **parameters
            )

            comparison = result["data"]

            # Keep the evidence compact enough
            # for downstream language models.
            # For a broad question, supply the
            # strongest and weakest comparisons.
            if (
                parameters["question_filter"] is None
                and len(comparison) > 12
            ):

                selected = pd.concat(
                    [
                        comparison.head(6),
                        comparison.tail(6)
                    ],
                    ignore_index=True
                ).drop_duplicates(
                    subset=["Question"]
                )

                coverage_note = (
                    f"Full analysis covered "
                    f"{len(comparison)} published "
                    "question/theme records. "
                    "The following table contains "
                    "the six lowest and six highest "
                    "comparisons.\n"
                )

            else:
                selected = comparison.copy()

                coverage_note = (
                    f"Analysis covered "
                    f"{len(comparison)} matching "
                    "published records.\n"
                )

            table_text = selected.to_csv(
                index=False
            )

            content = (
                "OFFICIAL PUBLIC DATA - "
                "OFFICE FOR STUDENTS.\n"
                "Dataset: National Student Survey.\n"
                f"Workbook: {result['workbook']}.\n"
                f"Study mode: {result['study_mode']}.\n"
                f"Level: {result['level']}.\n"
                "Institution: University of East London "
                "(UKPRN 10007144).\n"
                f"Comparator: "
                f"{result['comparison_label']}.\n"
                f"{coverage_note}"
                "Positivity percentages are published "
                "OfS values.\n"
                "Difference (pp) is UEL minus "
                "the comparator, using displayed "
                "rounded percentages.\n"
                "The UK aggregate is not the same "
                "as an individual provider or "
                "the OfS benchmark.\n"
                "These are descriptive comparisons, "
                "not statistical significance tests.\n\n"
                + table_text
            )

            evidence.append({
                "source_id": (
                    f"Source {len(evidence) + 1}"
                ),
                "document": result["workbook"],
                "section": (
                    "NSS full-time provider "
                    "comparison - All_subjects"
                ),
                "content": content,
                "distance": None,
                "source_type": "structured_analysis",
                "data_type": "official_public_data",
                "source_url": result["source_url"],
                "page_number": 0,
                "row_number": 0,
                "is_calculated": True,
                "publisher": result["publisher"],
                "sha256": result["sha256"],
            })

            print("\n[RETRIEVAL DEBUG]")
            print("Question:", question)
            print("Evidence count:", len(evidence))

            for item in evidence:
                print(
                    item["source_id"],
                    item["document"],
                    item.get("data_type")
                )

        # ------------------------------------------
        # 4. RETURN COMBINED EVIDENCE
        # ------------------------------------------


        return {
            "agent": self.name,
            "question": question,
            "evidence": evidence,
            "evidence_count": len(evidence)
        }


# --------------------------------------------------
# LOCAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    agent = RetrievalAgent(top_k=3)

    test_questions = [
        (
            "How has undergraduate retention changed "
            "across all five schools between "
            "2021/22 and 2024/25?"
        ),
        (
            "Compare UEL's NSS 2026 results "
            "with the UK for full-time students."
        ),
        (
            "How does UEL perform on NSS Theme 1 "
            "for first degree students?"
        ),
    ]

    for question in test_questions:

        print("\n" + "=" * 70)
        print("QUESTION:", question)

        result = agent.run(question)

        print(
            "Evidence count:",
            result["evidence_count"]
        )

        for item in result["evidence"]:

            print(
                f"\n{item['source_id']}: "
                f"{item['document']}"
            )

            print(
                "Data type:",
                item.get("data_type", "unknown")
            )

            if item.get("is_calculated"):
                print(item["content"])
