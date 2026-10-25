"""
nss_analysis.py

Analyse official NSS teaching-provider results.

Data source:
Office for Students, downloaded NSS full-time workbook.

All reported percentages come from published OfS data.
No synthetic data is used in this module.
"""

import argparse
import json
import re
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "external" / "nss"

UEL_UKPRN = "10007144"

REQUIRED_COLUMNS = [
    "UKPRN",
    "Provider name",
    "Level of study",
    "Question",
    "Responses",
    "Suppression reason",
    "Positivity measure (%)",
    "Benchmark (%)",
    "Difference (ppt)",
]


def find_latest_workbook():
    """Find the latest downloaded full-time teaching workbook."""

    files = list(
        DATA_DIR.glob("*Summary_Taught_Full-time.xlsx")
    )

    if not files:
        raise FileNotFoundError(
            "No NSS full-time teaching-provider workbook found "
            f"in {DATA_DIR}"
        )

    def release_year(path):
        match = re.search(
            r"NSS(\d{2})_Summary",
            path.name,
            flags=re.IGNORECASE
        )
        return int(match.group(1)) if match else -1

    return max(
        files,
        key=lambda path: (
            release_year(path),
            path.stat().st_mtime
        )
    )


def load_nss_data():
    """Load and validate the official All_subjects sheet."""

    workbook = find_latest_workbook()

    df = pd.read_excel(
        workbook,
        sheet_name="All_subjects",
        header=3,
        engine="openpyxl",
        dtype={"UKPRN": "string"}
    )

    df.columns = df.columns.astype(str).str.strip()

    missing = [
        name for name in REQUIRED_COLUMNS
        if name not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing NSS columns: {missing}"
        )

    for column in [
        "Positivity measure (%)",
        "Benchmark (%)",
        "Difference (ppt)",
        "Responses",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["UKPRN"] = df["UKPRN"].astype("string").str.strip()

    for column in [
        "Provider name",
        "Level of study",
        "Question",
    ]:
        df[column] = df[column].astype("string").str.strip()

    valid_percentages = df["Positivity measure (%)"].dropna()

    if not valid_percentages.between(0, 100).all():
        raise ValueError(
            "NSS data contains positivity values outside 0–100."
        )

    manifest_path = (
        workbook.parent
        / f"{workbook.stem}_manifest.json"
    )

    manifest = {}

    if manifest_path.exists():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

    return df, workbook, manifest


def analyse_nss(
    level="All undergraduates",
    question_filter=None,
    comparison_provider=None
):
    """
    Compare UEL with the UK aggregate or a named provider.

    The published positivity percentage is used directly.

    Suppressed or missing results are excluded.
    """

    df, workbook, manifest = load_nss_data()

    df = df[
        df["Level of study"].str.casefold()
        == level.casefold()
    ].copy()

    if df.empty:
        raise ValueError(
            f"No NSS records for level: {level}"
        )

    # Only analyse published, unsuppressed percentages.
    suppression = (
        df["Suppression reason"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        suppression.eq("")
        & df["Positivity measure (%)"].notna()
    ].copy()

    if question_filter:
        df = df[
            df["Question"].str.contains(
                question_filter,
                case=False,
                regex=False,
                na=False
            )
        ].copy()

    # Use the stable UK provider identifier rather
    # than relying solely on the provider's name.
    uel = df[
        df["UKPRN"] == UEL_UKPRN
    ].copy()

    if uel.empty:
        raise ValueError(
            "No matching, unsuppressed UEL results found."
        )

    if uel["Question"].duplicated().any():
        raise ValueError(
            "Duplicate UEL question records found."
        )

    if comparison_provider:
        other = df[
            df["Provider name"].str.casefold()
            == comparison_provider.casefold()
        ].copy()

        comparison_label = comparison_provider

        if other.empty:
            raise ValueError(
                "No matching comparison provider found: "
                f"{comparison_provider}"
            )

    else:
        # The workbook includes a published UK aggregate.
        # This is different from the OfS benchmark column.
        other = df[
            df["UKPRN"].str.upper() == "UK"
        ].copy()

        comparison_label = "UK aggregate"

        if other.empty:
            raise ValueError(
                "No published UK aggregate records found. "
                "A UK comparison cannot be calculated."
            )

    if other["Question"].duplicated().any():
        raise ValueError(
            "Duplicate comparison question records found."
        )

    comparison = uel[
        [
            "Question",
            "Positivity measure (%)",
            "Responses",
        ]
    ].merge(
        other[
            [
                "Question",
                "Positivity measure (%)",
            ]
        ],
        on="Question",
        how="inner",
        suffixes=("_UEL", "_Comparison"),
        validate="one_to_one"
    )

    if comparison.empty:
        raise ValueError(
            "UEL and comparison source have no common "
            "published questions."
        )

    comparison = comparison.rename(
        columns={
            "Positivity measure (%)_UEL":
                "UEL Positivity (%)",
            "Positivity measure (%)_Comparison":
                "Comparator Positivity (%)",
            "Responses":
                "UEL Responses"
        }
    )

    comparison["Difference (pp)"] = (
        comparison["UEL Positivity (%)"]
        - comparison["Comparator Positivity (%)"]
    ).round(1)

    comparison = comparison.sort_values(
        "Difference (pp)",
        ascending=True
    ).reset_index(drop=True)

    return {
        "data": comparison,
        "comparison_label": comparison_label,
        "level": level,
        "study_mode": "Full-time",
        "source_type": "official_public_data",
        "publisher": "Office for Students",
        "workbook": workbook.name,
        "source_url": manifest.get(
            "download_url", ""
        ),
        "source_page": manifest.get(
            "source_page", ""
        ),
        "sha256": manifest.get(
            "sha256", ""
        ),
        "questions_compared": len(comparison)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare UEL NSS results with published data."
    )

    parser.add_argument(
        "--level",
        default="All undergraduates"
    )

    parser.add_argument(
        "--question",
        default=None,
        help="Optional question or theme text."
    )

    parser.add_argument(
        "--provider",
        default=None,
        help="Optional provider name; defaults to UK aggregate."
    )

    args = parser.parse_args()

    result = analyse_nss(
        level=args.level,
        question_filter=args.question,
        comparison_provider=args.provider
    )

    comparison = result["data"]

    print("\nNSS PROVIDER COMPARISON")
    print("=" * 65)
    print("Publisher:", result["publisher"])
    print("Workbook:", result["workbook"])
    print("Study mode:", result["study_mode"])
    print("Level:", result["level"])
    print("Comparator:", result["comparison_label"])
    print("Questions compared:", result["questions_compared"])

    print("\nFIVE LOWEST DIFFERENCES")
    print("-" * 65)

    print(
        comparison.head(5).to_string(index=False)
    )

    print("\nFIVE HIGHEST DIFFERENCES")
    print("-" * 65)

    print(
        comparison.tail(5)
        .sort_values("Difference (pp)", ascending=False)
        .to_string(index=False)
    )

    print("\nSOURCE INFORMATION")
    print("-" * 65)
    print("URL:", result["source_url"] or "Not recorded")
    print("SHA-256:", result["sha256"] or "Not recorded")


if __name__ == "__main__":
    main()
