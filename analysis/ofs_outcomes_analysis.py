
"""
Dynamic Office for Students Student Outcomes adapter.

Supports provider-level OfS Student Outcomes CSV archives.

Features:
- No hard-coded provider UKPRN
- No hard-coded reporting year
- Configurable indicator, population, mode and level
- Suppression-aware numerical analysis
- Validated schema and provider matching
- Explicit publication provenance

A cached archive is used for reproducibility.
New official releases can be supplied using archive_path.

Run from project root:
    python -m analysis.ofs_outcomes_analysis
"""

import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTCOMES_CACHE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "source_hub"
    / "ofs"
)

SOURCE_PAGE = (
    "https://www.officeforstudents.org.uk/"
    "data-and-analysis/student-outcomes-data-dashboard/"
    "get-the-data/"
)

REQUIRED_COLUMNS = {
    "UKPRN_of_registering_provider",
    "POPULATION_TYPE",
    "INDICATOR_NAME",
    "MODE",
    "LEVEL",
    "SPLIT_IND_TYPE",
    "SPLIT_IND",
    "YEAR",
    "SUPP_REASON",
    "INDICATOR_VALUE",
    "DENOMINATOR",
}

MAX_PROVIDER_CSV_BYTES = 50_000_000


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def available_archives():
    """
    Find locally cached per-provider outcomes ZIP files.

    Does not guess which publication is newest from
    the filename. Release selection must be explicit
    if more than one matching archive exists.
    """

    if not OUTCOMES_CACHE.exists():
        return []

    return sorted(
        p for p in OUTCOMES_CACHE.glob("*.zip")
        if "student_outcomes_each_provider" in p.name.casefold()
    )


def resolve_archive(archive_path=None):

    if archive_path is not None:
        path = Path(archive_path).resolve()

    else:
        candidates = available_archives()

        if not candidates:
            raise FileNotFoundError(
                "No cached OfS per-provider outcomes "
                "archive was found. Download an official "
                "archive using the ingestion workflow."
            )

        if len(candidates) > 1:
            raise ValueError(
                "Multiple OfS outcomes releases are cached. "
                "Specify archive_path to select the "
                "intended publication:\n"
                + "\n".join(str(p) for p in candidates)
            )

        path = candidates[0].resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"OfS outcomes archive not found: {path}"
        )

    if not zipfile.is_zipfile(path):
        raise ValueError(
            "The selected OfS file is not a valid ZIP."
        )

    return path


def normalise(value):
    return str(value).strip().casefold()


def resolve_category(values, requested, dimension):
    """
    Validate a requested category against actual
    published values, case-insensitively.
    """

    options = sorted(
        {str(value).strip() for value in values if str(value).strip()}
    )

    if requested is None:
        return None

    matches = [
        option for option in options
        if normalise(option) == normalise(requested)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Unsupported {dimension}: {requested!r}. "
            f"Available values: {options}"
        )

    return matches[0]


def load_provider_outcomes(provider, archive_path=None):
    """
    Load one provider's published CSV directly
    from an official per-provider ZIP.

    provider must be a valid UKPRN.
    A provider-name resolution service can be
    connected later without changing this loader.
    """

    ukprn = str(provider).strip()

    if not re.fullmatch(r"\d{8}", ukprn):
        raise ValueError(
            "Provide an eight-digit UKPRN. "
            "Institution names must first be resolved "
            "to a verified UKPRN."
        )

    archive_path = resolve_archive(archive_path)

    expected_name = (
        f"student_outcomes__{ukprn}.csv"
    )

    with zipfile.ZipFile(archive_path) as archive:

        matches = [
            info
            for info in archive.infolist()
            if (
                not info.is_dir()
                and Path(info.filename).name.casefold()
                == expected_name.casefold()
            )
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected one provider file for "
                f"UKPRN {ukprn}, found {len(matches)}."
            )

        member = matches[0]

        if member.file_size > MAX_PROVIDER_CSV_BYTES:
            raise ValueError(
                "Provider CSV exceeds safe size limit."
            )

        with archive.open(member) as file:
            df = pd.read_csv(
                file,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )

    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Unsupported OfS outcomes schema. "
            f"Missing columns: {sorted(missing)}"
        )

    for column in REQUIRED_COLUMNS:
        df[column] = df[column].str.strip()

    if df.empty:
        raise ValueError(
            f"OfS provider file {ukprn} is empty."
        )

    if not df["UKPRN_of_registering_provider"].eq(ukprn).all():
        raise ValueError(
            "Provider identifier mismatch inside "
            "the OfS data file."
        )

    return df, {
        "publisher": "Office for Students",
        "ukprn": ukprn,
        "source_type": "official_public_data",
        "source_page": SOURCE_PAGE,
        "source_file": member.filename,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "acquisition_method": "cached_official_archive",
    }


def list_available_dimensions(provider, archive_path=None):
    """
    Show actual published options without making
    assumptions about study modes or cohort names.
    """

    df, source = load_provider_outcomes(
        provider,
        archive_path
    )

    columns = {
        "indicators": "INDICATOR_NAME",
        "populations": "POPULATION_TYPE",
        "modes": "MODE",
        "levels": "LEVEL",
        "years": "YEAR",
    }

    return {
        **source,
        "available": {
            name: sorted(
                df[column].unique().tolist()
            )
            for name, column in columns.items()
        },
    }


def analyse_ofs_outcomes(
    provider,
    indicator="Continuation",
    year=None,
    population=None,
    mode=None,
    level=None,
    archive_path=None,
    include_suppressed=False,
):
    """
    Analyse published overall-indicator records.

    All requested dimensions must match actual
    source categories.

    If optional dimensions are omitted, multiple
    matching records may be returned. The caller
    must choose a sufficiently specific result
    before stating one headline percentage.
    """

    df, source = load_provider_outcomes(
        provider=provider,
        archive_path=archive_path,
    )

    filters = [
        ("INDICATOR_NAME", indicator, "indicator"),
        ("POPULATION_TYPE", population, "population"),
        ("MODE", mode, "mode"),
        ("LEVEL", level, "level"),
        ("YEAR", year, "year"),
    ]

    # Validate requested values against the full
    # provider file, then narrow the result.
    for column, requested, label in filters:

        resolved = resolve_category(
            df[column],
            requested,
            label,
        )

        if resolved is not None:
            df = df[
                df[column].eq(resolved)
            ].copy()

        if df.empty:
            raise ValueError(
                "No records match the requested "
                "combination of OfS dimensions."
            )

    # Use only provider-level overall indicators,
    # not demographic subgroups.
    df = df[
        df["SPLIT_IND_TYPE"].str.casefold().eq("overall")
        & df["SPLIT_IND"].str.casefold().eq(
            "overall indicator"
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No overall-indicator records match "
            "the requested dimensions."
        )

    # Treat any suppression marker as disqualifying.
    # The core indicator suppression and additional
    # published suppression columns are preserved.
    suppression_columns = [
        col
        for col in df.columns
        if col.startswith("SUPP_REASON")
    ]

    df["is_suppressed"] = df[
        "SUPP_REASON"
    ].ne("")

    df["indicator_numeric"] = pd.to_numeric(
        df["INDICATOR_VALUE"],
        errors="coerce"
    )

    df["denominator_numeric"] = pd.to_numeric(
        df["DENOMINATOR"],
        errors="coerce"
    )

    df["is_valid"] = (
        ~df["is_suppressed"]
        & df["indicator_numeric"].notna()
        & df["indicator_numeric"].between(0, 100)
    )

    total_matches = len(df)
    suppressed_matches = int(
        df["is_suppressed"].sum()
    )

    if not include_suppressed:
        df = df[df["is_valid"]].copy()

    keep_columns = [
        "POPULATION_TYPE",
        "INDICATOR_NAME",
        "MODE",
        "LEVEL",
        "SPLIT_IND_TYPE",
        "SPLIT_IND",
        "YEAR",
        "INDICATOR_VALUE",
        "DENOMINATOR",
        "SUPP_REASON",
        "is_suppressed",
        "is_valid",
    ]

    df = df[keep_columns].sort_values(
        [
            "POPULATION_TYPE",
            "INDICATOR_NAME",
            "MODE",
            "LEVEL",
            "YEAR",
        ]
    ).reset_index(drop=True)

    return {
        **source,
        "indicator": indicator,
        "requested_year": year,
        "requested_population": population,
        "requested_mode": mode,
        "requested_level": level,
        "total_matching_overall_records": total_matches,
        "suppressed_records": suppressed_matches,
        "returned_records": len(df),
        "data": df,
        "limitations": (
            "OfS outcomes are defined indicators "
            "with distinct cohort periods and study "
            "populations. Suppressed results must "
            "not be treated as zero. Published "
            "indicator values are not interchangeable "
            "with internal retention statistics."
        ),
        "suppression_columns_present": (
            suppression_columns
        ),
    }


if __name__ == "__main__":

    print("\nOfS STUDENT OUTCOMES")
    print("=" * 70)

    # A known test provider only; the adapter itself
    # works with any valid UKPRN in the archive.
    ukprn = "10007144"

    dimensions = list_available_dimensions(ukprn)

    for name, values in dimensions["available"].items():
        print(f"\n{name}: {values}")

    result = analyse_ofs_outcomes(
        provider=ukprn,
        indicator="Continuation",
    )

    print("\nOverall continuation records:")
    print("Matching:", result["total_matching_overall_records"])
    print("Returned:", result["returned_records"])

    print(
        result["data"].head(30).to_string(
            index=False
        )
    )
