
"""
Dynamic HESA provider-enrolment analysis.

Supports any provider present in a validated HESA
provider-enrolment dataset, selected by UKPRN or name.

Supported schema:
HESA Student Statistics, provider-enrolment Figure 7.

The academic year is read from publication metadata.
No provider name, UKPRN or year is hard-coded.

This is a statistical adapter, not a release-discovery
system. A dataset for another academic year must be
provided or selected by the acquisition layer.
"""

import hashlib
import re
from pathlib import Path

from ingestion.source_hub import read_hesa_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "hesa_provider_enrolments.csv"
)


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def clean(value):
    return str(value or "").strip()


def available_providers(rows):
    """
    Return unique provider identifiers from the dataset.
    """

    providers = {
        (
            clean(row["UKPRN"]),
            clean(row["HE provider"])
        )
        for row in rows
        if clean(row.get("UKPRN"))
        and clean(row.get("HE provider"))
    }

    return [
        {"ukprn": ukprn, "provider": name}
        for ukprn, name in sorted(
            providers,
            key=lambda item: item[1].casefold()
        )
    ]


def resolve_provider(rows, provider):
    """
    Resolve a provider by exact name or UKPRN.

    Do not guess from partial or ambiguous names.
    """

    requested = clean(provider)

    if not requested:
        raise ValueError(
            "A provider name or UKPRN is required."
        )

    matches = [
        item
        for item in available_providers(rows)
        if (
            item["ukprn"] == requested
            or item["provider"].casefold()
            == requested.casefold()
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely resolve provider: "
            f"{requested!r}. Supply its official "
            "HESA provider name or UKPRN."
        )

    return matches[0]


def analyse_hesa_enrolments(
    provider,
    academic_year=None,
    level="All",
    mode="All",
    dataset_path=None,
    acquisition_method="provided_local_dataset",
):
    """
    Return a published provider enrolment count.

    Parameters
    ----------
    provider:
        Official provider name or UKPRN.

    academic_year:
        Optional requested year. Must match the
        selected dataset's published coverage.

    level:
        Exact published Level of study category.

    mode:
        Exact published Mode of study category.

    dataset_path:
        Path to a compatible, validated HESA CSV.

    acquisition_method:
        Describes how the dataset was obtained.

    Returns
    -------
    dict:
        Published count and complete provenance.
    """

    path = Path(
        dataset_path
        if dataset_path is not None
        else DEFAULT_DATASET
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"HESA dataset not found: {path}"
        )

    metadata, rows = read_hesa_table(path)

    published_year = clean(
        metadata.get("Academic years")
    )

    if not published_year:
        raise ValueError(
            "Missing published academic year metadata."
        )

    # This schema is a single-year provider table.
    # Do not silently treat it as a time series.
    if not re.fullmatch(
        r"\d{4}/\d{2}",
        published_year
    ):
        raise ValueError(
            "This HESA adapter requires a single-year "
            f"provider dataset. Found: {published_year!r}"
        )

    if (
        academic_year is not None
        and clean(academic_year) != published_year
    ):
        raise ValueError(
            f"Requested {academic_year}, but the "
            f"selected dataset covers {published_year}. "
            "Select the appropriate published release."
        )

    resolved = resolve_provider(
        rows,
        provider
    )

    provider_rows = [
        row for row in rows
        if clean(row["UKPRN"])
        == resolved["ukprn"]
    ]

    available_levels = sorted({
        clean(row["Level of study"])
        for row in provider_rows
    })

    available_modes = sorted({
        clean(row["Mode of study"])
        for row in provider_rows
    })

    def match(actual, requested):
        return (
            clean(actual).casefold()
            == clean(requested).casefold()
        )

    selected = [
        row
        for row in provider_rows
        if (
            match(row["Level of study"], level)
            and match(row["Mode of study"], mode)
            and match(
                row["Country of HE provider"],
                "All"
            )
            and match(
                row["Region of HE provider"],
                "All"
            )
            and match(
                row["Permanent address"],
                "Total"
            )
        )
    ]

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one published total "
            f"for {resolved['provider']}, "
            f"level={level!r}, mode={mode!r}; "
            f"found {len(selected)}. "
            f"Available levels: {available_levels}. "
            f"Available modes: {available_modes}."
        )

    value = clean(
        selected[0]["Number"]
    ).replace(",", "")

    if not re.fullmatch(r"\d+", value):
        raise ValueError(
            "The selected HESA count is missing, "
            "suppressed or not a valid integer."
        )

    return {
        "publisher": "HESA",
        "provider": resolved["provider"],
        "ukprn": resolved["ukprn"],
        "academic_year": published_year,
        "level": clean(
            selected[0]["Level of study"]
        ),
        "mode": clean(
            selected[0]["Mode of study"]
        ),
        "enrolments": int(value),
        "source_url": metadata.get(
            "Data file canonical link"
        ),
        "source_page": metadata.get(
            "Data source link"
        ),
        "licence": metadata.get(
            "Licence"
        ),
        "last_updated": metadata.get(
            "Last updated"
        ),
        "source_file": path.name,
        "sha256": file_sha256(path),
        "acquisition_method": acquisition_method,
        "source_type": "official_public_data",
        "limitations": (
            "Published HESA rounded enrolment count. "
            "This is not a retention or continuation "
            "rate. Overlapping subtotals are not added."
        ),
    }


def list_hesa_providers(dataset_path=None):
    """
    List available providers without knowing
    their UKPRNs in advance.
    """

    path = (
        Path(dataset_path)
        if dataset_path is not None
        else DEFAULT_DATASET
    )

    _, rows = read_hesa_table(path)

    return available_providers(rows)


if __name__ == "__main__":

    result = analyse_hesa_enrolments(
        provider="10007144"
    )

    for key, value in result.items():
        print(f"{key}: {value}")
