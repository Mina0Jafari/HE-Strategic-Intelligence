
"""
Shared acquisition interface for official evidence sources.

Reuses the existing source_hub functions.
Supports:
- HESA downloads with validated local fallback
- OfS ZIP/CSV/XLSX downloads
- GOV.UK policy documents

Statistical interpretation belongs in analysis/,
not in this acquisition module.
"""

from pathlib import Path

from requests.exceptions import HTTPError

from ingestion import source_hub


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HESA_PROVIDER_URL = (
    "https://www.hesa.ac.uk/"
    "data-and-analysis/sb273/figure-7.csv"
)

HESA_PROVIDER_FALLBACK = (
    PROJECT_ROOT / "data" / "hesa_provider_enrolments.csv"
)


def acquire_official_dataset(
    source,
    url,
    fallback_path=None
):
    """
    Acquire a dataset using the existing approved
    source_hub downloader.

    A local fallback is permitted only if it can
    be validated by the appropriate dataset adapter.
    """

    if source not in {"hesa", "ofs"}:
        raise ValueError(
            "Dataset source must be 'hesa' or 'ofs'."
        )

    try:
        result = source_hub.download(url, source)

        return {
            "source": source,
            "path": Path(result["path"]),
            "source_url": url,
            "acquisition_method": (
                "cached_download"
                if result.get("cached")
                else "official_download"
            ),
            "sha256": result.get("sha256"),
        }

    except HTTPError as error:
        status = getattr(
            error.response,
            "status_code",
            None
        )

        if status != 403 or fallback_path is None:
            raise

        fallback = Path(fallback_path).resolve()

        if not fallback.is_file():
            raise FileNotFoundError(
                "Official download returned HTTP 403 "
                "and the fallback file does not exist."
            ) from error

        # Reject arbitrary local fallback locations.
        data_root = (PROJECT_ROOT / "data").resolve()

        if not fallback.is_relative_to(data_root):
            raise ValueError(
                "Fallback must be inside the "
                "project data directory."
            )

        import hashlib

        sha256 = hashlib.sha256(
            fallback.read_bytes()
        ).hexdigest()

        return {
            "source": source,
            "path": fallback,
            "source_url": url,
            "acquisition_method": "local_fallback",
            "download_status": 403,
            "sha256": sha256,
            "validation_required": True,
        }


def acquire_hesa_provider_dataset():
    """
    Compatibility function for the existing HESA
    Figure 7 source.

    This is a named dataset fixture, not the final
    question-driven HESA dataset selector.
    """

    return acquire_official_dataset(
        source="hesa",
        url=HESA_PROVIDER_URL,
        fallback_path=HESA_PROVIDER_FALLBACK,
    )


def acquire_govuk_policy(url):
    """
    Retrieve a GOV.UK document through the
    existing official Content API integration.
    """

    return source_hub.govuk_policy(url)


def discover_official_datasets(
    source,
    keyword=""
):
    """
    Search a registered publication page for
    downloadable datasets.
    """

    if source not in {"hesa", "ofs"}:
        raise ValueError(
            "Discovery currently supports "
            "HESA and OfS."
        )

    return source_hub.discover(
        source,
        keyword=keyword
    )
