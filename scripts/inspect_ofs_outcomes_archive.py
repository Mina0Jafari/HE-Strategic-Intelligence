
"""Inspect the official OfS student outcomes per-provider CSV archive."""

import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

SOURCE_PAGE = (
    "https://www.officeforstudents.org.uk/data-and-analysis/"
    "student-outcomes-data-dashboard/get-the-data/"
)
ARCHIVE_URL = (
    "https://blobofsproduks.blob.core.windows.net/files/"
    "StudentOutcomesDashboard/2025/UpdateFeb2026/"
    "student_outcomes_each_provider.zip"
)

BASE = Path(__file__).resolve().parent
CACHE = BASE / "data" / "external" / "source_hub" / "ofs"
ARCHIVE = CACHE / "student_outcomes_each_provider_Feb2026.zip"
MANIFEST = CACHE / "student_outcomes_each_provider_Feb2026_manifest.json"

MAX_DOWNLOAD_BYTES = 1_500_000_000
MAX_PREVIEW_MEMBER_BYTES = 15_000_000


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_if_missing():
    CACHE.mkdir(parents=True, exist_ok=True)

    if ARCHIVE.is_file() and zipfile.is_zipfile(ARCHIVE):
        print("Using cached ZIP:", ARCHIVE)
        return "cached"

    if ARCHIVE.exists():
        ARCHIVE.unlink()

    temp = ARCHIVE.with_suffix(".part")

    if temp.exists():
        temp.unlink()

    print("Downloading official OfS ZIP (one-time download)...")

    try:
        with requests.get(
            ARCHIVE_URL,
            stream=True,
            timeout=(20, 120)
        ) as response:

            response.raise_for_status()

            parsed = urlparse(response.url)

            if (
                parsed.scheme != "https"
                or parsed.hostname != "blobofsproduks.blob.core.windows.net"
            ):
                raise ValueError("Unexpected download redirect host")

            declared_size = response.headers.get("Content-Length")

            if (
                declared_size
                and int(declared_size) > MAX_DOWNLOAD_BYTES
            ):
                raise ValueError("Archive exceeds download limit")

            size = 0

            with temp.open("wb") as output:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        size += len(chunk)

                        if size > MAX_DOWNLOAD_BYTES:
                            raise ValueError(
                                "Archive exceeds download limit"
                            )

                        output.write(chunk)

        if not zipfile.is_zipfile(temp):
            raise ValueError("Downloaded file is not a valid ZIP")

        temp.replace(ARCHIVE)
        return "downloaded"

    finally:
        if temp.exists():
            temp.unlink()


def preview_csv_member(archive, info):
    print("\nCSV PREVIEW:", info.filename)

    if info.file_size > MAX_PREVIEW_MEMBER_BYTES:
        print("Preview skipped: member too large.")
        return

    with archive.open(info) as raw:
        prefix = raw.read(min(info.file_size, 120_000))

    text = prefix.decode(
        "utf-8-sig",
        errors="replace"
    )

    rows = list(csv.reader(io.StringIO(text)))

    for index, row in enumerate(rows[:12], start=1):
        print(f"  Row {index:02d}: {row[:18]}")


def main():
    method = download_if_missing()
    checksum = file_sha256(ARCHIVE)

    print("Source:", SOURCE_PAGE)
    print("Acquisition:", method)
    print(
        "Archive size (MB):",
        round(ARCHIVE.stat().st_size / 1_000_000, 2)
    )
    print("SHA-256:", checksum)

    with zipfile.ZipFile(ARCHIVE) as zip_file:

        members = [
            m for m in zip_file.infolist()
            if not m.is_dir()
        ]

        print("\nFiles inside ZIP:", len(members))
        print("First 15 filenames:")

        for info in members[:15]:
            print(
                " ",
                info.filename,
                f"({info.file_size:,} bytes)"
            )

        matches = [
            m for m in members
            if (
                "10007144" in m.filename
                or "east london" in m.filename.casefold()
                or "east_london" in m.filename.casefold()
            )
        ]

        print("\nUEL filename matches:", len(matches))

        for info in matches[:20]:
            print(
                " ",
                info.filename,
                f"({info.file_size:,} bytes)"
            )

        csv_members = [
            m for m in members
            if m.filename.casefold().endswith(".csv")
        ]

        selected = [
            m for m in matches
            if m.filename.casefold().endswith(".csv")
        ]

        if selected:
            preview_csv_member(zip_file, selected[0])

        elif csv_members:
            print(
                "\nNo UEL filename match; "
                "previewing first CSV to inspect schema."
            )
            preview_csv_member(zip_file, csv_members[0])

        else:
            print("\nNo CSV members found.")

    manifest = {
        "publisher": "Office for Students",
        "source_page": SOURCE_PAGE,
        "download_url": ARCHIVE_URL,
        "local_archive": str(ARCHIVE),
        "acquisition_method": method,
        "sha256": checksum,
        "retrieved_or_inspected_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "archive_members": len(members),
        "uel_filename_matches": [
            m.filename for m in matches
        ],
        "note": (
            "Schema inspection only; "
            "no outcome measure calculated."
        ),
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\nSaved provenance manifest:", MANIFEST)


if __name__ == "__main__":
    main()
