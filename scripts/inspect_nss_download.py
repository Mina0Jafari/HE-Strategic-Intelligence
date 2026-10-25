
import argparse
import csv
import hashlib
import json
import re
import zipfile

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from openpyxl import load_workbook


# ----------------------------------------
# PROJECT PATHS
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CATALOGUE_DIR = (
    BASE_DIR / "evaluation" / "results"
    / "source_discovery"
)

DOWNLOAD_DIR = (
    BASE_DIR / "data" / "external" / "nss"
)

TRUSTED_HOST = (
    "blobofsproduks.blob.core.windows.net"
)


# ----------------------------------------
# FIND DISCOVERED NSS DATASETS
# ----------------------------------------

def find_nss_dataset():

    catalogues = sorted(
        CATALOGUE_DIR.glob("nss_*_catalogue.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not catalogues:
        raise FileNotFoundError(
            "No NSS discovery catalogue found."
        )

    for catalogue in catalogues:

        with catalogue.open(
            encoding="utf-8-sig",
            newline=""
        ) as file:
            records = list(csv.DictReader(file))

        matches = [
            row for row in records
            if "summary_taught_full-time.xlsx"
            in row.get("download_url", "").lower()
        ]

        if matches:

            def release_year(row):
                text = (
                    row.get("release_year_hint", "")
                    + " "
                    + row["download_url"]
                )

                years = re.findall(r"20\d{2}", text)

                return max(
                    (int(year) for year in years),
                    default=0
                )

            selected = max(
                matches,
                key=release_year
            )

            return catalogue, selected

    raise ValueError(
        "No full-time teaching-provider NSS workbook "
        "was found in the discovery catalogues."
    )


# ----------------------------------------
# INSPECT EXCEL WORKBOOK
# ----------------------------------------

def inspect_workbook(file_path):

    print("\nEXCEL WORKBOOK STRUCTURE")
    print("=" * 60)

    workbook = load_workbook(
        file_path,
        read_only=True,
        data_only=True
    )

    try:
        for sheet in workbook.worksheets:

            print(f"\nWorksheet: {sheet.title}")

            print(
                f"Reported rows: {sheet.max_row}"
            )

            print(
                f"Reported columns: {sheet.max_column}"
            )

            print("\nFirst five rows:")

            for row in sheet.iter_rows(
                min_row=1,
                max_row=5,
                values_only=True
            ):

                print([
                    str(value)[:80]
                    if value is not None else ""
                    for value in row[:12]
                ])

    finally:
        workbook.close()


# ----------------------------------------
# DOWNLOAD AND RECORD SOURCE
# ----------------------------------------

def download_dataset(record, max_mb=150):

    url = record["download_url"]

    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or parsed.hostname != TRUSTED_HOST
    ):
        raise ValueError(
            "Download URL is not on the approved "
            "OfS storage domain."
        )

    filename = Path(parsed.path).name

    if not filename.lower().endswith(".xlsx"):
        raise ValueError(
            "Expected an XLSX dataset."
        )

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = DOWNLOAD_DIR / filename

    temporary_file = destination.with_suffix(
        ".partial"
    )

    max_bytes = max_mb * 1024 * 1024

    file_hash = hashlib.sha256()

    total_bytes = 0

    print("\nDownloading dataset...")

    try:

        with requests.get(
            url,
            stream=True,
            timeout=(15, 120),
            allow_redirects=False
        ) as response:

            response.raise_for_status()

            if response.is_redirect:
                raise ValueError(
                    "Unexpected redirect."
                )

            expected_size = int(
                response.headers.get(
                    "Content-Length", "0"
                ) or 0
            )

            if expected_size > max_bytes:
                raise ValueError(
                    "Dataset exceeds the size limit."
                )

            with temporary_file.open("wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue

                    total_bytes += len(chunk)

                    if total_bytes > max_bytes:
                        raise ValueError(
                            "Download exceeds size limit."
                        )

                    file.write(chunk)

                    file_hash.update(chunk)

        if not zipfile.is_zipfile(temporary_file):
            raise ValueError(
                "Downloaded file is not a valid XLSX."
            )

        temporary_file.replace(destination)

    finally:

        if temporary_file.exists():
            temporary_file.unlink()

    metadata = {
        "publisher": "Office for Students",
        "dataset_title": record["dataset_title"],
        "source_page": record["source_page"],
        "download_url": url,
        "downloaded_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "filename": filename,
        "size_bytes": total_bytes,
        "sha256": file_hash.hexdigest(),
        "schema_status": "not_yet_validated"
    }

    metadata_path = DOWNLOAD_DIR / (
        destination.stem + "_manifest.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\nDownload completed successfully.")
    print("Saved to:", destination)
    print("Metadata:", metadata_path)

    return destination


# ----------------------------------------
# MAIN
# ----------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the discovered NSS workbook."
    )

    parser.add_argument(
        "--max-mb",
        type=int,
        default=150
    )

    args = parser.parse_args()

    catalogue, record = find_nss_dataset()

    print("\nNSS DATASET INSPECTION")
    print("=" * 60)

    print("Catalogue:", catalogue)
    print("Dataset:", record["dataset_title"])
    print("Download URL:", record["download_url"])

    url = record["download_url"]
    filename = Path(urlparse(url).path).name

    local_file = DOWNLOAD_DIR / filename

    if args.download:

        local_file = download_dataset(
            record,
            max_mb=args.max_mb
        )

    elif not local_file.exists():

        print(
            "\nDataset has not been downloaded."
        )

        print(
            "Run with --download to download it."
        )

        return

    inspect_workbook(local_file)


if __name__ == "__main__":
    main()
