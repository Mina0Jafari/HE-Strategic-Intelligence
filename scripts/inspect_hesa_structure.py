
"""
Inspect HESA CSV structures without assuming
that the first row contains column headings.
"""

import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"

FILES = [
    "hesa_enrolments.csv",
    "hesa_entrants.csv",
    "hesa_provider_enrolments.csv",
]


def inspect_file(path):

    print("\n" + "=" * 70)
    print("FILE:", path.name)
    print("=" * 70)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.reader(file)

        for number, row in enumerate(reader, start=1):

            if number > 35:
                break

            populated = [
                value.strip()
                for value in row
                if value.strip()
            ]

            print(
                f"Row {number:02d}:",
                row[:12]
            )

            if (
                len(populated) >= 3
                and any(
                    term in " ".join(populated).lower()
                    for term in (
                        "academic year",
                        "he provider",
                        "level of study",
                        "mode of study",
                        "number",
                        "enrolments",
                    )
                )
            ):
                print(
                    "  Possible metadata/header row"
                )


for filename in FILES:

    file_path = DATA_DIR / filename

    if file_path.exists():
        inspect_file(file_path)

    else:
        print(
            f"File not found: {filename}"
        )
