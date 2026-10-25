
from pathlib import Path
from openpyxl import load_workbook

DATA_DIR = Path(__file__).resolve().parent / "data" / "external" / "nss"

files = sorted(
    DATA_DIR.glob("*Summary_Taught_Full-time.xlsx"),
    key=lambda file: file.stat().st_mtime,
    reverse=True
)

if not files:
    raise FileNotFoundError(
        "Full-time NSS teaching-provider workbook not found."
    )

file_path = files[0]

print("WORKBOOK:", file_path.name)

workbook = load_workbook(
    file_path,
    read_only=True,
    data_only=True
)

try:
    sheet = workbook["All_subjects"]

    rows = sheet.iter_rows(values_only=True)

    for _ in range(3):
        next(rows)

    headers = [
        str(value).strip() if value is not None else ""
        for value in next(rows)
    ]

    print("\nALL COLUMN HEADERS")
    print("=" * 60)

    for index, name in enumerate(headers, start=1):
        print(f"{index}. {name}")

    provider_index = headers.index("Provider name")
    level_index = headers.index("Level of study")

    uel_records = []
    levels = set()

    for row in rows:
        provider = str(row[provider_index] or "")
        level = str(row[level_index] or "")

        if "university of east london" in provider.casefold():
            uel_records.append(row)
            levels.add(level)

    print("\nUEL RECORDS")
    print("=" * 60)
    print("Matching records:", len(uel_records))

    print("\nLevels of study:")
    for level in sorted(levels):
        print("-", level)

    print("\nFIRST FIVE UEL RECORDS")
    print("=" * 60)

    for row in uel_records[:5]:
        for column, value in zip(headers, row):
            print(f"{column}: {value}")
        print("-" * 60)

finally:
    workbook.close()
