
from pathlib import Path
import pandas as pd


# -----------------------------------
# PROJECT PATHS
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent

EXCEL_FILE = BASE_DIR / "data" / "institutional_performance.xlsx"


# -----------------------------------
# LOAD EXCEL WORKBOOK
# -----------------------------------
def load_excel_workbook(file_path=EXCEL_FILE):
    """
    Load every worksheet from an Excel workbook.

    Returns:
        Dictionary of Pandas DataFrames.
        Each key is a worksheet name.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Excel workbook not found: {file_path}"
        )

    worksheets = pd.read_excel(
        file_path,
        sheet_name=None,
        engine="openpyxl"
    )

    return worksheets


# -----------------------------------
# VALIDATE WORKSHEETS
# -----------------------------------
def validate_worksheets(worksheets):
    """
    Check basic data quality for each worksheet.
    """

    validation_results = []

    for sheet_name, dataframe in worksheets.items():

        validation_results.append({
            "worksheet": sheet_name,
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "missing_values": int(
                dataframe.isna().sum().sum()
            ),
            "duplicate_rows": int(
                dataframe.duplicated().sum()
            )
        })

    return pd.DataFrame(validation_results)


# -----------------------------------
# PREPARE STRUCTURED RECORDS
# -----------------------------------
def prepare_records(worksheets):
    """
    Convert worksheet rows into searchable records
    while retaining source metadata.
    """

    records = []

    for sheet_name, dataframe in worksheets.items():

        for row_number, (_, row) in enumerate(
            dataframe.iterrows(),
            start=2
        ):

            values = row.to_dict()

            text = (
                f"Worksheet: {sheet_name}\n"
                + "\n".join(
                    f"{column}: {value}"
                    for column, value in values.items()
                )
            )

            records.append({
                "source": EXCEL_FILE.name,
                "worksheet": sheet_name,
                "row_number": row_number,
                "content": text,
                "values": values
            })

    return records


# -----------------------------------
# TEST DATA LOADING
# -----------------------------------
if __name__ == "__main__":

    worksheets = load_excel_workbook()

    print("\nEXCEL WORKBOOK LOADED")
    print("=" * 50)

    for sheet_name, dataframe in worksheets.items():
        print(
            f"{sheet_name}: "
            f"{len(dataframe)} rows, "
            f"{len(dataframe.columns)} columns"
        )

    validation = validate_worksheets(worksheets)

    print("\nDATA QUALITY SUMMARY")
    print(validation.to_string(index=False))

    records = prepare_records(worksheets)

    print(f"\nSEARCHABLE RECORDS CREATED: {len(records)}")

    print("\nEXAMPLE RECORD")
    print(records[0]["content"])
