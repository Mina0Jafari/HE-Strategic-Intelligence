
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def inspect_file(file_path):

    extension = file_path.suffix.lower()

    result = {
        "File": file_path.name,
        "Type": extension,
        "Size (MB)": round(
            file_path.stat().st_size / (1024 * 1024), 2
        ),
        "Details": "",
        "Status": "OK"
    }

    try:
        if extension == ".pdf":

            reader = PdfReader(str(file_path))

            result["Details"] = (
                f"{len(reader.pages)} pages"
            )

        elif extension == ".xlsx":

            workbook = pd.ExcelFile(file_path)

            result["Details"] = (
                f"{len(workbook.sheet_names)} sheets: "
                + ", ".join(workbook.sheet_names[:5])
            )

        elif extension == ".csv":

            sample = pd.read_csv(
                file_path,
                nrows=5
            )

            result["Details"] = (
                f"{len(sample.columns)} columns"
            )

        elif extension == ".txt":

            text = file_path.read_text(
                encoding="utf-8"
            )

            result["Details"] = (
                f"{len(text)} characters"
            )

    except Exception as error:

        result["Status"] = "ERROR"
        result["Details"] = str(error)[:120]

    return result


def build_inventory():

    records = []

    for file_path in sorted(DATA_DIR.iterdir()):

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in [".pdf", ".xlsx", ".csv", ".txt"]
        ):

            records.append(
                inspect_file(file_path)
            )

    return pd.DataFrame(records)


if __name__ == "__main__":

    inventory = build_inventory()

    print("\nPROJECT DATA SOURCE INVENTORY")
    print("=" * 65)

    if inventory.empty:
        print("No supported data files found.")
    else:
        print(inventory.to_string(index=False))

        inventory.to_csv(
            BASE_DIR / "source_inventory.csv",
            index=False
        )

        print("\nInventory saved: source_inventory.csv")
