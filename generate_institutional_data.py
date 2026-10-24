
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "institutional_performance.xlsx"

# All data below is simulated and does not
# represent actual UEL performance.

retention = pd.DataFrame([
    ["2023/24", "Business", "Undergraduate", 85, 90],
    ["2024/25", "Business", "Undergraduate", 82, 90],
    ["2023/24", "Computing", "Undergraduate", 88, 90],
    ["2024/25", "Computing", "Undergraduate", 84, 90],
    ["2023/24", "Education", "Undergraduate", 90, 90],
    ["2024/25", "Education", "Undergraduate", 87, 90],
    ["2023/24", "Health", "Undergraduate", 89, 90],
    ["2024/25", "Health", "Undergraduate", 86, 90],
], columns=[
    "Academic Year", "School", "Level",
    "Retention Rate", "Target"
])

satisfaction = pd.DataFrame([
    ["2023/24", "Business", "Overall", 78],
    ["2024/25", "Business", "Overall", 74],
    ["2023/24", "Computing", "Overall", 80],
    ["2024/25", "Computing", "Overall", 76],
    ["2023/24", "Education", "Overall", 83],
    ["2024/25", "Education", "Overall", 80],
    ["2023/24", "Health", "Overall", 82],
    ["2024/25", "Health", "Overall", 79],
], columns=[
    "Academic Year", "School",
    "Category", "Satisfaction Rate"
])

programmes = pd.DataFrame([
    ["2024/25", "Business", "Business Management", 1250, 82, 78],
    ["2024/25", "Computing", "Computer Science", 980, 84, 81],
    ["2024/25", "Education", "Primary Education", 720, 87, 85],
    ["2024/25", "Health", "Nursing", 1100, 86, 83],
], columns=[
    "Academic Year", "School", "Programme",
    "Enrolments", "Continuation Rate",
    "Completion Rate"
])

finance = pd.DataFrame([
    ["2023/24", 175000000, 171000000],
    ["2024/25", 185000000, 181000000],
], columns=[
    "Academic Year", "Operating Income",
    "Operating Expenditure"
])

finance["Operating Surplus"] = (
    finance["Operating Income"]
    - finance["Operating Expenditure"]
)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    retention.to_excel(
        writer, sheet_name="Student Retention", index=False
    )

    satisfaction.to_excel(
        writer, sheet_name="Student Satisfaction", index=False
    )

    programmes.to_excel(
        writer, sheet_name="Programme Performance", index=False
    )

    finance.to_excel(
        writer, sheet_name="Financial Performance", index=False
    )

print("SIMULATED INSTITUTIONAL DATASET CREATED")
print(f"File: {OUTPUT_FILE}")
print("Worksheets: 4")
