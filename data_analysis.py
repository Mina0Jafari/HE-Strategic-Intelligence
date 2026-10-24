
import pandas as pd
from data_loader import load_excel_workbook


def analyse_retention(start_year=None, end_year=None):
    """
    Analyse undergraduate retention rates across schools
    and academic years using structured Excel data.

    Parameters
    ----------
    start_year : str, optional
        First academic year to include, e.g. "2022/23".
        Defaults to the earliest available year.

    end_year : str, optional
        Last academic year to include, e.g. "2024/25".
        Defaults to the latest available year.

    Returns
    -------
    pandas.DataFrame
        School-level retention rates for all selected years,
        plus the percentage-point change between the first
        and last selected years.

    Notes
    -----
    Academic years are discovered dynamically.
    No academic years or school names are hardcoded.
    """

    # -----------------------------------
    # 1. LOAD INSTITUTIONAL DATA
    # -----------------------------------
    worksheets = load_excel_workbook()

    if "Student Retention" not in worksheets:
        raise ValueError(
            "Student Retention worksheet not found."
        )

    df = worksheets["Student Retention"].copy()

    # -----------------------------------
    # 2. VALIDATE REQUIRED COLUMNS
    # -----------------------------------
    required_columns = [
        "Academic Year",
        "School",
        "Level",
        "Retention Rate",
        "Target"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # -----------------------------------
    # 3. FILTER UNDERGRADUATE RECORDS
    # -----------------------------------
    df = df[
        df["Level"].astype(str).str.strip().str.lower()
        == "undergraduate"
    ].copy()

    if df.empty:
        raise ValueError(
            "No undergraduate retention records found."
        )

    # -----------------------------------
    # 4. VALIDATE DATA
    # -----------------------------------
    df["School"] = df["School"].astype("string").str.strip()
    df["Academic Year"] = (
        df["Academic Year"].astype("string").str.strip()
    )

    if (
        df["School"].isna().any()
        or df["School"].eq("").any()
        or df["Academic Year"].isna().any()
        or df["Academic Year"].eq("").any()
    ):
        raise ValueError(
            "School or Academic Year contains missing values."
        )

    df["Retention Rate"] = pd.to_numeric(
        df["Retention Rate"],
        errors="raise"
    )

    df["Target"] = pd.to_numeric(
        df["Target"],
        errors="raise"
    )

    if (
        df["Retention Rate"].isna().any()
        or df["Target"].isna().any()
    ):
        raise ValueError(
            "Retention Rate or Target contains missing values."
        )

    if not df["Retention Rate"].between(0, 100).all():
        raise ValueError(
            "Retention rates must be between 0 and 100."
        )

    if not df["Target"].between(0, 100).all():
        raise ValueError(
            "Targets must be between 0 and 100."
        )

    # Prevent ambiguous school/year comparisons
    if df.duplicated(
        subset=["School", "Academic Year"]
    ).any():
        raise ValueError(
            "Duplicate school/year retention records found."
        )

    # -----------------------------------
    # 5. DISCOVER ACADEMIC YEARS
    # -----------------------------------
    def academic_year_start(year):
        """
        Extract the starting calendar year from
        an academic year such as '2024/25'.
        """
        try:
            return int(str(year).split("/")[0])
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid academic year format: {year}"
            )

    available_years = sorted(
        df["Academic Year"].unique().tolist(),
        key=academic_year_start
    )

    if not available_years:
        raise ValueError(
            "No academic years available."
        )

    # -----------------------------------
    # 6. SELECT REQUESTED PERIOD
    # -----------------------------------
    first_year = (
        str(start_year).strip()
        if start_year is not None
        else available_years[0]
    )

    last_year = (
        str(end_year).strip()
        if end_year is not None
        else available_years[-1]
    )

    if first_year not in available_years:
        raise ValueError(
            f"Start year '{first_year}' is unavailable. "
            f"Available years: {available_years}"
        )

    if last_year not in available_years:
        raise ValueError(
            f"End year '{last_year}' is unavailable. "
            f"Available years: {available_years}"
        )

    first_position = available_years.index(first_year)
    last_position = available_years.index(last_year)

    if first_position > last_position:
        raise ValueError(
            "Start year must not be later than end year."
        )

    selected_years = available_years[
        first_position:last_position + 1
    ]

    # -----------------------------------
    # 7. BUILD RETENTION COMPARISON
    # -----------------------------------
    comparison = df.pivot(
        index="School",
        columns="Academic Year",
        values="Retention Rate"
    )

    comparison = comparison[selected_years].copy()

    if comparison.isna().any().any():
        raise ValueError(
            "One or more schools are missing retention "
            "data for a selected academic year."
        )

    # -----------------------------------
    # 8. CALCULATE RETENTION CHANGES
    # -----------------------------------
    if len(selected_years) >= 2:

        comparison["Change (pp)"] = (
            comparison[last_year]
            - comparison[first_year]
        )

        comparison = comparison.sort_values(
            "Change (pp)",
            ascending=True
        )

    else:
        # A single-year request is a snapshot,
        # so no year-on-year change is calculated.
        comparison["Change (pp)"] = float("nan")

        comparison = comparison.sort_values(
            selected_years[0],
            ascending=True
        )

    # -----------------------------------
    # 9. RETURN RESULTS
    # -----------------------------------
    comparison = comparison.reset_index()

    comparison.columns.name = None

    return comparison


# -----------------------------------
# TEST THE ANALYSIS
# -----------------------------------
if __name__ == "__main__":

    print("\nUNDERGRADUATE RETENTION ANALYSIS")
    print("=" * 60)

    # Automatically analyse all available years
    results = analyse_retention()

    print("\nALL AVAILABLE ACADEMIC YEARS")
    print(results.to_string(index=False))

    if results["Change (pp)"].notna().any():

        largest_decline = results.iloc[0]

        print("\nLARGEST RETENTION DECLINE")

        print(
            f"{largest_decline['School']}: "
            f"{largest_decline['Change (pp)']:+.1f} "
            "percentage points"
        )

    # Demonstrate a selected period without hardcoding
    # any academic year in the analysis function.
    print("\nLATEST AVAILABLE ACADEMIC YEAR")

    latest_year = sorted(
        results.columns[
            results.columns.astype(str).str.match(
                r"^\d{4}/\d{2,4}$"
            )
        ].tolist(),
        key=lambda year: int(year.split("/")[0])
    )[-1]

    latest_results = analyse_retention(
        start_year=latest_year,
        end_year=latest_year
    )

    print(latest_results.to_string(index=False))
