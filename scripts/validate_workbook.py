"""Read-only checks for Version 2 simulated HE institutional workbook.
Run from project root: python validate_workbook.py
Requires: pandas, openpyxl
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / 'data' / 'institutional_performance.xlsx'
YEARS = {'2021/22', '2022/23', '2023/24', '2024/25'}
SCHOOLS = {'ACI', 'ACE', 'HSB', 'CSC', 'RDSBL'}
SPECS = {
    'Student Retention': (20, ['Academic Year', 'School', 'Level', 'Retention Rate', 'Target'], ['Academic Year', 'School', 'Level']),
    'Student Satisfaction': (20, ['Academic Year', 'School', 'Category', 'Satisfaction Rate'], ['Academic Year', 'School', 'Category']),
    'Programme Performance': (40, ['Academic Year', 'School', 'Programme', 'Enrolments', 'Continuation Rate', 'Completion Rate'], ['Academic Year', 'School', 'Programme']),
    'Financial Performance': (4, ['Academic Year', 'Operating Income', 'Operating Expenditure', 'Operating Surplus'], ['Academic Year']),
}
issues = []

def fail(sheet, message):
    issues.append(f'{sheet}: {message}')
    print(f'  FAIL: {message}')

def check_numeric(df, sheet, columns, *, lower=0, upper=None):
    for col in columns:
        if col not in df:
            continue
        values = pd.to_numeric(df[col], errors='coerce')
        invalid = values.isna() | (values < lower)
        if upper is not None:
            invalid |= values > upper
        if invalid.any():
            fail(sheet, f'{col}: invalid value(s) in Excel row(s) {[int(i)+2 for i in df.index[invalid]]}')
        df[col] = values

if not WORKBOOK.is_file():
    print(f'Workbook not found: {WORKBOOK}\nPut this script in the project root, beside your data/ folder.')
    sys.exit(2)

try:
    sheets = pd.read_excel(WORKBOOK, sheet_name=None, engine='openpyxl')
except Exception as exc:
    print(f'Could not open workbook: {exc}')
    sys.exit(2)

print(f'Validating: {WORKBOOK}')
print('NOTE: This validation checks simulated data consistency, not authenticity of UEL statistics.\n')

for sheet, (expected_rows, required, keys) in SPECS.items():
    print(f'[{sheet}]')
    if sheet not in sheets:
        fail(sheet, 'worksheet missing (check spelling)')
        continue
    df = sheets[sheet].copy()
    df.columns = [str(c).strip() for c in df.columns]
    absent = [c for c in required if c not in df.columns]
    if absent:
        fail(sheet, f'missing columns: {absent}')
        continue
    if len(df) != expected_rows:
        fail(sheet, f'expected {expected_rows} rows; found {len(df)}')
    for col in keys:
        if df[col].isna().any():
            fail(sheet, f'blank values in key column: {col}')
    duplicates = df.duplicated(keys, keep=False)
    if duplicates.any():
        fail(sheet, f'duplicate key entries at Excel rows {[int(i)+2 for i in df.index[duplicates]]}')
    years = set(df['Academic Year'].astype(str).str.strip())
    if years != YEARS:
        fail(sheet, f'year coverage differs: missing={sorted(YEARS-years)}, unexpected={sorted(years-YEARS)}')
    if sheet != 'Financial Performance':
        schools = set(df['School'].astype(str).str.strip())
        if schools != SCHOOLS:
            fail(sheet, f'school codes differ: missing={sorted(SCHOOLS-schools)}, unexpected={sorted(schools-SCHOOLS)}')
    if sheet in ('Student Retention', 'Student Satisfaction'):
        counts = df.groupby(['Academic Year', 'School'], dropna=False).size()
        if len(counts) != 20 or not (counts == 1).all():
            fail(sheet, 'expected exactly one record per academic year and school')
    if sheet == 'Student Retention':
        check_numeric(df, sheet, ['Retention Rate', 'Target'], upper=100)
        if not df['Level'].astype(str).str.strip().eq('Undergraduate').all():
            fail(sheet, 'expected Undergraduate for all rows')
    elif sheet == 'Student Satisfaction':
        check_numeric(df, sheet, ['Satisfaction Rate'], upper=100)
        if not df['Category'].astype(str).str.strip().eq('Overall').all():
            fail(sheet, 'expected Overall category for all rows')
    elif sheet == 'Programme Performance':
        check_numeric(df, sheet, ['Enrolments'])
        check_numeric(df, sheet, ['Continuation Rate', 'Completion Rate'], upper=100)
        if df['Enrolments'].notna().all() and (df['Enrolments'] % 1 != 0).any():
            fail(sheet, 'enrolment counts should be integers')
        counts = df.groupby(['Academic Year', 'School'], dropna=False).size()
        if len(counts) != 20 or not (counts == 2).all():
            fail(sheet, 'expected two programmes per school for each academic year')
        programmes_per_year = df.groupby('Academic Year')['Programme'].nunique()
        if not (programmes_per_year == 10).all():
            fail(sheet, 'expected 10 unique programme names each academic year')
        if (df.groupby('Programme')['School'].nunique() > 1).any():
            fail(sheet, 'some programmes belong to multiple schools')
    else:
        cols = ['Operating Income','Operating Expenditure','Operating Surplus']
        optional = ['Operating Surplus Margin','Staff Costs','Cash Reserves']
        check_numeric(df, sheet, cols + [c for c in optional if c in df.columns])
        if not df[cols].isna().any().any():
            differences = (df['Operating Income'] - df['Operating Expenditure'] - df['Operating Surplus']).abs()
            if (differences > 0.01).any():
                fail(sheet, 'operating surplus must equal income minus expenditure')
            if 'Operating Surplus Margin' in df.columns:
                expected_margin = 100 * df['Operating Surplus'] / df['Operating Income'].replace(0, float('nan'))
                bad = (df['Operating Surplus Margin'] - expected_margin).abs() > 0.02
                if bad.any():
                    fail(sheet, f'surplus margin differs from calculated percent at Excel row(s) {[int(i)+2 for i in df.index[bad]]}')
            if 'Staff Costs' in df and (df['Staff Costs'] > df['Operating Expenditure']).any():
                fail(sheet, 'staff costs exceed operating expenditure')
    print(f'  Rows checked: {len(df)}\n')

print('=' * 60)
if issues:
    print(f'VALIDATION FAILED: {len(issues)} issue(s). Fix these before rebuilding ChromaDB.')
    sys.exit(1)
print('VALIDATION PASSED: all four sheets meet the Version 2 consistency checks.')
print('Reminder: figures remain synthetic; comparisons with the earlier corpus are not like-for-like.')
