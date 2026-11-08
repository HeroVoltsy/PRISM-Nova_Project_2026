
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "Projects" / "NOVAProject"
DATA = ROOT / "data" / "ABIDE_I"
AUDIT = DATA / "connectivity" / "audit"

df = pd.read_csv(DATA / "participants.csv", dtype=str)
df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns.tolist())

print("\nFILE_ID frequency, top 15:")
print(df["FILE_ID"].value_counts(dropna=False).head(15).to_string())

placeholder = df[
    df["FILE_ID"].astype("string").str.strip().eq("no_filename")
].copy()

print("\nPlaceholder rows:", len(placeholder))
print("\nPotential identifying fields:")
candidate_cols = [
    c for c in df.columns
    if any(term in c.upper() for term in [
        "SUB", "ID", "FILE", "SCAN", "SITE", "DX", "DIAG"
    ])
]
print(placeholder[candidate_cols].to_string(index=False))

placeholder.to_csv(AUDIT / "placeholder_no_filename_rows.csv", index=False)
print("\nSaved full rows to:", AUDIT / "placeholder_no_filename_rows.csv")
