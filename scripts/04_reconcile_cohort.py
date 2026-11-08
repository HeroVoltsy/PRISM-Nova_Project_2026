
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "Projects" / "NOVAProject"
DATA = ROOT / "data" / "ABIDE_I"
CONN = DATA / "connectivity"
AUDIT = CONN / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

participants = pd.read_csv(DATA / "participants.csv", dtype={"FILE_ID": str})
metadata = pd.read_csv(CONN / "metadata.csv", dtype={"FILE_ID": str})
failures = pd.read_csv(CONN / "extraction_failures.csv", dtype={"FILE_ID": str})

for df in (participants, metadata, failures):
    df.columns = df.columns.str.strip()
    df["FILE_ID"] = df["FILE_ID"].astype("string").str.strip()

key_cols = ["FILE_ID", "DX_GROUP", "SITE_ID"]
missing = [c for c in key_cols if c not in participants.columns]
if missing:
    raise ValueError(f"Missing participant columns: {missing}")

# 1. Inspect duplicate phenotype records.
duplicate_rows = participants[
    participants.duplicated("FILE_ID", keep=False)
].sort_values("FILE_ID")

duplicate_summary = (
    duplicate_rows.groupby("FILE_ID", dropna=False)
    .agg(
        row_count=("FILE_ID", "size"),
        diagnosis_values=("DX_GROUP", lambda s: sorted(s.dropna().astype(str).unique())),
        site_values=("SITE_ID", lambda s: sorted(s.dropna().astype(str).unique())),
    )
    .reset_index()
)

duplicate_summary["metadata_conflict"] = (
    duplicate_summary["diagnosis_values"].map(len).gt(1)
    | duplicate_summary["site_values"].map(len).gt(1)
)

duplicate_rows.to_csv(AUDIT / "duplicate_phenotype_rows.csv", index=False)
duplicate_summary.to_csv(AUDIT / "duplicate_id_summary.csv", index=False)

# 2. Build a unique participant table, retaining only unambiguous IDs.
unique_counts = (
    participants.groupby("FILE_ID")[["DX_GROUP", "SITE_ID"]]
    .nunique(dropna=True)
)
conflicting_ids = set(
    unique_counts.index[
        (unique_counts["DX_GROUP"] > 1) | (unique_counts["SITE_ID"] > 1)
    ]
)

deduplicated = participants.drop_duplicates("FILE_ID", keep="first").copy()
deduplicated["METADATA_CONFLICT"] = deduplicated["FILE_ID"].isin(conflicting_ids)
deduplicated.to_csv(AUDIT / "participants_deduplicated_review.csv", index=False)

# 3. Reconcile participant IDs, feature metadata, and failures.
participant_ids = set(deduplicated["FILE_ID"].dropna())
extracted_ids = set(metadata["FILE_ID"].dropna())
failure_ids = set(failures["FILE_ID"].dropna())

print("=== ID RECONCILIATION ===")
print("Phenotype rows:", len(participants))
print("Unique phenotype IDs:", len(participant_ids))
print("Duplicate phenotype rows:", len(participants) - len(participant_ids))
print("IDs with conflicting diagnosis/site:", len(conflicting_ids))
print("Extracted metadata IDs:", len(extracted_ids))
print("Unique failure IDs:", len(failure_ids))
print("Failure log rows:", len(failures))

print("\n=== SET OVERLAPS ===")
print("Phenotype IDs absent from metadata:", len(participant_ids - extracted_ids))
print("Metadata IDs absent from phenotype:", len(extracted_ids - participant_ids))
print("Failure IDs absent from phenotype:", len(failure_ids - participant_ids))
print("Failure IDs also in metadata:", len(failure_ids & extracted_ids))
print("Phenotype IDs neither extracted nor logged failed:",
      len(participant_ids - extracted_ids - failure_ids))

# 4. Inspect failure reasons and duplicate failure entries.
print("\n=== FAILURE REASONS ===")
print(failures["ERROR"].value_counts(dropna=False).to_string())

print("\n=== DUPLICATE FAILURE IDs ===")
print(failures["FILE_ID"].value_counts().loc[lambda s: s > 1].to_string())

# 5. Join failure reasons to unique phenotype metadata.
phenotype_unique = deduplicated[
    ["FILE_ID", "DX_GROUP", "SITE_ID", "METADATA_CONFLICT"]
]
failure_unique = failures.drop_duplicates("FILE_ID", keep=False)

failure_audit = phenotype_unique.merge(
    failures, on="FILE_ID", how="inner", validate="one_to_many"
)
failure_audit.to_csv(AUDIT / "failure_reasons_with_phenotype.csv", index=False)

print("\n=== FAILURES BY SITE AND DIAGNOSIS ===")
print(
    failure_audit.groupby(["SITE_ID", "DX_GROUP", "ERROR"], dropna=False)
    .size().rename("N").reset_index().to_string(index=False)
)

# 6. Save a unique-ID cohort reconciliation table.
status = pd.DataFrame({"FILE_ID": sorted(participant_ids)})
status["IN_METADATA"] = status["FILE_ID"].isin(extracted_ids)
status["IN_FAILURE_LOG"] = status["FILE_ID"].isin(failure_ids)
status = status.merge(
    phenotype_unique, on="FILE_ID", how="left", validate="one_to_one"
)
status.to_csv(AUDIT / "unique_id_reconciliation.csv", index=False)

print("\nSaved reconciliation outputs to:", AUDIT)
