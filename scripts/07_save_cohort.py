
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "Projects" / "NOVAProject"
DATA = ROOT / "data" / "ABIDE_I"
CONN = DATA / "connectivity"
AUDIT = CONN / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

participants = pd.read_csv(DATA / "participants.csv", dtype=str)
metadata = pd.read_csv(CONN / "metadata.csv", dtype=str)
roi = pd.read_csv(AUDIT / "roi_quality_by_participant.csv", dtype=str)

for df in (participants, metadata, roi):
    df.columns = df.columns.str.strip()
    df["FILE_ID"] = df["FILE_ID"].astype("string").str.strip()

# Remove malformed placeholder rows before any participant-level accounting.
participants = participants[
    participants["FILE_ID"].notna()
    & participants["FILE_ID"].ne("")
    & participants["FILE_ID"].str.lower().ne("no_filename")
].copy()

# All remaining phenotype IDs should identify one participant each.
dupes = participants[participants["FILE_ID"].duplicated(keep=False)]
if not dupes.empty:
    dupes.to_csv(AUDIT / "remaining_duplicate_ids.csv", index=False)
    raise ValueError("Duplicate real FILE_IDs remain; inspect remaining_duplicate_ids.csv")

# Keep one ROI-audit row per actual participant.
roi = roi[
    roi["FILE_ID"].notna()
    & roi["FILE_ID"].ne("")
    & roi["FILE_ID"].str.lower().ne("no_filename")
].drop_duplicates("FILE_ID")

metadata_ids = set(metadata["FILE_ID"].dropna())
roi_status = roi.set_index("FILE_ID")["status"].to_dict()

manifest = participants[["FILE_ID", "DX_GROUP", "SITE_ID"]].copy()
manifest["ROI_STATUS"] = manifest["FILE_ID"].map(roi_status)
manifest["IN_FEATURE_MATRIX"] = manifest["FILE_ID"].isin(metadata_ids)

manifest["COHORT_STATUS"] = "review_required"
manifest.loc[
    manifest["IN_FEATURE_MATRIX"] & manifest["ROI_STATUS"].eq("valid"),
    "COHORT_STATUS"
] = "included"

manifest.loc[
    manifest["ROI_STATUS"].eq("constant_roi"),
    "COHORT_STATUS"
] = "excluded_constant_roi"

manifest.loc[
    manifest["ROI_STATUS"].eq("roi_file_missing"),
    "COHORT_STATUS"
] = "excluded_missing_roi_file"

# IDs present in extracted metadata should map to valid ROI audit records.
unmatched_extracted = manifest[
    manifest["IN_FEATURE_MATRIX"]
    & ~manifest["ROI_STATUS"].eq("valid")
]
if not unmatched_extracted.empty:
    unmatched_extracted.to_csv(
        AUDIT / "extracted_roi_status_mismatches.csv", index=False
    )
    raise ValueError(
        "Some extracted IDs do not have valid ROI status; inspect mismatch file."
    )

# Check that all actual IDs have a recognized status.
unresolved = manifest[manifest["COHORT_STATUS"].eq("review_required")]
if not unresolved.empty:
    unresolved.to_csv(AUDIT / "unresolved_cohort_records.csv", index=False)
    raise ValueError(
        "Unresolved actual participant records remain; inspect unresolved file."
    )

manifest.to_csv(AUDIT / "cohort_manifest.csv", index=False)

print("=== FROZEN COHORT COUNTS ===")
print("Identifiable phenotype IDs:", len(manifest))
print(manifest["COHORT_STATUS"].value_counts().to_string())

print("\n=== INCLUDED PARTICIPANTS BY SITE AND DIAGNOSIS ===")
print(
    manifest[manifest["COHORT_STATUS"].eq("included")]
    .groupby(["SITE_ID", "DX_GROUP"])
    .size().rename("N").reset_index().to_string(index=False)
)

print("\nSaved:", AUDIT / "cohort_manifest.csv")
