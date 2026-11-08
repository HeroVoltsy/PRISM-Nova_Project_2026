
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "Projects" / "NOVAProject"
DATA = ROOT / "data" / "ABIDE_I"
AUDIT = DATA / "connectivity" / "audit"

participants = pd.read_csv(DATA / "participants.csv", dtype={"FILE_ID": str})
roi = pd.read_csv(
    AUDIT / "roi_quality_by_participant.csv",
    dtype={"FILE_ID": str}
)

participants.columns = participants.columns.str.strip()
roi.columns = roi.columns.str.strip()
participants["FILE_ID"] = participants["FILE_ID"].astype("string").str.strip()
roi["FILE_ID"] = roi["FILE_ID"].astype("string").str.strip()

# Detect duplicate IDs with conflicting diagnosis or site labels.
key_cols = ["FILE_ID", "DX_GROUP", "SITE_ID"]
conflict_counts = participants.groupby("FILE_ID")[["DX_GROUP", "SITE_ID"]].nunique()
conflict_ids = conflict_counts.index[
    (conflict_counts["DX_GROUP"] > 1) | (conflict_counts["SITE_ID"] > 1)
].tolist()

print("=== CONFLICTING PHENOTYPE RECORDS ===")
print(
    participants[participants["FILE_ID"].isin(conflict_ids)]
    [key_cols].sort_values("FILE_ID").to_string(index=False)
)

# Use the participant-level ROI audit, not the placeholder failure log.
failure_manifest = roi[
    roi["status"].isin(["roi_file_missing", "constant_roi"])
].copy()

failure_manifest["FAILURE_CATEGORY"] = failure_manifest["status"].map({
    "roi_file_missing": "missing_file",
    "constant_roi": "constant_roi_signal",
})

failure_manifest.to_csv(AUDIT / "corrected_failure_manifest.csv", index=False)

print("\n=== PARTICIPANT-LEVEL FAILURE COUNTS ===")
print(failure_manifest["FAILURE_CATEGORY"].value_counts().to_string())

print("\n=== FAILURES BY SITE AND DIAGNOSIS ===")
print(
    failure_manifest.groupby(
        ["SITE_ID", "DX_GROUP", "FAILURE_CATEGORY"], dropna=False
    ).size().rename("N").reset_index().to_string(index=False)
)

print("\nCorrected manifest saved to:", AUDIT / "corrected_failure_manifest.csv")
