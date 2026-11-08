
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Projects" / "NOVAProject"
DATA = ROOT / "data" / "ABIDE_I"
ROI_DIR = DATA / "rois_cc200"
CONNECTIVITY_DIR = DATA / "connectivity"
AUDIT_DIR = CONNECTIVITY_DIR / "audit"

PARTICIPANTS_FILE = DATA / "participants.csv"
METADATA_FILE = CONNECTIVITY_DIR / "metadata.csv"
FEATURES_FILE = CONNECTIVITY_DIR / "X_connectivity.npy"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def find_failure_file():
    candidates = [
        p for p in CONNECTIVITY_DIR.glob("*.csv")
        if "fail" in p.name.lower() or "error" in p.name.lower()
    ]
    return candidates[0] if candidates else None


def normalize_id(value):
    if pd.isna(value):
        return None
    return str(value).strip()


def load_csv(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing columns {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def roi_path(file_id):
    return ROI_DIR / f"{file_id}_rois_cc200.1D"


def inspect_roi(path):
    try:
        ts = np.loadtxt(path, dtype=np.float32)
        if ts.ndim != 2:
            return {"status": "invalid_dimensions", "n_timepoints": None,
                    "n_rois": None, "nonfinite": None, "constant_rois": None}
        if ts.shape[1] != 200:
            return {"status": "wrong_roi_count", "n_timepoints": ts.shape[0],
                    "n_rois": ts.shape[1], "nonfinite": None,
                    "constant_rois": None}

        nonfinite = int((~np.isfinite(ts)).sum())
        constant_rois = int(np.sum(np.nanstd(ts, axis=0) == 0))
        if nonfinite:
            status = "nonfinite_values"
        elif constant_rois:
            status = "constant_roi"
        elif ts.shape[0] < 2:
            status = "too_few_timepoints"
        else:
            status = "valid"

        return {
            "status": status,
            "n_timepoints": int(ts.shape[0]),
            "n_rois": int(ts.shape[1]),
            "nonfinite": nonfinite,
            "constant_rois": constant_rois,
        }
    except Exception as exc:
        return {
            "status": f"read_error:{type(exc).__name__}",
            "n_timepoints": None,
            "n_rois": None,
            "nonfinite": None,
            "constant_rois": None,
        }


def main():
    participants = load_csv(PARTICIPANTS_FILE, "Participants file")
    metadata = load_csv(METADATA_FILE, "Extracted metadata")
    features = np.load(FEATURES_FILE, mmap_mode="r")

    require_columns(participants, ["FILE_ID", "DX_GROUP", "SITE_ID"], "Participants")
    require_columns(metadata, ["FILE_ID", "DX_GROUP", "SITE_ID"], "Metadata")

    participants["FILE_ID"] = participants["FILE_ID"].map(normalize_id)
    metadata["FILE_ID"] = metadata["FILE_ID"].map(normalize_id)

    participants = participants.dropna(subset=["FILE_ID"]).copy()
    metadata = metadata.dropna(subset=["FILE_ID"]).copy()

    if participants["FILE_ID"].duplicated().any():
        print("WARNING: duplicate FILE_IDs in participants.csv")
    if metadata["FILE_ID"].duplicated().any():
        print("WARNING: duplicate FILE_IDs in extracted metadata")

    print("\n=== INPUT SUMMARY ===")
    print(f"Phenotypic rows: {len(participants)}")
    print(f"Metadata rows:   {len(metadata)}")
    print(f"Feature shape:   {features.shape}")
    print(f"Feature dtype:   {features.dtype}")

    if features.ndim != 2 or features.shape[1] != 19900:
        raise ValueError(f"Unexpected feature matrix shape: {features.shape}")
    if features.shape[0] != len(metadata):
        raise ValueError(
            "Feature row count does not match metadata row count. "
            "Verify extraction row ordering before continuing."
        )

    # Confirm that each extracted feature row maps to its metadata record.
    feature_quality = {
        "n_rows": features.shape[0],
        "n_features": features.shape[1],
        "nonfinite_values": int((~np.isfinite(features)).sum()),
        "rows_with_nonfinite": int(
            np.sum(~np.isfinite(features).all(axis=1))
        ),
        "zero_variance_features": int(
            np.sum(np.nanstd(features, axis=0) == 0)
        ),
    }
    pd.DataFrame([feature_quality]).to_csv(
        AUDIT_DIR / "feature_matrix_quality.csv", index=False
    )

    # Compare extracted participants with the phenotype cohort.
    phenotype_ids = set(participants["FILE_ID"])
    extracted_ids = set(metadata["FILE_ID"])

    included = participants[participants["FILE_ID"].isin(extracted_ids)].copy()
    excluded = participants[~participants["FILE_ID"].isin(extracted_ids)].copy()

    included["EXTRACTION_STATUS"] = "included"
    excluded["EXTRACTION_STATUS"] = "not_in_feature_matrix"

    inclusion = pd.concat([included, excluded], ignore_index=True)
    inclusion.to_csv(AUDIT_DIR / "participant_inclusion.csv", index=False)

    site_dx = (
        inclusion.groupby(["SITE_ID", "DX_GROUP", "EXTRACTION_STATUS"],
                          dropna=False)
        .size().rename("N").reset_index()
    )
    site_dx.to_csv(AUDIT_DIR / "site_diagnosis_inclusion.csv", index=False)

    # Audit all available ROI files, not only those in the final matrix.
    records = []
    ids_to_check = sorted(phenotype_ids)
    for i, file_id in enumerate(ids_to_check, start=1):
        path = roi_path(file_id)
        info = inspect_roi(path) if path.exists() else {
            "status": "roi_file_missing",
            "n_timepoints": None,
            "n_rois": None,
            "nonfinite": None,
            "constant_rois": None,
        }
        records.append({"FILE_ID": file_id, **info})
        if i % 100 == 0:
            print(f"Inspected ROI files: {i}/{len(ids_to_check)}")

    roi_audit = pd.DataFrame(records)
    roi_audit = participants.merge(roi_audit, on="FILE_ID", how="left")
    roi_audit["IN_FEATURE_MATRIX"] = roi_audit["FILE_ID"].isin(extracted_ids)
    roi_audit.to_csv(AUDIT_DIR / "roi_quality_by_participant.csv", index=False)

    roi_summary = (
        roi_audit.groupby(["SITE_ID", "DX_GROUP", "status"], dropna=False)
        .size().rename("N").reset_index()
    )
    roi_summary.to_csv(AUDIT_DIR / "roi_quality_summary.csv", index=False)

    scan_lengths = (
        roi_audit[roi_audit["n_timepoints"].notna()]
        .groupby("SITE_ID")["n_timepoints"]
        .agg(["count", "min", "median", "mean", "max"])
        .reset_index()
    )
    scan_lengths.to_csv(AUDIT_DIR / "scan_length_by_site.csv", index=False)

    # Compare included and excluded diagnosis/site composition.
    inclusion_summary = (
        inclusion.groupby(["EXTRACTION_STATUS", "SITE_ID", "DX_GROUP"],
                          dropna=False)
        .size().rename("N").reset_index()
    )
    inclusion_summary.to_csv(
        AUDIT_DIR / "inclusion_by_site_diagnosis.csv", index=False
    )

    # Load the failure log, if present, and preserve its original fields.
    failure_file = find_failure_file()
    if failure_file:
        failures = pd.read_csv(failure_file)
        failures.to_csv(AUDIT_DIR / "failure_log_copy.csv", index=False)
        print(f"\nFailure log: {failure_file.name}")
        print(f"Failure rows: {len(failures)}")
        print(f"Failure columns: {failures.columns.tolist()}")
    else:
        print("\nNo failure CSV detected in connectivity directory.")

    print("\n=== FEATURE MATRIX QUALITY ===")
    for key, value in feature_quality.items():
        print(f"{key}: {value}")

    print("\n=== PARTICIPANT RETENTION ===")
    print(f"Phenotypic participants: {len(phenotype_ids)}")
    print(f"Included:                {len(extracted_ids & phenotype_ids)}")
    print(f"Not included:            {len(phenotype_ids - extracted_ids)}")
    print(f"Metadata IDs absent from phenotype: {len(extracted_ids - phenotype_ids)}")

    print("\n=== ROI QUALITY COUNTS ===")
    print(roi_audit["status"].value_counts(dropna=False).to_string())

    print("\n=== INCLUSION BY SITE AND DIAGNOSIS ===")
    print(site_dx.to_string(index=False))

    print("\n=== SCAN LENGTHS BY SITE ===")
    print(scan_lengths.to_string(index=False))

    print(f"\nAudit files saved to: {AUDIT_DIR}")


if __name__ == "__main__":
    main()
