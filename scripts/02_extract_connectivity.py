from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("data/ABIDE_I")
ROI_DIR = ROOT / "rois_cc200"
META_PATH = ROOT / "participants.csv"
OUT = ROOT / "connectivity"

OUT.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(META_PATH, dtype={"FILE_ID": str, "SITE_ID": str})
meta["FILE_ID"] = meta["FILE_ID"].str.strip()

features = []
records = []
failed = []

for row in tqdm(meta.itertuples(index=False), total=len(meta)):
    file_id = row.FILE_ID
    path = ROI_DIR / f"{file_id}_rois_cc200.1D"

    if not path.exists():
        failed.append((file_id, "missing_file"))
        continue

    try:
        ts = np.loadtxt(path, dtype=np.float64)

        if ts.ndim != 2 or ts.shape[1] != 200:
            raise ValueError(f"Unexpected shape: {ts.shape}")

        if not np.isfinite(ts).all():
            raise ValueError("Non-finite values")

        if np.any(np.std(ts, axis=0) == 0):
            raise ValueError("Constant ROI signal")

        corr = np.corrcoef(ts, rowvar=False)
        corr = np.clip(corr, -0.999999, 0.999999)
        z = np.arctanh(corr)

        upper = np.triu_indices(200, k=1)
        vector = z[upper].astype(np.float32)

        if not np.isfinite(vector).all():
            raise ValueError("Non-finite connectivity")

        features.append(vector)
        records.append({
            "FILE_ID": file_id,
            "DX_GROUP": row.DX_GROUP,
            "SITE_ID": row.SITE_ID,
            "N_TIMEPOINTS": ts.shape[0],
        })

    except Exception as exc:
        failed.append((file_id, str(exc)))

X = np.stack(features)
output_meta = pd.DataFrame(records)

np.save(OUT / "X_connectivity.npy", X)
output_meta.to_csv(OUT / "metadata.csv", index=False)
pd.DataFrame(failed, columns=["FILE_ID", "ERROR"]).to_csv(
    OUT / "extraction_failures.csv", index=False
)

print("Feature matrix:", X.shape)
print("Feature dtype:", X.dtype)
print("Metadata:", output_meta.shape)
print("Failures:", len(failed))
print("Saved to:", OUT.resolve())
print("\nDiagnosis counts:")
print(output_meta["DX_GROUP"].value_counts())
print("\nSite counts:")
print(output_meta["SITE_ID"].value_counts())
print("\nDiagnosis by site:")
print(pd.crosstab(output_meta["SITE_ID"], output_meta["DX_GROUP"]))
