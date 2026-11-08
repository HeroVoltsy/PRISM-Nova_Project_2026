
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path("data/ABIDE_I")
ROI_DIR = ROOT / "rois_cc200"
MANIFEST = ROOT / "download_manifest.csv"
PHENO = ROOT / "participants.csv"

manifest = pd.read_csv(MANIFEST)
pheno = pd.read_csv(PHENO)

files = list(ROI_DIR.glob("*.1D"))
results = []
dimensions = Counter()

for path in files:
    try:
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        valid = (
            data.ndim == 2
            and data.shape[0] > 0
            and data.shape[1] == 200
            and np.isfinite(data).all()
        )

        dimensions[str(data.shape)] += 1
        results.append({
            "FILE_ID": path.name.replace("_rois_cc200.1D", ""),
            "shape": str(data.shape),
            "finite": bool(np.isfinite(data).all()),
            "valid": valid,
        })

    except Exception as exc:
        results.append({
            "FILE_ID": path.name.replace("_rois_cc200.1D", ""),
            "shape": None,
            "finite": False,
            "valid": False,
        })
        print(f"ERROR {path.name}: {exc}")

checked = pd.DataFrame(results)
checked.to_csv(ROOT / "roi_validation.csv", index=False)

valid_ids = set(checked.loc[checked["valid"], "FILE_ID"])
pheno["downloaded_valid"] = pheno["FILE_ID"].isin(valid_ids)

print("\nTotal files:", len(files))
print("Valid files:", int(checked["valid"].sum()))
print("Invalid files:", int((~checked["valid"]).sum()))
print("\nROI dimensions:")
for shape, count in dimensions.most_common():
    print(shape, count)

print("\nDiagnosis distribution among valid files:")
print(pheno[pheno["downloaded_valid"]]["DX_GROUP"].value_counts())

print("\nSite distribution among valid files:")
print(pheno[pheno["downloaded_valid"]]["SITE_ID"].value_counts())

print("\nManifest statuses:")
print(manifest["STATUS"].value_counts())

print("\nValidation saved to:", ROOT / "roi_validation.csv")
