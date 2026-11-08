
import csv
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

ROOT = Path("data/ABIDE_I")
PHENO = ROOT / "phenotypic/Phenotypic_V1_0b_preprocessed1.csv"
OUT = ROOT / "rois_cc200"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/"
    "ABIDE_Initiative/Outputs/cpac/filt_noglobal/"
    "rois_cc200"
)

with PHENO.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

required = {"FILE_ID", "DX_GROUP", "SITE_ID"}
if not rows or not required.issubset(rows[0]):
    raise ValueError(f"Unexpected CSV columns: {list(rows[0]) if rows else 'empty'}")

participants = []
for row in rows:
    file_id = (row.get("FILE_ID") or "").strip()
    diagnosis = (row.get("DX_GROUP") or "").strip()
    site = (row.get("SITE_ID") or "").strip()

    if file_id and diagnosis in {"1", "2"}:
        participants.append({
            "FILE_ID": file_id,
            "DX_GROUP": diagnosis,
            "SITE_ID": site,
        })

def download(p):
    name = p["FILE_ID"]
    dest = OUT / f"{name}_rois_cc200.1D"
    url = f"{BASE}/{name}_rois_cc200.1D"

    if dest.exists() and dest.stat().st_size > 0:
        return name, "exists"

    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(15, 120))
            if response.status_code == 404:
                return name, "not_found"
            response.raise_for_status()

            temp = dest.with_suffix(dest.suffix + ".part")
            with temp.open("wb") as f:
                f.write(response.content)

            if temp.stat().st_size == 0:
                temp.unlink(missing_ok=True)
                raise IOError("Empty download")

            temp.replace(dest)
            return name, "downloaded"

        except Exception as exc:
            if attempt == 3:
                return name, f"failed: {exc}"
            time.sleep(2 ** attempt)

results = []
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(download, p) for p in participants]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())

with (ROOT / "download_manifest.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["FILE_ID", "STATUS"])
    writer.writerows(results)

with (ROOT / "participants.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=["FILE_ID", "DX_GROUP", "SITE_ID"])
    writer.writeheader()
    writer.writerows(participants)

from collections import Counter
print("Participants in metadata:", len(participants))
print("Download outcomes:", dict(Counter(status for _, status in results)))
print("Data directory:", OUT.resolve())
