#!/usr/bin/env python
"""Download TCIA imaging for both cohorts and store it in the canonical layout.

Canonical imaging layout (the one every downstream script expects):

    data/raw/NSCLC-Radiomics/<PatientID>/<SeriesInstanceUID>/*.dcm      (Lung1)
    data/raw/NSCLC-Radiogenomics/<PatientID>/<SeriesInstanceUID>/*.dcm  (RG)

`scan_series.py`, `select_ct_series.py` and `segmentation_pipeline.py`
all read this exact layout, so do not change it.

The TCIA API names its collections "NSCLC-Radiomics" and
"NSCLC Radiogenomics" (with a space).  We keep the API name for queries but
always write to the hyphenated directory names shown above.

Usage:
    python scripts/download_tcia.py            # both cohorts
    python scripts/download_tcia.py lung1      # NSCLC-Radiomics only (CT + SEG)
    python scripts/download_tcia.py rg         # NSCLC-Radiogenomics only (CT + PET)
"""
import glob
import json
import os
import shutil
import sys
import time

from tcia_utils import nbia

# which -> (TCIA API collection name, canonical output directory name)
COLLECTIONS = {
    "lung1": ("NSCLC-Radiomics", "NSCLC-Radiomics"),
    "rg": ("NSCLC Radiogenomics", "NSCLC-Radiogenomics"),
}
MODALITIES = {"lung1": ["CT", "SEG"], "rg": ["CT", "PET"]}

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")


def _series_rows(collection, modality):
    """Return list of dicts with PatientID / SeriesInstanceUID for one modality."""
    data = nbia.getSeries(collection=collection, modality=modality)
    if data is None:
        return []
    if hasattr(data, "to_dict"):  # pandas DataFrame
        rows = data.to_dict("records")
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    out = []
    for r in rows:
        pid = str(r.get("PatientID", "")).strip()
        suid = str(r.get("SeriesInstanceUID", "")).strip()
        if pid and suid:
            out.append({"patient_id": pid, "series_uid": suid})
    return out


def _series_downloaded(series_dir, patient_id, suid):
    """True if the series already exists in the canonical location with DICOMs."""
    d = os.path.join(series_dir, patient_id, suid)
    return os.path.isdir(d) and bool(glob.glob(os.path.join(d, "*.dcm")))


def _flatten_into(series_tmp, dest):
    """Move every DICOM under series_tmp (any nesting depth) into dest/."""
    os.makedirs(dest, exist_ok=True)
    dcms = glob.glob(os.path.join(series_tmp, "**", "*.dcm"), recursive=True)
    for d in dcms:
        shutil.move(d, os.path.join(dest, os.path.basename(d)))
    shutil.rmtree(series_tmp, ignore_errors=True)
    return len(dcms)


def download_cohort(which):
    api_name, dir_name = COLLECTIONS[which]
    series_dir = os.path.join(RAW, dir_name)
    os.makedirs(series_dir, exist_ok=True)

    log = {"collection": api_name, "modalities": {}, "elapsed_min": None}
    t0 = time.time()

    for mod in MODALITIES[which]:
        rows = _series_rows(api_name, mod)
        print(f"[{which}] {mod}: {len(rows)} series (API)", flush=True)

        # keep only series not yet present in the canonical layout
        missing = [
            r for r in rows
            if not _series_downloaded(series_dir, r["patient_id"], r["series_uid"])
        ]
        print(f"[{which}] {mod}: {len(missing)} to download, "
              f"{len(rows) - len(missing)} already present", flush=True)

        if missing:
            uids = [r["series_uid"] for r in missing]
            nbia.downloadSeries(
                uids, path=series_dir, input_type="list", max_workers=8
            )
            # tcia_utils unzips to <path>/<series_uid>/... ; reorganise into
            # the canonical <patient_id>/<series_uid>/ layout.
            for r in missing:
                suid = r["series_uid"]
                tmp = os.path.join(series_dir, suid)
                if not os.path.isdir(tmp):
                    print(f"  WARN {suid}: download produced no directory, skipped", flush=True)
                    continue
                dest = os.path.join(series_dir, r["patient_id"], suid)
                n = _flatten_into(tmp, dest)
                print(f"  moved {suid} -> {r['patient_id']}/{suid} ({n} dcm)", flush=True)

        log["modalities"][mod] = {
            "api_series": len(rows),
            "downloaded": len(missing),
        }

    log["elapsed_min"] = round((time.time() - t0) / 60, 1)
    log_path = os.path.join(RAW, f"download_log_{which}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[{which}] ALL DONE in {log['elapsed_min']:.1f} min -> "
          f"data/raw/{dir_name}/  (log: {log_path})", flush=True)


def main():
    args = sys.argv[1:] or ["lung1", "rg"]
    for which in args:
        if which not in COLLECTIONS:
            sys.exit(f"unknown cohort '{which}' (expected lung1 | rg)")
    for which in args:
        download_cohort(which)


if __name__ == "__main__":
    sys.exit(main())
