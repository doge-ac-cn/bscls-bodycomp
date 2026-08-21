#!/usr/bin/env python3
"""Download official clinical spreadsheets for both TCIA cohorts.

download_tcia.py fetches DICOM imaging only; clinical labels are hosted
separately on the TCIA website and must be placed under:

    data/raw/data/clinical/

This script automates that step. It downloads the two official files and
derives the two auxiliary tables that the legacy build pipeline expects
(the columns of `master_table_nsclc_radiogenomics.csv` and
`sarg_patients.parquet` are a subset of the official VA-R01 labels file,
so they are re-created losslessly from it for reproducibility).

Official sources (TCIA, public, CC BY 3.0):
- Lung1:  NSCLC-Radiomics collection, clinical v3 (Oct-2019)
  https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv
- RG:     NSCLC-Radiogenomics collection, R01 clinical labels
  https://www.cancerimagingarchive.net/wp-content/uploads/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv

The RG file uses the column name "Case ID"; the build scripts expect
"Patient ID", so the column is renamed on download.

Usage:
    python scripts/download_clinical.py            # download if missing
    python scripts/download_clinical.py --force    # re-download everything
"""
import argparse
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIN = os.path.join(BASE, "data", "raw", "data", "clinical")
DATA = os.path.join(BASE, "data", "raw", "data")
os.makedirs(CLIN, exist_ok=True)

URLS = {
    "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv": (
        "https://www.cancerimagingarchive.net/wp-content/uploads/"
        "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
    ),
    "NSCLC-Radiogenomics-VA-R01-labels.csv": (
        "https://www.cancerimagingarchive.net/wp-content/uploads/"
        "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
    ),
}

# R01 labels -> legacy master_table / sarg_patients column mapping.
# These legacy files were produced by earlier internal tooling; their columns
# are a lossless subset of the official VA-R01 labels spreadsheet, so we
# regenerate them rather than require a manual download that no longer exists.
MASTER_COLS = {
    "Case ID": "PatientID",
    "Age at Histological Diagnosis": "AgeAtDiagnosis",
    "Gender": "Sex",
    "Smoking status": "SmokingStatus",
    "Histology ": "Histology",          # official header has trailing space
    "Survival Status": "SurvivalStatus",
    "Time to Death (days)": "TimeToDeathDays",
    "Recurrence": "Recurrence",
    "EGFR mutation status": "EGFR",
    "KRAS mutation status": "KRAS",
    "ALK translocation status": "ALK",
}
SARG_COLS = {
    "Case ID": "patient_id",
    "Pathological N stage": "pathological_n_stage",
    "Weight (lbs)": "weight_lbs",
    "Date of Death": "date_of_death",
    "Date of Last Known Alive": "date_of_last_known_alive",
    "Time to Death (days)": "time_to_death_days",
    "CT Date": "ct_date",
}


def download(url: str, dest: str, force: bool) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
        print(f"  [skip] {os.path.basename(dest)} exists")
        return False
    print(f"  [get ] {os.path.basename(dest)}")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    print(f"         -> {dest} ({len(data)/1024:.1f} KiB)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    print("== Downloading official clinical spreadsheets ==")
    changed = False
    for fname, url in URLS.items():
        dest = os.path.join(CLIN, fname)
        if download(url, dest, args.force):
            changed = True

    # The RG file uses "Case ID"; all downstream scripts expect "Patient ID".
    va_dest = os.path.join(CLIN, "NSCLC-Radiogenomics-VA-R01-labels.csv")
    if changed or args.force:
        import pandas as pd
        df = pd.read_csv(va_dest)
        if "Case ID" in df.columns and "Patient ID" not in df.columns:
            df = df.rename(columns={"Case ID": "Patient ID"})
            df.to_csv(va_dest, index=False)
            print(f"  [fix ] renamed 'Case ID' -> 'Patient ID' ({len(df)} rows)")

    # ---- derive legacy auxiliary tables from VA labels ----
    import pandas as pd
    va = pd.read_csv(va_dest)
    # After the rename above the canonical patient-id column is "Patient ID".
    va = va.rename(columns={"Patient ID": "Case ID"})

    mt_path = os.path.join(DATA, "master_table_nsclc_radiogenomics.csv")
    mt = va[list(MASTER_COLS)].rename(columns=MASTER_COLS)
    mt.to_csv(mt_path, index=False)
    print(f"  [gen ] master_table_nsclc_radiogenomics.csv ({len(mt)} x {mt.shape[1]})")

    sp_path = os.path.join(CLIN, "sarg_patients.parquet")
    sp = va[list(SARG_COLS)].rename(columns=SARG_COLS)
    sp.to_parquet(sp_path, index=False)
    print(f"  [gen ] sarg_patients.parquet ({len(sp)} x {sp.shape[1]})")

    print("\nDone. Files under data/raw/data/clinical/ ready for "
          "prepare_clinical_build.py -> prepare_clinical_time.py -> "
          "prepare_clinical_v3.py -> prepare_clinical_fix.py")


if __name__ == "__main__":
    sys.exit(main())
