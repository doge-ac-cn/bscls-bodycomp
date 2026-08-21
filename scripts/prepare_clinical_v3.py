#!/usr/bin/env python3
"""Final clinical table: fix stage semantics and fill official fields.

- RG overall stage is unavailable; pathological T/N/M are kept separately.
- RG weight is recovered from the official VA labels file (152/211).
- Additional official fields are carried over (treatment, pack-years,
  pathology details, dates).

Output: data/clinical_master.csv
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "raw")
V2 = os.path.join(BASE, "data", "clinical_master.csv")
VA = os.path.join(DATA, "data/clinical/NSCLC-Radiogenomics-VA-R01-labels.csv")
L1 = os.path.join(DATA, "data/clinical/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
OUT = os.path.join(BASE, "data", "clinical_master.csv")

NA_TOKENS = {"not collected", "not assessed", "unknown", "none", "n/a", "not recorded in database"}


def na(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return np.nan if s.lower() in NA_TOKENS else s


def parse_date(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def lbs_to_kg(x):
    try:
        return round(float(x) * 0.4536, 1)
    except Exception:
        return np.nan


def main():
    m = pd.read_csv(V2)
    va = pd.read_csv(VA)
    l1 = pd.read_csv(L1)

    # Make the script idempotent: drop columns this step regenerates so a
    # re-run over its own output does not create *_x/*_y merge collisions.
    for c in ["weight_kg_va", "days_ct_to_surgery",
              "lung1_t_stage", "lung1_n_stage", "lung1_m_stage", "lung1_stage_detail",
              "patient_id_r01", "rg_pt", "rg_pn", "rg_pm", "rg_grade", "rg_lvi",
              "rg_pleural_invasion", "rg_adjuvant", "rg_chemotherapy", "rg_radiation",
              "rg_recurrence_location", "ethnicity", "pct_gg", "pack_years",
              "quit_smoking_year", "ct_date", "pet_date", "date_of_recurrence",
              "date_of_last_alive", "date_of_death"]:
        if c in m.columns:
            m = m.drop(columns=[c])
    # Also strip merge-suffix leftovers from previous runs (_va/_x/_y).
    m = m.loc[:, [c for c in m.columns
                  if not (c.endswith("_va") or c.endswith("_x") or c.endswith("_y"))]]

    # ---- RG field completion (VA-R01-labels as source) ----
    # VA Patient ID maps 1:1 to master patient_id (49 AMC-xxx + 162 R01-xxx)
    va["patient_id"] = va["Patient ID"].astype(str).str.strip()
    va["weight_kg"] = va["Weight (lbs)"].apply(lambda x: lbs_to_kg(na(x)))
    va["ct_date"] = va["CT Date"].apply(parse_date)
    va["pet_date"] = va["PET Date"].apply(parse_date)
    va["date_of_recurrence"] = va["Date of Recurrence"].apply(parse_date)
    va["date_of_last_alive"] = va["Date of Last Known Alive"].apply(parse_date)
    va["date_of_death"] = va["Date of Death"].apply(parse_date)
    va["pct_gg"] = va["%GG"].apply(na)
    va["pack_years"] = pd.to_numeric(va["Pack Years"].apply(na), errors="coerce")
    va["quit_smoking_year"] = pd.to_numeric(va["Quit Smoking Year"].apply(na), errors="coerce")
    va["rg_pt"] = va["Pathological T stage"].apply(na)
    va["rg_pn"] = va["Pathological N stage"].apply(na)
    va["rg_pm"] = va["Pathological M stage"].apply(na)
    va["rg_grade"] = va["Histopathological Grade"].apply(na)
    va["rg_lvi"] = va["Lymphovascular invasion"].apply(na)
    va["rg_pleural_invasion"] = va["Pleural invasion (elastic, visceral, or parietal)"].apply(na)
    va["rg_adjuvant"] = va["Adjuvant Treatment"].apply(na)
    va["rg_chemotherapy"] = va["Chemotherapy"].apply(na)
    va["rg_radiation"] = va["Radiation"].apply(na)
    va["rg_recurrence_location"] = va["Recurrence Location"].apply(na)
    va["ethnicity"] = va["Ethnicity"].apply(na)
    va["days_ct_to_surgery"] = pd.to_numeric(va["Days between CT and surgery"].apply(na), errors="coerce")

    rg_cols = ["patient_id", "weight_kg", "ct_date", "pet_date",
               "date_of_recurrence", "date_of_last_alive", "date_of_death", "pct_gg",
               "pack_years", "quit_smoking_year", "rg_pt", "rg_pn", "rg_pm", "rg_grade",
               "rg_lvi", "rg_pleural_invasion", "rg_adjuvant", "rg_chemotherapy",
               "rg_radiation", "rg_recurrence_location", "ethnicity", "days_ct_to_surgery"]
    va_sub = va[rg_cols].drop_duplicates(subset="patient_id", keep="first")

    # join on patient_id (verified 211/211)
    m = m.merge(va_sub, on="patient_id", how="left", suffixes=("", "_va"))
    # fill weight from VA (66 in v2 -> 152 from VA)
    m["weight_kg"] = m["weight_kg_va"]
    m = m.drop(columns=["weight_kg_va"])

    # R01 unified alias: AMC-xxx -> R01-xxx (used for RG time merge)
    m["patient_id_r01"] = np.where(
        m.cohort == "RG",
        "R01-" + m.patient_id.astype(str).str.replace("AMC-", "", regex=False),
        m.patient_id)

    # fix patient_id_r01 double-prefix bug (R01-R01-xxx -> R01-xxx)
    m.loc[m.cohort == "RG", "patient_id_r01"] = (
        "R01-" + m.loc[m.cohort == "RG", "patient_id_r01"]
        .str.replace("R01-", "", regex=False).str.replace("AMC-", "", regex=False))

    # 1. RG stage semantics: original value (pN) moved to rg_pn; stage set to NaN
    m.loc[m.cohort == "RG", "stage"] = np.nan

    # ---- Lung1 detailed staging ----
    l1_map = l1.rename(columns={
        "PatientID": "patient_id", "clinical.T.Stage": "lung1_t_stage",
        "Clinical.N.Stage": "lung1_n_stage", "Clinical.M.Stage": "lung1_m_stage",
        "Overall.Stage": "lung1_stage_detail"})
    l1_map = l1_map[["patient_id", "lung1_t_stage", "lung1_n_stage",
                     "lung1_m_stage", "lung1_stage_detail"]]
    m = m.merge(l1_map, on="patient_id", how="left")

    m.to_csv(OUT, index=False)
    print(f"Saved: {OUT} ({m.shape[0]} x {m.shape[1]})")

    # ---- verification ----
    print("\n=== Missing by cohort ===")
    for c in ["Lung1", "RG"]:
        sub = m[m.cohort == c]
        miss = sub.isna().sum()
        print(f"\n--- {c} n={len(sub)} ---")
        print(miss[miss > 0].to_string())
        print("complete:", list(miss[miss == 0].index))

    print("\n=== Key fixes verification ===")
    rg = m[m.cohort == "RG"]
    print(f"RG stage non-null (expected 0): {rg.stage.notna().sum()}")
    print(f"RG rg_pn non-null (expected ~162): {rg.rg_pn.notna().sum()} distribution: {rg.rg_pn.value_counts(dropna=False).to_dict()}")
    print(f"RG weight_kg non-null (expected ~152): {rg.weight_kg.notna().sum()}")
    print(f"RG pack_years non-null (expected ~155): {rg.pack_years.notna().sum()}")
    print(f"RG treatment chemo/radio/adjuvant non-null: {rg.rg_chemotherapy.notna().sum()}/{rg.rg_radiation.notna().sum()}/{rg.rg_adjuvant.notna().sum()}")
    print(f"RG ct_date non-null: {rg.ct_date.notna().sum()} | pet_date: {rg.pet_date.notna().sum()}")
    l1sub = m[m.cohort == "Lung1"]
    print(f"Lung1 detailed T/N/M non-null: {l1sub.lung1_t_stage.notna().sum()}/{l1sub.lung1_n_stage.notna().sum()}/{l1sub.lung1_m_stage.notna().sum()}")
    print(f"Lung1 stage_detail sample: {l1sub.lung1_stage_detail.value_counts(dropna=False).head(6).to_dict()}")
    # time fields unchanged
    print(f"OS events Lung1={int((m.os_event==1).sum())} (expected 305+63=368? actual={int((m[m.cohort=='Lung1'].os_event==1).sum())}+{int((m[m.cohort=='RG'].os_event==1).sum())})")
    print(f"RFS events RG={int((m[m.cohort=='RG'].rfs_event==1).sum())} (expected 54)")


if __name__ == "__main__":
    main()
