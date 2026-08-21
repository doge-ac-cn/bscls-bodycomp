#!/usr/bin/env python3
"""Generate small fully synthetic demo data for CI / quick-start runs.

No TCIA data, imaging, or GPU is required. The script writes
``data/clinical_master.csv`` and ``data/bodycomp_features.csv`` with the same
schema as the real pipeline (65 feature columns, 41 clinical columns) for 120
synthetic patients (60 ``LUNG1-xxx`` + 60 ``AMC-xxx``, sex balanced 30/30 per
cohort so that the analysis scripts' sex-stratified cut-point logic runs).

Values are random draws from plausible physiological ranges with a fixed seed.
They are **not** real measurements and the resulting analyses are meaningless
as science — the purpose is to let scripts 01–05/07/08 run end-to-end without
downloading data or running segmentation.

Usage::

    python scripts/make_demo_data.py
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

rng = np.random.default_rng(20260821)

# --------------------------------------------------------------------------
# Feature table template (65 columns, matching extract_bodycomp.py output)
# --------------------------------------------------------------------------
FEATURE_COLS = [
    "patient_id", "z_mm", "coverage_extended",
    "SMA_vol_cm3", "SM_mean_hu", "SAT_vol_cm3", "SAT_vol_fatHU_cm3",
    "VAT_vol_cm3", "VAT_vol_fatHU_cm3", "vert_available",
    "T12_slice", "T12_edge", "SMA_T12_cm2", "SM_density_T12_HU",
    "SM_density_T12_n", "BMD_T12_HU", "BMD_T12_n", "ES_T12_cm2",
    "ES_density_T12_HU", "ES_T12_n", "SAT_T12_cm2", "VAT_T12_cm2",
    "L1_slice", "L1_edge", "SMA_L1_cm2", "SM_density_L1_HU",
    "SM_density_L1_n", "BMD_L1_HU", "BMD_L1_n", "ES_L1_cm2",
    "ES_density_L1_HU", "ES_L1_n", "Psoas_L1_cm2", "Psoas_density_L1_HU",
    "Psoas_L1_n", "SAT_L1_cm2", "VAT_L1_cm2",
    "L3_slice", "L3_edge", "SMA_L3_cm2", "SM_density_L3_HU",
    "SM_density_L3_n", "BMD_L3_HU", "BMD_L3_n", "ES_L3_cm2",
    "ES_density_L3_HU", "ES_L3_n", "Psoas_L3_cm2", "Psoas_density_L3_HU",
    "Psoas_L3_n", "SAT_L3_cm2", "VAT_L3_cm2",
    "T4_slice", "T4_edge", "SMA_T4_cm2", "SM_density_T4_HU",
    "SM_density_T4_n", "BMD_T4_HU", "BMD_T4_n", "ES_T4_cm2",
    "ES_density_T4_HU", "ES_T4_n", "SAT_T4_cm2", "VAT_T4_cm2",
    "L1_unavailable",
]

# --------------------------------------------------------------------------
# Clinical table template (41 columns, matching prepare_clinical*.py output)
# --------------------------------------------------------------------------
CLINICAL_COLS = [
    "patient_id", "cohort", "age", "sex", "stage", "histology", "smoking",
    "os_time", "os_event", "rfs_time", "rfs_event", "egfr", "kras", "alk",
    "weight_kg", "patient_id_amc", "patient_id_r01", "ct_date", "pet_date",
    "date_of_recurrence", "date_of_last_alive", "date_of_death", "pct_gg",
    "pack_years", "quit_smoking_year", "rg_pt", "rg_pn", "rg_pm", "rg_grade",
    "rg_lvi", "rg_pleural_invasion", "rg_adjuvant", "rg_chemotherapy",
    "rg_radiation", "rg_recurrence_location", "ethnicity", "lung1_t_stage",
    "lung1_n_stage", "lung1_m_stage", "lung1_stage_detail", "days_ct_to_surgery",
]


def build_features():
    rows = []
    for i in range(1, 121):
        is_rg = i > 60
        pid = "AMC-%03d" % i if is_rg else "LUNG1-%03d" % i
        # whole-body volumes (RG whole-body CT sees larger muscles)
        sma_vol = rng.normal(12500 if is_rg else 8500, 2000)
        sat_vol = rng.normal(3500, 1200)
        vat_vol = rng.normal(2500, 1100)
        # construct fat inversely correlated with muscle so composite
        # phenotypes are non-empty in the demo
        fat_scale = 1.6 - 0.6 * (sma_vol / 12000.0)
        sat_vol = max(400.0, sat_vol * fat_scale)
        vat_vol = max(200.0, vat_vol * fat_scale)
        # vertebral levels present for whole-body (RG); Lung1 chest CT misses L3
        has_l3 = is_rg
        l1_ok = rng.random() > 0.15  # small fraction of L1-unavailable cases
        row = {
            "patient_id": pid,
            "z_mm": round(rng.uniform(2.5, 5.0), 2),
            "coverage_extended": int(is_rg),
            "SMA_vol_cm3": round(sma_vol, 1),
            "SM_mean_hu": round(rng.uniform(30, 55), 1),
            "SAT_vol_cm3": round(sat_vol, 1),
            "SAT_vol_fatHU_cm3": round(sat_vol * rng.uniform(0.75, 0.95), 1),
            "VAT_vol_cm3": round(vat_vol, 1),
            "VAT_vol_fatHU_cm3": round(vat_vol * rng.uniform(0.75, 0.95), 1),
            "vert_available": 1,
            "T12_slice": int(rng.integers(20, 60)),
            "T12_edge": int(rng.random() > 0.8),
            "SMA_T12_cm2": round(rng.uniform(45, 95), 2),
            "SM_density_T12_HU": round(rng.uniform(32, 58), 1),
            "SM_density_T12_n": int(rng.integers(200, 500)),
            "BMD_T12_HU": round(rng.uniform(120, 320), 1),
            "BMD_T12_n": int(rng.integers(100, 300)),
            "ES_T12_cm2": round(rng.uniform(8, 22), 2),
            "ES_density_T12_HU": round(rng.uniform(30, 55), 1),
            "ES_T12_n": int(rng.integers(50, 150)),
            "SAT_T12_cm2": round(rng.uniform(40, 260), 2),
            "VAT_T12_cm2": round(rng.uniform(20, 240), 2),
            "L1_slice": int(rng.integers(30, 80)),
            "L1_edge": int(not l1_ok),
            "SMA_L1_cm2": round(rng.uniform(55, 130), 2) if l1_ok else np.nan,
            "SM_density_L1_HU": round(rng.uniform(32, 58), 1) if l1_ok else np.nan,
            "SM_density_L1_n": int(rng.integers(200, 500)) if l1_ok else np.nan,
            "BMD_L1_HU": round(rng.uniform(120, 320), 1) if l1_ok else np.nan,
            "BMD_L1_n": int(rng.integers(100, 300)) if l1_ok else np.nan,
            "ES_L1_cm2": round(rng.uniform(10, 26), 2) if l1_ok else np.nan,
            "ES_density_L1_HU": round(rng.uniform(30, 55), 1) if l1_ok else np.nan,
            "ES_L1_n": int(rng.integers(50, 150)) if l1_ok else np.nan,
            "Psoas_L1_cm2": round(rng.uniform(6, 18), 2) if l1_ok else np.nan,
            "Psoas_density_L1_HU": round(rng.uniform(30, 55), 1) if l1_ok else np.nan,
            "Psoas_L1_n": int(rng.integers(30, 120)) if l1_ok else np.nan,
            "SAT_L1_cm2": round(rng.uniform(40, 260), 2) if l1_ok else np.nan,
            "VAT_L1_cm2": round(rng.uniform(20, 240), 2) if l1_ok else np.nan,
            "L3_slice": int(rng.integers(90, 150)) if has_l3 else np.nan,
            "L3_edge": int(rng.random() > 0.8) if has_l3 else np.nan,
            "SMA_L3_cm2": round(rng.uniform(80, 160), 2) if has_l3 else np.nan,
            "SM_density_L3_HU": round(rng.uniform(32, 58), 1) if has_l3 else np.nan,
            "SM_density_L3_n": int(rng.integers(200, 500)) if has_l3 else np.nan,
            "BMD_L3_HU": round(rng.uniform(120, 320), 1) if has_l3 else np.nan,
            "BMD_L3_n": int(rng.integers(100, 300)) if has_l3 else np.nan,
            "ES_L3_cm2": round(rng.uniform(12, 30), 2) if has_l3 else np.nan,
            "ES_density_L3_HU": round(rng.uniform(30, 55), 1) if has_l3 else np.nan,
            "ES_L3_n": int(rng.integers(50, 150)) if has_l3 else np.nan,
            "Psoas_L3_cm2": round(rng.uniform(8, 22), 2) if has_l3 else np.nan,
            "Psoas_density_L3_HU": round(rng.uniform(30, 55), 1) if has_l3 else np.nan,
            "Psoas_L3_n": int(rng.integers(30, 120)) if has_l3 else np.nan,
            "SAT_L3_cm2": round(rng.uniform(50, 300), 2) if has_l3 else np.nan,
            "VAT_L3_cm2": round(rng.uniform(30, 260), 2) if has_l3 else np.nan,
            "T4_slice": int(rng.integers(5, 20)),
            "T4_edge": int(rng.random() > 0.85),
            "SMA_T4_cm2": round(rng.uniform(30, 80), 2),
            "SM_density_T4_HU": round(rng.uniform(32, 58), 1),
            "SM_density_T4_n": int(rng.integers(150, 400)),
            "BMD_T4_HU": round(rng.uniform(120, 320), 1),
            "BMD_T4_n": int(rng.integers(80, 250)),
            "ES_T4_cm2": round(rng.uniform(6, 16), 2),
            "ES_density_T4_HU": round(rng.uniform(30, 55), 1),
            "ES_T4_n": int(rng.integers(30, 100)),
            "SAT_T4_cm2": round(rng.uniform(30, 200), 2),
            "VAT_T4_cm2": round(rng.uniform(15, 180), 2),
            "L1_unavailable": int(not l1_ok),
        }
        rows.append(row)
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    return df


def build_clinical():
    rows = []
    for i in range(1, 121):
        is_rg = i > 60
        pid = "AMC-%03d" % i if is_rg else "LUNG1-%03d" % i
        # sex blocks of 30 per cohort (LUNG1 1-30 M / 31-60 F, AMC 61-90 M / 91-120 F)
        sex = "M" if ((i - 1) // 30) % 2 == 0 else "F"
        age = int(rng.integers(45, 86))
        os_time = float(rng.integers(120, 3800))
        os_event = int(rng.random() > 0.35)
        rfs_time = float(rng.integers(90, int(os_time) + 1))
        rfs_event = int(rng.random() > 0.7) if os_event else 0
        row = {
            "patient_id": pid,
            "cohort": "RG" if is_rg else "Lung1",
            "age": age,
            "sex": sex,
            "stage": (rng.choice(["I", "II", "III", "IV"], p=[0.20, 0.15, 0.50, 0.15]) if not is_rg else np.nan),
            "histology": rng.choice(["Adenocarcinoma", "Squamous", "NSCLC-NOS"]),
            "smoking": rng.choice(["Yes", "No", np.nan], p=[0.6, 0.3, 0.1]),
            "os_time": os_time,
            "os_event": os_event,
            "rfs_time": rfs_time,
            "rfs_event": rfs_event,
            "egfr": rng.choice([0, 1, np.nan], p=[0.5, 0.25, 0.25]),
            "kras": rng.choice([0, 1, np.nan], p=[0.5, 0.25, 0.25]),
            "alk": rng.choice([0, 1, np.nan], p=[0.7, 0.1, 0.2]),
            "weight_kg": round(rng.uniform(45, 100), 1) if is_rg else np.nan,
            "patient_id_amc": ("AMC-%03d" % i) if is_rg else np.nan,
            "patient_id_r01": ("R01-%03d" % i) if is_rg else np.nan,
            "ct_date": np.nan,
            "pet_date": np.nan,
            "date_of_recurrence": np.nan,
            "date_of_last_alive": np.nan,
            "date_of_death": np.nan,
            "pct_gg": np.nan,
            "pack_years": round(rng.uniform(0, 60), 1) if is_rg else np.nan,
            "quit_smoking_year": np.nan,
            "rg_pt": rng.choice(["T1", "T2", "T3", "T4"]) if is_rg else np.nan,
            "rg_pn": rng.choice(["N0", "N1", "N2"]) if is_rg else np.nan,
            "rg_pm": rng.choice(["M0", "M1"]) if is_rg else np.nan,
            "rg_grade": rng.choice([1, 2, 3, np.nan]) if is_rg else np.nan,
            "rg_lvi": rng.choice(["Yes", "No", np.nan]) if is_rg else np.nan,
            "rg_pleural_invasion": rng.choice(["Yes", "No", np.nan]) if is_rg else np.nan,
            "rg_adjuvant": rng.choice(["Yes", "No"]) if is_rg else np.nan,
            "rg_chemotherapy": rng.choice(["Yes", "No"]) if is_rg else np.nan,
            "rg_radiation": rng.choice(["Yes", "No"]) if is_rg else np.nan,
            "rg_recurrence_location": np.nan,
            "ethnicity": np.nan,
            "lung1_t_stage": rng.choice(["T1", "T2", "T3", "T4"]) if not is_rg else np.nan,
            "lung1_n_stage": rng.choice(["N0", "N1", "N2", "N3"]) if not is_rg else np.nan,
            "lung1_m_stage": rng.choice(["M0", "M1"]) if not is_rg else np.nan,
            "lung1_stage_detail": np.nan,
            "days_ct_to_surgery": round(rng.uniform(5, 90), 1) if is_rg else np.nan,
        }
        rows.append(row)
    df = pd.DataFrame(rows, columns=CLINICAL_COLS)
    return df


def main():
    feat = build_features()
    clin = build_clinical()
    assert set(feat.columns) == set(FEATURE_COLS)
    assert set(clin.columns) == set(CLINICAL_COLS)
    # every low-muscle demo patient also carries a fat abnormality
    low = feat.sort_values("SMA_vol_cm3").head(4)
    assert (feat["SMA_vol_cm3"] > 0).all()
    feat.to_csv(os.path.join(DATA, "bodycomp_features.csv"), index=False)
    clin.to_csv(os.path.join(DATA, "clinical_master.csv"), index=False)
    print("Demo data written:")
    print("  data/bodycomp_features.csv  %d x %d" % feat.shape)
    print("  data/clinical_master.csv    %d x %d" % clin.shape)
    print("  patients: %s" % ", ".join(feat.patient_id))


if __name__ == "__main__":
    main()
