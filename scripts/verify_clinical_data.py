#!/usr/bin/env python3
"""Verify the clinical master table against official source files."""
import pandas as pd
import numpy as np
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "raw")
CLIN = os.path.join(BASE, "data", "clinical_master.csv")
VA = os.path.join(DATA, "data/clinical/NSCLC-Radiogenomics-VA-R01-labels.csv")
L1 = os.path.join(DATA, "data/clinical/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")

clin = pd.read_csv(CLIN)
va = pd.read_csv(VA)
l1 = pd.read_csv(L1)

errors = []
def check(cond, msg):
    if cond:
        print(f"  [OK] {msg}")
    else:
        errors.append(msg)
        print(f"  [FAIL] {msg}")

print("=== 1. Basic structure ===")
check(clin.shape[0] == 633, f"rows = 633 (got {clin.shape[0]})")
check(clin.shape[1] == 41, f"cols = 41 (got {clin.shape[1]})")
check(clin.patient_id.nunique() == 633, "patient_id unique")
check(clin.patient_id.isna().sum() == 0, "patient_id no missing")
print(f"  columns: {list(clin.columns)}")

print("\n=== 2. Cohort composition ===")
lc = clin.cohort.value_counts().to_dict()
check(lc.get("Lung1") == 422 and lc.get("RG") == 211, f"Lung1=422 RG=211 (got {lc})")

print("\n=== 3. RG stage semantics ===")
rg = clin[clin.cohort == "RG"]
check(rg.stage.isna().all(), f"RG stage all NaN (got {rg.stage.notna().sum()} non-null)")
check(rg.rg_pn.notna().sum() >= 150, f"RG rg_pn >= 150 (got {rg.rg_pn.notna().sum()})")
check(rg.rg_pt.notna().sum() >= 150, f"RG rg_pt >= 150 (got {rg.rg_pt.notna().sum()})")
check(rg.rg_pm.notna().sum() >= 150, f"RG rg_pm >= 150 (got {rg.rg_pm.notna().sum()})")

print("\n=== 4. RG weight recovery ===")
check(rg.weight_kg.notna().sum() == 152, f"RG weight non-null = 152 (got {rg.weight_kg.notna().sum()})")

print("\n=== 5. patient_id_r01 alias ===")
bad = rg.patient_id_r01[rg.patient_id_r01.str.contains("R01-R01-", na=False)]
check(len(bad) == 0, f"no double-prefix R01-R01 (got {len(bad)})")
r01_all = rg.patient_id_r01.str.startswith("R01-").all()
check(r01_all, "patient_id_r01 all R01- prefixed")
def to_r01_alias(pid):
    return "R01-" + str(pid).replace("AMC-", "", 1).replace("R01-", "", 1)
alias_ok = (rg.patient_id_r01 == rg.patient_id.apply(to_r01_alias)).all()
check(alias_ok, "patient_id_r01 reversible to patient_id")

print("\n=== 6. Official VA label linkage ===")
va["patient_id"] = va["Patient ID"].astype(str).str.strip()
va_ids = set(va.patient_id)
rg_ids = set(rg.patient_id)
check(rg_ids == va_ids, f"RG patient_id matches VA official 211 (diff RG-only={rg_ids-va_ids} VA-only={va_ids-rg_ids})")
na_tokens = {"not collected", "not assessed", "unknown", "none", "n/a"}
def safe_w(x):
    s = str(x).strip().lower()
    if s in na_tokens or not s:
        return np.nan
    try:
        return round(float(x) * 0.4536, 1)
    except Exception:
        return np.nan
va_w = va.set_index("patient_id")["Weight (lbs)"].apply(safe_w)
rg_w = rg.set_index("patient_id")["weight_kg"]
matched = 0
mismatch = []
for pid in va_w.index:
    if pd.notna(va_w[pid]) and pd.notna(rg_w.get(pid, np.nan)):
        if abs(va_w[pid] - rg_w[pid]) < 0.15:
            matched += 1
        else:
            mismatch.append((pid, va_w[pid], rg_w[pid]))
check(matched == 152 and len(mismatch) == 0, f"weight matches official conversion 152/152 (matched {matched}, mismatch {mismatch[:3]})")

print("\n=== 7. Lung1 detailed staging ===")
l1s = clin[clin.cohort == "Lung1"]
check(l1s.lung1_t_stage.notna().sum() >= 400, f"Lung1 T stage >= 400 (got {l1s.lung1_t_stage.notna().sum()})")
check(l1s.lung1_n_stage.notna().sum() >= 400, f"Lung1 N stage >= 400 (got {l1s.lung1_n_stage.notna().sum()})")
check(l1s.lung1_m_stage.notna().sum() >= 400, f"Lung1 M stage >= 400 (got {l1s.lung1_m_stage.notna().sum()})")
check(l1s.lung1_stage_detail.notna().sum() >= 400, f"Lung1 stage_detail >= 400 (got {l1s.lung1_stage_detail.notna().sum()})")

print("\n=== 8. Outcome events ===")
os_l1 = int((clin[clin.cohort == "Lung1"].os_event == 1).sum())
os_rg = int((clin[clin.cohort == "RG"].os_event == 1).sum())
rfs_rg = int((clin[clin.cohort == "RG"].rfs_event == 1).sum())
check(os_l1 == 373 and os_rg == 63, f"Lung1 OS=373 RG OS=63 (got {os_l1}/{os_rg})")
check(rfs_rg == 54, f"RG RFS=54 (got {rfs_rg})")

print("\n=== 9. RG treatment/smoking fields ===")
check(rg.rg_chemotherapy.notna().sum() >= 200, f"RG chemo >= 200 (got {rg.rg_chemotherapy.notna().sum()})")
check(rg.rg_radiation.notna().sum() >= 200, f"RG radiation >= 200 (got {rg.rg_radiation.notna().sum()})")
check(rg.rg_adjuvant.notna().sum() >= 200, f"RG adjuvant >= 200 (got {rg.rg_adjuvant.notna().sum()})")
check(rg.pack_years.notna().sum() >= 150, f"RG pack_years >= 150 (got {rg.pack_years.notna().sum()})")
check(rg.ct_date.notna().sum() >= 200, f"RG ct_date >= 200 (got {rg.ct_date.notna().sum()})")
check(rg.days_ct_to_surgery.notna().sum() == 211, f"RG days_ct_to_surgery = 211 (got {rg.days_ct_to_surgery.notna().sum()})")
check(clin[clin.cohort == "Lung1"].days_ct_to_surgery.isna().all(), "Lung1 days_ct_to_surgery all NaN (RG-only field)")

print("\n=== 9b. Mutation status values (official labels) ===")
alk_pos = rg.alk.dropna()
check(int((alk_pos == 1).sum()) == 2, f"RG ALK+ = 2 (got {(alk_pos == 1).sum()})")
egfr_pos = rg.egfr.dropna()
check(int((egfr_pos == 1).sum()) == 43, f"RG EGFR+ = 43 (got {(egfr_pos == 1).sum()})")
kras_pos = rg.kras.dropna()
check(int((kras_pos == 1).sum()) == 38, f"RG KRAS+ = 38 (got {(kras_pos == 1).sum()})")
# official ALK "Translocated" must be 1, never NaN
va["key"] = va["Patient ID"].astype(str).str.strip()
trans = va.loc[va["ALK translocation status"].astype(str).str.strip() == "Translocated", "key"]
trans_master = set(rg.loc[rg.alk == 1, "patient_id"].astype(str).str.strip())
check(set(trans) == trans_master, f"ALK Translocated {sorted(trans)} all encoded as 1 (got {sorted(trans_master)})")

print("\n=== 10. Official Lung1 patient_id linkage ===")
l1_ids = set(l1["PatientID"].astype(str).str.strip())
l1m_ids = set(clin[clin.cohort == "Lung1"].patient_id)
check(l1m_ids == l1_ids, f"Lung1 patient_id matches official (diff {l1m_ids-l1_ids} / {l1_ids-l1m_ids})")

print("\n" + "=" * 50)
if errors:
    print(f"Conclusion: {len(errors)} issue(s) found")
    for e in errors:
        print("  -", e)
    sys.exit(1)
else:
    print("Conclusion: clinical_master.csv all checks passed")
