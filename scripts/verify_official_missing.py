#!/usr/bin/env python3
"""Verify that every missing value in the clinical master table
matches an officially missing value in the source files."""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "raw")
VA = os.path.join(DATA, "data/clinical/NSCLC-Radiogenomics-VA-R01-labels.csv")
L1 = os.path.join(DATA, "data/clinical/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
M = os.path.join(BASE, "data/clinical_master.csv")

NA_TOKENS = {"not collected", "not assessed", "unknown", "none", "n/a", "not recorded in database"}

def na(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return np.nan if s.lower() in NA_TOKENS else s

def report(desc, master_miss, off_miss, master_ids, off_ids):
    """master_miss/off_miss: sets of patient ids; compare differences."""
    only_master = sorted(master_miss - off_miss)   # master missing but official present -> we lost data
    only_off = sorted(off_miss - master_miss)      # official missing but master present -> we over-filled
    status = "OK" if not only_master and not only_off else "MISMATCH"
    print(f"[{status}] {desc}: master missing {len(master_miss)} / official missing {len(off_miss)}")
    if only_master:
        print(f"   [X] master missing but official present ({len(only_master)}): {only_master[:15]}")
    if only_off:
        print(f"   [!] official missing but master present ({len(only_off)}): {only_off[:15]}")
    return status

def main():
    m = pd.read_csv(M)
    va = pd.read_csv(VA)
    l1 = pd.read_csv(L1)

    # ---------- RG ----------
    rg = m[m.cohort == "RG"].copy()
    rg["key"] = rg["patient_id"].astype(str).str.strip()
    va["key"] = va["Patient ID"].astype(str).str.strip()
    mm = rg.merge(va, on="key", how="left", suffixes=("", "_off"))
    assert len(mm) == 211, f"RG merge rows {len(mm)} != 211"

    # key mapping: official key -> master patient_id
    print("=" * 70)
    print("RG cohort (n=211) vs official VA-R01-labels.csv")
    print("=" * 70)

    # 1. weight: official 'Not Collected' -> missing
    off_wt = set(mm.loc[mm["Weight (lbs)"].apply(lambda x: na(x) is np.nan), "key"])
    m_wt = set(mm.loc[mm["weight_kg"].isna(), "key"])
    report("weight_kg", m_wt, off_wt, None, None)

    # 2. pack_years: official NaN or 'Not Collected' -> missing
    off_py = set(mm.loc[mm["Pack Years"].apply(lambda x: na(x) is np.nan), "key"])
    m_py = set(mm.loc[mm["pack_years"].isna(), "key"])
    report("pack_years", m_py, off_py, None, None)

    # 3. quit_smoking_year
    off_qy = set(mm.loc[mm["Quit Smoking Year"].apply(lambda x: na(x) is np.nan), "key"])
    m_qy = set(mm.loc[mm["quit_smoking_year"].isna(), "key"])
    report("quit_smoking_year", m_qy, off_qy, None, None)

    # 4. EGFR: Unknown/Not collected -> missing
    off_egfr = set(mm.loc[mm["EGFR mutation status"].isin(["Unknown", "Not collected"]), "key"])
    m_egfr = set(mm.loc[mm["egfr"].isna(), "key"])
    report("egfr", m_egfr, off_egfr, None, None)

    # 5. KRAS
    off_kras = set(mm.loc[mm["KRAS mutation status"].isin(["Unknown", "Not collected"]), "key"])
    m_kras = set(mm.loc[mm["kras"].isna(), "key"])
    report("kras", m_kras, off_kras, None, None)

    # 6. ALK: Translocated=1, Wildtype=0, Unknown/Not collected -> missing
    off_alk = set(mm.loc[mm["ALK translocation status"].isin(["Unknown", "Not collected"]), "key"])
    m_alk = set(mm.loc[mm["alk"].isna(), "key"])
    report("alk", m_alk, off_alk, None, None)

    # 7. pathological T/N/M
    for mc, oc in [("rg_pt", "Pathological T stage"), ("rg_pn", "Pathological N stage"), ("rg_pm", "Pathological M stage")]:
        off_miss = set(mm.loc[mm[oc].apply(lambda x: na(x) is np.nan), "key"])
        m_miss = set(mm.loc[mm[mc].isna(), "key"])
        report(mc, m_miss, off_miss, None, None)

    # 8. grade / lvi / pleural
    for mc, oc in [("rg_grade", "Histopathological Grade"), ("rg_lvi", "Lymphovascular invasion"),
                   ("rg_pleural_invasion", "Pleural invasion (elastic, visceral, or parietal)")]:
        off_miss = set(mm.loc[mm[oc].apply(lambda x: na(x) is np.nan), "key"])
        m_miss = set(mm.loc[mm[mc].isna(), "key"])
        report(mc, m_miss, off_miss, None, None)

    # 9. treatment: Adjuvant/Chemo/Radiation ('Not Collected' -> missing)
    for mc, oc in [("rg_adjuvant", "Adjuvant Treatment"), ("rg_chemotherapy", "Chemotherapy"), ("rg_radiation", "Radiation")]:
        off_miss = set(mm.loc[mm[oc].apply(lambda x: na(x) is np.nan), "key"])
        m_miss = set(mm.loc[mm[mc].isna(), "key"])
        report(mc, m_miss, off_miss, None, None)

    # 10. pct_gg / ethnicity
    for mc, oc in [("pct_gg", "%GG"), ("ethnicity", "Ethnicity")]:
        off_miss = set(mm.loc[mm[oc].apply(lambda x: na(x) is np.nan), "key"])
        m_miss = set(mm.loc[mm[mc].isna(), "key"])
        report(mc, m_miss, off_miss, None, None)

    # 11. recurrence: rfs_event vs official Recurrence
    off_rec = set(mm.loc[mm["Recurrence"].apply(lambda x: na(x) is np.nan), "key"])
    m_rec = set(mm.loc[mm["rfs_event"].isna(), "key"])
    report("rfs_event", m_rec, off_rec, None, None)
    # value agreement: yes=1, no=0
    rec_map = {"yes": 1, "no": 0}
    mismatch_val = mm[mm["Recurrence"].map(rec_map).notna() & mm["rfs_event"].notna() &
                      (mm["Recurrence"].map(rec_map) != mm["rfs_event"])]
    print(f"   rfs_event value mismatch: {len(mismatch_val)} cases {mismatch_val['key'].tolist()[:10]}")

    # 12. dates (ct_date/pet_date/date_of_recurrence/date_of_death/date_of_last_alive)
    for mc, oc in [("ct_date", "CT Date"), ("pet_date", "PET Date"),
                   ("date_of_recurrence", "Date of Recurrence"),
                   ("date_of_death", "Date of Death"),
                   ("date_of_last_alive", "Date of Last Known Alive")]:
        off_miss = set(mm.loc[mm[oc].apply(lambda x: na(x) is np.nan), "key"])
        m_miss = set(mm.loc[mm[mc].isna(), "key"])
        report(mc, m_miss, off_miss, None, None)

    # 13. days_ct_to_surgery
    off_dct = set(mm.loc[mm["Days between CT and surgery"].apply(lambda x: na(x) is np.nan), "key"])
    m_dct = set(mm.loc[mm["days_ct_to_surgery"].isna(), "key"])
    report("days_ct_to_surgery", m_dct, off_dct, None, None)

    # 14. survival status -> os_event agreement
    ss_map = {"Dead": 1, "Alive": 0}
    mis_os = mm[mm["Survival Status"].map(ss_map).notna() & mm["os_event"].notna() &
                (mm["Survival Status"].map(ss_map) != mm["os_event"])]
    print(f"   os_event vs Survival Status mismatch: {len(mis_os)} cases {mis_os['key'].tolist()[:10]}")

    # 15. Time to Death -> os_time rough check (Dead with official values only)
    dead = mm[mm["Survival Status"] == "Dead"]
    ttd = pd.to_numeric(dead["Time to Death (days)"], errors="coerce")
    print(f"   Dead n={len(dead)}, official Time to Death non-null {ttd.notna().sum()}, master os_time non-null {dead['os_time'].notna().sum()}")
    if ttd.notna().sum() > 0:
        sub = dead[ttd.notna() & dead["os_time"].notna()]
        diff = (pd.to_numeric(sub["Time to Death (days)"], errors="coerce") - sub["os_time"]).abs()
        print(f"   diff>2 days: {(diff > 2).sum()} cases, max diff {diff.max():.0f} days")

    # ---------- Lung1 ----------
    print()
    print("=" * 70)
    print("Lung1 cohort (n=422) vs official NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
    print("=" * 70)
    l1c = m[m.cohort == "Lung1"].copy()
    l1c["key"] = l1c["patient_id"].astype(str).str.strip()
    l1["key"] = l1["PatientID"].astype(str).str.strip()
    l1m = l1c.merge(l1, on="key", how="left", suffixes=("", "_off"))
    assert len(l1m) == 422, f"Lung1 merge rows {len(l1m)} != 422"

    for mc, oc in [("age", "age"), ("histology", "Histology"), ("stage", "Overall.Stage"),
                   ("lung1_t_stage", "clinical.T.Stage"), ("lung1_n_stage", "Clinical.N.Stage"),
                   ("lung1_m_stage", "Clinical.M.Stage")]:
        off_miss = set(l1m.loc[l1m[oc].isna(), "key"])
        m_miss = set(l1m.loc[l1m[mc].isna(), "key"])
        report(mc, m_miss, off_miss, None, None)

    # age value agreement (non-missing)
    sub = l1m[l1m["age"].notna() & l1m["age_off"].notna()]
    agediff = (sub["age"] - sub["age_off"]).abs()
    print(f"   age diff>0.01: {(agediff > 0.01).sum()} cases")

    # OS event/time agreement
    mis_os_l1 = l1m[l1m["deadstatus.event"].notna() & l1m["os_event"].notna() &
                    (l1m["deadstatus.event"] != l1m["os_event"])]
    print(f"   os_event vs deadstatus.event mismatch: {len(mis_os_l1)} cases")
    sub2 = l1m[l1m["Survival.time"].notna() & l1m["os_time"].notna()]
    sd = (sub2["Survival.time"] - sub2["os_time"]).abs()
    print(f"   os_time vs Survival.time diff>2 days: {(sd > 2).sum()} cases (official unit=months, master=days)")


if __name__ == "__main__":
    main()
