#!/usr/bin/env python3
"""Encode time-to-event fields (OS/RFS) in days.

OS: dead -> time to death; censored -> last known alive - CT date.
RFS: recurrence -> recurrence date - CT date; censored -> last known alive.

Output: data/clinical_master.csv
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(BASE, "data", "clinical_master.csv")
SARG = os.path.join(BASE, "data", "raw", "data", "clinical", "NSCLC-Radiogenomics-VA-R01-labels.csv")
OUT = os.path.join(BASE, "data", "clinical_master.csv")


def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue
    return pd.to_datetime(s, errors="coerce")


def main():
    master = pd.read_csv(MASTER)
    sarg = pd.read_csv(SARG)

    # parse SARG time fields
    sarg["CT_date"] = sarg["CT Date"].apply(parse_date)
    sarg["death_date"] = sarg["Date of Death"].apply(parse_date)
    sarg["last_alive"] = sarg["Date of Last Known Alive"].apply(parse_date)
    sarg["recur_date"] = sarg["Date of Recurrence"].apply(parse_date)
    sarg["t_death"] = pd.to_numeric(sarg["Time to Death (days)"], errors="coerce")

    # SARG Patient ID is 'AMC-XXX'; master may use R01-XXX
    sarg["patient_id"] = sarg["Patient ID"].astype(str).str.strip()
    sarg["patient_id_amc"] = sarg["patient_id"]
    sarg["patient_id_r01"] = "R01-" + sarg["patient_id"].str.replace("AMC-", "", regex=False)

    # build RG time table
    rg_times = []
    for _, r in sarg.iterrows():
        ct = r["CT_date"]
        if pd.isna(ct):
            continue
        # OS
        if pd.notna(r["t_death"]):
            os_t = r["t_death"]
            os_e = 1
        elif pd.notna(r["last_alive"]):
            os_t = (r["last_alive"] - ct).days
            os_e = 0
        else:
            os_t, os_e = np.nan, 0
        # RFS
        if pd.notna(r["recur_date"]):
            rfs_t = (r["recur_date"] - ct).days
            rfs_e = 1
        elif pd.notna(r["last_alive"]):
            rfs_t = (r["last_alive"] - ct).days
            rfs_e = 0
        else:
            rfs_t, rfs_e = np.nan, 0
        rg_times.append({
            "patient_id_amc": r["patient_id_amc"],
            "patient_id_r01": r["patient_id_r01"],
            "os_time_rg": os_t, "os_event_rg": os_e,
            "rfs_time_rg": rfs_t, "rfs_event_rg": rfs_e,
        })
    rg_df = pd.DataFrame(rg_times)

    # merge back to master
    m = master.copy()
    m = m.merge(rg_df, left_on="patient_id", right_on="patient_id_amc", how="left", suffixes=("", "_rg"))
    # R01 patients
    missing = m[m["os_time_rg"].isna() & (m.cohort == "RG")]
    if len(missing):
        m2 = missing.merge(rg_df, left_on="patient_id", right_on="patient_id_r01", how="left", suffixes=("", "_r01"))
        for col in ["os_time_rg", "os_event_rg", "rfs_time_rg", "rfs_event_rg"]:
            m.loc[m2.index, col] = m2[col].fillna(m.loc[m2.index, col])

    # overwrite with fixed values
    m.loc[m.cohort == "RG", "os_time"] = m.loc[m.cohort == "RG", "os_time_rg"]
    m.loc[m.cohort == "RG", "os_event"] = m.loc[m.cohort == "RG", "os_event_rg"]
    m.loc[m.cohort == "RG", "rfs_time"] = m.loc[m.cohort == "RG", "rfs_time_rg"]
    m.loc[m.cohort == "RG", "rfs_event"] = m.loc[m.cohort == "RG", "rfs_event_rg"]
    m = m.drop(columns=["os_time_rg", "os_event_rg", "rfs_time_rg", "rfs_event_rg"])

    m.to_csv(OUT, index=False)
    print(f"Saved {OUT}: {m.shape[0]} x {m.shape[1]}")

    # verification
    for c in ["Lung1", "RG"]:
        sub = m[m.cohort == c]
        print(f"=== {c} ===")
        print(f"  os_time non-null: {sub.os_time.notna().sum()}/{len(sub)} | os events: {(sub.os_event==1).sum()}")
        print(f"  rfs_time non-null: {sub.rfs_time.notna().sum()}/{len(sub)} | rfs events: {(sub.rfs_event==1).sum()}")
        if sub.rfs_time.notna().sum():
            print(f"  rfs_time distribution: median={sub.rfs_time.median():.0f}, range=[{sub.rfs_time.min():.0f}, {sub.rfs_time.max():.0f}]")
    # spot-check AMC-001
    row = m[m.patient_id == "AMC-001"]
    if len(row):
        print("\nAMC-001 check:", row[["os_time", "os_event", "rfs_time", "rfs_event"]].to_dict("records"))


if __name__ == "__main__":
    main()
