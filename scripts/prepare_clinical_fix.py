#!/usr/bin/env python3
"""Fix three data-entry issues in the clinical master table.

1. ALK translocations (official value 'Translocated') were silently set to
   missing; recompute EGFR/KRAS/ALK from the official labels file.
2. AMC-049 recurrence field was officially 'Not collected' but encoded as
   no recurrence; set RFS fields to missing.
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "raw")
VA = os.path.join(DATA, "data/clinical/NSCLC-Radiogenomics-VA-R01-labels.csv")
M = os.path.join(BASE, "data/clinical_master.csv")

NA_TOKENS = {"not collected", "not assessed", "unknown", "none", "n/a", "not recorded in database"}

def na(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return np.nan if s.lower() in NA_TOKENS else s

def main():
    m = pd.read_csv(M)
    va = pd.read_csv(VA)
    va["patient_id"] = va["Patient ID"].astype(str).str.strip()

    # ---- 1. recompute egfr/kras/alk from official VA ----
    va_map = va.set_index("patient_id")
    # keep official tokens (incl. Unknown/Not collected)
    egfr_off = va_map["EGFR mutation status"].apply(na)
    kras_off = va_map["KRAS mutation status"].apply(na)
    alk_off = va_map["ALK translocation status"].apply(na)

    egfr_new = egfr_off.map({"Mutant": 1, "Wildtype": 0})
    kras_new = kras_off.map({"Mutant": 1, "Wildtype": 0})
    alk_new = alk_off.map({"Translocated": 1, "Wildtype": 0})

    rg_mask = m.cohort == "RG"
    m.loc[rg_mask, "egfr"] = m.loc[rg_mask, "patient_id"].map(egfr_new).values
    m.loc[rg_mask, "kras"] = m.loc[rg_mask, "patient_id"].map(kras_new).values
    m.loc[rg_mask, "alk"] = m.loc[rg_mask, "patient_id"].map(alk_new).values

    # ---- 2. AMC-049 rfs mis-entry fix ----
    m.loc[(m.patient_id.str.strip() == "AMC-049") & rg_mask, ["rfs_event", "rfs_time"]] = np.nan

    m.to_csv(M, index=False)
    print("Saved:", M)

    # ---- verification ----
    rg = m[m.cohort == "RG"]
    print("\nALK fix verification:")
    print(rg[rg.patient_id.isin(["R01-012", "R01-130"])][["patient_id", "egfr", "kras", "alk"]].to_string())
    print("ALK value distribution:", rg["alk"].value_counts(dropna=False).to_dict())
    print("EGFR value distribution:", rg["egfr"].value_counts(dropna=False).to_dict())
    print("KRAS value distribution:", rg["kras"].value_counts(dropna=False).to_dict())
    print("\nAMC-049 fix verification:")
    print(rg[rg.patient_id.str.strip() == "AMC-049"][["patient_id", "rfs_event", "rfs_time"]].to_string())
    print("RG rfs_event distribution:", rg["rfs_event"].value_counts(dropna=False).to_dict())

if __name__ == "__main__":
    main()
