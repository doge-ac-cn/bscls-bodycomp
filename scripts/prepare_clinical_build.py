"""Build a unified clinical master table (Lung1 + RG).

Standardises column names across cohorts and merges official clinical
spreadsheets. Source files are expected under data/raw/data/clinical/.

Output: data/clinical_master.csv
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data", "clinical_master.csv")

def prep_lung1():
    df = pd.read_csv(os.path.join(DATA, "data/clinical/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"))
    df = df.rename(columns={"PatientID": "patient_id", "age": "age", "gender": "sex",
                            "Overall.Stage": "stage", "Histology": "histology",
                            "Survival.time": "os_time", "deadstatus.event": "os_event"})
    df["cohort"] = "Lung1"
    df["sex"] = df["sex"].map({"male": "M", "female": "F"})
    df["stage"] = df["stage"].astype(str).str.replace("IIIb", "III").replace("IIIa", "III")
    df["stage"] = df["stage"].where(df["stage"].isin(["I", "II", "III", "IV"]), np.nan)
    df["os_event"] = df["os_event"].astype(int)
    df["os_time"] = df["os_time"].astype(float)
    df["rfs_time"], df["rfs_event"] = np.nan, np.nan
    df["egfr"], df["kras"], df["alk"] = np.nan, np.nan, np.nan
    df["smoking"] = np.nan
    df["weight_kg"] = np.nan
    return df[["patient_id", "cohort", "age", "sex", "stage", "histology", "smoking",
               "os_time", "os_event", "rfs_time", "rfs_event", "egfr", "kras", "alk", "weight_kg"]]

def prep_rg():
    mt = pd.read_csv(os.path.join(DATA, "data/master_table_nsclc_radiogenomics.csv"))
    sar = pd.read_parquet(os.path.join(DATA, "data/clinical/sarg_patients.parquet"))
    # SARG dedup (first row per patient_id); 105-case subset
    sar = sar.drop_duplicates(subset="patient_id", keep="first")
    df = mt.rename(columns={"PatientID": "patient_id", "AgeAtDiagnosis": "age", "Sex": "sex",
                            "SmokingStatus": "smoking", "Histology": "histology",
                            "SurvivalStatus": "surv_status", "TimeToDeathDays": "os_time",
                            "Recurrence": "rfs_event_raw"})
    df["cohort"] = "RG"
    df["sex"] = df["sex"].map({"Male": "M", "Female": "F"})
    df["smoking"] = df["smoking"].astype(str)
    # fill pathological stage, weight, survival from SARG
    sar2 = sar[["patient_id", "pathological_n_stage", "weight_lbs", "date_of_death", "date_of_last_known_alive", "time_to_death_days"]].copy()
    df = df.merge(sar2, on="patient_id", how="left")
    df["stage"] = df["pathological_n_stage"].map(lambda x: x if x in ("N0", "N1", "N2") else np.nan)
    def lbs_to_kg(x):
        try:
            return round(float(x) * 0.4536, 1)
        except Exception:
            return np.nan
    df["weight_kg"] = df["weight_lbs"].apply(lbs_to_kg)
    # OS time: dead -> time_to_death_days (SARG) or master TimeToDeathDays; alive -> SARG follow-up
    df["os_event"] = (df["surv_status"].astype(str).str.lower() == "dead").astype(int)
    df["os_time"] = pd.to_numeric(df["time_to_death_days"].fillna(df["os_time"]), errors="coerce")
    # alive follow-up: date_of_last_known_alive - ct_date (approximation)
    if "ct_date" in sar.columns:
        sar_dates = sar[["patient_id", "ct_date", "date_of_last_known_alive"]].copy()
        df = df.merge(sar_dates, on="patient_id", how="left", suffixes=("", "_sarg"))
        try:
            dlo = pd.to_datetime(df["date_of_last_known_alive"], errors="coerce")
            ctd = pd.to_datetime(df["ct_date"], errors="coerce")
            fup = (dlo - ctd).dt.days
            df["os_time"] = df["os_time"].where(df["os_event"] == 1, fup)
        except Exception:
            pass
    # RFS：recurrence yes→1, no→0, Not collected→NA
    df["rfs_event"] = df["rfs_event_raw"].map({"yes": 1, "no": 0, "Not collected": np.nan}).astype("Int64")
    df["rfs_time"] = np.nan  # recurrence time set later (date_of_recurrence - ct_date)
    for c in ["EGFR", "KRAS", "ALK"]:
        # Official labels encode ALK positivity as "Translocated" (not "Mutant").
        # Mapping only {"Mutant":1} would silently drop the 2 ALK+ cases.
        mapping = {"Mutant": 1, "Wildtype": 0}
        if c == "ALK":
            mapping["Translocated"] = 1
        df[c.lower()] = df[c].map(mapping).astype("Int64")
    return df[["patient_id", "cohort", "age", "sex", "stage", "histology", "smoking",
               "os_time", "os_event", "rfs_time", "rfs_event", "egfr", "kras", "alk", "weight_kg"]]

def main():
    l1 = prep_lung1()
    rg = prep_rg()
    m = pd.concat([l1, rg], ignore_index=True)
    m.to_csv(OUT, index=False)
    print(f"Unified clinical table: {OUT} ({len(m)} patients)")
    print(m.groupby("cohort").agg(n=("patient_id", "count"),
                                  os_events=("os_event", "sum"),
                                  rfs_events=("rfs_event", "sum"),
                                  age_mean=("age", "mean"),
                                  male=("sex", lambda s: (s == "M").sum()),
                                  wt_avail=("weight_kg", lambda s: s.notna().sum())).to_string())
    print("\nMissing rates:")
    print((m.isna().mean() * 100).round(1).to_string())

if __name__ == "__main__":
    main()
