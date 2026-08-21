#!/usr/bin/env python3
"""Recurrence-free survival analysis with competing risks (Fine-Gray).

Death is a competing risk for recurrence; standard Cox can overestimate
cumulative incidence. Fine-Gray subdistribution hazards are estimated with
an IPCW-weighted partial likelihood (scripts/finegray.py), with standard
Cox (death censored) as comparison.

Input : data/bodycomp_features.csv + data/clinical_master.csv (RG cohort)
Output: outputs/finegray_rfs/
"""
import os
import json
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "finegray_rfs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(BASE, "scripts"))
from finegray import FineGray  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402

STAGE_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}


def prep():
    feat = pd.read_csv(os.path.join(BASE, "data", "bodycomp_features.csv"))
    clin = pd.read_csv(os.path.join(BASE, "data", "clinical_master.csv"))
    df = feat.merge(clin, on="patient_id", how="inner")
    df = df[df.cohort == "RG"].copy()
    df["sex_m"] = (df["sex"] == "M").astype(int)
    df["stage_n"] = df["stage"].map(STAGE_MAP)
    df["log_SMA_vol_cm3"] = np.log(df["SMA_vol_cm3"].clip(lower=1))
    df["log_SAT_vol_cm3"] = np.log(df["SAT_vol_cm3"].clip(lower=1))
    df["log_VAT_vol_cm3"] = np.log(df["VAT_vol_cm3"].clip(lower=1))
    df["SMA_primary_cm2"] = df["SMA_L1_cm2"].fillna(df["SMA_T12_cm2"])
    return df


def build_rfs_competing(d):
    """Build competing-risk outcome for RFS:
    1 = recurrence (rfs_event==1, time=rfs_time)
    2 = death without recurrence (os_event==1 & rfs_event==0, time=rfs_time)
    0 = censored (rfs_event==0 & os_event==0, time=rfs_time)
    """
    d = d.copy()
    # drop cases with missing RFS outcome (official Recurrence='Not collected', e.g. AMC-049);
    # not treated as censoring (consistent with RFS Cox analysis)
    d = d[d["rfs_event"].notna()]
    d["fg_event"] = 0
    d.loc[d["rfs_event"] == 1, "fg_event"] = 1
    d.loc[(d["rfs_event"] == 0) & (d["os_event"] == 1), "fg_event"] = 2
    d["fg_time"] = d["rfs_time"].fillna(d["os_time"])
    return d


def main():
    d = prep()
    d = build_rfs_competing(d)
    need = ["fg_time", "fg_event", "log_SMA_vol_cm3", "age", "sex_m"]
    sub = d[need].dropna()
    print(f"RG RFS analysis n={len(sub)}")
    print(f"  recurrence={int((sub.fg_event == 1).sum())}, competing death={int((sub.fg_event == 2).sum())}, "
          f"censored={int((sub.fg_event == 0).sum())}")

    # sex-stratified lowest tertile -> low muscle
    sub["muscle_low"] = np.nan
    for s in [0, 1]:
        vals = sub.loc[sub.sex_m == s, "log_SMA_vol_cm3"].dropna()
        if len(vals) >= 20:
            cut = vals.quantile(1 / 3)
            sub.loc[sub.sex_m == s, "muscle_low"] = (sub.loc[sub.sex_m == s, "log_SMA_vol_cm3"] < cut).astype(int)

    X = sub[["muscle_low", "age", "sex_m"]].values
    keep = ~np.isnan(X).any(axis=1)
    sub = sub[keep]
    X = sub[["muscle_low", "age", "sex_m"]].values

    # Fine-Gray
    fg = FineGray(max_iter=100)
    fg.fit(sub["fg_time"].values, sub["fg_event"].values, X)
    fg_sum = fg.summary(["muscle_low", "age", "sex_m"])

    # comparison: standard Cox (death censored)
    cph = CoxPHFitter()
    cdf = sub[["fg_time", "fg_event", "muscle_low", "age", "sex_m"]].copy()
    cdf["c_event"] = (cdf["fg_event"] == 1).astype(int)
    cph.fit(cdf[["fg_time", "c_event", "muscle_low", "age", "sex_m"]],
            duration_col="fg_time", event_col="c_event")
    cox_hr = float(cph.hazard_ratios_["muscle_low"])
    cox_p = float(cph.summary.loc["muscle_low", "p"])
    lo, hi = np.exp(cph.confidence_intervals_.loc["muscle_low", "95% lower-bound"]), \
             np.exp(cph.confidence_intervals_.loc["muscle_low", "95% upper-bound"])

    fg_row = [r for r in fg_sum if r["var"] == "muscle_low"][0]
    result = {
        "n": len(sub),
        "events_recurrence": int((sub.fg_event == 1).sum()),
        "competing_death": int((sub.fg_event == 2).sum()),
        "censored": int((sub.fg_event == 0).sum()),
        "muscle_low_n": int(sub.muscle_low.sum()),
        "finegray": {"HR": round(float(fg_row["HR"]), 3), "p": round(float(fg_row["p"]), 4),
                     "se": round(float(fg_row["se"]), 4)},
        "cox_death_as_censor": {"HR": round(cox_hr, 3), "p": round(cox_p, 4),
                                "CI95": [round(float(lo), 3), round(float(hi), 3)]},
        "note": "Difference between Fine-Gray and Cox HRs reflects competing-risk impact; "
                "if Fine-Gray is significant but Cox is not, death as competing risk distorts recurrence estimates"
    }
    with open(os.path.join(OUT, "rg_finegray_rfs.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
