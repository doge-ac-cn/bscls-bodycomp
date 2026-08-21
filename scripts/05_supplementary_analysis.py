#!/usr/bin/env python3
"""Supplementary analyses for the full dataset.

  A. Bootstrap stability (1000x) for phenotype Cox models, full and reduced
     adjustment sets, proportional-hazards diagnostics, C-index increment,
     Kaplan-Meier curves and decision-curve analysis (DCA).
  B. Multiple-testing control: BH-FDR across all body-composition metrics
     (univariate OS Cox, both cohorts).
  C. Median follow-up time (reverse KM).
  D. Segmentation QC summary: missing patterns / L1->T12 fallback rate /
     edge markers / coverage.
  E. Chemotherapy-stratified muscle phenotype analysis (RG cohort).
  F. Pooled-cohort sarcopenic obesity (within-cohort stratified definition).

Output: outputs/supplementary/
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "supplementary")
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

ADJ_L1 = ["age", "sex_m", "stage_n"]
ADJ_RG = ["age", "sex_m"]

BC_METRICS = [
    ("log_SMA_vol_cm3", "3D muscle volume (log)"),
    ("SMA_primary_cm2", "L1/T12 muscle area"),
    ("SM_mean_hu", "mean muscle density"),
    ("log_SAT_vol_cm3", "SAT volume (log)"),
    ("log_VAT_vol_cm3", "VAT volume (log)"),
    ("SM_density_L1_HU", "L1 muscle density"),
    ("BMD_L1_HU", "L1 bone density"),
    ("SAT_L1_cm2", "L1 SAT area"),
    ("VAT_L1_cm2", "L1 VAT area"),
    ("ES_T12_cm2", "T12 erector spinae area"),
    ("BMD_T12_HU", "T12 bone density"),
    ("SMA_T4_cm2", "T4 muscle area"),
    ("Psoas_L1_cm2", "L1 psoas area"),
    ("Psoas_density_L1_HU", "L1 psoas density"),
    ("SMA_L3_cm2", "L3 muscle area (RG)"),
]


def prep():
    feat = pd.read_csv(os.path.join(BASE, "data", "bodycomp_features.csv"))
    clin = pd.read_csv(os.path.join(BASE, "data", "clinical_master.csv"))
    df = feat.merge(clin, on="patient_id", how="inner")
    df["sex_m"] = (df["sex"] == "M").astype(int)
    df["stage_n"] = df["stage"].map({"I": 1, "II": 2, "III": 3, "IV": 4})
    for c in ["log_SMA_vol_cm3", "log_SAT_vol_cm3", "log_VAT_vol_cm3"]:
        base = c.replace("log_", "")
        df[c] = np.log(df[base].clip(lower=1))
    df["SMA_primary_cm2"] = df["SMA_L1_cm2"].fillna(df["SMA_T12_cm2"])
    df["L1_unavailable"] = df["L1_unavailable"].fillna(0)
    return df


def sex_cutoff(d, col, q):
    cuts = {}
    for s in [0, 1]:
        sub = d[d.sex_m == s][col].dropna()
        if len(sub) >= 30:
            cuts[s] = sub.quantile(q)
    return cuts


def make_binary(d, col, q=1 / 3):
    label = f"{col}_low{q}"
    d[label] = np.nan
    for s, c in sex_cutoff(d, col, q).items():
        m = d.sex_m == s
        d.loc[m, label] = (d.loc[m, col] < c).astype(int)
    return d, label


def build_phenotypes(d):
    d2, lb = make_binary(d.copy(), "log_SMA_vol_cm3", q=1 / 3)
    d3, _ = make_binary(d2, "log_VAT_vol_cm3", q=2 / 3)
    d4, ls = make_binary(d3, "log_SAT_vol_cm3", q=1 / 3)
    vat_hi = [c for c in d4.columns if c.startswith("log_VAT_vol_cm3_low")][0]
    d4["pheno_sarc_obese"] = ((d4[lb] == 1) & (d4[vat_hi] == 0)).astype(int)
    d4["pheno_cachexia"] = ((d4[lb] == 1) & (d4[ls] == 1)).astype(int)
    d4["pheno_low_muscle_only"] = ((d4[lb] == 1) & (d4[ls] == 0) & (d4[vat_hi] == 1)).astype(int)
    return d4, lb


def cox_fit(sub, label, covs):
    cols = ["os_time", "os_event", label] + list(covs)
    sub = sub[cols].dropna()
    keep = [c for c in cols if sub[c].nunique() > 1]
    sub = sub[keep]
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="os_time", event_col="os_event")
    return cph, sub


def full_adjusted_model(d, label, covs):
    cph, sub = cox_fit(d, label, covs)
    rows = []
    for c in covs:
        if c not in cph.summary.index:
            continue
        hr = float(cph.hazard_ratios_[c])
        p = float(cph.summary.loc[c, "p"])
        lo, hi = np.exp(cph.confidence_intervals_.loc[c, "95% lower-bound"]), np.exp(cph.confidence_intervals_.loc[c, "95% upper-bound"])
        rows.append({"covariate": c, "HR": round(hr, 3), "p": round(p, 4),
                     "CI95": [round(float(lo), 3), round(float(hi), 3)]})
    hr = float(cph.hazard_ratios_[label])
    p = float(cph.summary.loc[label, "p"])
    lo, hi = np.exp(cph.confidence_intervals_.loc[label, "95% lower-bound"]), np.exp(cph.confidence_intervals_.loc[label, "95% upper-bound"])
    return {"label": label, "n": len(sub), "events": int(sub.os_event.sum()),
            "pheno_n": int(sub[label].sum()), "HR": round(hr, 3), "p": round(p, 4),
            "CI95": [round(float(lo), 3), round(float(hi), 3)],
            "c_index": round(float(cph.concordance_index_), 4),
            "covariates": rows}


def bootstrap_hr(d, label, B=1000, adj_cols=ADJ_RG):
    sub = d[["os_time", "os_event", label] + list(adj_cols)].dropna()
    hrs = []
    for _ in range(B):
        idx = np.random.choice(len(sub), len(sub), replace=True)
        boot = sub.iloc[idx]
        if boot[label].sum() < 5 or boot[label].nunique() < 2:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(boot, duration_col="os_time", event_col="os_event")
            hrs.append(float(cph.hazard_ratios_[label]))
        except Exception:
            continue
    hrs = np.array(hrs)
    if len(hrs) < 100:
        return None
    return {"B_ok": len(hrs), "HR_median": round(float(np.median(hrs)), 3),
            "CI95_percentile": [round(float(np.percentile(hrs, 2.5)), 3),
                                round(float(np.percentile(hrs, 97.5)), 3)],
            "prop_hr_gt1": round(float((hrs > 1).mean()), 3),
            "fail_rate": round(1 - len(hrs) / B, 3)}


def check_ph(d, label, adj_cols=ADJ_RG):
    cph, sub = cox_fit(d, label, adj_cols)
    from scipy.stats import linregress
    rows = []
    try:
        res = cph.compute_residuals(sub, kind="schoenfeld")
        for var in cph.summary.index:
            if var not in res.columns:
                continue
            r = res[var].dropna()
            if len(r) < 10:
                continue
            idx = r.index
            t = sub.loc[idx, "os_time"].values
            r_scaled = r.values + float(cph.hazard_ratios_[var])
            slope, intercept, rval, pval, se = linregress(t, r_scaled)
            rows.append({"variable": var, "p_schoenfeld": round(float(pval), 4)})
    except Exception as e:
        rows.append({"variable": "all", "error": str(e)})
    return rows


def cindex_increment(d, lb, covs):
    base_cols = ["os_time", "os_event"] + list(covs)
    base = d[base_cols].dropna()
    cph_base = CoxPHFitter()
    cph_base.fit(base, duration_col="os_time", event_col="os_event")
    ci_base = cph_base.concordance_index_
    cols = base_cols + [lb]
    full = d[cols].dropna()
    cph_full = CoxPHFitter()
    cph_full.fit(full, duration_col="os_time", event_col="os_event")
    ci_full = cph_full.concordance_index_
    return {"ci_clinical_base": round(float(ci_base), 4),
            "ci_clinical_plus_muscle": round(float(ci_full), 4),
            "delta": round(float(ci_full - ci_base), 4)}


def km_absolute_diff(d, label, years=(3, 5)):
    d2 = d.dropna(subset=["os_time", "os_event", label])
    m0, m1 = d2[label] == 0, d2[label] == 1
    kmf0, kmf1 = KaplanMeierFitter(), KaplanMeierFitter()
    kmf0.fit(d2.os_time[m0], d2.os_event[m0], label="normal")
    kmf1.fit(d2.os_time[m1], d2.os_event[m1], label="phenotype")
    lr = logrank_test(d2.os_time[m0], d2.os_time[m1], d2.os_event[m0], d2.os_event[m1])
    rows = []
    for yr in years:
        t = yr * 365
        s0 = float(kmf0.predict(t))
        s1 = float(kmf1.predict(t))
        rows.append({"time_yr": yr, "survival_normal": round(s0, 3),
                     "survival_pheno": round(s1, 3),
                     "absolute_diff": round(s1 - s0, 3)})
    return rows, float(lr.p_value)


def dca_3y(d, label, covs, horizon=1095):
    cols = ["os_time", "os_event", label] + list(covs)
    sub = d[cols].dropna()
    keep = [c for c in cols if sub[c].nunique() > 1]
    sub = sub[keep].copy()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="os_time", event_col="os_event")
    surv = cph.predict_survival_function(sub, times=[horizon]).T
    sub["p_risk_3y"] = 1 - surv.values[:, 0]
    cens = (sub.os_event == 0).astype(int)
    kmc = KaplanMeierFitter()
    kmc.fit(sub.os_time, cens)
    g_pred = np.clip(kmc.predict(sub.os_time).values, 0.05, None)
    sub["ipcw"] = 1.0 / g_pred
    sub["event_3y"] = ((sub.os_time <= horizon) & (sub.os_event == 1)).astype(float)
    thresh = np.linspace(0.02, 0.7, 50)
    nb_model, nb_all = [], []
    for pt in thresh:
        hi = (sub.p_risk_3y >= pt).astype(float)
        nb_model.append(float(np.mean(sub.ipcw * hi * (sub.event_3y - pt / (1 - pt) * (1 - sub.event_3y)))))
        nb_all.append(float(np.mean(sub.ipcw * (sub.event_3y - pt / (1 - pt) * (1 - sub.event_3y)))))
    best = max(range(len(thresh)), key=lambda i: nb_model[i])
    return {"threshold_best": round(float(thresh[best]), 3),
            "nb_best": round(float(nb_model[best]), 4),
            "nb_treat_all_at_best": round(float(nb_all[best]), 4)}


def fdr_by_cohort(df):
    rows = []
    for cohort in ["Lung1", "RG"]:
        d = df[df.cohort == cohort].copy()
        for col, lab in BC_METRICS:
            if col not in d.columns:
                continue
            sub = d[["os_time", "os_event", col]].dropna()
            if len(sub) < 30 or sub[col].nunique() < 3 or sub.os_event.sum() < 10:
                continue
            cph = CoxPHFitter()
            cph.fit(sub, duration_col="os_time", event_col="os_event")
            hr = float(cph.hazard_ratios_[col])
            p = float(cph.summary.loc[col, "p"])
            lo, hi = np.exp(cph.confidence_intervals_.loc[col, "95% lower-bound"]), np.exp(cph.confidence_intervals_.loc[col, "95% upper-bound"])
            rows.append({"cohort": cohort, "metric": col, "label": lab,
                         "n": len(sub), "events": int(sub.os_event.sum()),
                         "HR": hr, "p": p,
                         "CI95_lo": float(lo), "CI95_hi": float(hi)})
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    for cohort in out.cohort.unique():
        m = out.cohort == cohort
        pvals = out.loc[m, "p"].values
        n = len(pvals)
        if n <= 1:
            continue
        order = np.argsort(pvals)
        p_fdr = np.empty(n)
        running = 1.0
        for i in range(n - 1, -1, -1):
            idx = order[i]
            running = min(running, pvals[idx] * n / (i + 1))
            p_fdr[idx] = running
        out.loc[m, "p_fdr"] = p_fdr
    out["significant_fdr005"] = out["p_fdr"] < 0.05
    out.to_csv(os.path.join(OUT, "fdr_os_associations.csv"), index=False)
    return out


def median_followup(df):
    res = {}
    for cohort in ["Lung1", "RG"]:
        d = df[df.cohort == cohort].dropna(subset=["os_time", "os_event"])
        cens = (d.os_event == 0).astype(int)
        kmc = KaplanMeierFitter()
        kmc.fit(d.os_time, cens)
        med = float(kmc.median_survival_time_) if kmc.median_survival_time_ == kmc.median_survival_time_ else None
        res[f"{cohort}_os"] = {"n": len(d), "events": int(d.os_event.sum()),
                               "median_followup_days": med,
                               "median_followup_years": round(med / 365, 2) if med else None}
        if cohort == "RG":
            d2 = df[df.cohort == cohort].dropna(subset=["rfs_time", "rfs_event"])
            cens2 = (d2.rfs_event == 0).astype(int)
            kmc2 = KaplanMeierFitter()
            kmc2.fit(d2.rfs_time, cens2)
            med2 = float(kmc2.median_survival_time_) if kmc2.median_survival_time_ == kmc2.median_survival_time_ else None
            res[f"{cohort}_rfs"] = {"n": len(d2), "events": int(d2.rfs_event.sum()),
                                    "median_followup_days": med2,
                                    "median_followup_years": round(med2 / 365, 2) if med2 else None}
    return res


def qc_summary(df):
    res = {}
    for cohort in ["Lung1", "RG"]:
        d = df[df.cohort == cohort]
        res[cohort] = {
            "n": int(len(d)),
            "L1_unavailable_T12fallback": {"n": int(d.L1_unavailable.sum()),
                                           "pct": round(100 * d.L1_unavailable.mean(), 1)},
            "L1_edge": {"n": int(d.L1_edge.fillna(0).sum()),
                        "pct": round(100 * d.L1_edge.fillna(0).mean(), 1)},
            "T12_edge": {"n": int(d.T12_edge.fillna(0).sum()),
                         "pct": round(100 * d.T12_edge.fillna(0).mean(), 1)},
            "coverage_extended": int(d.coverage_extended.fillna(0).sum()),
            "SMA_vol_missing": int(d.SMA_vol_cm3.isna().sum()),
            "SMA_L1_missing": int(d.SMA_L1_cm2.isna().sum()),
            "SAT_vol_missing": int(d.SAT_vol_cm3.isna().sum()),
            "VAT_vol_missing": int(d.VAT_vol_cm3.isna().sum()),
            "SM_mean_hu_missing": int(d.SM_mean_hu.isna().sum()),
            "L3_available": int(d.SMA_L3_cm2.notna().sum()),
            "T4_available": int(d.SMA_T4_cm2.notna().sum()),
            "age_missing": int(d.age.isna().sum()),
        }
    return res


def chemo_strat(df):
    d = df[df.cohort == "RG"].copy()
    d, lb = make_binary(d, "log_SMA_vol_cm3", q=1 / 3)
    d["chemo"] = (d["rg_chemotherapy"] == "Yes").astype(int)
    out = {"chemo_yes_n": int((d.chemo == 1).sum()),
           "chemo_no_n": int((d.chemo == 0).sum())}
    for name, sub in [("chemo_yes", d[d.chemo == 1]), ("chemo_no", d[d.chemo == 0])]:
        cph, s = cox_fit(sub, lb, ["age", "sex_m"])
        hr = float(cph.hazard_ratios_[lb])
        p = float(cph.summary.loc[lb, "p"])
        lo, hi = np.exp(cph.confidence_intervals_.loc[lb, "95% lower-bound"]), np.exp(cph.confidence_intervals_.loc[lb, "95% upper-bound"])
        out[name] = {"n": len(s), "events": int(s.os_event.sum()),
                     "low_muscle_n": int(s[lb].sum()), "HR": round(hr, 3),
                     "p": round(p, 4), "CI95": [round(float(lo), 3), round(float(hi), 3)]}
    cols = ["os_time", "os_event", lb, "chemo", "age", "sex_m"]
    s = d[cols].dropna()
    keep = [c for c in cols if s[c].nunique() > 1]
    s = s[keep]
    s["muscle_x_chemo"] = s[lb] * s["chemo"]
    try:
        cph = CoxPHFitter()
        cph.fit(s, duration_col="os_time", event_col="os_event")
        out["interaction"] = {
            "n": len(s), "events": int(s.os_event.sum()),
            "HR_low_muscle": round(float(cph.hazard_ratios_[lb]), 3),
            "p_low_muscle": round(float(cph.summary.loc[lb, "p"]), 4),
            "HR_chemo": round(float(cph.hazard_ratios_["chemo"]), 3),
            "p_chemo": round(float(cph.summary.loc["chemo", "p"]), 4),
            "HR_interaction": round(float(cph.hazard_ratios_["muscle_x_chemo"]), 3),
            "p_interaction": round(float(cph.summary.loc["muscle_x_chemo", "p"]), 4)}
    except Exception as e:
        out["interaction"] = {"error": str(e)}
    return out


def merged_sarc_obese(df):
    d = df.copy()
    d["cohort_num"] = (d["cohort"] == "RG").astype(int)
    for cohort in ["Lung1", "RG"]:
        m = d.cohort == cohort
        d.loc[m, "log_SMA_vol_cm3"] = np.log(d.loc[m, "SMA_vol_cm3"].clip(lower=1))
        d.loc[m, "log_VAT_vol_cm3"] = np.log(d.loc[m, "VAT_vol_cm3"].clip(lower=1))
        d.loc[m, "log_SAT_vol_cm3"] = np.log(d.loc[m, "SAT_vol_cm3"].clip(lower=1))
        for col, q in [("log_SMA_vol_cm3", 1 / 3), ("log_VAT_vol_cm3", 2 / 3), ("log_SAT_vol_cm3", 1 / 3)]:
            for s in [0, 1]:
                sub = d[(d.cohort == cohort) & (d.sex_m == s)][col].dropna()
                if len(sub) >= 30:
                    d.loc[(d.cohort == cohort) & (d.sex_m == s) & d[col].notna(), col + "_q"] = \
                        (d.loc[(d.cohort == cohort) & (d.sex_m == s) & d[col].notna(), col] < sub.quantile(q)).astype(int)
    d["low_muscle"] = d["log_SMA_vol_cm3_q"]
    d["high_vat"] = (d["log_VAT_vol_cm3_q"] == 0).astype(float)
    d["low_sat"] = d["log_SAT_vol_cm3_q"]
    d["sarc_obese"] = ((d.low_muscle == 1) & (d.high_vat == 1)).astype(int)
    d["cachexia"] = ((d.low_muscle == 1) & (d.low_sat == 1)).astype(int)
    res = {"n": int(len(d)), "sarc_obese_n": int(d.sarc_obese.sum()),
           "cachexia_n": int(d.cachexia.sum())}
    for pheno in ["sarc_obese", "cachexia"]:
        r = full_adjusted_model(d, pheno, ["age", "sex_m", "cohort_num"])
        res[pheno] = r
        res[pheno + "_by_cohort"] = {c: int((d[d.cohort == c][pheno] == 1).sum())
                                     for c in ["Lung1", "RG"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pheno, title in [(axes[0], "sarc_obese", "Sarcopenic obesity"), (axes[1], "cachexia", "Cachexia-like")]:
        d2 = d.dropna(subset=["os_time", "os_event", pheno])
        m0, m1 = d2[pheno] == 0, d2[pheno] == 1
        kmf0, kmf1 = KaplanMeierFitter(), KaplanMeierFitter()
        kmf0.fit(d2.os_time[m0], d2.os_event[m0], label="others")
        kmf1.fit(d2.os_time[m1], d2.os_event[m1], label="phenotype")
        kmf0.plot_survival_function(ax=ax)
        kmf1.plot_survival_function(ax=ax)
        lr = logrank_test(d2.os_time[m0], d2.os_time[m1], d2.os_event[m0], d2.os_event[m1])
        ax.set_title("Merged OS by %s\nlogrank p=%.4f" % (title, lr.p_value))
        ax.set_xlabel("OS (days)")
        ax.set_ylabel("Survival probability")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "km_merged_phenotypes.png"), dpi=150)
    plt.close(fig)
    return res


def main():
    df = prep()
    print("merged df: n=%d" % len(df))
    results = {}

    # A. RG supplementary analyses
    rg = df[df.cohort == "RG"].copy()
    print("RG n=%d, OS events=%d" % (len(rg), rg.os_event.sum()))
    d_rg, lb_rg = build_phenotypes(rg)
    results["rg_ecr"] = {"cohort": "RG", "adjustment": ADJ_RG}
    for pheno in ["pheno_sarc_obese", "pheno_cachexia", "pheno_low_muscle_only"]:
        results["rg_ecr"]["full_model_" + pheno] = full_adjusted_model(d_rg, pheno, ADJ_RG)
        print("[A1] RG %s: %s" % (pheno, results["rg_ecr"]["full_model_" + pheno]))
    results["rg_ecr"]["bootstrap"] = {}
    for pheno in ["pheno_sarc_obese", "pheno_cachexia"]:
        results["rg_ecr"]["bootstrap"][pheno] = bootstrap_hr(d_rg, pheno, B=1000, adj_cols=ADJ_RG)
        print("[A2] RG bootstrap %s: %s" % (pheno, results["rg_ecr"]["bootstrap"][pheno]))
    results["rg_ecr"]["ph_check"] = {}
    for pheno in ["pheno_sarc_obese", "pheno_cachexia"]:
        results["rg_ecr"]["ph_check"][pheno] = check_ph(d_rg, pheno, ADJ_RG)
    results["rg_ecr"]["cindex"] = cindex_increment(d_rg, lb_rg, ADJ_RG)
    print("[A3] RG C-index: %s" % results["rg_ecr"]["cindex"])
    results["rg_ecr"]["km_diff"] = {}
    for pheno in ["pheno_sarc_obese", "pheno_cachexia"]:
        rows, lrp = km_absolute_diff(d_rg, pheno)
        results["rg_ecr"]["km_diff"][pheno] = {"rows": rows, "logrank_p": round(lrp, 4)}
        print("[A4] RG KM %s: %s (logrank p=%.4f)" % (pheno, rows, lrp))
    results["rg_ecr"]["dca"] = {}
    for pheno in ["pheno_sarc_obese", "pheno_cachexia"]:
        results["rg_ecr"]["dca"][pheno] = dca_3y(d_rg, pheno, ADJ_RG)
        print("[A5] RG DCA %s: %s" % (pheno, results["rg_ecr"]["dca"][pheno]))

    # B. FDR
    results["fdr"] = fdr_by_cohort(df).to_dict(orient="records")
    print("[B] FDR table rows=%d" % len(results["fdr"]))

    # C. median follow-up
    results["median_followup"] = median_followup(df)
    print("[C] %s" % json.dumps(results["median_followup"], ensure_ascii=False))

    # D. QC summary
    results["qc"] = qc_summary(df)
    print("[D] %s" % json.dumps(results["qc"], ensure_ascii=False))

    # E. chemotherapy stratification
    results["chemo_strat"] = chemo_strat(df)
    print("[E] %s" % json.dumps(results["chemo_strat"], ensure_ascii=False))

    # F. pooled cohort
    results["merged_sarc_obese"] = merged_sarc_obese(df)
    print("[F] %s" % json.dumps(results["merged_sarc_obese"], ensure_ascii=False))

    with open(os.path.join(OUT, "supplementary_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved: %s" % OUT)


if __name__ == "__main__":
    main()
