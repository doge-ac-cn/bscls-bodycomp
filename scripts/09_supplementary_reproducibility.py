#!/usr/bin/env python3
"""Reproducibility supplement: dynamic computation of every manuscript number.

Computes all numbers reported in the manuscript that are not already produced
by scripts 01-08, so the full results section can be regenerated from the two
data tables alone (data/bodycomp_features.csv + data/clinical_master.csv).

Outputs (under outputs/supplementary/):
  reproducibility_results.json   all sections below
  Table3_phenotypes.csv          rows for Table 3 (read by 08)
  Table4_stage_os.csv            rows for Table 4 (read by 08)

Sections:
  A. Continuous per-SD HR for the primary muscle metric (Lung1/RG/Pooled)
  B. Stage-adjusted sensitivity (Lung1 primary tertile)
  C. Per-tertile dose-response trend (Lung1)
  D. Cohort x muscle interaction (pooled)
  E. Lung1 phenotype models (cachexia / sarcopenic obesity / low-muscle-only)
  F. Lung1 phenotype bootstrap stability + 5-y OS + NNH
  G. Stage-stratified analyses (Table 4; full-cohort cutoffs)
  H. Cutoff robustness: quartile cutoff (Lung1)
  I. Stage-stratified Cox models with strata=stage
  J. C-index optimism-corrected increment (Lung1/Pooled, B=400)
  K. 3-year IPCW Brier score (Lung1/Pooled)
  L. Decision-curve analysis (Lung1 full + early I-II + bootstrap)
  M. Coverage sensitivity + T4 common-landmark analysis
  N. RG pack-years sensitivity
  O. Multiple imputation for missing age (Lung1, M=10, Rubin)
  P. EGFR mutant vs wild-type body composition (RG, age/sex adjusted)
  Q. Phenotype-family BH-FDR (12 tests)

Small-sample safety: every section degrades gracefully (returns None / skips)
when the available data cannot support the analysis, so the script also runs
end-to-end on the synthetic demo data used by CI.
"""
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy import stats as sstats
from scipy.stats import mannwhitneyu

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "supplementary")
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

ADJ_L1 = ["age", "sex_m", "stage_n"]
ADJ_RG = ["age", "sex_m"]
ADJ_POOL = ["age", "sex_m", "cohort_num"]


def prep():
    feat = pd.read_csv(os.path.join(BASE, "data", "bodycomp_features.csv"))
    clin = pd.read_csv(os.path.join(BASE, "data", "clinical_master.csv"))
    df = feat.merge(clin, on="patient_id", how="inner")
    df["sex_m"] = (df["sex"] == "M").astype(int)
    df["cohort_num"] = (df["cohort"] == "RG").astype(int)
    df["stage_n"] = df["stage"].map({"I": 1, "II": 2, "III": 3, "IV": 4})
    for c in ["log_SMA_vol_cm3", "log_SAT_vol_cm3", "log_VAT_vol_cm3"]:
        base = c.replace("log_", "")
        df[c] = np.log(df[base].clip(lower=1))
    df["SMA_primary_cm2"] = df["SMA_L1_cm2"].fillna(df["SMA_T12_cm2"])
    df["L1_unavailable"] = df["L1_unavailable"].fillna(0)
    df["coverage_extended"] = df["coverage_extended"].fillna(0)
    df["pack_years"] = pd.to_numeric(df.get("pack_years"), errors="coerce")
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
    d.loc[d[col].isna(), label] = np.nan
    return d, label


def build_phenotypes(d):
    d2, lb = make_binary(d.copy(), "log_SMA_vol_cm3", q=1 / 3)
    d3, _ = make_binary(d2, "log_VAT_vol_cm3", q=2 / 3)
    d4, ls = make_binary(d3, "log_SAT_vol_cm3", q=1 / 3)
    # make_binary uses "< quantile", so the VAT column equals 1 when VAT is
    # below the 2/3 quantile (i.e. NOT high VAT) and 0 when VAT >= 2/3.
    vat_low_mid = [c for c in d4.columns if c.startswith("log_VAT_vol_cm3_low")][0]
    d4["pheno_sarc_obese"] = ((d4[lb] == 1) & (d4[vat_low_mid] == 0)).astype(int)
    d4["pheno_cachexia"] = ((d4[lb] == 1) & (d4[ls] == 1)).astype(int)
    d4["pheno_low_muscle_only"] = ((d4[lb] == 1) & (d4[ls] == 0) & (d4[vat_low_mid] == 1)).astype(int)
    return d4, lb


def build_phenotypes_pooled(d):
    """Pooled-cohort phenotype labels: cutoffs computed WITHIN each cohort,
    labels concatenated (merged-cohort cutoffs are invalid)."""
    parts = []
    for cname in ["Lung1", "RG"]:
        sub = d[d.cohort == cname]
        dd, _ = build_phenotypes(sub)
        parts.append(dd[["patient_id", "pheno_cachexia", "pheno_sarc_obese",
                         "pheno_low_muscle_only", "log_SMA_vol_cm3_low0.3333333333333333"]])
    merged = parts[0].merge(parts[1], on="patient_id", how="outer", suffixes=("", "_rg"))
    for col in ["pheno_cachexia", "pheno_sarc_obese", "pheno_low_muscle_only",
                "log_SMA_vol_cm3_low0.3333333333333333"]:
        rg_col = col + "_rg"
        if rg_col in merged.columns:
            merged[col] = merged[col].fillna(merged[rg_col])
            merged.drop(columns=[rg_col], inplace=True)
    out = d.merge(merged, on="patient_id", how="left")
    return out, "log_SMA_vol_cm3_low0.3333333333333333"


def cox_fit(sub, label, covs):
    cols = ["os_time", "os_event", label] + list(covs)
    sub = sub[cols].dropna()
    keep = [c for c in cols if sub[c].nunique() > 1]
    sub = sub[keep]
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="os_time", event_col="os_event")
    return cph, sub


def hr_row(cph, lab):
    lo, hi = np.exp(cph.confidence_intervals_.loc[lab, "95% lower-bound"]), np.exp(
        cph.confidence_intervals_.loc[lab, "95% upper-bound"])
    n_ev = cph.event_observed
    return {"label": lab, "n": int(len(n_ev)), "events": int(n_ev.sum()),
            "HR": float(cph.hazard_ratios_[lab]), "p": float(cph.summary.loc[lab, "p"]),
            "CI95": [float(lo), float(hi)], "c_index": float(cph.concordance_index_)}


# ----------------------------------------------------------------------------
# A. Continuous per-SD HR (primary muscle metric)
# ----------------------------------------------------------------------------
def sec_A(df):
    out = {"Lung1": None, "RG": None, "Pooled": None}
    for cname, sub, covs in [("Lung1", df[df.cohort == "Lung1"], ADJ_L1),
                             ("RG", df[df.cohort == "RG"], ADJ_RG)]:
        tmp = sub[["os_time", "os_event", "log_SMA_vol_cm3"] + covs].dropna()
        if len(tmp) < 30 or tmp["log_SMA_vol_cm3"].nunique() < 5:
            continue
        cph = CoxPHFitter()
        cph.fit(tmp, duration_col="os_time", event_col="os_event")
        sd = tmp["log_SMA_vol_cm3"].std()
        coef = float(cph.summary.loc["log_SMA_vol_cm3", "coef"]) * sd
        se = float(cph.summary.loc["log_SMA_vol_cm3", "se(coef)"]) * sd
        hr = float(np.exp(coef))
        ci = [float(np.exp(coef - 1.96 * se)), float(np.exp(coef + 1.96 * se))]
        p = float(2 * (1 - sstats.norm.cdf(abs(coef / se))))
        out[cname] = {"metric": "log_SMA_vol_cm3", "per_SD": float(sd), "n": int(tmp.shape[0]),
                      "events": int(tmp["os_event"].sum()), "HR_per_SD": hr,
                      "CI95": ci, "p": p}
    tmp = df[["os_time", "os_event", "log_SMA_vol_cm3", "cohort_num", "age", "sex_m"]].dropna()
    if len(tmp) >= 30 and tmp["log_SMA_vol_cm3"].nunique() >= 5:
        cph = CoxPHFitter()
        cph.fit(tmp, duration_col="os_time", event_col="os_event")
        sd = tmp["log_SMA_vol_cm3"].std()
        coef = float(cph.summary.loc["log_SMA_vol_cm3", "coef"]) * sd
        se = float(cph.summary.loc["log_SMA_vol_cm3", "se(coef)"]) * sd
        hr = float(np.exp(coef))
        ci = [float(np.exp(coef - 1.96 * se)), float(np.exp(coef + 1.96 * se))]
        p = float(2 * (1 - sstats.norm.cdf(abs(coef / se))))
        out["Pooled"] = {"metric": "log_SMA_vol_cm3", "per_SD": float(sd), "n": int(tmp.shape[0]),
                         "events": int(tmp["os_event"].sum()), "HR_per_SD": hr,
                         "CI95": ci, "p": p}
    return out


# ----------------------------------------------------------------------------
# B. Stage-adjusted sensitivity (Lung1 primary tertile)
# ----------------------------------------------------------------------------
def sec_B(df):
    lung1 = df[df.cohort == "Lung1"].copy()
    d, lb = build_phenotypes(lung1)
    try:
        cph, _ = cox_fit(d, lb, ADJ_L1)
        return hr_row(cph, lb)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# C. Per-tertile dose-response trend (Lung1)
# ----------------------------------------------------------------------------
def sec_C(df):
    """Per-tertile dose-response trend (Lung1), unadjusted; within-sex tertile
    labels from the sex-stratified 1/3, 2/3 quantile boundaries (T=1,2,3)
    entered as a continuous term."""
    lung1 = df[df.cohort == "Lung1"].copy()
    d2 = lung1.dropna(subset=["log_SMA_vol_cm3"]).copy()
    d2["T"] = np.nan
    for s in [0, 1]:
        sub = d2[d2.sex_m == s]
        if len(sub) < 30:
            continue
        q1, q2 = sub["log_SMA_vol_cm3"].quantile([1 / 3, 2 / 3])
        m = d2.sex_m == s
        d2.loc[m & (d2["log_SMA_vol_cm3"] < q1), "T"] = 1
        d2.loc[m & (d2["log_SMA_vol_cm3"] >= q1) & (d2["log_SMA_vol_cm3"] < q2), "T"] = 2
        d2.loc[m & (d2["log_SMA_vol_cm3"] >= q2), "T"] = 3
    sub = d2[["os_time", "os_event", "T"]].dropna()
    if len(sub) < 30 or sub["T"].nunique() < 2:
        return None
    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col="os_time", event_col="os_event")
        return hr_row(cph, "T")
    except Exception:
        return None


# ----------------------------------------------------------------------------
# D. Cohort x muscle interaction (pooled)
# ----------------------------------------------------------------------------
def sec_D(df):
    d_all, lb = build_phenotypes_pooled(df)
    d_all["muscle_x_rg"] = d_all[lb] * d_all["cohort_num"]
    cols = ["os_time", "os_event", lb, "muscle_x_rg", "cohort_num"] + ["age", "sex_m"]
    sub = d_all[cols].dropna()
    if len(sub) < 60 or sub[lb].nunique() < 2 or sub["muscle_x_rg"].nunique() < 2:
        return None
    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col="os_time", event_col="os_event")
        return {"n": int(sub.shape[0]), "events": int(sub.os_event.sum()),
                "muscle_HR": float(cph.hazard_ratios_[lb]),
                "muscle_p": float(cph.summary.loc[lb, "p"]),
                "interaction_HR": float(cph.hazard_ratios_["muscle_x_rg"]),
                "interaction_p": float(cph.summary.loc["muscle_x_rg", "p"]),
                "cohort_HR": float(cph.hazard_ratios_["cohort_num"]),
                "cohort_p": float(cph.summary.loc["cohort_num", "p"])}
    except Exception:
        return None


# ----------------------------------------------------------------------------
# E. Lung1 phenotype models
# ----------------------------------------------------------------------------
PHENO_LABELS = ["pheno_cachexia", "pheno_sarc_obese", "pheno_low_muscle_only"]


def sec_E(df):
    lung1 = df[df.cohort == "Lung1"].copy()
    d, _ = build_phenotypes(lung1)
    out = {}
    for lab in PHENO_LABELS:
        try:
            cph, _ = cox_fit(d, lab, ADJ_L1)
            out[lab] = hr_row(cph, lab)
        except Exception:
            out[lab] = None
    return out


# ----------------------------------------------------------------------------
# F. Lung1 bootstrap stability + 5-y OS + NNH (cachexia-like)
# ----------------------------------------------------------------------------
def bootstrap_hr(d, label, B=1000, adj_cols=ADJ_L1):
    sub = d[["os_time", "os_event", label] + list(adj_cols)].dropna()
    if len(sub) < 40 or sub[label].sum() < 5:
        return None
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
    return {"B_ok": int(len(hrs)), "HR_median": round(float(np.median(hrs)), 3),
            "CI95_percentile": [round(float(np.percentile(hrs, 2.5)), 3),
                                round(float(np.percentile(hrs, 97.5)), 3)],
            "prop_hr_gt1": round(float((hrs > 1).mean()), 3),
            "fail_rate": round(1 - len(hrs) / B, 3)}


def km_absolute_diff(d, label, years=(5,), t0=1825):
    d2 = d.dropna(subset=["os_time", "os_event", label])
    m0, m1 = d2[label] == 0, d2[label] == 1
    if m0.sum() < 10 or m1.sum() < 3:
        return None, None
    kmf0, kmf1 = KaplanMeierFitter(), KaplanMeierFitter()
    kmf0.fit(d2.os_time[m0], d2.os_event[m0], label="normal")
    kmf1.fit(d2.os_time[m1], d2.os_event[m1], label="phenotype")
    lr = logrank_test(d2.os_time[m0], d2.os_time[m1], d2.os_event[m0], d2.os_event[m1])
    rows = []
    for yr in years:
        t = yr * 365.25
        s0 = float(kmf0.predict(t))
        s1 = float(kmf1.predict(t))
        diff = s1 - s0
        nnh = abs(1.0 / diff) if abs(diff) > 1e-9 else None
        rows.append({"time_yr": yr, "survival_normal": round(s0, 3),
                     "survival_pheno": round(s1, 3),
                     "absolute_diff_pp": round(diff * 100, 1),
                     "nnh": round(nnh, 1) if nnh else None})
    return rows, float(lr.p_value)


def sec_F(df):
    lung1 = df[df.cohort == "Lung1"].copy()
    d, _ = build_phenotypes(lung1)
    out = {"bootstrap": {}, "km": {}, "nnh": {}}
    for lab in ["pheno_cachexia"]:
        np.random.seed(42)  # reproducible bootstrap (independent of other sections)
        out["bootstrap"][lab] = bootstrap_hr(d, lab, B=1000, adj_cols=ADJ_L1)
        rows, lrp = km_absolute_diff(d, lab, years=(5,))
        out["km"][lab] = {"rows": rows, "logrank_p": round(lrp, 4) if lrp is not None else None}
        if rows:
            r5 = rows[0]
            out["nnh"][lab] = {"nnh_5y": r5["nnh"], "abs_diff_pp": r5["absolute_diff_pp"]}
        else:
            out["nnh"][lab] = None
    return out


# ----------------------------------------------------------------------------
# G. Stage-stratified analyses (Table 4; full-cohort cutoffs)
# ----------------------------------------------------------------------------
def sec_G(df):
    lung1 = df[df.cohort == "Lung1"].copy()
    d, lb = build_phenotypes(lung1)  # full-cohort cutoffs, then split by stage
    out = {"low_muscle": {}, "cachexia": {}, "interaction": None}
    for name, sel in [("stage_I_II", d.stage_n <= 2), ("stage_III", d.stage_n == 3)]:
        sub = d[sel].copy()
        if len(sub) < 30 or sub[lb].nunique() < 2:
            out["low_muscle"][name] = None
            out["cachexia"][name] = None
            continue
        for key, lab in [("low_muscle", lb), ("cachexia", "pheno_cachexia")]:
            try:
                cph, _ = cox_fit(sub, lab, ["age", "sex_m"])
                out[key][name] = hr_row(cph, lab)
            except Exception:
                out[key][name] = None
        # KM for cachexia (Table 4)
        rows, lrp = km_absolute_diff(sub, "pheno_cachexia", years=(5,))
        out["cachexia"][name + "_km"] = {"rows": rows, "logrank_p": round(lrp, 4) if lrp is not None else None}
    # stage x muscle interaction (full cohort; low muscle x stage I-II)
    d["stage_I_II"] = (d.stage_n <= 2).astype(int)
    d["muscle_x_stage12"] = d[lb] * d["stage_I_II"]
    cols = ["os_time", "os_event", lb, "muscle_x_stage12", "stage_I_II"] + ["age", "sex_m"]
    sub = d[cols].dropna()
    if len(sub) >= 60 and sub["muscle_x_stage12"].nunique() >= 2:
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col="os_time", event_col="os_event")
            out["interaction"] = {"n": int(sub.shape[0]), "events": int(sub.os_event.sum()),
                                  "muscle_HR": float(cph.hazard_ratios_[lb]),
                                  "muscle_p": float(cph.summary.loc[lb, "p"]),
                                  "interaction_HR": float(cph.hazard_ratios_["muscle_x_stage12"]),
                                  "interaction_p": float(cph.summary.loc["muscle_x_stage12", "p"]),
                                  "stage_I_II_HR": float(cph.hazard_ratios_["stage_I_II"]),
                                  "stage_I_II_p": float(cph.summary.loc["stage_I_II", "p"])}
        except Exception:
            out["interaction"] = None
    return out


# ----------------------------------------------------------------------------
# H. Cutoff robustness: quartile cutoff (Lung1)
# ----------------------------------------------------------------------------
def sec_H(df):
    lung1 = df[df.cohort == "Lung1"].copy()
    d, lbq = make_binary(lung1, "log_SMA_vol_cm3", q=1 / 4)
    try:
        cph, _ = cox_fit(d, lbq, ["age", "sex_m"])
        return hr_row(cph, lbq)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# I. Stage-stratified Cox models with strata=stage (Lung1 phenotypes)
# ----------------------------------------------------------------------------
def sec_I(df):
    lung1 = df[df.cohort == "Lung1"].copy()
    d, _ = build_phenotypes(lung1)
    out = {}
    for lab in ["pheno_cachexia", "pheno_sarc_obese"]:
        cols = ["os_time", "os_event", lab, "age", "sex_m", "stage_n"]
        sub = d[cols].dropna()
        if len(sub) < 40 or sub[lab].nunique() < 2 or sub["stage_n"].nunique() < 2:
            out[lab] = None
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col="os_time", event_col="os_event", strata=["stage_n"])
            out[lab] = hr_row(cph, lab)
        except Exception:
            out[lab] = None
    return out


# ----------------------------------------------------------------------------
# J. C-index bootstrap + optimism correction (B=400)
# ----------------------------------------------------------------------------
def sec_J(df, B=400):
    out = {}
    np.random.seed(42)  # reproducible bootstrap (matches the manuscript run)
    for name, sub, covs, pooled in [
        ("Lung1", df[df.cohort == "Lung1"], ADJ_L1, False),
        ("Pooled", df, ADJ_POOL, True),
    ]:
        d, lb = (build_phenotypes_pooled(sub) if pooled else build_phenotypes(sub))
        cols = ["os_time", "os_event"] + covs + [lb]
        dat = d[cols].dropna()
        if len(dat) < 40 or dat[lb].nunique() < 2:
            out[name] = None
            continue
        risk_cols = covs
        try:
            cph_b = CoxPHFitter().fit(dat, duration_col="os_time", event_col="os_event",
                                      formula=" + ".join(risk_cols))
            c_b = cph_b.concordance_index_
            cph_f = CoxPHFitter().fit(dat, duration_col="os_time", event_col="os_event",
                                      formula=lb + " + " + " + ".join(risk_cols))
            c_f = cph_f.concordance_index_
        except Exception:
            out[name] = None
            continue
        apparent_delta = c_f - c_b
        deltas, optims = [], []
        for b in range(B):
            idx = np.random.choice(dat.index, size=len(dat), replace=True)
            boot = dat.loc[idx]
            try:
                cph_boot_b = CoxPHFitter().fit(boot, duration_col="os_time", event_col="os_event",
                                               formula=" + ".join(risk_cols))
                cph_boot_f = CoxPHFitter().fit(boot, duration_col="os_time", event_col="os_event",
                                               formula=lb + " + " + " + ".join(risk_cols))
                risk_b = -cph_boot_b.predict_partial_hazard(dat)
                risk_f = -cph_boot_f.predict_partial_hazard(dat)
                ci_b_test = concordance_index(dat["os_time"], risk_b, dat["os_event"])
                ci_f_test = concordance_index(dat["os_time"], risk_f, dat["os_event"])
                ci_b_app = concordance_index(boot["os_time"], -cph_boot_b.predict_partial_hazard(boot), boot["os_event"])
                ci_f_app = concordance_index(boot["os_time"], -cph_boot_f.predict_partial_hazard(boot), boot["os_event"])
                delta_test = ci_f_test - ci_b_test
                optim = (ci_f_app - ci_b_app) - delta_test
                deltas.append(delta_test)
                optims.append(optim)
            except Exception:
                continue
        if len(deltas) < 100:
            out[name] = None
            continue
        deltas = np.array(deltas)
        optims = np.array(optims)
        opt_corr = apparent_delta - optims.mean()
        out[name] = {"n": int(dat.shape[0]), "events": int(dat["os_event"].sum()),
                     "c_clinical": float(c_b), "c_clinical_plus_muscle": float(c_f),
                     "apparent_delta": float(apparent_delta),
                     "optimism_corrected_delta": float(opt_corr),
                     "delta_CI95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                     "B_ok": int(len(deltas)),
                     "c_clinical_plus_muscle_optimism_corrected": float(c_f - optims.mean())}
    return out


# ----------------------------------------------------------------------------
# K. 3-year IPCW Brier score (Lung1/Pooled)
# ----------------------------------------------------------------------------
def sec_K(df):
    t3 = 3 * 365.25
    out = {}
    for name, sub, covs, pooled in [
        ("Lung1", df[df.cohort == "Lung1"], ADJ_L1, False),
        ("Pooled", df, ADJ_POOL, True),
    ]:
        d, lb = (build_phenotypes_pooled(sub) if pooled else build_phenotypes(sub))
        cols = ["os_time", "os_event"] + covs + [lb]
        dat = d[cols].dropna()
        if len(dat) < 40 or dat[lb].nunique() < 2:
            out[name] = None
            continue
        try:
            kmf_c = KaplanMeierFitter().fit(dat["os_time"], dat["os_event"] == 0)
            g = kmf_c.predict(dat["os_time"].clip(upper=t3 - 1e-9)).values
            g_t = float(kmf_c.predict(t3))
            w = np.where(dat["os_time"] > t3, 1.0 / g_t, 1.0 / np.maximum(g, 1e-6))
            y = (dat["os_time"] > t3).astype(float)
            rows_b = []
            for label, fml in [("clinical", None), ("clinical_plus_muscle", lb + " + " + " + ".join(covs))]:
                cph = CoxPHFitter().fit(dat, duration_col="os_time", event_col="os_event",
                                        formula=fml if fml else " + ".join(covs))
                surv = cph.predict_survival_function(dat, times=[t3]).T.iloc[:, 0].values
                brier = float(np.mean(w * (y - surv) ** 2))
                rows_b.append({"model": label, "brier_3y": brier})
            out[name] = {"rows": rows_b,
                         "delta_brier": round(rows_b[1]["brier_3y"] - rows_b[0]["brier_3y"], 4)}
        except Exception:
            out[name] = None
    return out


# ----------------------------------------------------------------------------
# L. Decision-curve analysis (Lung1 full + early I-II + bootstrap)
# ----------------------------------------------------------------------------
def _nb_from_risk(sub, thresh, horizon):
    cens = (sub.os_event == 0).astype(int)
    kmc = KaplanMeierFitter()
    kmc.fit(sub.os_time, cens)
    g_pred = np.clip(kmc.predict(sub.os_time).values, 0.05, None)
    sub = sub.assign(ipcw=1.0 / g_pred,
                     event_3y=((sub.os_time <= horizon) & (sub.os_event == 1)).astype(float))
    rows = []
    for pt in thresh:
        hi = (sub.p_risk_3y >= pt).astype(float)
        nb_model = float(np.mean(sub.ipcw * hi * (sub.event_3y - pt / (1 - pt) * (1 - sub.event_3y))))
        nb_all = float(np.mean(sub.ipcw * (sub.event_3y - pt / (1 - pt) * (1 - sub.event_3y))))
        rows.append({"threshold": round(float(pt), 3), "nb_model": round(nb_model, 5),
                     "nb_treat_all": round(nb_all, 5)})
    return pd.DataFrame(rows)


def _fit_risk(d, cols, label=None, horizon=1095):
    cols = list(cols)
    if label is not None:
        cols = cols + [label]
    sub = d[["os_time", "os_event"] + cols].dropna()
    keep = [c for c in cols if sub[c].nunique() > 1]
    sub = sub[["os_time", "os_event"] + keep].copy()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="os_time", event_col="os_event")
    surv = cph.predict_survival_function(sub, times=[horizon]).T
    sub["p_risk_3y"] = 1 - surv.values[:, 0]
    sub["event_3y"] = ((sub.os_time <= horizon) & (sub.os_event == 1)).astype(float)
    return sub, keep


def dca_pair(d, label, covs, horizon=1095):
    sub_full, _ = _fit_risk(d, covs, label=label, horizon=horizon)
    sub_base, _ = _fit_risk(d, covs, label=None, horizon=horizon)
    thresh = np.linspace(0.02, 0.7, 69)
    curve_full = _nb_from_risk(sub_full, thresh, horizon)
    curve_base = _nb_from_risk(sub_base, thresh, horizon)
    curve_full["nb_increment_vs_clinical"] = (curve_full.nb_model - curve_base.nb_model).round(5)
    best = curve_full.loc[curve_full.nb_model.idxmax()]
    return curve_full, curve_base, {"n": len(sub_full), "events_3y": int(sub_full.event_3y.sum()),
                                    "threshold_best": float(best.threshold),
                                    "nb_best": float(best.nb_model),
                                    "nb_increment_best": float(best.nb_increment_vs_clinical)}


def sec_L(df, B_boot=500):
    out = {"full": {}, "early": {}, "bootstrap_early": {}}
    lung1 = df[df.cohort == "Lung1"].copy()
    for scope, sel, key in [("Lung1_full", df.cohort == "Lung1", "full"),
                            ("Lung1_early", (df.cohort == "Lung1") & (df.stage_n <= 2), "early")]:
        d0 = build_phenotypes(df[sel].copy())[0]
        if len(d0) < 30:
            out[key] = {"error": "insufficient n"}
            continue
        for lab in ["log_SMA_vol_cm3_low0.3333333333333333", "pheno_cachexia"]:
            try:
                curve, curve_base, meta = dca_pair(d0, lab, ["age", "sex_m", "stage_n"])
                key_rows = {}
                for pt in [0.2, 0.3, 0.4, 0.5, 0.6]:
                    r = curve[curve.threshold.round(2) == round(pt, 2)]
                    rb = curve_base[curve_base.threshold.round(2) == round(pt, 2)]
                    if len(r) and len(rb):
                        key_rows[str(pt)] = {"nb_increment": float(r.iloc[0].nb_increment_vs_clinical),
                                             "nb_model": float(r.iloc[0].nb_model),
                                             "nb_clinical": float(rb.iloc[0].nb_model)}
                out[key][lab] = {"n": meta["n"], "events_3y": meta["events_3y"],
                                 "threshold_best": meta["threshold_best"], "nb_best": meta["nb_best"],
                                 "nb_increment_best": meta["nb_increment_best"],
                                 "key_thresholds": key_rows}
            except Exception as e:
                out[key][lab] = {"error": str(e)}
    # bootstrap CI for early-stage increment at pt=0.5
    early = build_phenotypes(lung1[lung1.stage_n <= 2].copy())[0]
    if len(early) >= 30:
        np.random.seed(42)  # reproducible bootstrap (matches the manuscript run)
        incs = {lab: [] for lab in ["log_SMA_vol_cm3_low0.3333333333333333", "pheno_cachexia"]}
        for b in range(B_boot):
            idx            = np.random.choice(len(early), size=len(early), replace=True)
            boot = early.iloc[idx].reset_index(drop=True)
            for lab in incs:
                try:
                    curve, cb, meta = dca_pair(boot, lab, ["age", "sex_m", "stage_n"])
                    r = curve[curve.threshold.round(2) == 0.5]
                    if len(r):
                        incs[lab].append(float(r.iloc[0].nb_increment_vs_clinical))
                except Exception:
                    pass
        for lab, v in incs.items():
            v = np.array(v)
            if len(v) < 100:
                out["bootstrap_early"][lab] = None
                continue
            out["bootstrap_early"][lab] = {
                "B": int(len(v)), "mean_increment": round(float(v.mean()), 4),
                "CI95": [round(float(np.percentile(v, 2.5)), 4),
                         round(float(np.percentile(v, 97.5)), 4)],
                "prop_gt0": round(float((v > 0).mean()), 3)}
    return out


# ----------------------------------------------------------------------------
# M. Coverage sensitivity + T4 common-landmark analysis
# ----------------------------------------------------------------------------
def sec_M(df):
    out = {}
    d_all, lb = build_phenotypes_pooled(df)
    cols = ["os_time", "os_event", lb, "age", "sex_m", "cohort_num", "coverage_extended"]
    tmp = d_all[cols].dropna()
    if len(tmp) >= 60 and tmp[lb].nunique() >= 2:
        try:
            cph0 = CoxPHFitter().fit(tmp.drop(columns=["coverage_extended"]),
                                     duration_col="os_time", event_col="os_event")
            cph1 = CoxPHFitter().fit(tmp, duration_col="os_time", event_col="os_event")
            out["pooled_adj_coverage"] = {
                "HR_without_coverage": float(cph0.hazard_ratios_[lb]),
                "HR_with_coverage": float(cph1.hazard_ratios_[lb]),
                "coverage_HR": float(cph1.hazard_ratios_["coverage_extended"]),
                "coverage_p": float(cph1.summary.loc["coverage_extended", "p"])}
        except Exception:
            out["pooled_adj_coverage"] = None
    # T4 landmark
    t4_rows = []
    for cname, sub in [("Lung1", df[df.cohort == "Lung1"]), ("RG", df[df.cohort == "RG"])]:
        covs = ADJ_L1 if cname == "Lung1" else ADJ_RG
        sub = sub.dropna(subset=["SMA_T4_cm2"])
        if len(sub) < 30:
            continue
        d2, lb2 = make_binary(sub.copy(), "SMA_T4_cm2", q=1 / 3)
        try:
            cph, _ = cox_fit(d2, lb2, covs)
            r = hr_row(cph, lb2)
            r["cohort"] = cname
            t4_rows.append(r)
        except Exception:
            continue
    # pooled T4: within-cohort cutoffs concatenated
    t4_parts = []
    for cname, sub in [("Lung1", df[df.cohort == "Lung1"]), ("RG", df[df.cohort == "RG"])]:
        sub = sub.dropna(subset=["SMA_T4_cm2"])
        if len(sub) < 30:
            continue
        d2, lb2 = make_binary(sub.copy(), "SMA_T4_cm2", q=1 / 3)
        t4_parts.append(d2[["patient_id", lb2]])
    if len(t4_parts) == 2:
        t4_merge = t4_parts[0].merge(t4_parts[1], on="patient_id", how="outer", suffixes=("", "_rg"))
        lb3 = t4_parts[0].columns[1]
        if lb3 + "_rg" in t4_merge.columns:
            t4_merge[lb3] = t4_merge[lb3].fillna(t4_merge[lb3 + "_rg"])
            t4_merge.drop(columns=[lb3 + "_rg"], inplace=True)
        d3 = df.merge(t4_merge, on="patient_id", how="left")
        try:
            cph, _ = cox_fit(d3, lb3, ADJ_POOL)
            t4_pool = hr_row(cph, lb3)
            t4_pool["cohort"] = "Pooled"
            t4_rows.append(t4_pool)
        except Exception:
            pass
    out["T4_landmark"] = t4_rows
    out["coverage_counts"] = {
        "Lung1_non_extended": int(((df.cohort == "Lung1") & (df.coverage_extended == 0)).sum()),
        "RG_extended": int(((df.cohort == "RG") & (df.coverage_extended == 1)).sum()),
    }
    out["T4_availability"] = {
        "n_available": int(df["SMA_T4_cm2"].notna().sum()),
        "n_total": int(len(df)),
        "pct_available": round(100 * df["SMA_T4_cm2"].notna().mean(), 1),
    }
    return out


# ----------------------------------------------------------------------------
# N. RG pack-years sensitivity
# ----------------------------------------------------------------------------
def sec_N(df):
    rg = df[df.cohort == "RG"].copy()
    d3, lb3 = make_binary(rg, "log_SMA_vol_cm3", q=1 / 3)
    if len(rg) < 30:
        return None
    try:
        cph_agesex, sub_ax = cox_fit(d3, lb3, ["age", "sex_m"])
        cph_py, sub_py = cox_fit(d3, lb3, ["age", "sex_m", "pack_years"])
        return {"primary_age_sex": hr_row(cph_agesex, lb3),
                "plus_packyears": hr_row(cph_py, lb3),
                "packyears_n_available_in_model": int(sub_py.shape[0]),
                "packyears_HR": float(cph_py.hazard_ratios_["pack_years"]),
                "packyears_p": float(cph_py.summary.loc["pack_years", "p"])}
    except Exception:
        return None


# ----------------------------------------------------------------------------
# O. Multiple imputation for missing age (Lung1, M=10, Rubin)
# ----------------------------------------------------------------------------
def sec_O(df):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    lung1 = df[df.cohort == "Lung1"].copy()
    d, lb = build_phenotypes(lung1)
    mi_cols = ["age", "sex_m", "stage_n", "log_SMA_vol_cm3", "os_time", "os_event"]
    dat = d[mi_cols + [lb]].copy()
    dat["os_event"] = dat["os_event"].astype(float)
    n_missing = int(dat["age"].isna().sum())
    if n_missing == 0:
        return {"n_missing_age": 0, "note": "no missing age; MI not needed"}
    M = 10
    pooled = {c: [] for c in [lb, "age", "sex_m", "stage_n"]}
    n_used = []
    for m in range(M):
        imp = IterativeImputer(max_iter=20, random_state=42 + m, sample_posterior=True)
        arr = imp.fit_transform(dat[mi_cols])
        tmp = dat.copy()
        tmp["age"] = arr[:, 0]
        fit_dat = tmp[["os_time", "os_event", lb, "age", "sex_m", "stage_n"]].dropna()
        n_used.append(len(fit_dat))
        cph = CoxPHFitter().fit(fit_dat, duration_col="os_time", event_col="os_event")
        for c in pooled:
            pooled[c].append(float(cph.summary.loc[c, "coef"]))
    rubin = {}
    for c in pooled:
        coefs = np.array(pooled[c])
        m = len(coefs)
        Q = coefs.mean()
        U = 0.0
        for m_ in range(M):
            tmp = dat.copy()
            tmp["age"] = IterativeImputer(max_iter=20, random_state=42 + m_, sample_posterior=True).fit_transform(dat[mi_cols])[:, 0]
            fit_dat = tmp[["os_time", "os_event", lb, "age", "sex_m", "stage_n"]].dropna()
            cph = CoxPHFitter().fit(fit_dat, duration_col="os_time", event_col="os_event")
            U += float(cph.summary.loc[c, "se(coef)"]) ** 2
        U /= m
        Bm = coefs.var(ddof=1)
        T = U + (1 + 1 / m) * Bm
        se = np.sqrt(T)
        df_mi = (m - 1) * (1 + U / ((1 + 1 / m) * Bm)) ** 2
        p = 2 * (1 - sstats.t.cdf(abs(Q / se), df=df_mi))
        rubin[c] = {"coef": float(Q), "se": float(se), "HR": float(np.exp(Q)),
                    "CI95": [float(np.exp(Q - 1.96 * se)), float(np.exp(Q + 1.96 * se))], "p": float(p)}
    cc = dat.dropna(subset=["age"])
    cph_cc = CoxPHFitter().fit(cc[["os_time", "os_event", lb, "age", "sex_m", "stage_n"]].dropna(),
                               duration_col="os_time", event_col="os_event")
    return {"n_missing_age": n_missing, "M": M, "n_per_imputation": int(np.mean(n_used)),
            "rubin": rubin,
            "complete_case_HR": float(cph_cc.hazard_ratios_[lb]),
            "complete_case_p": float(cph_cc.summary.loc[lb, "p"]),
            "mi_HR": rubin[lb]["HR"], "mi_p": rubin[lb]["p"],
            "mi_CI95": rubin[lb]["CI95"]}


# ----------------------------------------------------------------------------
# P. EGFR mutant vs wild-type body composition (RG, age/sex adjusted)
# ----------------------------------------------------------------------------
def sec_P(df):
    rg = df[df.cohort == "RG"].copy()
    m = rg.dropna(subset=["egfr", "log_SMA_vol_cm3", "age", "sex_m"]).copy()
    if len(m) < 30 or m["egfr"].nunique() < 2:
        return None
    out = {"n": int(len(m)), "mutant_n": int(m["egfr"].sum())}
    # simple linear regression y ~ egfr + age + sex_m (closed-form OLS)
    for var, label in [("log_SMA_vol_cm3", "log muscle volume"),
                       ("SAT_vol_cm3", "SAT volume"),
                       ("VAT_vol_cm3", "VAT volume")]:
        m2 = m.dropna(subset=[var]).copy()
        if len(m2) < 30:
            continue
        X = np.column_stack([np.ones(len(m2)), m2["egfr"], m2["age"], m2["sex_m"]])
        y = m2[var].values
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        sigma2 = resid @ resid / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = float(np.sqrt(cov[1, 1]))
        p = float(2 * (1 - sstats.t.cdf(abs(beta[1] / se), df=dof)))
        out[label] = {"beta_egfr": float(beta[1]), "se": se, "p": round(p, 3),
                      "CI95": [float(beta[1] - 1.96 * se), float(beta[1] + 1.96 * se)]}
        med_mut = m2.loc[m2.egfr == 1, var].median()
        med_wt = m2.loc[m2.egfr == 0, var].median()
        out[label]["median_mutant"] = float(med_mut)
        out[label]["median_wild"] = float(med_wt)
    return out


# ----------------------------------------------------------------------------
# Q. Phenotype-family BH-FDR (12 tests)
# ----------------------------------------------------------------------------
def sec_Q(df):
    pheno_labels = [None, "pheno_cachexia", "pheno_sarc_obese", "pheno_low_muscle_only"]
    tests = []
    for cname, sub in [("Lung1", df[df.cohort == "Lung1"]), ("RG", df[df.cohort == "RG"])]:
        covs = ADJ_L1 if cname == "Lung1" else ADJ_RG
        d, lb = build_phenotypes(sub)
        labels = [lb] + [x for x in pheno_labels if x]
        for lab in labels:
            try:
                cph, _ = cox_fit(d, lab, covs)
            except Exception:
                continue
            r = hr_row(cph, lab)
            r.update({"cohort": cname, "family": "phenotype"})
            tests.append(r)
    d_all, lb = build_phenotypes_pooled(df)
    labels = [lb] + [x for x in pheno_labels if x]
    for lab in labels:
        try:
            cph, _ = cox_fit(d_all, lab, ADJ_POOL)
        except Exception:
            continue
        r = hr_row(cph, lab)
        r.update({"cohort": "Pooled", "family": "phenotype"})
        tests.append(r)
    if not tests:
        return None
    pvals = np.array([t["p"] for t in tests])
    n_tests = len(pvals)
    order = np.argsort(pvals)
    q_sorted = pvals[order] * n_tests / np.arange(1, n_tests + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    qvals = np.empty_like(pvals)
    qvals[order] = q_sorted
    qvals = np.clip(qvals, 0, 1)
    for i, t in enumerate(tests):
        t["q_BH"] = float(qvals[i])
        t["BH_sig"] = bool(qvals[i] < 0.05)
    return {"n_tests": n_tests, "tests": tests,
            "n_sig_after_fdr": int(sum(qvals < 0.05))}


# ----------------------------------------------------------------------------
# Table 3 / Table 4 data (read by 08)
# ----------------------------------------------------------------------------
def table3_rows(df):
    rows = []
    # Lung1 (age, sex, stage), RG (age, sex), Pooled (age, sex, cohort)
    specs = [("Lung1", df[df.cohort == "Lung1"], ADJ_L1),
             ("RG", df[df.cohort == "RG"], ADJ_RG)]
    for cname, sub, covs in specs:
        d, _ = build_phenotypes(sub)
        for lab in PHENO_LABELS:
            try:
                cph, subfit = cox_fit(d, lab, covs)
                r = hr_row(cph, lab)
                pheno_n = int(subfit[lab].sum())
            except Exception:
                continue
            label = {"pheno_cachexia": "Cachexia-like (low muscle + low subcutaneous fat)",
                     "pheno_sarc_obese": "Sarcopenic obesity (imaging-defined; low muscle + high VAT)",
                     "pheno_low_muscle_only": "Low-muscle-only (low muscle, normal fat)"}[lab]
            adj = "age, sex, stage" if cname == "Lung1" else "age, sex"
            rows.append([label, cname, r["n"], r["events"], pheno_n,
                         f"{r['HR']:.2f}", f"{r['CI95'][0]:.2f}-{r['CI95'][1]:.2f}",
                         f"{r['p']:.3f}", adj])
    d_all, _ = build_phenotypes_pooled(df)
    for lab in PHENO_LABELS:
        try:
            cph, subfit = cox_fit(d_all, lab, ADJ_POOL)
            r = hr_row(cph, lab)
            pheno_n = int(subfit[lab].sum())
        except Exception:
            continue
        label = {"pheno_cachexia": "Cachexia-like (low muscle + low subcutaneous fat)",
                 "pheno_sarc_obese": "Sarcopenic obesity (imaging-defined; low muscle + high VAT)",
                 "pheno_low_muscle_only": "Low-muscle-only (low muscle, normal fat)"}[lab]
        rows.append([label, "Pooled", r["n"], r["events"], pheno_n,
                     f"{r['HR']:.2f}", f"{r['CI95'][0]:.2f}-{r['CI95'][1]:.2f}",
                     f"{r['p']:.3f}", "age, sex, cohort"])
    T = pd.DataFrame(rows, columns=["Phenotype", "Cohort", "n", "Events", "Phenotype n",
                                    "HR", "95% CI", "p", "Adjustment"])
    return T


def table4_rows(df):
    lung1 = df[df.cohort == "Lung1"].copy()
    d, _ = build_phenotypes(lung1)
    out = []
    for name, sel in [("Stage I-II", d.stage_n <= 2), ("Stage III", d.stage_n == 3)]:
        sub = d[sel].copy()
        if len(sub) < 30 or sub["pheno_cachexia"].nunique() < 2:
            continue
        rows_km, lrp = km_absolute_diff(sub, "pheno_cachexia", years=(5,))
        try:
            cph, subfit = cox_fit(sub, "pheno_cachexia", ["age", "sex_m"])
            hr = hr_row(cph, "pheno_cachexia")
            cox_pheno_n = int(subfit["pheno_cachexia"].sum())
        except Exception:
            hr = None
            cox_pheno_n = None
        if rows_km:
            r5 = rows_km[0]
            out.append({"Stage": name, "n_km": int(len(sub)),
                        "n_pheno_km": int(sub["pheno_cachexia"].sum()),
                        "os5_pheno_pct": round(r5["survival_pheno"] * 100, 1),
                        "os5_rest_pct": round(r5["survival_normal"] * 100, 1),
                        "abs_diff_pp": r5["absolute_diff_pp"],
                        "logrank_p": round(lrp, 4) if lrp is not None else None,
                        "cox_n": hr["n"] if hr else None,
                        "cox_pheno_n": cox_pheno_n,
                        "HR": round(hr["HR"], 3) if hr else None,
                        "CI95": [round(x, 3) for x in hr["CI95"]] if hr else None,
                        "p": round(hr["p"], 4) if hr else None})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
def main():
    df = prep()
    results = {}
    print("merged df: n=%d" % len(df))

    results["A_continuous_perSD"] = sec_A(df)
    print("[A] per-SD:", {k: (v and {kk: v[kk] for kk in ("HR_per_SD", "CI95", "p")} or None)
                          for k, v in results["A_continuous_perSD"].items()})
    results["B_stage_adjusted"] = sec_B(df)
    print("[B] stage-adjusted:", {k: results["B_stage_adjusted"][k] for k in ("HR", "CI95", "p", "n")}
          if results["B_stage_adjusted"] else None)
    results["C_trend"] = sec_C(df)
    print("[C] trend:", results["C_trend"] and {k: results["C_trend"][k] for k in ("HR", "p", "n")})
    results["D_cohort_interaction"] = sec_D(df)
    print("[D] interaction:", results["D_cohort_interaction"] and
          {k: results["D_cohort_interaction"][k] for k in ("interaction_HR", "interaction_p", "n")})
    results["E_lung1_phenotypes"] = sec_E(df)
    print("[E] Lung1 phenotypes:", {k: (v and {kk: v[kk] for kk in ("HR", "CI95", "p", "n")} or None)
                                    for k, v in results["E_lung1_phenotypes"].items()})
    results["F_lung1_bootstrap_km"] = sec_F(df)
    print("[F] Lung1 bootstrap/KM:", results["F_lung1_bootstrap_km"])
    results["G_stage_stratified"] = sec_G(df)
    print("[G] stage-stratified:", json.dumps(results["G_stage_stratified"], default=str)[:500])
    results["H_quartile"] = sec_H(df)
    print("[H] quartile:", results["H_quartile"] and {k: results["H_quartile"][k] for k in ("HR", "CI95", "p", "n")})
    results["I_strata_stage"] = sec_I(df)
    print("[I] strata=stage:", {k: (v and {kk: v[kk] for kk in ("HR", "p", "n")} or None)
                                for k, v in results["I_strata_stage"].items()})
    results["J_cindex_bootstrap"] = sec_J(df, B=400)
    print("[J] C-index:", {k: (v and {kk: v[kk] for kk in ("optimism_corrected_delta", "delta_CI95", "B_ok")} or None)
                           for k, v in results["J_cindex_bootstrap"].items()})
    results["K_brier"] = sec_K(df)
    print("[K] Brier:", {k: (v and v["delta_brier"] or None) for k, v in results["K_brier"].items()})
    results["L_dca"] = sec_L(df, B_boot=500)
    print("[L] DCA:", json.dumps(results["L_dca"], default=str)[:300])
    results["M_coverage_t4"] = sec_M(df)
    print("[M] coverage/T4:", json.dumps(results["M_coverage_t4"], default=str)[:400])
    results["N_packyears"] = sec_N(df)
    print("[N] pack-years:", results["N_packyears"] and
          {k: results["N_packyears"][k] for k in ("plus_packyears", "packyears_n_available_in_model")})
    results["O_mi_age"] = sec_O(df)
    print("[O] MI:", results["O_mi_age"] and {k: results["O_mi_age"].get(k) for k in
                                              ("n_missing_age", "mi_HR", "mi_p", "mi_CI95")})
    results["P_egfr"] = sec_P(df)
    print("[P] EGFR:", results["P_egfr"])
    results["Q_phenotype_fdr"] = sec_Q(df)
    print("[Q] phenotype FDR:", results["Q_phenotype_fdr"] and
          {k: results["Q_phenotype_fdr"][k] for k in ("n_tests", "n_sig_after_fdr")})

    with open(os.path.join(OUT, "reproducibility_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    T3 = table3_rows(df)
    T3.to_csv(os.path.join(OUT, "Table3_phenotypes.csv"), index=False)
    print("\nTable 3 rows:")
    print(T3.to_string(index=False))

    T4 = table4_rows(df)
    if len(T4):
        T4.to_csv(os.path.join(OUT, "Table4_stage_os.csv"), index=False)
        print("\nTable 4 rows:")
        print(T4.to_string(index=False))
    else:
        print("\nTable 4: no rows (insufficient stage-stratified data)")

    print("\nResults saved: %s" % OUT)


if __name__ == "__main__":
    main()
