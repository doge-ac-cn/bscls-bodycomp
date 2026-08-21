#!/usr/bin/env python3
"""Primary survival analysis: body composition vs overall survival (Cox).

L1-level features are the primary exposure set (robust to CT coverage
differences); 3D whole-body volumes are included as sensitivity.

Input : data/bodycomp_features.csv + data/clinical_master.csv
Output: outputs/primary_survival/
  - univariate_cox.csv / multivariate_cox.csv / cindex_compare.json / km_sma_l1.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "primary_survival")
os.makedirs(OUT, exist_ok=True)

STAGE_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}

# L1 primary features + 3D volume secondary
L1_FEATURES = ["SMA_L1_cm2", "SM_density_L1_HU", "BMD_L1_HU", "SAT_L1_cm2", "VAT_L1_cm2"]
VOL_FEATURES = ["SMA_vol_cm3", "SM_mean_hu", "SAT_vol_cm3", "VAT_vol_cm3"]
ADJ = ["age", "sex_m", "stage_n"]


def prep():
    feat = pd.read_csv(os.path.join(BASE, "data", "bodycomp_features.csv"))
    clin = pd.read_csv(os.path.join(BASE, "data", "clinical_master.csv"))
    df = feat.merge(clin, on="patient_id", how="inner")
    # 3D volume comparability: exclude extended-coverage scans (volume-only analysis)
    df["coverage_extended"] = df.get("coverage_extended", 0)
    df["sex_m"] = (df["sex"] == "M").astype(int)
    df["stage_n"] = df["stage"].map(STAGE_MAP)
    # log-transformed volumes
    for c in VOL_FEATURES:
        df[f"log_{c}"] = np.log(df[c].clip(lower=1))
    # fallback L1 -> T12 when L1 unavailable (primary-analysis sensitivity)
    df["SMA_primary_cm2"] = df["SMA_L1_cm2"].fillna(df["SMA_T12_cm2"])
    df["use_T12"] = df["SMA_L1_cm2"].isna().astype(int)
    return df


def run_cox(df, features, label, adj=True):
    rows = []
    for f in features:
        cols = ["os_time", "os_event", f] + (ADJ if adj else [])
        d = df[cols].dropna()
        if len(d) < 30 or d[f].nunique() < 10:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(d, duration_col="os_time", event_col="os_event")
            hr = float(cph.hazard_ratios_[f])
            p = float(cph.summary.loc[f, "p"])
            # confidence_intervals_ are on log-HR scale; exponentiate
            ci = [float(np.exp(cph.confidence_intervals_.loc[f, "95% lower-bound"])),
                  float(np.exp(cph.confidence_intervals_.loc[f, "95% upper-bound"]))]
            rows.append({"feature": f, "n": len(d), "events": int(d.os_event.sum()),
                         "HR": round(hr, 3), "p": round(p, 4),
                         "CI95": [round(x, 3) for x in ci],
                         "adjusted": adj})
        except Exception as e:
            print(f"[{f}] Cox failed: {e}")
    return pd.DataFrame(rows)


def cindex_cv(df, feature_sets, n_folds=5, seed=42):
    """Compare C-index of clinical baseline vs clinical + body composition (stratified 5-fold CV)."""
    from sklearn.model_selection import StratifiedKFold
    d = df.dropna(subset=["os_time", "os_event", "age", "sex_m", "stage_n", *feature_sets])
    if len(d) < 40:
        return None
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    res = {name: [] for name in feature_sets}
    res["clinical"] = []
    for tr, te in skf.split(d, d["os_event"]):
        Xtr, Xte = d.iloc[tr], d.iloc[te]
        # clinical baseline
        cph = CoxPHFitter()
        cph.fit(Xtr[["os_time", "os_event", *ADJ]], duration_col="os_time", event_col="os_event")
        risk_clin = cph.predict_partial_hazard(Xte)
        res["clinical"].append(concordance_index(Xte.os_time, -risk_clin, Xte.os_event))
        # clinical + body composition
        for name in feature_sets:
            cph2 = CoxPHFitter(penalizer=0.05)
            cph2.fit(Xtr[["os_time", "os_event", *ADJ, name]], duration_col="os_time", event_col="os_event")
            risk = cph2.predict_partial_hazard(Xte)
            res[name].append(concordance_index(Xte.os_time, -risk, Xte.os_event))
    out = {k: {"cindex_mean": round(float(np.mean(v)), 3),
               "cindex_std": round(float(np.std(v)), 3)} for k, v in res.items()}
    return out


def bootstrap_cindex(df, feature_cols, n_boot=200, seed=42):
    """Bootstrap C-index: clinical baseline vs +body composition (increment stability)."""
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=["os_time", "os_event", "age", "sex_m", "stage_n", *feature_cols])
    if len(d) < 40:
        return None
    out = {c: [] for c in feature_cols}
    out["clinical"] = []
    n = len(d)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        # need enough events
        if d.iloc[idx].os_event.sum() < 10:
            continue
        tr = idx
        te = np.arange(n)
        try:
            cph = CoxPHFitter()
            cph.fit(d.iloc[tr][["os_time", "os_event", *ADJ]], duration_col="os_time", event_col="os_event")
            out["clinical"].append(concordance_index(d.iloc[te].os_time, -cph.predict_partial_hazard(d.iloc[te]), d.iloc[te].os_event))
            for c in feature_cols:
                cph2 = CoxPHFitter(penalizer=0.05)
                cph2.fit(d.iloc[tr][["os_time", "os_event", *ADJ, c]], duration_col="os_time", event_col="os_event")
                out[c].append(concordance_index(d.iloc[te].os_time, -cph2.predict_partial_hazard(d.iloc[te]), d.iloc[te].os_event))
        except Exception:
            continue
    res = {}
    for k, v in out.items():
        if v:
            res[k] = {"cindex_mean": round(float(np.mean(v)), 3),
                      "cindex_ci": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]}
    return res


def km_plot(df, col, out_path, title):
    d = df.dropna(subset=["os_time", "os_event", col])
    if len(d) < 30:
        return
    q = d[col].quantile([1 / 3, 2 / 3]).values
    grp = np.where(d[col] <= q[0], "T1", np.where(d[col] <= q[1], "T2", "T3"))
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))
    for g in ["T1", "T2", "T3"]:
        m = grp == g
        kmf.fit(d.os_time[m], d.os_event[m], label=f"{g} (n={m.sum()})")
        kmf.plot_survival_function(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("OS (days)")
    ax.set_ylabel("Survival probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = prep()
    print(f"Sample: {len(df)} (Lung1={df.cohort.eq('Lung1').sum()}, RG={df.cohort.ne('Lung1').sum()}), OS events {df.os_event.sum()}")
    df.to_csv(os.path.join(OUT, "model_df.csv"), index=False)

    # Lung1 primary analysis
    lung1 = df[df.cohort == "Lung1"]
    print(f"\n=== Lung1 (n={len(lung1)}, events {lung1.os_event.sum()}) ===")

    uv = run_cox(lung1, L1_FEATURES + [f"log_{c}" for c in VOL_FEATURES] + ["SMA_primary_cm2"], "univariate", adj=False)
    print("\nUnivariate Cox (Lung1):")
    print(uv.to_string(index=False) if len(uv) else "none")
    uv.to_csv(os.path.join(OUT, "univariate_cox_lung1.csv"), index=False)

    mv = run_cox(lung1, L1_FEATURES + [f"log_{c}" for c in VOL_FEATURES] + ["SMA_primary_cm2"], "multivariable (age/sex/stage)")
    print("\nMultivariable Cox (Lung1, adjusted for age/sex/stage):")
    print(mv.to_string(index=False) if len(mv) else "none")
    mv.to_csv(os.path.join(OUT, "multivariate_cox_lung1.csv"), index=False)

    # C-index comparison (stratified 5-fold CV + bootstrap)
    print("\nC-index stratified 5-fold CV (Lung1):")
    cc = cindex_cv(lung1, L1_FEATURES[:1] + [f"log_{c}" for c in VOL_FEATURES[:1]], n_folds=5)
    if cc:
        for k, v in cc.items():
            print(f"  {k}: {v['cindex_mean']} ± {v['cindex_std']}")
        json.dump(cc, open(os.path.join(OUT, "cindex_compare_stratified.json"), "w"), indent=2, ensure_ascii=False)
    print("\nC-index bootstrap (Lung1, 200 reps):")
    bc = bootstrap_cindex(lung1, L1_FEATURES[:1] + [f"log_{c}" for c in VOL_FEATURES[:1]], n_boot=200)
    if bc:
        for k, v in bc.items():
            print(f"  {k}: {v['cindex_mean']} CI {v['cindex_ci']}")
        json.dump(bc, open(os.path.join(OUT, "cindex_bootstrap.json"), "w"), indent=2, ensure_ascii=False)

    km_plot(lung1, "SMA_primary_cm2", os.path.join(OUT, "km_sma_primary.png"),
            "Lung1 SMA tertiles (L1 or T12) - OS")
    km_plot(lung1, "SM_density_L1_HU", os.path.join(OUT, "km_sm_density.png"),
            "Lung1 muscle density tertiles - OS")

    # RG (fewer events, descriptive only)
    rg = df[df.cohort != "Lung1"]
    print(f"\n=== RG (n={len(rg)}, events {rg.os_event.sum()}) — descriptive only ===")
    if len(rg) > 0:
        print(rg[["patient_id", "SMA_primary_cm2", "SM_density_L1_HU", "SMA_vol_cm3"]].to_string(index=False))

    print(f"\nAll outputs in {OUT}/")


if __name__ == "__main__":
    main()
