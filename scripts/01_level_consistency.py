"""Cross-level consistency of body-composition measurements (L1 vs L3).

Runs on the RG cohort only (whole-body CT covers both L1 and L3).
Evaluates agreement via Bland-Altman plots, Pearson/Spearman correlation,
and categorical agreement (Cohen's kappa) for paired L1/L3 features.

Input : data/bodycomp_features.csv
Output: outputs/level_consistency/ (bland_altman.png + consistency_results.json)
"""
import pandas as pd
import numpy as np
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "data", "bodycomp_features.csv")
OUT = os.path.join(BASE, "outputs", "level_consistency")
os.makedirs(OUT, exist_ok=True)

PAIRS = [
    ("SMA_L1_cm2", "SMA_L3_cm2", "SMA (cm2)"),
    ("SM_density_L1_HU", "SM_density_L3_HU", "Muscle density (HU)"),
    ("BMD_L1_HU", "BMD_L3_HU", "Bone density (HU)"),
    ("SAT_L1_cm2", "SAT_L3_cm2", "SAT (cm2)"),
    ("VAT_L1_cm2", "VAT_L3_cm2", "VAT (cm2)"),
]

def bland_altman(a, b):
    diff = a - b
    mean = (a + b) / 2
    md = np.nanmean(diff)
    sd = np.nanstd(diff, ddof=1)
    loa_lo, loa_hi = md - 1.96 * sd, md + 1.96 * sd
    return mean, diff, md, sd, loa_lo, loa_hi

def sex_stratified_tertile_label(s, x):
    """Sex-stratified lowest tertile -> binary low-muscle label (within-cohort cutoff)."""
    cuts = {}
    for g in np.unique(s.dropna()):
        cuts[g] = x[s == g].quantile(1 / 3)
    out = pd.Series(np.nan, index=x.index)
    for g, c in cuts.items():
        m = s == g
        out.loc[m] = (x[m] < c).astype(int)
    return out

def main():
    df = pd.read_csv(CSV)
    rg = df[df["patient_id"].str.startswith(("R01-", "AMC-"))].copy()
    rg = rg.dropna(subset=["SMA_L1_cm2", "SMA_L3_cm2"]).reset_index(drop=True)
    sex = pd.read_csv(os.path.join(BASE, "data", "clinical_master.csv")) \
        .set_index("patient_id")["sex"].reindex(rg["patient_id"]).reset_index(drop=True)
    rg["sex_m"] = (sex == "M").astype(float)
    print(f"RG cases with both L1 and L3 available: {len(rg)}")

    results = {"n_rg_with_L1_L3": int(len(rg))}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    for i, (c1, c2, label) in enumerate(PAIRS):
        a = rg[c1].astype(float)
        b = rg[c2].astype(float)
        valid = a.notna() & b.notna()
        a, b = a[valid], b[valid]
        if len(a) < 3:
            results[label] = {"n": len(a), "note": "insufficient sample"}
            continue
        r = np.corrcoef(a, b)[0, 1]
        rho = a.corr(b, method="spearman")
        mean, diff, md, sd, lo_lo, lo_hi = bland_altman(a, b)
        entry = {"n": int(len(a)), "pearson_r": round(float(r), 3),
                 "spearman_rho": round(float(rho), 3),
                 "bias": round(float(md), 3), "sd": round(float(sd), 3),
                 "LoA": [round(float(lo_lo), 3), round(float(lo_hi), 3)]}
        # categorical agreement (low-muscle Kappa): SMA only
        if c1 == "SMA_L1_cm2" and len(a) >= 30 and sex is not None:
            s = (sex[valid] == "M").astype(float)
            lab1 = sex_stratified_tertile_label(s, a)
            lab3 = sex_stratified_tertile_label(s, b)
            ok = lab1.notna() & lab3.notna() & s.notna()
            k = cohen_kappa_score(lab1[ok], lab3[ok])
            agree = float((lab1[ok] == lab3[ok]).mean())
            entry["classification_kappa_low_tertile"] = round(float(k), 3)
            entry["classification_agreement"] = round(agree, 3)
            entry["cutoff_note"] = "sex-stratified within-cohort lowest tertile (used for categorical-agreement argument)"
        results[label] = entry
        ax = axes[i]
        ax.scatter(mean, diff, alpha=0.6, s=25)
        ax.axhline(md, color="tab:red", ls="--", label=f"bias={md:.2f}")
        ax.axhline(lo_lo, color="tab:gray", ls=":", label=f"LoA [{lo_lo:.2f},{lo_hi:.2f}]")
        ax.axhline(lo_hi, color="tab:gray", ls=":")
        ax.set_xlabel(f"Mean of L1/L3 {label}")
        ax.set_ylabel("L1 − L3")
        ax.set_title(f"{label}\nr={r:.3f}, bias={md:.2f}")
        ax.legend(fontsize=7)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bland_altman.png"), dpi=150)

    with open(os.path.join(OUT, "consistency_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nPlot saved: {OUT}/bland_altman.png")

if __name__ == "__main__":
    main()
