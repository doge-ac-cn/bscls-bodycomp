#!/usr/bin/env python
# Dual-cohort complementary analysis
# Pre-specified design:
#   - Cohorts are COMPLEMENTARY, not train/validate:
#       Lung1 (n=422, 373 events): discovery/power — multi-phenotype screening, OS
#       RG    (n=211, ~58 events): exploratory validation + mechanism (chemo, BMI, RFS, drivers)
#   - Primary shared metric : log_SMA_vol_cm3 (whole-body 3D volume; immune to slice-position
#                             and CT-coverage differences between chest CT and whole-body CT)
#   - Cutoffs               : WITHIN-cohort sex-stratified median + tertile (NO cross-cohort transfer;
#                             (L1 area distributions differ ~25% between cohorts)
#   - Model                 : within-cohort Cox (unadj + age/sex) -> random-effects meta-analysis
#                             (DerSimonian-Laird), I^2 heterogeneity, forest plot
#   - Verdict (pre-registered):
#       pooled HR >= 1.3, direction consistent, I^2 moderate   -> complementary evidence
#       pooled HR 1.1-1.3, direction consistent                -> weak complementary evidence
#       pooled HR < 1.1 or reversed + high I^2                 -> heterogeneity finding (report as is)
#   - Secondary             : SMA_L1_cm2 within-cohort cutoffs (exploratory only)
# Idempotent: rerun after full RG segmentation+feature extraction completes.
import os, json
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "dual_cohort_meta")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK"]
plt.rcParams["axes.unicode_minus"] = False

feat = pd.read_csv(os.path.join(ROOT, "data", "bodycomp_features.csv"))
clin = pd.read_csv(os.path.join(ROOT, "data", "clinical_master.csv"))
d = feat.merge(clin, on="patient_id", how="inner").copy()
d["event"] = d["os_event"].astype(int)
d["event_t"] = d["os_time"].astype(float)
d["age"] = d["age"].astype(float)
d["sex"] = (d["sex"] == "M").astype(int)
d["cohort"] = np.where(d["patient_id"].str.startswith("LUNG1"), "Lung1", "RG")
d["log_vol"] = np.log(d["SMA_vol_cm3"])
if "rg_chemotherapy" in d.columns:
    d["chemo"] = (d["rg_chemotherapy"].fillna("No") == "Yes").astype(int)

print(f"Merged: n={len(d)}  Lung1={int((d['cohort']=='Lung1').sum())}  RG={int((d['cohort']=='RG').sum())}")
for c in ["Lung1", "RG"]:
    sub = d[d["cohort"] == c]
    print(f"  {c}: events={sub['event'].sum()}, M/F={int((sub['sex']==1).sum())}/{int((sub['sex']==0).sum())}, "
          f"age={sub['age'].median():.0f}, median OS={sub['event_t'].median():.0f}d")

METRICS = [
    ("log_vol", "3D whole-body muscle volume log(cm3)", "primary"),
    ("SMA_L1_cm2", "L1 muscle area cm2", "secondary"),
]

def sex_cut(df, col, q):
    return df.groupby("sex")[col].quantile(q)

def run_cox(sub, col, low_col, covs):
    cols = covs + [low_col, "event", "event_t"]
    s = sub.dropna(subset=cols).copy()
    if s["event"].sum() < 8 or len(s) < 25:
        return None
    cph = CoxPHFitter()
    try:
        cph.fit(s[cols], duration_col="event_t", event_col="event")
    except Exception as e:
        print(f"    [Cox failed: {e}]")
        return None
    return {
        "n": len(s), "events": int(s["event"].sum()),
        "HR": float(np.exp(cph.params_[low_col])),
        "p": float(cph.summary.loc[low_col, "p"]),
        "CI95": [float(cph.summary.loc[low_col, "exp(coef) lower 95%"]),
                 float(cph.summary.loc[low_col, "exp(coef) upper 95%"])],
    }

def dl_pool(rows):
    """Dual-cohort complementary analysis with random-effects meta-analysis.

The two cohorts (discovery and exploratory validation) are treated as
complementary, not as train/validation. The primary shared metric is the
log-transformed 3D whole-body muscle volume (immune to slice-position and
CT-coverage differences between chest CT and whole-body CT). Cutoffs are
within-cohort sex-stratified median/tertile (no cross-cohort transfer).
Within-cohort Cox models (unadjusted + age/sex) are pooled with a
DerSimonian-Laird random-effects meta-analysis; heterogeneity is reported
as I^2. A forest plot is generated.

Secondary : SMA_L1_cm2 within-cohort cutoffs (exploratory only).
"""
    k = len(rows)
    if k == 0:
        return None
    logs = np.array([np.log(r["HR"]) for r in rows])
    v = np.array([((np.log(r["CI95"][1]) - np.log(r["CI95"][0])) / (2 * 1.96)) ** 2 for r in rows])
    if k == 1:  # single study: report its own estimate, no heterogeneity
        return {
            "k": 1, "HR": float(np.exp(logs[0])), "p": rows[0]["p"],
            "CI95": list(rows[0]["CI95"]), "I2": np.nan, "tau2": 0.0, "Q": 0.0,
        }
    w = 1.0 / v
    logbar = float(np.sum(w * logs) / np.sum(w))
    Q = float(np.sum(w * (logs - logbar) ** 2))
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if Q > (k - 1) else 0.0
    w_star = 1.0 / (v + tau2)
    log_pool = float(np.sum(w_star * logs) / np.sum(w_star))
    se_pool = float(np.sqrt(1.0 / np.sum(w_star)))
    I2 = float(max(0.0, (Q - (k - 1)) / Q * 100)) if Q > 0 else 0.0
    z = log_pool / se_pool
    p_pool = float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(z))))
    return {
        "k": k, "HR": float(np.exp(log_pool)), "p": p_pool,
        "CI95": [float(np.exp(log_pool - 1.96 * se_pool)), float(np.exp(log_pool + 1.96 * se_pool))],
        "I2": I2, "tau2": tau2, "Q": Q,
    }

results = []
forest_data = {}

for col, label, role in METRICS:
    for qname, q in [("median", 0.5), ("tertile", 1 / 3)]:
        rows = []
        for c in ["Lung1", "RG"]:
            sub = d[d["cohort"] == c]
            cut = sex_cut(sub, col, q)
            sub = sub.copy()
            sub["low"] = (sub[col] < sub["sex"].map(cut)).astype(int)
            for covs, cname in [(["age"], "unadjusted"), (["age", "sex"], "adjusted age+sex")]:
                r = run_cox(sub, col, "low", covs)
                if r is None:
                    results.append({"metric": label, "role": role, "cohort": c, "cutoff": qname,
                                    "model": cname, "n": len(sub), "events": int(sub["event"].sum()),
                                    "low_n": int(sub["low"].sum()), "HR": None})
                    continue
                r.update({"metric": label, "role": role, "cohort": c, "cutoff": qname, "model": cname,
                          "low_n": int(sub["low"].sum())})
                results.append(r)
                if cname == "adjusted age+sex":
                    rows.append({"cohort": c, **r})
        if rows:
            pool = dl_pool(rows)
            pool.update({"metric": label, "role": role, "cutoff": qname, "model": "random-effects meta",
                         "cohort": "Pooled", "low_n": None})
            results.append(pool)
            forest_data[f"{label}|{qname}"] = {"rows": rows, "pool": pool}
            print(f"\n[{label} | {qname}] pooled HR={pool['HR']:.2f} (95%CI {pool['CI95'][0]:.2f}-{pool['CI95'][1]:.2f}), "
                  f"p={pool['p']:.3f}, I²={pool['I2']:.1f}%")
            for r in rows:
                print(f"    {r['cohort']}: HR={r['HR']:.2f} ({r['CI95'][0]:.2f}-{r['CI95'][1]:.2f}), "
                      f"events={r['events']}/{r['n']}")

out = pd.DataFrame(results)
out.to_csv(os.path.join(OUT, "dual_cohort_results.csv"), index=False)
print("\n=== All results ===")
print(out.to_string(index=False))

# verdict on primary metric (sex-stratified tertile, adjusted model)
prim_key = "3D whole-body muscle volume log(cm3)|tertile"
verdict = "insufficient data"
if prim_key in forest_data:
    pool = forest_data[prim_key]["pool"]
    hr, i2 = pool["HR"], pool["I2"]
    dirs = [r["HR"] for r in forest_data[prim_key]["rows"]]
    consistent = all((h - 1) * (hr - 1) > 0 for h in dirs) if len(dirs) > 1 else True
    if len(dirs) < 2:
        verdict = f"single-cohort data only (Lung1 HR={hr:.2f})"
    elif hr >= 1.3 and consistent and i2 < 60:
        verdict = "complementary evidence (pooled HR>=1.3, consistent direction, moderate heterogeneity)"
    elif hr >= 1.1 and consistent:
        verdict = "weak complementary evidence (1.1-1.3, consistent direction)"
    elif hr < 1.1 or not consistent:
        verdict = f"no pooled effect / reversed (HR={hr:.2f}, I2={i2:.0f}%) — report heterogeneity as is"
with open(os.path.join(OUT, "verdict.json"), "w") as fh:
    json.dump({"verdict": verdict, "primary_key": prim_key, "pool": pool if prim_key in forest_data else None},
              fh, indent=2, ensure_ascii=False, default=str)
print(f"\n=== Primary verdict: {verdict} ===")

# ---- forest plot ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (key, fd) in zip(axes, forest_data.items()):
    label, qname = key.split("|")
    rows, pool = fd["rows"], fd["pool"]
    ypos = np.arange(len(rows))[::-1]
    ax.axvline(1, color="grey", lw=0.8, ls="--")
    for y, r in zip(ypos, rows):
        hr, lo, hi = r["HR"], r["CI95"][0], r["CI95"][1]
        ax.plot([lo, hi], [y, y], color="steelblue", lw=2)
        ax.plot(hr, y, "s", color="steelblue", ms=8)
        ax.text(0.03, y, f"{r['cohort']} (n={r['n']}, ev={r['events']})", va="center",
                transform=ax.get_yaxis_transform())
        ax.text(0.98, y, f"{hr:.2f} [{lo:.2f}, {hi:.2f}]", va="center", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=9)
    yp = len(rows)
    hr, lo, hi = pool["HR"], pool["CI95"][0], pool["CI95"][1]
    ax.plot([lo, hi], [yp, yp], color="crimson", lw=2.5)
    ax.plot(hr, yp, "D", color="crimson", ms=9)
    ax.text(0.03, yp, f"Pooled (RE, I²={pool['I2']:.0f}%)", va="center", fontweight="bold",
            transform=ax.get_yaxis_transform())
    ax.text(0.98, yp, f"{hr:.2f} [{lo:.2f}, {hi:.2f}]  p={pool['p']:.3f}", va="center", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=9, fontweight="bold")
    ax.set_yticks([])
    ax.set_xscale("log")
    ax.set_xlim(0.4, 4.0)
    ax.set_xlabel("HR (95% CI, log scale)")
    ax.set_title(f"{label}\n({qname} split, adjusted age+sex)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
plt.suptitle("Dual-cohort complementary analysis: within-cohort split + random-effects meta (DerSimonian-Laird)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "forest_dual_cohort.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Forest plot saved:", os.path.join(OUT, "forest_dual_cohort.png"))
