#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate manuscript figures and tables.

All numbers are read from already-computed result files (outputs of the
numbered analysis scripts), not re-estimated here.

Output: outputs/manuscript_figures/
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ----------------------------------------------------------------------------
# Unified style
# ----------------------------------------------------------------------------
for f in font_manager.fontManager.ttflist:
    if "Liberation Sans" in f.name:
        plt.rcParams["font.family"] = "Liberation Sans"
        break
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.edgecolor": "#222222", "savefig.dpi": 300,
    "figure.dpi": 100, "pdf.fonttype": 42, "ps.fonttype": 42,
})
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "red": "#D55E00", "sky": "#56B4E9", "vermillion": "#D55E00",
         "yellow": "#F0E442", "grey": "#999999", "black": "#000000"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "manuscript_figures")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png/pdf")

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
clin = pd.read_csv(os.path.join(ROOT, "data", "clinical_master.csv"))
body = pd.read_csv(os.path.join(ROOT, "data", "bodycomp_features.csv"))
df = clin.merge(body, on="patient_id", how="left")
df["log_vol"] = np.log(np.clip(df["SMA_vol_cm3"], 1, None))

# sex-stratified tertile of log muscle volume (within cohort)
def sex_tertile_lo(g):
    return g.groupby("sex")["log_vol"].transform(
        lambda x: pd.qcut(x, 3, labels=[0, 1, 2], duplicates="drop")).astype(int) == 0

df["low_vol_tertile"] = df.groupby("cohort", group_keys=False).apply(sex_tertile_lo)

# ----------------------------------------------------------------------------
# Figure 1 — study design
# ----------------------------------------------------------------------------
def fig1_study_design():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF2F8", ec="#2C3E50", fs=8.2, bold=False):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                           fc=fc, ec=ec, lw=1.1)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#111111")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.0, color="#333333"))

    ax.text(5, 9.55, "Fully automated CT body-composition phenotyping in NSCLC",
            ha="center", fontsize=11, fontweight="bold", color="#1F3864")

    # Cohorts
    box(0.5, 7.1, 4.1, 1.5,
        "Cohort 1 — NSCLC-Radiomics (Lung1)\nChest CT · n = 422\n"
        "Stage III 68% · 373 deaths · median FU 9.65 y", fc="#EAF2F8")
    box(5.4, 7.1, 4.1, 1.5,
        "Cohort 2 — NSCLC-Radiogenomics (RG)\nWhole-body PET/CT · n = 211\n"
        "Surgical cohort · 63 deaths · median FU 4.15 y", fc="#EAF2F8")
    arrow(2.55, 7.1, 3.3, 5.6); arrow(7.45, 7.1, 6.7, 5.6)

    # Pipeline
    box(1.0, 4.2, 3.6, 1.3,
        "dcm2niix conversion\nTotalSegmentator v2.18.0\n(total + tissue_types)",
        fc="#FDF2E9", ec="#935116")
    box(5.4, 4.2, 3.6, 1.3,
        "Feature extraction\n3D muscle volume · L1/L3 area · SAT/VAT\nBMD · muscle density",
        fc="#FDF2E9", ec="#935116")
    arrow(4.6, 4.85, 5.4, 4.85)

    # Phenotypes
    box(0.5, 2.2, 4.1, 1.5,
        "Composite phenotypes (mutually exclusive)\n"
        "Cachexia-like (low muscle + low fat)  n=102\n"
        "Sarcopenic obesity (low muscle + high VAT)  n=36\n"
        "Low-muscle-only  n=75",
        fc="#E8F8F5", ec="#117A65")
    box(5.4, 2.2, 4.1, 1.5,
        "Analysis\nCox OS · random-effects pooling\nStage interaction · Fine–Gray RFS\n"
        "ICC (30-pt test–retest) · FDR · DCA\nEGFR exploratory",
        fc="#E8F8F5", ec="#117A65")
    arrow(2.8, 4.2, 2.5, 3.7); arrow(7.2, 4.2, 7.5, 3.7)

    # Conclusion
    box(2.0, 0.35, 6.0, 1.1,
        "Pooled 3D muscle-volume tertile HR 1.38 (95% CI 1.12–1.70, p=0.002, I²=0%);\n"
        "cachexia-like HR 1.48 (p=0.003); all 633 patients analysed",
        fc="#FEF9E7", ec="#7D6608", fs=8.6, bold=True)
    arrow(3.8, 2.2, 4.4, 1.45)

    fig.tight_layout()
    save(fig, "Figure1_study_design")

# ----------------------------------------------------------------------------
# Figure 2 — Kaplan-Meier OS by muscle-volume tertile
# ----------------------------------------------------------------------------
def _fmt_p(p):
    """Format log-rank p: show <0.001 instead of 0.000."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def fig2_km():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    for ax, cohort in zip(axes, ["Lung1", "RG"]):
        d = df[df["cohort"] == cohort].dropna(subset=["os_time", "log_vol"]).copy()
        d["lo"] = d["low_vol_tertile"].astype(bool)
        lo, hi = d[d["lo"]], d[~d["lo"]]
        km_lo, km_hi = KaplanMeierFitter(), KaplanMeierFitter()
        km_lo.fit(lo["os_time"], lo["os_event"], label="Lowest tertile")
        km_hi.fit(hi["os_time"], hi["os_event"], label="Upper two tertiles")
        lr = logrank_test(lo["os_time"], hi["os_time"], lo["os_event"], hi["os_event"])
        ax.step(km_lo.survival_function_.index / 365.25, km_lo.survival_function_["Lowest tertile"],
                where="post", color=OKABE["red"], lw=1.8)
        ax.step(km_hi.survival_function_.index / 365.25, km_hi.survival_function_["Upper two tertiles"],
                where="post", color=OKABE["blue"], lw=1.8)
        ax.set_title(f"{cohort}  (n={len(d)})", fontweight="bold")
        ax.set_xlabel("Years since pre-treatment CT" if cohort == "Lung1" else "Years",
                      labelpad=42)
        if cohort == "Lung1":
            ax.set_ylabel("Overall survival probability")
        ax.set_xlim(0, 10); ax.set_ylim(0, 1.02)
        ax.axhline(0.5, ls="--", lw=0.6, color="#999999")
        # risk table (placed below axis; reserve space via ylim + labelpad)
        times = np.array([0, 2, 4, 6, 8, 10]) * 365.25
        n_lo = [int((lo["os_time"] >= t).sum()) for t in times]
        n_hi = [int((hi["os_time"] >= t).sum()) for t in times]
        for i, t in enumerate(times):
            ax.text(t / 365.25, -0.18, f"{n_lo[i]}", ha="center", fontsize=7, color=OKABE["red"])
            ax.text(t / 365.25, -0.30, f"{n_hi[i]}", ha="center", fontsize=7, color=OKABE["blue"])
        ax.text(0.02, 0.93, f"log-rank p = {_fmt_p(lr.p_value)}",
                transform=ax.transAxes, fontsize=8)
        ax.text(0.02, 0.84, "lowest", transform=ax.transAxes, fontsize=7.5, color=OKABE["red"])
        ax.text(0.02, 0.78, "upper two", transform=ax.transAxes, fontsize=7.5, color=OKABE["blue"])
    fig.text(0.5, 0.015, "Sex-stratified tertiles of 3D whole-body skeletal-muscle volume",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "Figure2_km_os_muscle_tertile")

# ----------------------------------------------------------------------------
# Figure 3 — forest plot: primary metric + phenotypes
# ----------------------------------------------------------------------------
def fig3_forest():
    rows = []
    def add(label, cohort, hr, lo, hi, p, n, style="main", group=None):
        rows.append(dict(label=label, cohort=cohort, hr=hr, lo=lo, hi=hi, p=p,
                         n=n, style=style, group=group))

    # primary metric (from dual_cohort_meta results)
    d04 = pd.read_csv(os.path.join(ROOT, "outputs", "dual_cohort_meta",
                                   "dual_cohort_results.csv"))
    d04 = d04[(d04["metric"] == "3D whole-body muscle volume log(cm3)") &
              (d04["cutoff"] == "tertile") & (d04["model"].isin(["adjusted age+sex", "random-effects meta"]))]
    for _, r in d04.iterrows():
        def _ci(v):
            try:
                nums = [float(x.strip()) for x in str(v).strip("[]").split(",")]
                return nums[0], nums[1]
            except Exception:
                return float(r["HR"]), float(r["HR"])
        lo, hi = _ci(r["CI95"])
        add("3D muscle volume, lowest tertile", str(r["cohort"]),
            float(r["HR"]), lo, hi, float(r["p"]), int(r["n"]) if r["n"] == r["n"] else 0,
            group="A")

    # phenotypes (Lung1 adjusted age/sex/stage; RG adjusted age/sex; merged age/sex/cohort)
    # read from the reproducibility supplement (script 09) so the figure cannot
    # drift from Table 3 / the manuscript numbers.
    t3 = _load_table3()
    short = {"Cachexia-like (low muscle + low subcutaneous fat)": "Cachexia-like",
             "Sarcopenic obesity (imaging-defined; low muscle + high VAT)": "Sarcopenic obesity",
             "Low-muscle-only (low muscle, normal fat)": "Low-muscle-only"}
    for _, r in t3.iterrows():
        hr = float(r["HR"])
        ci_str = str(r["95% CI"])
        sep = "–" if "–" in ci_str else "-"
        try:
            lo = float(ci_str.split(sep)[0])
            hi = float(ci_str.split(sep)[1])
        except Exception:
            lo, hi = hr, hr
        add(short.get(str(r["Phenotype"]), str(r["Phenotype"])), str(r["Cohort"]),
            hr, lo, hi, float(r["p"]), int(r["n"]), group="B" if r["Phenotype"].startswith(("Cachexia", "Sarcopenic")) else "C")

    R = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ypos = np.arange(len(R))[::-1]
    colors = {"Lung1": OKABE["blue"], "RG": OKABE["orange"], "Pooled": OKABE["black"]}
    lw = {"Lung1": 2.2, "RG": 2.2, "Pooled": 3.0}
    for y, r in zip(ypos, R.itertuples()):
        c = colors[r.cohort]
        ax.plot([np.log(r.lo), np.log(r.hi)], [y, y], color=c, lw=1.4, zorder=2)
        ax.scatter([np.log(r.hr)], [y], marker="D" if r.cohort == "Pooled" else "o",
                   s=22 if r.cohort == "Pooled" else 16, color=c, zorder=3)
    ax.axvline(0, ls="--", lw=0.8, color="#666666")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.label}  ·  {r.cohort}" for r in R.itertuples()], fontsize=7.5)
    ax.set_xlim(np.log(0.25), np.log(4.5))
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks([np.log(t) for t in ticks])
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel("Hazard ratio (95% CI), log scale")
    # group separators (12 rows -> y 11..0; boundaries at 8.5, 5.5, 2.5)
    for gpos in [8.5, 5.5, 2.5]:
        ax.axhline(gpos, ls=":", lw=0.6, color="#aaaaaa")
    ax.text(np.log(0.25), 10.2, "A  Primary analysis (age + sex adjusted)", fontsize=7.5, style="italic")
    ax.text(np.log(0.25), 7.0, "B  Composite phenotypes (adjusted; pooled: age + sex + cohort)", fontsize=7.5, style="italic")
    ax.text(np.log(0.25), 1.0, "C  Reference phenotype (low muscle, normal fat)", fontsize=7.5, style="italic")
    # HR text
    for y, r in zip(ypos, R.itertuples()):
        ax.text(np.log(r.hi) + 0.08, y, f"{r.hr:.2f} ({r.lo:.2f}–{r.hi:.2f})",
                va="center", fontsize=6.8, color="#333333")
    ax.set_xlim(np.log(0.25), np.log(5.2))
    ax.grid(axis="x", ls=":", lw=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "Figure3_forest_primary_phenotypes")

# ----------------------------------------------------------------------------
# Table 1 — baseline
# ----------------------------------------------------------------------------
def table1_baseline():
    def med_iqr(s):
        s = s.dropna()
        return f"{s.median():.1f} ({s.quantile(0.25):.1f}–{s.quantile(0.75):.1f})"
    def n_pct(s):
        s = s.dropna()
        return f"{int(s.sum())} ({s.mean()*100:.1f}%)"

    rows = []
    for cohort in ["Lung1", "RG"]:
        d = df[df["cohort"] == cohort]
        st = d["stage"].dropna()
        if len(st):
            iii_iv = st.astype(str).str.startswith(("III", "IV")).sum()
            stage_iii_iv = f"{int(iii_iv)} ({iii_iv/len(d)*100:.1f}%)"
        else:
            stage_iii_iv = "NA"
        rows.append({
            "Cohort": cohort, "N": len(d),
            "Male sex, n (%)": n_pct(d["sex"] == "M"),
            "Age, y, median (IQR)": med_iqr(d["age"]),
            "Stage III–IV, n (%)": stage_iii_iv,
            "OS events, n (%)": n_pct(d["os_event"] == 1),
            "Median follow-up, y": f"{9.65 if cohort == 'Lung1' else 4.15:.2f}",
            "3D muscle volume, cm³, median (IQR)": med_iqr(d["SMA_vol_cm3"]),
            "SAT volume, cm³, median (IQR)": med_iqr(d["SAT_vol_cm3"]),
            "VAT volume, cm³, median (IQR)": med_iqr(d["VAT_vol_cm3"]),
            "Muscle density, HU, median (IQR)": med_iqr(d["SM_mean_hu"]),
            "Vertebral BMD, HU, median (IQR)": med_iqr(d["BMD_L1_HU"].fillna(d["BMD_T12_HU"])),
            "L1 unavailable (T12 fallback), n (%)": n_pct(d["SMA_L1_cm2"].isna()),
            "EGFR mutant, n (%)": n_pct(d["egfr"] == 1),
        })
    T = pd.DataFrame(rows).set_index("Cohort").T
    T.to_csv(os.path.join(OUT, "Table1_baseline.csv"))
    print(T.to_string())

# ----------------------------------------------------------------------------
# Table 2 — primary Cox results
# ----------------------------------------------------------------------------
def _load_repro_results():
    p = os.path.join(ROOT, "outputs", "supplementary", "reproducibility_results.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Run scripts/09_supplementary_reproducibility.py first.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def table2_primary():
    d = pd.read_csv(os.path.join(ROOT, "outputs", "dual_cohort_meta",
                                 "dual_cohort_results.csv"))
    d = d[d["model"].isin(["adjusted age+sex", "random-effects meta"])].copy()
    d = d[d["cutoff"].isin(["median", "tertile"])]

    def fmt_ci(s):
        try:
            nums = [float(x.strip()) for x in s.strip("[]").split(",")]
            return f"{nums[0]:.2f}–{nums[1]:.2f}"
        except Exception:
            return s

    keep = d[["metric", "cutoff", "cohort", "n", "events", "HR", "p", "CI95"]].copy()
    keep["HR (95% CI)"] = keep.apply(lambda r: f"{r.HR:.2f} ({fmt_ci(r.CI95)})", axis=1)
    keep["p"] = keep["p"].map(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001")
    keep["n"] = keep["n"].fillna("—")
    keep["events"] = keep["events"].fillna("—")
    out = keep.drop(columns=["HR", "CI95"]).rename(columns={"metric": "Metric", "cutoff": "Cutoff",
                                                            "cohort": "Cohort"})

    # Prepend the per-SD continuous rows (computed by script 09)
    rep = _load_repro_results()
    per_sd = rep.get("A_continuous_perSD", {})
    sd_rows = []
    for cohort in ["Lung1", "RG", "Pooled"]:
        r = per_sd.get(cohort)
        if not r:
            continue
        sd_rows.append({
            "Metric": "3D whole-body muscle volume log(cm3)",
            "Cutoff": "Continuous, per SD",
            "Cohort": cohort,
            "n": r["n"], "events": r["events"],
            "p": f"{r['p']:.3f}" if r["p"] >= 0.001 else "<0.001",
            "HR (95% CI)": f"{r['HR_per_SD']:.2f} ({r['CI95'][0]:.2f}–{r['CI95'][1]:.2f})",
        })
    if sd_rows:
        out = pd.concat([pd.DataFrame(sd_rows), out], ignore_index=True)

    out.to_csv(os.path.join(OUT, "Table2_primary_cox.csv"), index=False)
    print(out.to_string(index=False))

# ----------------------------------------------------------------------------
# Table 3 — phenotype results
# ----------------------------------------------------------------------------
# Table 3 is read from the reproducibility supplement (script 09), which
# recomputes every phenotype row from the data tables. This keeps the table
# in sync with the manuscript numbers without hard-coding them here.
# ----------------------------------------------------------------------------
def _load_table3():
    p = os.path.join(ROOT, "outputs", "supplementary", "Table3_phenotypes.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Run scripts/09_supplementary_reproducibility.py first.")
    return pd.read_csv(p)


def table3_phenotypes():
    T = _load_table3()
    T.to_csv(os.path.join(OUT, "Table3_phenotypes.csv"), index=False)
    print(T.to_string(index=False))

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Figure 1 — study design"); fig1_study_design()
    print("Figure 2 — KM curves"); fig2_km()
    print("Figure 3 — forest plot"); fig3_forest()
    print("Table 1 — baseline"); table1_baseline()
    print("Table 2 — primary Cox"); table2_primary()
    print("Table 3 — phenotypes"); table3_phenotypes()
    print("DONE ->", OUT)
