# Fully automated CT body-composition phenotyping in NSCLC

[![CI](https://github.com/doge-ac-cn/bscls-bodycomp/actions/workflows/ci.yml/badge.svg)](https://github.com/doge-ac-cn/bscls-bodycomp/actions/workflows/ci.yml)

This repository reproduces the complete analysis pipeline of the manuscript
*"Fully automated CT-derived body composition phenotypes predict overall
survival in non-small cell lung cancer: a dual-cohort study of 633 patients"*.

We derive body-composition phenotypes (3D skeletal-muscle volume, subcutaneous
and visceral adipose tissue, muscle density, vertebral bone density, and
level-specific areas) fully automatically from routine staging CT scans with
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator), and evaluate
their association with overall survival in two independent NSCLC cohorts using
Cox regression, random-effects meta-analysis, and competing-risk (Fine-Gray)
models.

## Cohorts and data

| Cohort | n | Modality | Median follow-up | Source |
|---|---|---|---|---|
| NSCLC-Radiomics (Lung1) | 422 | Chest CT | 9.65 y | [TCIA](https://www.cancerimagingarchive.net/) |
| NSCLC-Radiogenomics (RG) | 211 | Whole-body PET/CT | 4.15 y | [TCIA](https://www.cancerimagingarchive.net/) |

Both datasets are public:

- **NSCLC-Radiomics (Lung1)** — Aerts HJWL et al. *Nat Commun.* 2014;5:4006.
  TCIA DOI: `10.7937/K9/TCIA.2015.PF0M9REI`
- **NSCLC-Radiogenomics** — Bakr S et al. *Sci Data.* 2018;5:180202.
  TCIA DOI: `10.7937/K9/TCIA.2017.7hs46erv`

Imaging (DICOM) and clinical spreadsheets must be downloaded from TCIA and
placed under `data/raw/` preserving the TCIA package layout:

```
data/raw/
├── NSCLC-Radiomics/          # Lung1 DICOM series
├── NSCLC-Radiogenomics/      # RG DICOM series
└── data/
    ├── clinical/
    │   ├── NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv
    │   ├── NSCLC-Radiogenomics-VA-R01-labels.csv
    │   └── sarg_patients.parquet
    └── master_table_nsclc_radiogenomics.csv
```

**Automated imaging download** — `download_tcia.py` fetches the DICOM
series of both cohorts and writes them in the canonical layout shown above
(patient → series directories, exactly what `scan_series.py` and
`segmentation_pipeline.py` expect):

```bash
python scripts/download_tcia.py            # both cohorts
python scripts/download_tcia.py lung1      # NSCLC-Radiomics only (CT + SEG)
python scripts/download_tcia.py rg         # NSCLC-Radiogenomics only (CT + PET)
```

The script is idempotent: already-downloaded series are detected in the
canonical location and skipped. It reorganises the flat
`tcia-utils` output into `<PatientID>/<SeriesInstanceUID>/` and flattens any
nesting introduced by the TCIA zip layout, so the result always matches the
tree above.

**Automated clinical download** — `download_tcia.py` fetches imaging only;
the clinical spreadsheets are hosted separately on the TCIA website. Run

```bash
python scripts/download_clinical.py   # downloads both official CSVs and derives the two auxiliary tables
```

`download_clinical.py` fetches the two official files (Lung1 clinical v3 and
the RG R01 labels spreadsheet), renames the RG `Case ID` column to
`Patient ID` (the name every downstream script expects), and derives
`master_table_nsclc_radiogenomics.csv` / `sarg_patients.parquet` from the
official labels — those two auxiliary tables are a lossless column subset of
the VA-R01 labels file, so they are regenerated instead of requiring a
manual download that is no longer exposed on the TCIA website. Use
`--force` to re-download everything.

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU with ≥ 8 GB VRAM (recommended for segmentation)
- [dcm2niix](https://github.com/rordenlab/dcm2niix) on `PATH` (or set
  `DCM2NIIX=/path/to/dcm2niix`)
- `tcia-utils` (optional; only needed for automatic TCIA download)

```bash
pip install -r requirements.txt
```

**Analysis-only mode** (no GPU, no segmentation, no TCIA data): install
`requirements-core.txt` and use the synthetic demo data described below.

## Quick start with synthetic demo data

No imaging data, GPU, or TCIA download is needed to exercise the analysis
layer (scripts 01–05, 07, 08):

```bash
pip install -r requirements-core.txt
python scripts/make_demo_data.py        # 120 synthetic patients (60 Lung1 + 60 RG, sex-balanced)
python scripts/01_level_consistency.py
python scripts/02_primary_survival.py
python scripts/03_finegray_rfs.py
python scripts/04_dual_cohort_meta.py
python scripts/05_supplementary_analysis.py
python scripts/07_missingness.py
python scripts/08_manuscript_figures.py
```

`make_demo_data.py` writes `data/clinical_master.csv` and
`data/bodycomp_features.csv` with the same schema as the real pipeline
(41 + 65 columns). The values are random draws from plausible ranges with a
fixed seed; they carry **no scientific meaning** and only verify that the
pipeline executes end-to-end. Script 06 (ICC reproducibility) requires real
segmentation masks and is skipped in demo mode. The same sequence runs in
GitHub Actions CI on every push.

## Pipeline

Run the steps in order from the repository root. Each script writes to
`data/` or `outputs/` and is idempotent (safe to re-run).

### 1. Clinical master table

```bash
python scripts/download_clinical.py       # official clinical spreadsheets from TCIA (auto)
python scripts/prepare_clinical_build.py  # merge official spreadsheets -> data/clinical_master.csv
python scripts/prepare_clinical_time.py   # encode OS/RFS time-to-event in days
python scripts/prepare_clinical_v3.py     # fix stage semantics, recover weight, add official fields
python scripts/prepare_clinical_fix.py    # fix ALK mapping and AMC-049 recurrence mis-entry
python scripts/verify_clinical_data.py    # consistency checks vs official sources
python scripts/verify_official_missing.py # per-field missing-value audit vs official files
```

The clinical pipeline needs `pandas`, `numpy` and `pyarrow` (for the
`sarg_patients.parquet` auxiliary table).

### 2. CT series selection and segmentation

```bash
python scripts/scan_series.py              # scan DICOM -> data/series_manifest.json
python scripts/select_ct_series.py         # pick one CT per patient -> data/selected_ct.json
python scripts/segmentation_pipeline.py --start 0 --end 633   # dcm2niix + TotalSegmentator
```

`segmentation_pipeline.py` converts DICOM to NIfTI and runs TotalSegmentator
(`total` and `tissue_types` tasks) serially with resumable `--start/--end`
ranges. It is designed for memory-limited machines: run in chunks if needed,
and do **not** launch multiple workers against the same output directory.

**Segmentation outputs are not shipped in this repository** (~16 GB for 633
cases). To verify that locally produced masks match the ones used in the
manuscript, `data/mask_manifest.json` contains the SHA-256 digest of every
mask file (`{total,tissue_types}` per patient). Recompute the digest of your
own `data/ts_masks/` and compare:

```bash
python - <<'PY'
import hashlib, json, os
ref = json.load(open("data/mask_manifest.json"))
ok = bad = 0
for row in ref["files"]:
    p = os.path.join("data/ts_masks", row["mask_dir"], row["file"])
    h = hashlib.sha256()
    if not os.path.exists(p):
        bad += 1; continue
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    (ok if h.hexdigest() == row["sha256"] else bad.__add__)(1)
print(f"{ok}/{len(ref['files'])} masks match")
PY
```

Segmentation is deterministic (TotalSegmentator / nnU-Net inference), so a
full match is expected on the same input NIfTI. `mask_manifest.json` is
generated by `scripts/make_mask_manifest.py`.

### 3. Feature extraction

```bash
python scripts/extract_bodycomp.py         # -> data/bodycomp_features.csv
```

Extracts whole-body 3D volumes (skeletal muscle, SAT, VAT), vertebral-level
2D areas/densities (L1/L3/T12 with T4/T11 sensitivity), and muscle-specific
metrics (erector spinae, psoas) with HU filtering (muscle −29 to 150 HU,
bone 0–600 HU, fat −190 to −30 HU).

### 4. Analyses

| Script | Purpose |
|---|---|
| `01_level_consistency.py` | L1 vs L3 measurement agreement (Bland-Altman, correlation, kappa) |
| `02_primary_survival.py` | Primary OS Cox models + C-index + KM curves |
| `03_finegray_rfs.py` | Recurrence-free survival with competing risks (Fine-Gray) |
| `04_dual_cohort_meta.py` | Within-cohort Cox + DerSimonian-Laird random-effects pooling + forest plot |
| `05_supplementary_analysis.py` | Bootstrap stability, FDR control, median follow-up, QC, chemo stratification, pooled phenotypes |
| `06_icc_reproducibility.py` | Reproducibility: software test-retest (Dice/ICC) + slice-position sensitivity |
| `07_missingness.py` | Missing-data pattern analysis |

```bash
python scripts/01_level_consistency.py
python scripts/02_primary_survival.py
python scripts/03_finegray_rfs.py
python scripts/04_dual_cohort_meta.py
python scripts/05_supplementary_analysis.py
python scripts/06_icc_reproducibility.py --analyze-a --analyze-b
python scripts/07_missingness.py
```

`06_icc_reproducibility.py` has four modes: `--sample`, `--rerun`
(re-runs TotalSegmentator on the 30-case sample; slow), `--analyze-a`,
`--analyze-b`.

### 5. Manuscript figures and tables

```bash
python scripts/08_manuscript_figures.py    # -> outputs/manuscript_figures/
```

Reads already-computed result files (notably `outputs/dual_cohort_meta/
dual_cohort_results.csv`) and renders the study-design diagram, KM curves,
forest plot, and baseline/primary/phenotype tables.

## Phenotype definitions

All cutoffs are **within-cohort, sex-stratified** (no cross-cohort transfer):

- **Low muscle**: lowest tertile of log 3D whole-body skeletal-muscle volume.
- **Cachexia-like**: low muscle + lowest tertile of subcutaneous adipose
  tissue volume.
- **Sarcopenic obesity**: low muscle + above two-thirds quantile of visceral
  adipose tissue volume.
- **Low-muscle-only**: low muscle with neither fat abnormality.

## Primary results (for verification)

- Pooled 3D muscle-volume tertile: HR 1.38 (95% CI 1.12–1.70, p = 0.002,
  I² = 0%)
- Cachexia-like pooled: HR 1.48 (95% CI 1.14–1.92, p = 0.003)
- Sarcopenic obesity pooled: HR 1.55 (95% CI 1.05–2.30, p = 0.029)
- Low-muscle-only pooled: HR 0.95 (95% CI 0.71–1.29, p = 0.751)
- Recurrence (Fine-Gray): no association (HR 1.04, p = 0.90)
- Software test-retest: Dice 0.984–0.991; feature ICC 0.987–1.000

## Reproducibility notes

- All random processes use fixed seeds.
- Segmentation is deterministic (nnU-Net inference); the ICC script re-runs it
  to demonstrate test-retest stability.
- Phenotype and cutoff code lives inside each analysis script; cross-checking
  against `outputs/` JSON files is recommended after each run.
- Known limitations reported in the manuscript: RG single-cohort effect is
  directionally consistent but not independently significant (powered for
  pooled analysis); recurrence (RFS) is negative; C-index increment is small;
  height was unavailable, so SMI (cm²/m²) could not be computed — absolute
  volumes with sex stratification are used instead.

## Citation

If you use this pipeline, please cite the manuscript:

> Fully automated CT-derived body composition phenotypes predict overall
> survival in non-small cell lung cancer: a dual-cohort study of 633 patients.
> *Manuscript in preparation* (target journal: European Journal of Radiology).

```bibtex
@unpublished{bodycomp2026,
  title  = {Fully automated {CT}-derived body composition phenotypes predict
            overall survival in non-small cell lung cancer: a dual-cohort
            study of 633 patients},
  author = {},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

Please also cite the underlying public datasets (see Cohorts and data section)
and [TotalSegmentator](https://github.com/wasserth/TotalSegmentator).

## License

Code: MIT (see LICENSE).

The datasets remain under the TCIA Data Usage Agreement of their respective
collections; derived feature tables produced by this pipeline may be subject
to those terms.
