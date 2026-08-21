#!/usr/bin/env python3
"""Extract body-composition features from TotalSegmentator masks.

Whole-body 3D volumes (skeletal muscle, SAT, VAT) and vertebral-level
2D areas/densities (L1/L3/T12, with T4/T11 for sensitivity), plus
muscle-specific metrics at selected levels (erector spinae, psoas).
HU filtering: muscle (-29,150), bone (0,600), fat (-190,-30).

Input : data/nifti/*.nii.gz + data/ts_masks/<pid>_{total,tissue_types}/
Output: data/bodycomp_features.csv
"""
import nibabel as nib
import numpy as np
import pandas as pd
import os, glob, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIFTI_DIR = os.path.join(BASE, "data", "nifti")
MASK_DIR = os.path.join(BASE, "data", "ts_masks")
OUT_CSV = os.path.join(BASE, "data", "bodycomp_features.csv")
MASK_MAP = json.load(open(os.path.join(BASE, "data", "ts_mask_map.json"))) if os.path.exists(os.path.join(BASE, "data", "ts_mask_map.json")) else None

MUSCLE_HU = (-29, 150)
BONE_HU = (0, 600)
FAT_HU = (-190, -30)
MIN_VERT_VOXELS = 8000
VERT_LEVELS = ["vertebrae_T4", "vertebrae_T11", "vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3"]


def find_mask(pid, task_suffixes, cls_name):
    """Find the mask file across possible directory suffixes."""
    for suf in task_suffixes:
        p = os.path.join(MASK_DIR, f"{pid}_{suf}", f"{cls_name}.nii.gz")
        if os.path.exists(p):
            return p
    return None


def vert_center_slice(mask):
    zs = np.where(mask.any(axis=(0, 1)))[0]
    if zs.size == 0:
        return None
    weights = mask[:, :, zs].sum(axis=(0, 1)).astype(float)
    if weights.sum() == 0:
        return int(zs.mean())
    return int(round(np.average(zs, weights=weights)))


def hu_stats(mask, hu, hu_range):
    vals = hu[mask]
    vals = vals[(vals >= hu_range[0]) & (vals <= hu_range[1])]
    if vals.size == 0:
        return np.nan, 0
    return float(vals.mean()), int(vals.size)


def area_at(mask, z_center, hu, spacing, hu_range=None, n_side=1):
    lo, hi = max(0, z_center - n_side), min(hu.shape[2] - 1, z_center + n_side)
    sl = slice(lo, hi + 1)
    n_layers = hi - lo + 1
    m = mask[:, :, sl]
    area = float(m.sum() * spacing[0] * spacing[1] / 100.0) / n_layers
    if hu_range is not None:
        vals = hu[:, :, sl][m]
        vals = vals[(vals >= hu_range[0]) & (vals <= hu_range[1])]
        density = float(vals.mean()) if vals.size else np.nan
        n = int(vals.size)
        return area, density, n
    return area, np.nan, 0


def vol_cm3(mask, sp):
    return float(mask.sum() * np.prod(sp) / 1000.0)


def merge_pair(pid, suffix_list, left_name, right_name):
    """Merge left/right symmetric masks into one bool mask; None if missing."""
    pl = find_mask(pid, suffix_list, left_name)
    pr = find_mask(pid, suffix_list, right_name)
    m = None
    if pl:
        m = nib.load(pl).get_fdata().astype(bool)
    if pr:
        a = nib.load(pr).get_fdata().astype(bool)
        m = a if m is None else (m | a)
    return m


def extract_patient(pid, tissue_suffix, total_suffix):
    nii = os.path.join(NIFTI_DIR, f"{pid}.nii.gz")
    if not os.path.exists(nii):
        return None
    img = nib.load(nii)
    hu = img.get_fdata()
    sp = img.header.get_zooms()[:3]
    row = {"patient_id": pid}
    # ---- scan coverage QC: prerequisite for 3D volume comparability ----
    row["z_mm"] = round(float(img.shape[2] * sp[2]), 1)
    row["coverage_extended"] = 1 if row["z_mm"] > 600 else 0

    # ---- body-composition masks (tissue_types) ----
    sm_path = find_mask(pid, tissue_suffix, "skeletal_muscle")
    if sm_path is None:
        return None  # no muscle mask, cannot extract
    sm = nib.load(sm_path).get_fdata().astype(bool)
    row["SMA_vol_cm3"] = vol_cm3(sm, sp)
    # muscle density (after HU filtering)
    row["SM_mean_hu"] = hu_stats(sm, hu, MUSCLE_HU)[0]

    sat_path = find_mask(pid, tissue_suffix, "subcutaneous_fat")
    vat_path = find_mask(pid, tissue_suffix, "torso_fat")
    sat = nib.load(sat_path).get_fdata().astype(bool) if sat_path else None
    vat = nib.load(vat_path).get_fdata().astype(bool) if vat_path else None
    if sat is not None:
        row["SAT_vol_cm3"] = vol_cm3(sat, sp)
        row["SAT_vol_fatHU_cm3"] = vol_cm3(sat & (hu >= FAT_HU[0]) & (hu <= FAT_HU[1]), sp)
    if vat is not None:
        row["VAT_vol_cm3"] = vol_cm3(vat, sp)
        row["VAT_vol_fatHU_cm3"] = vol_cm3(vat & (hu >= FAT_HU[0]) & (hu <= FAT_HU[1]), sp)

    # ---- vertebrae (total) ----
    verts = {}
    for v in VERT_LEVELS:
        p = find_mask(pid, total_suffix, v)
        if p:
            m = nib.load(p).get_fdata().astype(bool)
            if m.sum() >= MIN_VERT_VOXELS:
                verts[v] = m
    row["vert_available"] = ",".join(v.replace("vertebrae_", "") for v in verts)

    # ---- specific muscles (total) ----
    es = merge_pair(pid, total_suffix, "autochthon_left", "autochthon_right")   # erector spinae group
    ps = merge_pair(pid, total_suffix, "iliopsoas_left", "iliopsoas_right")     # psoas (+iliacus)

    def level_cols(vname, tag, with_psoas=False):
        """Extract at vertebral centre slice: whole-muscle area/density + BMD + erector spinae + psoas (lumbar only) + fat area."""
        m = verts.get(vname)
        if m is None:
            return {}
        zs = np.where(m.any(axis=(0, 1)))[0]
        nz = hu.shape[2]
        cz = vert_center_slice(m)
        if cz is None:
            return {}
        r = {f"{tag}_slice": cz}
        r[f"{tag}_edge"] = 1 if (zs.min() < 3 or zs.max() > nz - 3) else 0
        r[f"SMA_{tag}_cm2"], r[f"SM_density_{tag}_HU"], r[f"SM_density_{tag}_n"] = area_at(sm, cz, hu, sp, MUSCLE_HU)
        bmd, bn = hu_stats(m, hu, BONE_HU)
        r[f"BMD_{tag}_HU"], r[f"BMD_{tag}_n"] = bmd, bn
        if es is not None:
            r[f"ES_{tag}_cm2"], r[f"ES_density_{tag}_HU"], r[f"ES_{tag}_n"] = area_at(es, cz, hu, sp, MUSCLE_HU)
        if with_psoas and ps is not None:
            r[f"Psoas_{tag}_cm2"], r[f"Psoas_density_{tag}_HU"], r[f"Psoas_{tag}_n"] = area_at(ps, cz, hu, sp, MUSCLE_HU)
        if sat is not None:
            r[f"SAT_{tag}_cm2"], _, _ = area_at(sat, cz, hu, sp)
        if vat is not None:
            r[f"VAT_{tag}_cm2"], _, _ = area_at(vat, cz, hu, sp)
        return r

    # ---- vertebral levels ----
    l1 = verts.get("vertebrae_L1")
    if l1 is None:
        row["L1_unavailable"] = 1
    row.update(level_cols("vertebrae_T4", "T4"))
    row.update(level_cols("vertebrae_T12", "T12"))
    row.update(level_cols("vertebrae_L1", "L1", with_psoas=True))
    row.update(level_cols("vertebrae_L3", "L3", with_psoas=True))
    return row


def main():
    # determine per-pid directory suffix
    pids = sorted(os.path.basename(f).replace(".nii.gz", "") for f in glob.glob(os.path.join(NIFTI_DIR, "*.nii.gz")))
    tissue_suffix = ["tissue_types"]
    total_suffix = ["total_all", "total"]
    rows = []
    for pid in pids:
        try:
            r = extract_patient(pid, tissue_suffix, total_suffix)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"[{pid}] EXTRACT FAIL: {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}: {df.shape[0]} patients x {df.shape[1]} cols", flush=True)


if __name__ == '__main__':
    main()
