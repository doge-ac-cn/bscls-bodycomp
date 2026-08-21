#!/usr/bin/env python3
"""Generate data/mask_manifest.json: SHA-256 of every TotalSegmentator mask.

The segmentation outputs (~16 GB for 633 patients) are not shipped in the
repository. This script produces a small manifest that lets users verify that
their locally produced masks match the ones used in the manuscript.

Usage:
    python scripts/make_mask_manifest.py
"""
import hashlib
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_DIR = os.path.join(BASE, "data", "ts_masks")
OUT = os.path.join(BASE, "data", "mask_manifest.json")

# Only the directories actually consumed by extract_bodycomp.py:
#   *_tissue_types (skeletal_muscle, subcutaneous_fat, torso_fat)
#   total task      (vertebrae, autochthon, iliopsoas, ...)
# The total task may live under *_total_all (with a *_total symlink) or as a
# real *_total directory when the *_total_all alias was never created.
# Intermediate dirs (*_total_dcm, *_total_dcm2niix, *_fusion_all) are excluded.
KEEP_SUFFIXES = ("_tissue_types", "_total_all")
TOTAL_REAL_SUFFIX = "_total"


def main():
    rows, total = [], 0
    for pid in sorted(os.listdir(MASK_DIR)):
        d = os.path.join(MASK_DIR, pid)
        if not os.path.isdir(d) or os.path.islink(d):
            continue
        if pid.endswith(KEEP_SUFFIXES):
            dirs = [d]
        elif pid.endswith(TOTAL_REAL_SUFFIX) and not os.path.exists(pid[:-len(TOTAL_REAL_SUFFIX)] + "_total_all"):
            dirs = [d]  # real *_total directory, no *_total_all alias
        else:
            continue
        for dd in dirs:
            for f in sorted(os.listdir(dd)):
                if not f.endswith(".nii.gz"):
                    continue
                p = os.path.join(dd, f)
                sz = os.path.getsize(p)
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                rows.append({"mask_dir": pid, "file": f, "size_bytes": sz, "sha256": h.hexdigest()})
                total += sz
    manifest = {
        "n_dirs": len({r["mask_dir"] for r in rows}),
        "n_files": len(rows),
        "total_bytes": total,
        "files": rows,
    }
    with open(OUT, "w") as fh:
        json.dump(manifest, fh)
    print(f"OK rows={len(rows)} dirs={manifest['n_dirs']} total_GB={total / 1e9:.2f}")


if __name__ == "__main__":
    main()
