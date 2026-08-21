#!/usr/bin/env python
"""Download TCIA imaging for both cohorts: NSCLC-Radiomics (Lung1) / NSCLC Radiogenomics (RG)."""
import os, sys, json, time
from tcia_utils import nbia

which = sys.argv[1]  # lung1 | rg
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", which)
os.makedirs(OUT, exist_ok=True)

COLL = {"lung1": "NSCLC-Radiomics", "rg": "NSCLC Radiogenomics"}[which]
MODS = {"lung1": ["CT", "SEG"], "rg": ["CT", "PET"]}[which]

log = []
t0 = time.time()
for mod in MODS:
    series = nbia.getSeries(collection=COLL, modality=mod)
    if series is None or len(series) == 0:
        log.append(f"{mod}: no series")
        continue
    print(f"[{which}] {mod}: {len(series)} series, downloading to {OUT}/{mod}/", flush=True)
    df = nbia.downloadSeries(series, path=os.path.join(OUT, mod), max_workers=8)
    log.append(f"{mod}: {len(series)} series done")

elapsed = (time.time() - t0) / 60
print(f"[{which}] ALL DONE in {elapsed:.1f} min", flush=True)
with open(os.path.join(OUT, "download_log.json"), "w") as f:
    json.dump({"collection": COLL, "modalities": MODS, "elapsed_min": round(elapsed, 1), "log": log}, f, indent=2)
