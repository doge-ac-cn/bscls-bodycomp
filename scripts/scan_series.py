"""Scan all CT series in both cohorts; build a series manifest for the segmentation pipeline."""
import pydicom, glob, os, json, sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "series_manifest.json")

def is_usable_ct(d, exclude_desc=None):
    """Whether the series is a usable axial CT (excludes scouts, coronal/sagittal reformats, non-CT)."""
    if d.get('Modality') != 'CT':
        return False
    desc = str(d.get('SeriesDescription', '')).lower()
    # exclude scouts/coronal/sagittal/dose reports
    for bad in ['scout', 'topogram', 'surview', 'localizer', 'coronal', 'sagittal',
                'dose', 'drr', 'mip', 'vrt', 'curve']:
        if bad in desc:
            return False
    st = d.get('SliceThickness')
    if st is not None:
        try:
            if float(st) > 5.0:  # too thick for body composition
                return False
        except (TypeError, ValueError):
            pass
    return True

def scan_dataset(ds_name, out_key):
    base = os.path.join(ROOT, ds_name)
    patients = sorted(os.listdir(base))
    manifest = {}
    multi_ct = Counter()
    for pid in patients:
        pdir = os.path.join(base, pid)
        if not os.path.isdir(pdir):
            continue
        series_list = []
        for suid in sorted(os.listdir(pdir)):
            sdir = os.path.join(pdir, suid)
            dcms = glob.glob(os.path.join(sdir, '*.dcm'))
            if not dcms:
                continue
            try:
                d = pydicom.dcmread(dcms[0], stop_before_pixels=True)
                if is_usable_ct(d):
                    st_raw = d.get('SliceThickness')
                    st = float(st_raw) if st_raw is not None else None
                    series_list.append({
                        "series_uid": suid,
                        "n_files": len(dcms),
                        "slice_thickness": st,
                        "span_mm": (st * len(dcms)) if st else None,
                        "desc": str(d.get('SeriesDescription', '')),
                        "patient": pid,
                    })
            except Exception as e:
                print(f"  WARN {pid}/{suid}: {type(e).__name__}: {e}", file=sys.stderr)
        manifest[pid] = series_list
        if len(series_list) > 1:
            multi_ct[pid] = [s['series_uid'][:8] + f"({s['n_files']}f)" for s in series_list]
    n_total = len(manifest)
    n_with_ct = sum(1 for v in manifest.values() if v)
    n_multi = len(multi_ct)
    print(f"[{out_key}] patients={n_total}, with_CT={n_with_ct}, multi_CT={n_multi}")
    if n_multi:
        for pid, su in list(multi_ct.items())[:10]:
            print(f"  multi {pid}: {su}")
    return manifest

m = {}
m['lung1'] = scan_dataset('NSCLC-Radiomics', 'Lung1')
m['radiogenomics'] = scan_dataset('NSCLC-Radiogenomics', 'RG')

with open(OUT, 'w') as f:
    json.dump(m, f, ensure_ascii=False, indent=1)
print(f"\nmanifest saved: {OUT}")

if __name__ == '__main__':
    pass  # module importable; scanning runs in main flow
