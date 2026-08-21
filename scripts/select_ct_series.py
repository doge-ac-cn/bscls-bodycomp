"""Select one CT series per patient for body-composition analysis.

Selection strategy: prefer the widest coverage, then the thinnest slice
thickness, then the largest number of slices.
"""
import json, os, sys

MANIFEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "series_manifest.json")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "selected_ct.json")

def pick(series_list):
    if not series_list:
        return None
    if len(series_list) == 1:
        return series_list[0]
    # multiple series: widest coverage (whole-body > chest) -> thinnest slice -> most slices
    return max(series_list, key=lambda s: (
        s.get('span_mm', 0) or 0,
        -(s.get('slice_thickness') if s.get('slice_thickness') is not None else 99),
        s['n_files'],
    ))

def main():
    m = json.load(open(MANIFEST))
    sel = {}
    for ds_key, patients in m.items():
        sel[ds_key] = {}
        for pid, series_list in patients.items():
            s = pick(series_list)
            if s is not None:
                sel[ds_key][pid] = s
    json.dump(sel, open(OUT, 'w'), ensure_ascii=False, indent=1)
    for ds_key, patients in sel.items():
        print(f"[{ds_key}] selected {len(patients)} patients")
        # slice-thickness distribution
        thicks = sorted({round(s['slice_thickness'], 2) for s in patients.values() if s.get('slice_thickness')})
        print(f"  slice-thickness distribution: {thicks}")

if __name__ == '__main__':
    main()
