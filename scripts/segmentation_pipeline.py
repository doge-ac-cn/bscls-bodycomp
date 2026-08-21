"""Run TotalSegmentator whole-body segmentation for a set of NIfTI scans.

Serial, resumable execution (--start/--end) designed for memory-limited
machines; multi-worker parallel execution is not recommended (high peak
memory). Produces both total and tissue-type masks.

Usage: python segmentation_pipeline.py --start 0 --end 633
"""
import json, os, sys, time, glob, subprocess, argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE, "data", "raw")
NIFTI_DIR = os.path.join(BASE, "data", "nifti")
MASK_DIR = os.path.join(BASE, "data", "ts_masks")
DCM2NIIX = os.environ.get("DCM2NIIX", "dcm2niix")


def convert(pid, dicom_dir):
    tmp = os.path.join(NIFTI_DIR, f".tmp_{pid}")
    os.makedirs(tmp, exist_ok=True)
    r = subprocess.run([DCM2NIIX, "-z", "y", "-o", tmp, "-f", "%p_%s", dicom_dir],
                       capture_output=True, text=True)
    nii = glob.glob(os.path.join(tmp, "*.nii.gz"))
    if not nii:
        return False, (r.stderr or r.stdout)[-200:]
    out = os.path.join(NIFTI_DIR, f"{pid}.nii.gz")
    os.replace(nii[0], out)
    for extra in glob.glob(os.path.join(tmp, "*")):
        os.remove(extra)
    os.rmdir(tmp)
    return True, ""


def run_ts(pid, task, minfiles):
    nii = os.path.join(NIFTI_DIR, f"{pid}.nii.gz")
    out = os.path.join(MASK_DIR, f"{pid}_{task}")
    if os.path.exists(out) and len(os.listdir(out)) >= minfiles:
        return "skip", 0
    from totalsegmentator.python_api import totalsegmentator
    t0 = time.time()
    totalsegmentator(nii, out, task=task, nora_tag=False, quiet=True,
                     force_split=True, nr_thr_saving=2)
    return "ok", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0, help="job index start (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="job index end (exclusive)")
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--skip-total", action="store_true")
    ap.add_argument("--skip-tissue", action="store_true")
    args = ap.parse_args()

    sel = json.load(open(os.path.join(BASE, "data", "selected_ct.json")))
    os.makedirs(NIFTI_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    jobs = []
    for ds_key, patients in sel.items():
        ds_dir = 'NSCLC-Radiomics' if ds_key == 'lung1' else 'NSCLC-Radiogenomics'
        for pid, s in patients.items():
            if args.patients and pid not in args.patients.split(','):
                continue
            jobs.append((ds_dir, pid, s))
            if args.limit and len(jobs) >= args.limit:
                break
        if args.limit and len(jobs) >= args.limit:
            break

    # sharding: disjoint case ranges when running parallel workers to avoid concurrent writes
    jobs = jobs[args.start:args.end]

    print(f"Jobs: {len(jobs)}", flush=True)
    t0 = time.time()
    for i, (ds_dir, pid, s) in enumerate(jobs, 1):
        dicom_dir = os.path.join(DATA_ROOT, ds_dir, pid, s['series_uid'])
        # 1) conversion
        if not args.skip_convert:
            nii = os.path.join(NIFTI_DIR, f"{pid}.nii.gz")
            if not os.path.exists(nii):
                ok, err = convert(pid, dicom_dir)
                if not ok:
                    print(f"[{i}/{len(jobs)}] {pid} CONVERT FAIL: {err}", flush=True)
                    continue
        # 2) total
        if not args.skip_total:
            try:
                st, dur = run_ts(pid, "total", 5)
                print(f"[{i}/{len(jobs)}] {pid} total: {st} ({dur:.0f}s, {time.time()-t0:.0f}s total)", flush=True)
            except Exception as e:
                print(f"[{i}/{len(jobs)}] {pid} total FAIL: {type(e).__name__}: {str(e)[:150]}", flush=True)
        # 3) tissue
        if not args.skip_tissue:
            try:
                st, dur = run_ts(pid, "tissue_types", 3)
                print(f"[{i}/{len(jobs)}] {pid} tissue: {st} ({dur:.0f}s)", flush=True)
            except Exception as e:
                print(f"[{i}/{len(jobs)}] {pid} tissue FAIL: {type(e).__name__}: {str(e)[:150]}", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
