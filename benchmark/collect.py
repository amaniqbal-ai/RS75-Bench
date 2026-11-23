#!/usr/bin/env python3
"""
Pack everything needed for the paper and the code release into 2 zip files.
Started by COLLECT.bat. Reads only; nothing in the benchmark folder is changed or deleted.

  COLLECT_OUTPUT/SatBench_results_<run>.zip  (small)  results (no .pt), logs, image lists, splits,
                                                      manifests, selection, settings, model info
  COLLECT_OUTPUT/SatBench_models_<run>.zip   (~5 GB)  the fine-tuned models (best.pt, all seeds)

Secrets (hf_token.txt, rclone.conf, notify.txt) are NEVER included.
"""

import hashlib
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "COLLECT_OUTPUT"
SECRETS = {"hf_token.txt", "rclone.conf", "notify.txt"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def gather():
    res, mod = [], []
    for p in sorted((HERE / "results").rglob("*")) if (HERE / "results").exists() else []:
        if p.is_file() and not p.name.endswith((".tmp", ".tmp.npy")):
            (mod if p.suffix == ".pt" else res).append(p)
    if (HERE / "logs").exists():
        res += sorted(p for p in (HERE / "logs").rglob("*") if p.is_file())
    for pat in (
        "data/ready.json",
        "data/index_*.csv",
        "data/split_*.npz",
        "data/upload_package/clean_manifest.csv",
        "data/upload_package/raw_manifest.csv",
        "state/selection.json",
        "state/run_name.txt",
        "state/FINISHED",
        "features/*/*/meta.json",
        "settings.ini",
        "README.txt",
        "main.py",
        "satbench/*.py",
        "setup.ps1",
        "RUN.bat",
        "STOP.bat",
        "COLLECT.bat",
        "collect.py",
    ):
        res += sorted(p for p in HERE.glob(pat) if p.is_file())
    res = [p for p in dict.fromkeys(res) if p.name not in SECRETS]
    mod = [p for p in mod if p.name not in SECRETS]
    return res, mod


def make_zip(path, files, compress):
    tmp = path.with_suffix(".zip.tmp")
    sums, total = [], 0
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(tmp, "w", mode, allowZip64=True) as z:
        for i, p in enumerate(files, 1):
            rel = p.relative_to(HERE).as_posix()
            z.write(p, rel)
            total += p.stat().st_size
            sums.append(f"{sha256(p)}  {rel}")
            if i % 200 == 0 or i == len(files):
                print(f"   {i}/{len(files)} files ({total / 1e9:.2f} GB)", flush=True)
        z.writestr("checksums_sha256.txt", "\n".join(sums) + "\n")
    os.replace(tmp, path)
    return total


def main():
    run = (
        (HERE / "state" / "run_name.txt").read_text().strip()
        if (HERE / "state" / "run_name.txt").exists()
        else "run"
    )
    if (HERE / "state" / "main.pid").exists():
        print("NOTE: the benchmark seems to be RUNNING. Collecting anyway (results so far).")
    if not (HERE / "state" / "FINISHED").exists():
        print("NOTE: the benchmark has not finished yet - the zip contains the results so far.")
    OUT.mkdir(exist_ok=True)
    res, mod = gather()
    n_res = sum(1 for p in res if p.name == "result.json")
    print(
        f"\nRun: {run}\n  result/log/list files: {len(res)}  (finished jobs: {n_res})\n  model files (best.pt): {len(mod)}\n"
    )

    print("1/2  Creating the RESULTS zip ...")
    z1 = OUT / f"SatBench_results_{run}.zip"
    make_zip(z1, res, compress=True)
    print(f"     -> {z1.name}  ({z1.stat().st_size / 1e6:.1f} MB)\n")

    z2 = None
    if mod:
        print("2/2  Creating the MODELS zip (large, please wait) ...")
        z2 = OUT / f"SatBench_models_{run}.zip"
        make_zip(z2, mod, compress=False)
        print(f"     -> {z2.name}  ({z2.stat().st_size / 1e9:.2f} GB)\n")
    else:
        print("2/2  No fine-tuned models (best.pt) found - skipped.\n")

    # quick check: open both zips again
    for z in (z1, z2):
        if z:
            with zipfile.ZipFile(z) as zz:
                bad = zz.testzip()
            print(f"Check {z.name}: {'OK' if bad is None else 'DAMAGED FILE: ' + bad}")
    print(
        f"\nDONE. Please send the files in:\n  {OUT}\n"
        f"Small results zip: e-mail / WhatsApp / Google Drive.  Models zip: Google Drive or USB stick."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nPROBLEM: {type(e).__name__}: {e}\nPlease send a screenshot of this window.")
        sys.exit(1)
