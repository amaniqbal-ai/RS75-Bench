"""
Rebuild both datasets as image folders from the original public datasets and the package:
  <out>/clean/<75 classes>/...   (identical to the curated dataset, checked by SHA-256)
  <out>/raw/<132 classes>/...    (original images with the raw folder-name labels)

Needs: the 15 original datasets (datasets_initail) and the upload_package folder.

Usage:
  python build_datasets.py --orig /path/datasets_initail --package /path/upload_package --out /path/data
  add --verify to re-check every clean file against its SHA-256 (slower, recommended once)
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True)
ap.add_argument("--package", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--verify", action="store_true")
a = ap.parse_args()
orig, pkg, out = Path(a.orig).expanduser(), Path(a.package).expanduser(), Path(a.out).expanduser()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build(manifest, cls_col, target, verify):
    m = pd.read_csv(pkg / manifest)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} is not empty - choose a new --out")
    missing, bad = 0, 0
    for r in tqdm(m.itertuples(), total=len(m), desc=target.name):
        src = pkg / r.source if str(r.source).startswith("extras/") else orig / r.source
        dst = target / str(getattr(r, cls_col)) / r.file_name
        if not src.exists():
            missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if verify and hasattr(r, "sha256") and sha(dst) != r.sha256:
            bad += 1
    n = sum(1 for _ in target.rglob("*") if _.is_file())
    k = sum(1 for _ in target.iterdir() if _.is_dir())
    print(
        f"{target.name}: {n} images in {k} classes   (missing sources: {missing}"
        + (f", SHA mismatches: {bad})" if verify else ")")
    )


build("clean_manifest.csv", "clean_class", out / "clean", a.verify)
build("raw_manifest.csv", "raw_class", out / "raw", False)
print("done - expected about: clean 238,620 images / 75 classes, raw about 216,470 images / 132 classes")
