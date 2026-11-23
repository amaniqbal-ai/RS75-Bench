"""
Build the RAW dataset as image folders (132 classes) from the ORIGINAL files.

Reads raw_match/raw_labels_final.csv (made by finalize_raw_labels.py) and puts every original
image into   <out>/<raw_class>/<dataset>__<original_file_name>
 - each original file is used only once (identical duplicates listed twice are kept once)
 - the 467 unmatched images are not included
 - the original folders are never changed

Modes:
  --mode copy      real copies (needs about 14 GB free space)            [default]
  --mode hardlink  no extra space, same disk only (looks like normal files)
  --mode symlink   no extra space, shortcuts to the originals

Usage:
  python build_raw_folders.py --labels raw_match/raw_labels_final.csv --orig datasets_initail --out raw_132
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--orig", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
args = ap.parse_args()

orig = Path(args.orig).expanduser().resolve()
out = Path(args.out).expanduser()
if out.exists() and any(out.iterdir()):
    raise SystemExit(f"{out} is not empty - choose a new folder so nothing is overwritten")
out.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(Path(args.labels).expanduser())
before = len(d)
d = d.drop_duplicates("original_path", keep="first")
print(f"rows: {before}  ->  unique original images: {len(d)}  (removed {before - len(d)} repeated originals)")

# free-space check for copy mode
if args.mode == "copy":
    need = sum(os.path.getsize(p) for p in d.original_path if os.path.exists(p))
    free = shutil.disk_usage(out).free
    print(f"space needed: {need / 1e9:.1f} GB   free: {free / 1e9:.1f} GB")
    if need > free * 0.95:
        raise SystemExit("not enough free space - use --mode hardlink (same disk) or another drive")

missing, done, rows = 0, 0, []
for r in tqdm(d.itertuples(), total=len(d)):
    src = Path(r.original_path)
    if not src.exists():
        missing += 1
        continue
    dataset = src.resolve().relative_to(orig).parts[0]
    dst_dir = out / str(r.raw_label)
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / f"{dataset}__{src.name}"
    k = 1
    while dst.exists():  # same name twice in one class -> add a number
        dst = dst_dir / f"{dataset}__{src.stem}_{k}{src.suffix}"
        k += 1
    if args.mode == "copy":
        shutil.copy2(src, dst)
    elif args.mode == "hardlink":
        os.link(src, dst)
    else:
        os.symlink(src.resolve(), dst)
    rows.append((str(dst), str(src), r.raw_label))
    done += 1

pd.DataFrame(rows, columns=["new_path", "original_path", "raw_class"]).to_csv(
    out.parent / f"{out.name}_manifest.csv", index=False
)
classes = sorted(p.name for p in out.iterdir() if p.is_dir())
print(f"\nimages placed: {done}   missing originals: {missing}")
print(f"class folders: {len(classes)}")
print(f"manifest: {out.parent / (out.name + '_manifest.csv')}")
