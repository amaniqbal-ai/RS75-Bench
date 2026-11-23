"""
Create a small package so that BOTH datasets
(clean 75 classes, raw 132 classes) can be rebuilt anywhere from the 15 original datasets.

Package contents (--out):
  clean_manifest.csv   clean_class, file_name, source  (source = path inside datasets_initail, or extras/...)
  raw_manifest.csv     raw_class, file_name, source
  extras/              clean images that do not exist in the originals (augmented copies etc.), copied as-is
  README.txt

Nothing in DataSet or datasets_initail is changed.

Usage:
  python make_package.py --clean DataSet --orig datasets_initail \
      --raw-labels raw_match/raw_labels_final.csv --out upload_package
"""

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def list_images(root):
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in EXT:
                out.append(os.path.join(dp, f))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--clean", required=True)
ap.add_argument("--orig", required=True)
ap.add_argument("--raw-labels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
clean, orig, out = Path(a.clean).expanduser(), Path(a.orig).expanduser().resolve(), Path(a.out).expanduser()
if out.exists() and any(out.iterdir()):
    raise SystemExit(f"{out} is not empty - choose a new folder")
(out / "extras").mkdir(parents=True, exist_ok=True)

# ---------- clean: match every DataSet file to an identical original by SHA-256
o_files = list_images(orig)
c_files = [p for p in list_images(clean) if os.path.dirname(p) != str(clean)]  # skip files lying in the root
print(f"originals: {len(o_files)}   clean images: {len(c_files)}")
with ThreadPoolExecutor(a.workers) as ex:
    o_hash = dict(zip(o_files, tqdm(ex.map(sha, o_files), total=len(o_files), desc="hash originals")))
    c_hash = dict(zip(c_files, tqdm(ex.map(sha, c_files), total=len(c_files), desc="hash clean")))
by_hash = {}
for p, h in o_hash.items():
    by_hash.setdefault(h, p)

rows, n_extra = [], 0
for p in tqdm(c_files, desc="clean manifest"):
    cls = os.path.relpath(os.path.dirname(p), clean)
    name = os.path.basename(p)
    src = by_hash.get(c_hash[p])
    if src:
        rows.append((cls, name, os.path.relpath(src, orig), c_hash[p]))
    else:
        dst = out / "extras" / cls / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        rows.append((cls, name, f"extras/{cls}/{name}", c_hash[p]))
        n_extra += 1
cm = pd.DataFrame(rows, columns=["clean_class", "file_name", "source", "sha256"])
cm.to_csv(out / "clean_manifest.csv", index=False)

# ---------- raw: from raw_labels_final.csv (each original once)
r = pd.read_csv(Path(a.raw_labels).expanduser()).drop_duplicates("original_path")
r = r[r.original_path.map(os.path.exists)]
r["source"] = r.original_path.map(lambda p: os.path.relpath(Path(p).resolve(), orig))
r["file_name"] = r.source.map(lambda s: s.split(os.sep)[0] + "__" + os.path.basename(s))
r["file_name"] = r.groupby(["raw_label", "file_name"]).file_name.transform(
    lambda s: [n if i == 0 else f"{Path(n).stem}_{i}{Path(n).suffix}" for i, n in enumerate(s)]
)
rm = r.rename(columns={"raw_label": "raw_class"})[["raw_class", "file_name", "source"]]
rm.to_csv(out / "raw_manifest.csv", index=False)

extra_gb = sum(f.stat().st_size for f in (out / "extras").rglob("*") if f.is_file()) / 1e9
txt = [
    f"clean images: {len(cm)} in {cm.clean_class.nunique()} classes "
    f"({len(cm) - n_extra} from originals, {n_extra} in extras/)",
    f"raw images:   {len(rm)} in {rm.raw_class.nunique()} classes (all from originals)",
    f"extras size:  {extra_gb:.1f} GB",
    "",
    "Upload: this whole folder + the datasets_initail folder (15 original datasets).",
    "Rebuild:  python build_datasets.py --orig <datasets_initail> --package <this folder> --out <target>",
]
(out / "README.txt").write_text("\n".join(txt))
print("\n" + "\n".join(txt))
