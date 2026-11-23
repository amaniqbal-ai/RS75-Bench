"""
Find the same image in DIFFERENT source datasets.

Reads the ORIGINAL dataset folders (01_RSI_CB128, 03_EuroSAT, ...,
RSICD_images_correct, RSITMD, UCMerced_LandUse, WHU-RS19). Never changes any file.

Two checks:
  1. EXACT  : identical file bytes (SHA-256)
  2. NEAR   : same picture re-saved / re-compressed / resized (perceptual hash, pHash,
              Hamming distance <= --max-dist, default 3)

Outputs (in --out):
  all_images.csv                one row per image: dataset, class, path, sha256, phash
  exact_groups.csv              every group of identical files (with datasets and classes)
  near_pairs.csv                near-duplicate pairs from different datasets (+ distance)
  matrix_exact.csv              how many images of dataset A (row) have an exact copy in dataset B (col)
  matrix_near.csv               same for near-duplicates
  label_conflicts.csv           duplicates whose class names do not match (after simple name cleaning)
  summary.txt                   the main numbers

Usage:
    pip install pillow imagehash numpy pandas tqdm
    python find_duplicates.py --root /path/to/folder_with_the_15_datasets --out dup_report
"""

import argparse
import hashlib
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def hash_one(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        import imagehash
        from PIL import Image

        with Image.open(path) as im:
            ph = int(str(imagehash.phash(im.convert("RGB"))), 16)
        return path, h.hexdigest(), ph, ""
    except Exception as e:  # unreadable image
        return path, "", -1, str(e)[:120]


def norm_name(s):
    """Simple class-name cleaning so 'golf_course', 'Golf Course', 'golfcourse' compare equal."""
    s = s.split("/")[-1].lower()
    return re.sub(r"[^a-z]", "", s)


def popcount64(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101 & 0xFFFFFFFFFFFFFFFF) >> 56


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder that contains 01_RSI_CB128, 03_EuroSAT, ...")
    ap.add_argument("--out", default="dup_report")
    ap.add_argument(
        "--max-dist",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="pHash Hamming distance for 'near duplicate' (0-3; 3 = re-saved copies)",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- 1. list images
    rows = []
    for ds in sorted(p for p in root.iterdir() if p.is_dir()):
        for dirpath, _, files in os.walk(ds):
            for f in files:
                if os.path.splitext(f)[1].lower() in EXT:
                    full = os.path.join(dirpath, f)
                    cls = os.path.relpath(dirpath, ds).replace(os.sep, "/")
                    if cls == ".":  # images directly in the top folder: the folder itself is the class
                        cls = ds.name
                    rows.append((ds.name, cls, full))
    print(f"found {len(rows)} images in {len(set(r[0] for r in rows))} datasets", flush=True)

    # ---------- 2. hash (parallel)
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for p, sha, ph, err in tqdm(ex.map(hash_one, [r[2] for r in rows], chunksize=64), total=len(rows)):
            res[p] = (sha, ph, err)
    df = pd.DataFrame(rows, columns=["dataset", "cls", "path"])
    df["sha256"] = df.path.map(lambda p: res[p][0])
    df["phash"] = df.path.map(lambda p: res[p][1])
    df["error"] = df.path.map(lambda p: res[p][2])
    df["cls_norm"] = df.cls.map(norm_name)
    bad = df[df.error != ""]
    df = df[df.error == ""].reset_index(drop=True)
    df.assign(phash_hex=df.phash.map(lambda x: f"{x:016x}")).drop(columns="phash").to_csv(
        out / "all_images.csv", index=False
    )
    print(f"hashed in {time.time() - t0:.0f}s; {len(bad)} unreadable files", flush=True)
    dsets = sorted(df.dataset.unique())

    # ---------- 3. exact duplicates
    g = df.groupby("sha256")
    exact = df[df.sha256.isin(g.size()[g.size() > 1].index)].copy()
    exact["group"] = exact.groupby("sha256").ngroup()
    exact.sort_values(["group", "dataset"]).to_csv(out / "exact_groups.csv", index=False)
    mx_exact = pd.DataFrame(0, index=dsets, columns=dsets)
    conflicts = []
    for sha, grp in exact.groupby("sha256"):
        ds_here = grp.dataset.unique()
        for a in ds_here:
            n_a = (grp.dataset == a).sum()
            for b in ds_here:
                if a != b:
                    mx_exact.loc[a, b] += n_a
        if grp.cls_norm.nunique() > 1:
            conflicts.append(
                dict(
                    kind="exact",
                    group=sha[:12],
                    items=" | ".join(f"{d}:{c}" for d, c in zip(grp.dataset, grp.cls)),
                )
            )
    mx_exact.to_csv(out / "matrix_exact.csv")

    # ---------- 4. near duplicates across datasets (pigeonhole on 4 x 16-bit chunks; exact for dist <= 3)
    ph = df.phash.values.astype(np.uint64)
    ds_arr = df.dataset.values
    sha_arr = df.sha256.values
    cand = set()
    for k in range(4):
        chunk = (ph >> np.uint64(16 * k)) & np.uint64(0xFFFF)
        buckets = defaultdict(list)
        for i, c in enumerate(chunk):
            buckets[int(c)].append(i)
        for idx in buckets.values():
            if len(idx) < 2:
                continue
            if len(idx) > 2000:  # a flat/blank image hash shared by thousands: compare within, but cap
                idx = idx[:2000]
            for a_i in range(len(idx)):
                ia = idx[a_i]
                for ib in idx[a_i + 1 :]:
                    if ds_arr[ia] != ds_arr[ib] and sha_arr[ia] != sha_arr[ib]:
                        cand.add((ia, ib) if ia < ib else (ib, ia))
    near = []
    for ia, ib in cand:
        d = int(popcount64(int(ph[ia]) ^ int(ph[ib])))
        if d <= args.max_dist:
            near.append((ia, ib, d))
    near_df = pd.DataFrame(
        [
            dict(
                dataset_a=ds_arr[a],
                class_a=df.cls[a],
                path_a=df.path[a],
                dataset_b=ds_arr[b],
                class_b=df.cls[b],
                path_b=df.path[b],
                distance=d,
            )
            for a, b, d in near
        ]
    )
    near_df.to_csv(out / "near_pairs.csv", index=False)
    mx_near = pd.DataFrame(0, index=dsets, columns=dsets)
    if len(near_df):
        for (a, b), grp in near_df.groupby(["dataset_a", "dataset_b"]):
            mx_near.loc[a, b] += grp.path_a.nunique()
            mx_near.loc[b, a] += grp.path_b.nunique()
        for r in near_df.itertuples():
            if norm_name(r.class_a) != norm_name(r.class_b):
                conflicts.append(
                    dict(
                        kind=f"near(d={r.distance})",
                        group="",
                        items=f"{r.dataset_a}:{r.class_a} | {r.dataset_b}:{r.class_b}",
                    )
                )
    mx_near.to_csv(out / "matrix_near.csv")
    pd.DataFrame(conflicts).to_csv(out / "label_conflicts.csv", index=False)

    # ---------- 5. summary
    cross_exact = exact.groupby("sha256").dataset.nunique()
    lines = [
        f"images hashed: {len(df)} (unreadable: {len(bad)})",
        f"exact duplicate groups (any dataset): {exact.sha256.nunique()} groups, {len(exact)} files",
        f"exact groups spanning 2+ datasets: {(cross_exact > 1).sum()}",
        f"near-duplicate pairs across datasets (pHash dist <= {args.max_dist}): {len(near_df)}",
        f"duplicates with different class names (conflicts): {len(conflicts)}",
        "",
        "images of each dataset with an exact copy in another dataset:",
        mx_exact.sum(axis=1).to_string(),
        "",
        "see matrix_exact.csv / matrix_near.csv for the full dataset x dataset tables",
    ]
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
