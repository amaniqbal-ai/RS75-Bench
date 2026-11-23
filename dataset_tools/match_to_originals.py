"""
Match the flattened raw-label copy back to the original files.

For every image in flat_dataset2 (132 raw classes, compressed JPG copies), find the SAME
picture among the original files in datasets_initail, so the raw benchmark can use the
original images with exactly the same raw labels.

Matching: perceptual hash (pHash) + difference hash (dHash), searched only inside the same
source dataset and source folder (taken from the flat file name, e.g.
01_RSI_CB128__cultivated_land__003274.jpg -> dataset 01_RSI_CB128, folder "cultivated land").
Nothing is copied, moved or changed.

Output (in --out):
  raw_original_labels.csv   original_path, raw_label, flat_path, split, distance, margin, status
  unmatched_flat.csv        flat images with no good match
  summary.txt

Usage:
  python match_to_originals.py --flat flat_dataset2 --orig datasets_initail --out raw_match
"""

import argparse
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
SKIP = {"images", "tests", "sample_data", "baseline"}  # non-scene raw folders
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def hashes(path):
    try:
        import imagehash
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            return path, int(str(imagehash.phash(im)), 16), int(str(imagehash.dhash(im)), 16)
    except Exception:
        return path, -1, -1


def popcount(a):
    a = a.astype(np.uint64)
    a = a - ((a >> np.uint64(1)) & np.uint64(0x5555555555555555))
    a = (a & np.uint64(0x3333333333333333)) + ((a >> np.uint64(2)) & np.uint64(0x3333333333333333))
    a = (a + (a >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((a * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def hash_all(paths, workers):
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for p, ph, dh in tqdm(ex.map(hashes, paths, chunksize=64), total=len(paths)):
            out[p] = (ph, dh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--out", default="raw_match")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-dist", type=int, default=12, help="max combined pHash+dHash distance for a match")
    args = ap.parse_args()
    flat, orig, out = Path(args.flat).expanduser(), Path(args.orig).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # ---- originals, indexed by (dataset, last folder name)
    orig_ds = {norm(p.name): p.name for p in orig.iterdir() if p.is_dir()}
    o_rows = []
    for ds in sorted(p for p in orig.iterdir() if p.is_dir()):
        for dp, _, fs in os.walk(ds):
            for f in fs:
                if os.path.splitext(f)[1].lower() in EXT:
                    o_rows.append((ds.name, norm(os.path.basename(dp)), os.path.join(dp, f)))
    # ---- flat images: .../<split?>/<raw_label>/<dataset>__<folder>__<id>.ext
    f_rows = []
    for dp, _, fs in os.walk(flat):
        for f in fs:
            if os.path.splitext(f)[1].lower() not in EXT:
                continue
            label = os.path.basename(dp)
            if label in SKIP or "__" not in f:
                continue
            parts = f.rsplit(".", 1)[0].split("__")
            split = os.path.basename(os.path.dirname(dp))
            f_rows.append(
                (label, parts[0], norm(parts[1]) if len(parts) > 2 else "", os.path.join(dp, f), split)
            )
    print(f"originals: {len(o_rows)}   flat images (scene classes): {len(f_rows)}", flush=True)

    def map_ds(name):  # 'RSICD' -> 'RSICD_images_correct'
        n = norm(name)
        if n in orig_ds:
            return orig_ds[n]
        c = [v for k, v in orig_ds.items() if k.startswith(n) or n.startswith(k)]
        return c[0] if len(c) == 1 else None

    t0 = time.time()
    H = hash_all([r[2] for r in o_rows] + [r[3] for r in f_rows], args.workers)
    print(f"hashed in {time.time() - t0:.0f}s", flush=True)

    idx = defaultdict(list)
    for ds, fold, p in o_rows:
        idx[(ds, fold)].append(p)
    groups = defaultdict(list)
    for r in f_rows:
        groups[(map_ds(r[1]), r[2])].append(r)

    res, unmatched = [], []
    for (ds, fold), rows in tqdm(groups.items(), desc="matching"):
        cands = [p for p in idx.get((ds, fold), []) if H[p][0] >= 0]  # skip unreadable originals
        if not cands:
            unmatched += [
                dict(flat_path=r[3], raw_label=r[0], reason=f"no original folder {ds}/{fold}") for r in rows
            ]
            continue
        cp = np.array([H[p][0] for p in cands], dtype=np.uint64)
        cd = np.array([H[p][1] for p in cands], dtype=np.uint64)
        for r in rows:
            fp, fd = H[r[3]]
            if fp < 0:
                unmatched.append(dict(flat_path=r[3], raw_label=r[0], reason="unreadable"))
                continue
            dist = popcount(cp ^ np.uint64(fp)) + popcount(cd ^ np.uint64(fd))
            o = np.argsort(dist)[:2]
            d1 = int(dist[o[0]])
            d2 = int(dist[o[1]]) if len(o) > 1 else 999
            ok = d1 <= args.max_dist and (d2 - d1) >= 3
            rec = dict(
                original_path=cands[o[0]],
                raw_label=r[0],
                flat_path=r[3],
                split=r[4],
                distance=d1,
                margin=d2 - d1,
                status="ok" if ok else "check",
            )
            res.append(rec)
            if not ok:
                unmatched.append(
                    dict(flat_path=r[3], raw_label=r[0], reason=f"weak match d={d1} margin={d2 - d1}")
                )

    df = pd.DataFrame(res)
    df.to_csv(out / "raw_original_labels.csv", index=False)
    pd.DataFrame(unmatched).to_csv(out / "unmatched_flat.csv", index=False)
    ok = df[df.status == "ok"]
    dup_orig = ok.original_path.duplicated(keep=False).sum()
    lines = [
        f"flat images (scene classes): {len(f_rows)}",
        f"matched with confidence: {len(ok)} ({100 * len(ok) / max(len(f_rows), 1):.2f}%)",
        f"weak / unmatched: {len(f_rows) - len(ok)}",
        f"raw classes covered: {ok.raw_label.nunique()}",
        f"originals used by more than one flat image: {dup_orig}",
        f"median match distance: {ok.distance.median() if len(ok) else 'n/a'}",
    ]
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
