"""Data preparation: download, unzip, image lists and fixed splits.

Only ORIGINAL images are split (70/15/15 per class). Augmented copies follow their source image into the
training set of the fine-tuning experiments; validation and test sets contain real images only."""

import csv
import os
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .config import AUG_RE, CFG, DATA, DATA_VERSION, ROOT, TEST_MODE, read_kv
from .utils import read_json, write_json


def stage_data(log, note):
    ready = DATA / "ready.json"
    if ready.exists() and read_json(ready, {}).get("version") == DATA_VERSION:
        return read_json(ready)
    log("STAGE 0 - preparing data")
    zips = {}
    for z in ("datasets_initail.zip", "upload_package.zip"):
        local = ROOT / z
        zips[z] = local if local.exists() else DATA / "download" / z
    missing = [
        z
        for z, p in zips.items()
        if not p.exists() and not (DATA / z.replace(".zip", "") / ".unzipped").exists()
    ]
    if missing:
        log(
            f"Downloading {', '.join(missing)} from Hugging Face ({CFG['hf_repo']}) - about 15 GB, this can take a while"
        )
        note.send("Downloading data", f"Downloading {len(missing)} file(s) from Hugging Face")
        token = None
        tf_ = ROOT / "hf_token.txt"
        if tf_.exists():
            txt = tf_.read_text(encoding="utf-8-sig").strip()
            token = read_kv(tf_).get("token") or (txt if txt.startswith("hf_") else None)
        token = token or os.environ.get("HF_TOKEN") or None
        from huggingface_hub import hf_hub_download

        for z in missing:
            for attempt in range(1, 6):
                try:
                    p = hf_hub_download(
                        CFG["hf_repo"], z, repo_type="dataset", token=token, local_dir=str(DATA / "download")
                    )
                    zips[z] = Path(p)
                    break
                except Exception as e:
                    log(f"download attempt {attempt} for {z} failed: {e}")
                    if "401" in str(e) or "403" in str(e) or "Repository Not Found" in str(e):
                        raise RuntimeError(
                            "Hugging Face refused access. Put your token in hf_token.txt next to RUN.bat "
                            "(one line: token=hf_...), or put the two zip files next to RUN.bat."
                        )
                    if attempt == 5:
                        raise RuntimeError(
                            f"Could not download {z}. Check the internet, or put the zip next to RUN.bat."
                        )
                    time.sleep(30 * attempt)
    for z, p in zips.items():
        target = DATA / z.replace(".zip", "")
        if (target / ".unzipped").exists():
            continue
        log(f"Unzipping {z} ...")
        with zipfile.ZipFile(p) as zf:
            zf.extractall(DATA)
        (target / ".unzipped").write_text("ok")
    pkg = DATA / "upload_package"
    if not (pkg / "clean_manifest.csv").exists():
        raise RuntimeError(
            "upload_package/clean_manifest.csv is missing - the zip is not the expected package."
        )

    def src_path(s):
        return f"upload_package/{s}" if s.startswith("extras/") else f"datasets_initail/{s}"

    # ---- clean index
    rows = []
    with open(pkg / "clean_manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((src_path(r["source"]), r["clean_class"], r["file_name"]))
    tim = int(CFG["test_images_per_class"])
    if TEST_MODE:
        rows = _subsample(rows, tim)
    orig_key = {}
    for p, c, n in rows:
        if not AUG_RE.search(Path(n).stem):
            orig_key[(c, n)] = True
    clean = []
    for p, c, n in rows:
        stem, suf = Path(n).stem, Path(n).suffix
        if AUG_RE.search(stem):
            src = AUG_RE.sub("", stem) + suf
            group = f"{c}/{src}" if (c, src) in orig_key else "ORPHAN"
            clean.append((p, c, 1, group))
        else:
            clean.append((p, c, 0, f"{c}/{n}"))
    clean += top_up(clean, log)
    # ---- raw index
    raw = []
    with open(pkg / "raw_manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw.append(
                (f"datasets_initail/{r['source']}", r["raw_class"], 0, f"{r['raw_class']}/{r['file_name']}")
            )
    if TEST_MODE:
        raw = _subsample(raw, tim, key=1)
    cnt = Counter(r[1] for r in raw)
    small = {c for c, n in cnt.items() if n < (4 if TEST_MODE else 10)}
    raw = [r for r in raw if r[1] not in small]
    # ---- check files
    miss = [r[0] for r in clean + raw if not (DATA / r[0]).exists()]
    if miss:
        raise RuntimeError(f"{len(miss)} image files are missing after unzipping, e.g. {miss[:3]}")
    cleanlp = [r for r in clean if r[2] == 0]  # linear probe: real images only
    for name, idx in (("clean", clean), ("cleanlp", cleanlp), ("raw", raw)):
        with open(DATA / f"index_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "is_aug", "group"])
            w.writerows(idx)
    # ---- fixed splits
    nruns = int(CFG["lp_runs"])
    for name, idx in (("clean", clean), ("cleanlp", cleanlp), ("raw", raw)):
        for seed in range(max(nruns, int(CFG["ft_seeds"]))):
            tr, va, te = make_split(idx, seed)
            import numpy as np

            np.savez(
                DATA / f"split_{name}_s{seed}.npz",
                train=np.array(tr, dtype=np.int64),
                val=np.array(va, dtype=np.int64),
                test=np.array(te, dtype=np.int64),
            )
    info = dict(
        version=DATA_VERSION,
        clean_images=len(clean),
        clean_classes=len({r[1] for r in clean}),
        clean_topup_generated=sum(1 for r in clean if "/topup/" in "/" + r[0]),
        clean_originals=sum(1 for r in clean if r[2] == 0),
        clean_augmented=sum(1 for r in clean if r[2] == 1),
        clean_orphan_augmented=sum(1 for r in clean if r[3] == "ORPHAN"),
        raw_images=len(raw),
        raw_classes=len({r[1] for r in raw}),
        raw_dropped_small_classes=sorted(small),
    )
    write_json(ready, info)
    log(
        f"Data ready: clean {info['clean_images']} images / {info['clean_classes']} classes "
        f"({info['clean_augmented']} augmented), raw {info['raw_images']} images / {info['raw_classes']} classes"
    )
    return info


def top_up(clean, log):
    """Classes that stayed below the augmentation target (because images were removed after the original
    augmentation) are topped up again. Every new copy gets at least one random change; copies are named
    <original>_topupNNNN and are used for FINE-TUNING TRAINING only (leakage-free, like the other copies)."""
    import random

    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    target, thr = int(CFG["topup_target"]), int(CFG["topup_threshold"])
    if TEST_MODE:
        target = 24
    real, total = defaultdict(list), Counter()
    for p, c, a, g in clean:
        total[c] += 1
        if a == 0:
            real[c].append(p)
    rng = random.Random(42)
    nrng = np.random.RandomState(42)
    out = []
    for c in sorted(real):
        need = target - total[c]
        if need <= 0 or len(real[c]) >= thr:
            continue
        log(f"Top-up: {c} has {total[c]} images ({len(real[c])} real) -> adding {need} augmented copies")
        src = list(real[c])
        rng.shuffle(src)
        used = Counter()
        (DATA / "topup" / c).mkdir(parents=True, exist_ok=True)
        for k in range(need):
            sp = src[k % len(src)]
            stem, suf = Path(sp).stem, Path(sp).suffix
            name = f"{stem}_topup{used[sp]:04d}{suf}"
            used[sp] += 1
            dst = DATA / "topup" / c / name
            if not dst.exists():
                with Image.open(DATA / sp) as im:
                    im = im.convert("RGB")
                changed = False
                while not changed:
                    if rng.random() < 0.5:
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)
                        changed = True
                    if rng.random() < 0.3:
                        im = im.transpose(Image.FLIP_TOP_BOTTOM)
                        changed = True
                    if rng.random() < 0.7:
                        im = im.rotate(90 * rng.randint(1, 3), expand=True)
                        changed = True
                    if rng.random() < 0.1:
                        w, h = im.size
                        z = rng.uniform(1.05, 1.2)
                        cw, ch = int(w / z), int(h / z)
                        im = im.crop(
                            ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)
                        ).resize((w, h))
                        changed = True
                    if rng.random() < 0.2:
                        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
                        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
                        changed = True
                    if rng.random() < 0.4:
                        im = ImageEnhance.Color(im).enhance(rng.uniform(0.7, 1.3))
                        changed = True
                    if rng.random() < 0.1:
                        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.0)))
                        changed = True
                    if rng.random() < 0.1:
                        a = np.asarray(im).astype(np.float32) + nrng.normal(
                            0, rng.uniform(3, 10), (im.size[1], im.size[0], 3)
                        )
                        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
                        changed = True
                tmp = dst.with_name("tmp_" + dst.name)
                im.save(tmp, quality=95) if suf.lower() in (".jpg", ".jpeg") else im.save(tmp)
                os.replace(tmp, dst)
            out.append((f"topup/{c}/{name}", c, 1, f"{c}/{Path(sp).name}"))
    return out


def _subsample(rows, k, key=1):
    """TEST_MODE MODE only: keep k originals per class (+ some of their augmented copies)."""
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    out = []
    for c in sorted(by):
        if len(by[c][0]) == 3:  # clean rows: (path, class, file_name)
            orig = [r for r in by[c] if not AUG_RE.search(Path(r[2]).stem)][:k]
            names = {r[2] for r in orig}
            aug = [
                r
                for r in by[c]
                if AUG_RE.search(Path(r[2]).stem)
                and AUG_RE.sub("", Path(r[2]).stem) + Path(r[2]).suffix in names
            ][: k // 2]
            out += orig + aug
        else:
            out += by[c][:k]
    return out


def make_split(idx, seed, frac=(0.70, 0.15, 0.15)):
    """Split ORIGINALS per class 70/15/15; augmented copies follow their source into TRAIN only."""
    import numpy as np

    rng = np.random.RandomState(seed)
    by = defaultdict(list)
    for i, (p, c, a, g) in enumerate(idx):
        if a == 0:
            by[c].append(i)
    where = {}
    tr, va, te = [], [], []
    for c in sorted(by):
        ids = np.array(by[c])
        rng.shuffle(ids)
        n = len(ids)
        n_va = max(1, int(round(frac[1] * n)))
        n_te = max(1, int(round(frac[2] * n)))
        n_tr = n - n_va - n_te
        for part, lst in ((ids[:n_tr], tr), (ids[n_tr : n_tr + n_va], va), (ids[n_tr + n_va :], te)):
            lst.extend(int(i) for i in part)
    for i in tr:
        where[idx[i][3]] = "train"
    for i, (p, c, a, g) in enumerate(idx):
        if a == 1 and (g == "ORPHAN" or where.get(g) == "train"):
            tr.append(i)
    return sorted(tr), sorted(va), sorted(te)


def index_of(set_name):
    """Linear probe on the clean list uses REAL images only (no augmented copies)."""
    return "cleanlp" if set_name == "clean" else set_name


def load_index(name):
    rows = list(csv.DictReader(open(DATA / f"index_{name}.csv", encoding="utf-8")))
    return rows
