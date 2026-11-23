"""Model creation, image loading and feature extraction (resumable, with automatic batch reduction)."""

import math
import os
import time

from .config import CFG, DATA, FEAT, STATE, TEST_MODE, short_name
from .data import index_of, load_index
from .hardware import gpu_count
from .utils import is_oom, read_json, write_json


def torch_setup():
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch, dev


def n_workers():
    if TEST_MODE:
        return 0
    if CFG["workers_per_job"] != "auto":
        return int(CFG["workers_per_job"])
    slots = max(1, gpu_count())
    return max(2, min(16, (os.cpu_count() or 8) // slots - 2))


class RandRot90:
    """Satellite images have no 'up': rotate by 0/90/180/270 degrees at random."""

    def __call__(self, im):
        import random

        k = random.randint(0, 3)
        return im.rotate(90 * k, expand=True) if k else im


class ImageList:
    def __init__(self, paths, labels, tf):
        self.p, self.y, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        from PIL import Image

        try:
            with Image.open(DATA / self.p[i]) as im:
                im = im.convert("RGB")
        except Exception:
            im = Image.new("RGB", (256, 256))
        return self.tf(im), self.y[i]


def create_model(name, num_classes=0, **kw):
    import timm

    pretrained = not TEST_MODE
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kw)
    except TypeError:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def extract_features(model_name, set_name, log):
    """Features for every image of a set, saved in chunks (resumable). Returns (X float32, meta)."""
    import numpy as np
    import timm

    torch, dev = torch_setup()
    outdir = FEAT / set_name / short_name(model_name)
    outdir.mkdir(parents=True, exist_ok=True)
    final = outdir / "features.npy"
    idx = load_index(index_of(set_name))
    if (
        final.exists()
        and (outdir / "meta.json").exists()
        and read_json(outdir / "meta.json", {}).get("n") == len(idx)
    ):
        return np.load(final, mmap_mode="r"), read_json(outdir / "meta.json")
    paths = [r["path"] for r in idx]
    model = create_model(model_name).eval().to(dev)
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    chunk = 400 if TEST_MODE else 40000
    n_chunks = math.ceil(len(paths) / chunk)
    bs = int(CFG["feature_batch"])
    t0 = time.time()
    done_imgs = 0
    size = int(cfg["input_size"][-1])
    if size > 224:  # big inputs (e.g. EfficientNet-B7 at 600 px): start with a proportionally smaller batch
        bs = max(8, int(bs * (224 / size) ** 2) // 8 * 8)
        log(f"  input {size}px -> starting with batch {bs}")
    use_amp = dev.type == "cuda"
    for ci in range(n_chunks):
        cf = outdir / f"chunk_{ci:04d}.npy"
        if cf.exists():
            continue
        sl = list(range(ci * chunk, min(len(paths), (ci + 1) * chunk)))
        while True:
            try:
                dl = torch.utils.data.DataLoader(
                    ImageList([paths[i] for i in sl], [0] * len(sl), tf),
                    batch_size=bs,
                    num_workers=n_workers(),
                    pin_memory=dev.type == "cuda",
                    persistent_workers=False,
                )
                out = []
                with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                    for x, _ in dl:
                        f = model(x.to(dev, non_blocking=True))
                        out.append(f.float().cpu().numpy())
                arr = np.concatenate(out).astype(np.float32)
                if not np.isfinite(arr).all():
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                np.save(cf.with_suffix(".tmp.npy"), arr)
                os.replace(cf.with_suffix(".tmp.npy"), cf)
                done_imgs += len(sl)
                log(
                    f"  features {set_name} {short_name(model_name)}: chunk {ci + 1}/{n_chunks} "
                    f"({done_imgs / max(time.time() - t0, 1e-6):.0f} img/s, batch {bs})"
                )
                write_json(STATE / "progress" / f"{os.getpid()}.json", {"chunk": ci + 1, "chunks": n_chunks})
                break
            except Exception as e:
                # out of memory, or other GPU/cuDNN size limits (e.g. 'canUse32BitIndexMath', CUDNN_STATUS_*):
                # retry the same chunk with half the batch
                if (is_oom(e) or isinstance(e, RuntimeError)) and bs > 4:
                    first = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
                    bs //= 2
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    log(f"  GPU problem ({first}) -> retrying this chunk with batch {bs}")
                    continue
                raise
    X = np.concatenate([np.load(outdir / f"chunk_{ci:04d}.npy") for ci in range(n_chunks)])
    np.save(outdir / "features.tmp.npy", X)
    os.replace(outdir / "features.tmp.npy", final)
    for ci in range(n_chunks):
        (outdir / f"chunk_{ci:04d}.npy").unlink(missing_ok=True)
    meta = dict(
        model=model_name,
        dim=int(X.shape[1]),
        n=int(X.shape[0]),
        params_m=round(params, 2),
        input_size=int(cfg["input_size"][-1]),
        extract_seconds=round(time.time() - t0, 1),
        gpu_images_per_s=speed_test(model, cfg, dev),
    )
    write_json(outdir / "meta.json", meta)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(final, mmap_mode="r"), meta


def speed_test(model, cfg, dev):
    import torch

    try:
        bs = 4 if dev.type == "cpu" else 64
        it = 2 if dev.type == "cpu" else 30
        x = torch.randn(bs, *cfg["input_size"], device=dev)
        with torch.inference_mode(), torch.autocast(
            dev.type, dtype=torch.float16, enabled=dev.type == "cuda"
        ):
            for _ in range(3 if dev.type == "cuda" else 1):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            for _ in range(it):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        return round(bs * it / (time.time() - t), 1)
    except Exception:
        return None
