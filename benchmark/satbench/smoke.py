"""Short end-to-end check that runs before the long experiments."""

import math
from collections import defaultdict

from .config import DATA, RES, STATE, models_by_family
from .data import load_index
from .linear_probe import svm_predict, svm_train
from .models import ImageList, create_model, n_workers, torch_setup
from .utils import write_json


def smoke_job(log):
    """Fast checks so problems show up in the first hour, not after days."""
    import timm

    torch, dev = torch_setup()
    rep = dict(
        device=str(dev), gpus=torch.cuda.device_count() if dev.type == "cuda" else 0, models={}, problems=[]
    )
    log(f"Smoke test on {dev} ({rep['gpus']} GPU visible to this job)")
    if dev.type == "cuda":
        a = torch.randn(2048, 2048, device=dev)
        float((a @ a).sum())
    idx = load_index("clean")
    sample = [r["path"] for r in idx if r["is_aug"] == "0"][:2]
    fams = models_by_family()
    for m in fams["CNN"] + fams["ViT"]:
        try:
            net = create_model(m).eval().to(dev)
            cfg = timm.data.resolve_model_data_config(net)
            tf = timm.data.create_transform(**cfg)
            from PIL import Image

            x = torch.stack([tf(Image.open(DATA / p).convert("RGB")) for p in sample]).to(dev)
            with torch.inference_mode():
                f = net(x)
            assert f.ndim == 2 and torch.isfinite(f).all()
            rep["models"][m] = dict(ok=True, dim=int(f.shape[1]), input=int(cfg["input_size"][-1]))
            log(f"  model OK  {m}  (features {f.shape[1]}, input {cfg['input_size'][-1]})")
            del net
        except Exception as e:
            rep["models"][m] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
            log(f"  model FAILED {m}: {type(e).__name__}: {str(e)[:300]}")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    # mini linear probe + mini fine-tune on a temporary tiny copy of the clean index
    ok_models = [m for m, v in rep["models"].items() if v["ok"]]
    if not ok_models:
        raise RuntimeError(
            "No model could be loaded - check the internet connection (weights come from Hugging Face)."
        )
    tiny = min(ok_models, key=lambda m: rep["models"][m]["dim"])
    by = defaultdict(list)
    for i, r in enumerate(idx):
        if r["is_aug"] == "0":
            by[r["label"]].append(i)
    keep = [i for c in sorted(by) for i in by[c][:8]]
    rows = [idx[i] for i in keep]
    xs = []
    net = create_model(tiny).eval().to(dev)
    tf = timm.data.create_transform(**timm.data.resolve_model_data_config(net))
    dl = torch.utils.data.DataLoader(
        ImageList([r["path"] for r in rows], [0] * len(rows), tf),
        batch_size=64,
        num_workers=min(2, n_workers()),
    )
    with torch.inference_mode():
        for x, _ in dl:
            xs.append(net(x.to(dev)).float())
    X = torch.cat(xs)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    cls = sorted({r["label"] for r in rows})
    yv = torch.tensor([cls.index(r["label"]) for r in rows], device=dev)
    W, b = svm_train(torch, X, yv, [1e-4], len(cls), 0, 3, 5e-3, 256)
    acc = float((svm_predict(torch, X, W, b)[0] == yv).float().mean())
    assert math.isfinite(acc), "linear probe produced NaN"
    log(f"  mini linear probe OK (train acc {acc:.2f})")
    net = create_model(tiny, num_classes=len(cls)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-4)
    tf2 = timm.data.create_transform(**timm.data.resolve_model_data_config(net))
    dl = torch.utils.data.DataLoader(
        ImageList([r["path"] for r in rows[:64]], [cls.index(r["label"]) for r in rows[:64]], tf2),
        batch_size=16,
        shuffle=True,
        num_workers=min(2, n_workers()),
    )
    net.train()
    for x, yy in dl:
        loss = torch.nn.functional.cross_entropy(net(x.to(dev)), yy.to(dev))
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.save(net.state_dict(), STATE / "smoke_ckpt.pt")
    net.load_state_dict(torch.load(STATE / "smoke_ckpt.pt", map_location=dev))
    (STATE / "smoke_ckpt.pt").unlink(missing_ok=True)
    log(f"  mini fine-tune OK (loss {float(loss):.3f})")
    rep["failed_models"] = [m for m, v in rep["models"].items() if not v["ok"]]
    out = RES / "smoke"
    out.mkdir(exist_ok=True)
    write_json(out / "result.json", rep)
