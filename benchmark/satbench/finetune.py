"""Full fine-tuning of a backbone on the clean list (resumable after every epoch)."""

import math
import os
import time

from .config import CFG, DATA, RES, STATE, TEST_MODE, family_of, short_name
from .data import load_index
from .linear_probe import metrics
from .models import ImageList, RandRot90, create_model, n_workers, torch_setup
from .utils import is_oom, now, write_json


def finetune_job(model_name, seed, log):
    import numpy as np
    import timm
    from torchvision import transforms as T

    torch, dev = torch_setup()
    out = RES / f"ft_clean__{model_name}__s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    idx = load_index("clean")
    classes = sorted({r["label"] for r in idx})
    c2i = {c: i for i, c in enumerate(classes)}
    paths = [r["path"] for r in idx]
    y = np.array([c2i[r["label"]] for r in idx])
    n_cls = len(classes)
    sp = np.load(DATA / f"split_clean_s{seed}.npz")
    tr, va, te = sp["train"], sp["val"], sp["test"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = create_model(model_name, num_classes=n_cls, drop_path_rate=0.1).to(dev)
    cfg = timm.data.resolve_model_data_config(model)
    size = int(cfg["input_size"][-1])
    tf_train = T.Compose(
        [
            RandRot90(),
            T.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(cfg["mean"], cfg["std"]),
        ]
    )
    tf_eval = timm.data.create_transform(**cfg)
    epochs = int(CFG["ft_epochs"])
    target_bs = 8 if TEST_MODE else int(CFG["ft_batch"])
    ck = out / "ckpt_last.pt"
    state = torch.load(ck, map_location="cpu", weights_only=False) if ck.exists() else None
    bs = state["bs"] if state else target_bs
    bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG["ft_lr"]), weight_decay=0.05)
    start_ep, best_val, hist = 0, -1.0, []
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_ep, best_val, hist = state["epoch"] + 1, state["best_val"], state["hist"]
        log(f"  resuming fine-tune from epoch {start_ep}")
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def loader(ids, tf, shuffle, b):
        return torch.utils.data.DataLoader(
            ImageList([paths[i] for i in ids], y[ids].tolist(), tf),
            batch_size=b,
            shuffle=shuffle,
            num_workers=n_workers(),
            pin_memory=dev.type == "cuda",
            drop_last=shuffle and len(ids) > b,
            persistent_workers=False,
        )

    def evaluate(ids):
        model.eval()
        preds = []
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
            for x, _ in loader(ids, tf_eval, False, bs * 2):
                preds.append(model(x.to(dev)).argmax(1).cpu())
        return torch.cat(preds).numpy()

    steps_per_ep = max(1, len(tr) // target_bs)
    total = steps_per_ep * epochs
    warm = steps_per_ep
    ep = start_ep
    while ep < epochs:
        try:
            model.train()
            accum = max(1, target_bs // bs)
            step = ep * steps_per_ep
            t0 = time.time()
            seen = 0
            opt.zero_grad(set_to_none=True)
            for i, (x, yy) in enumerate(loader(tr, tf_train, True, bs)):
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                    loss = crit(model(x.to(dev, non_blocking=True)), yy.to(dev)) / accum
                loss.backward()
                seen += len(x)
                if (i + 1) % accum == 0:
                    lr_t = float(CFG["ft_lr"]) * (
                        step / warm
                        if step < warm
                        else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                    )
                    for gp in opt.param_groups:
                        gp["lr"] = lr_t
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
            pv = evaluate(va)
            vacc = float((pv == y[va]).mean())
            hist.append(
                dict(
                    epoch=ep,
                    val_acc=vacc,
                    seconds=round(time.time() - t0, 1),
                    images_per_s=round(seen / (time.time() - t0), 1),
                )
            )
            if vacc > best_val:
                best_val = vacc
                torch.save(model.state_dict(), out / "best.pt")
            torch.save(
                dict(
                    model=model.state_dict(),
                    opt=opt.state_dict(),
                    epoch=ep,
                    best_val=best_val,
                    hist=hist,
                    bs=bs,
                ),
                out / "ckpt.tmp",
            )
            os.replace(out / "ckpt.tmp", ck)
            log(
                f"  fine-tune {short_name(model_name)} seed {seed} epoch {ep + 1}/{epochs}: val acc {vacc:.4f} (batch {bs})"
            )
            write_json(STATE / "progress" / f"{os.getpid()}.json", {"epoch": ep + 1, "epochs": epochs})
            ep += 1
        except Exception as e:
            if is_oom(e) and bs > 2:
                bs //= 2
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log(f"  out of GPU memory -> repeating epoch {ep + 1} with batch {bs} (same effective batch)")
                continue
            raise
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    pt = evaluate(te)
    m, rc_c, f1_c, cm = metrics(y[te], pt, n_cls)
    np.save(out / "confusion.npy", cm)
    res = dict(
        job=out.name,
        model=model_name,
        family=family_of(model_name),
        seed=seed,
        set="clean",
        n_classes=n_cls,
        test=m,
        best_val_acc=best_val,
        history=hist,
        input_size=size,
        epochs=epochs,
        finished=now(),
        per_class_recall=dict(zip(classes, np.round(rc_c, 4).tolist())),
    )
    write_json(out / "result.json", res)
    for f in ("ckpt_last.pt",):
        (out / f).unlink(missing_ok=True)
    log(
        f"DONE fine-tune {short_name(model_name)} seed {seed}: test acc {m['accuracy']:.4f} macroF1 {m['f1_macro']:.4f}"
    )
