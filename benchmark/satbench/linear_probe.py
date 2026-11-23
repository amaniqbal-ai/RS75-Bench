"""Linear probe: one-vs-rest linear SVM (hinge loss, L2) on frozen features, trained on the GPU."""

import math
import time

from .config import CFG, DATA, RES, TEST_MODE, family_of, short_name
from .data import index_of, load_index
from .models import extract_features, torch_setup
from .utils import is_oom, now, write_json


def svm_train(torch, X, y, alphas, n_cls, seed, epochs, lr, bs):
    """One-vs-rest linear SVM (hinge + L2), all alphas at once, mini-batch Adam with cosine decay."""
    dev = X.device
    n, d = X.shape
    A = len(alphas)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    al = torch.tensor(alphas, device=dev, dtype=torch.float32).view(A, 1, 1)
    W = torch.zeros(A, d, n_cls, device=dev)
    b = torch.zeros(A, 1, n_cls, device=dev)
    mW, vW, mb, vb = torch.zeros_like(W), torch.zeros_like(W), torch.zeros_like(b), torch.zeros_like(b)
    spe = math.ceil(n / bs)
    total = spe * epochs
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for s in range(spe):
            bi = perm[s * bs : (s + 1) * bs]
            Xb = X[bi]
            yb = y[bi]
            nb = Xb.shape[0]
            Y = torch.full((nb, n_cls), -1.0, device=dev)
            Y[torch.arange(nb, device=dev), yb] = 1.0
            Sc = torch.matmul(Xb.unsqueeze(0), W) + b
            G = -(Y * ((Y * Sc) < 1.0).float()) / nb
            gW = torch.matmul(Xb.t().unsqueeze(0), G) + al * W
            gb = G.sum(1, keepdim=True)
            step += 1
            lr_t = lr * 0.5 * (1 + math.cos(math.pi * (step - 1) / total))
            mW.mul_(0.9).add_(gW, alpha=0.1)
            vW.mul_(0.999).addcmul_(gW, gW, value=0.001)
            mb.mul_(0.9).add_(gb, alpha=0.1)
            vb.mul_(0.999).addcmul_(gb, gb, value=0.001)
            c1, c2 = 1 - 0.9**step, 1 - 0.999**step
            W -= lr_t * (mW / c1) / ((vW / c2).sqrt() + 1e-8)
            b -= lr_t * (mb / c1) / ((vb / c2).sqrt() + 1e-8)
    return W, b


def svm_predict(torch, X, W, b, chunk=65536):
    return torch.cat(
        [
            (torch.matmul(X[i : i + chunk].unsqueeze(0), W) + b).argmax(-1)
            for i in range(0, X.shape[0], chunk)
        ],
        dim=1,
    )


def metrics(y, p, n_cls):
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    _, rc_c, f1_c, _ = precision_recall_fscore_support(y, p, labels=np.arange(n_cls), zero_division=0)
    return (
        dict(
            accuracy=float(accuracy_score(y, p)),
            precision_macro=float(pr),
            recall_macro=float(rc),
            f1_macro=float(f1),
        ),
        rc_c,
        f1_c,
        confusion_matrix(y, p, labels=np.arange(n_cls)),
    )


def linear_probe_job(model_name, set_name, log):
    import numpy as np

    torch, dev = torch_setup()
    X, meta = extract_features(model_name, set_name, log)
    idx = load_index(index_of(set_name))
    classes = sorted({r["label"] for r in idx})
    c2i = {c: i for i, c in enumerate(classes)}
    y_all = np.array([c2i[r["label"]] for r in idx])
    n_cls = len(classes)
    alphas = [float(a) for a in CFG["alphas"].split(",")]
    epochs = 3 if TEST_MODE else int(CFG["svm_epochs"])
    lr = float(CFG["svm_lr"])
    sbs = int(CFG["svm_batch"])
    Xg = torch.tensor(np.asarray(X), dtype=torch.float32, device=dev)
    yg = torch.tensor(y_all, device=dev)
    runs, cms, rcs, f1s = [], [], [], []
    for seed in range(int(CFG["lp_runs"])):
        t0 = time.time()
        sp = np.load(DATA / f"split_{index_of(set_name)}_s{seed}.npz")
        tr, va, te = sp["train"], sp["val"], sp["test"]
        T = lambda a: torch.tensor(a, device=dev)
        mu = Xg[T(tr)].mean(0)
        sd = Xg[T(tr)].std(0) + 1e-6
        Z = lambda a: (Xg[T(a)] - mu) / sd
        while True:
            try:
                W, b = svm_train(torch, Z(tr), yg[T(tr)], alphas, n_cls, seed, epochs, lr, sbs)
                pv = svm_predict(torch, Z(va), W, b)
                accs = (pv == yg[T(va)].unsqueeze(0)).float().mean(1).tolist()
                best = max(range(len(alphas)), key=lambda k: (accs[k], -alphas[k]))
                trva = np.concatenate([tr, va])
                mu2 = Xg[T(trva)].mean(0)
                sd2 = Xg[T(trva)].std(0) + 1e-6
                W2, b2 = svm_train(
                    torch,
                    (Xg[T(trva)] - mu2) / sd2,
                    yg[T(trva)],
                    [alphas[best]],
                    n_cls,
                    seed,
                    epochs,
                    lr,
                    sbs,
                )
                pt = svm_predict(torch, (Xg[T(te)] - mu2) / sd2, W2, b2)[0].cpu().numpy()
                break
            except Exception as e:
                if is_oom(e) and sbs > 128:
                    sbs //= 2
                    torch.cuda.empty_cache()
                    log(f"  out of GPU memory in classifier -> batch {sbs}")
                    continue
                raise
        m, rc_c, f1_c, cm = metrics(y_all[te], pt, n_cls)
        m.update(
            run=seed,
            best_alpha=alphas[best],
            n_train=int(len(tr)),
            n_val=int(len(va)),
            n_test=int(len(te)),
            seconds=round(time.time() - t0, 1),
        )
        runs.append(m)
        cms.append(cm)
        rcs.append(rc_c)
        f1s.append(f1_c)
        log(
            f"  {short_name(model_name)} {set_name} run {seed}: acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f} alpha={alphas[best]:g}"
        )
    out = RES / f"lp_{set_name}__{model_name}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "confusion_mean.npy", np.mean(cms, 0))
    keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    summ = {f"{k}_mean": float(np.mean([r[k] for r in runs])) for k in keys}
    summ.update(
        {f"{k}_std": float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0 for k in keys}
    )
    res = dict(
        job=f"lp_{set_name}__{model_name}",
        model=model_name,
        family=family_of(model_name),
        set=set_name,
        n_classes=n_cls,
        classes=classes,
        runs=runs,
        summary=summ,
        **{k: v for k, v in meta.items() if k != "model"},
        per_class_recall=dict(zip(classes, np.mean(rcs, 0).round(4).tolist())),
        per_class_f1=dict(zip(classes, np.mean(f1s, 0).round(4).tolist())),
        finished=now(),
    )
    write_json(out / "result.json", res)
    log(
        f"DONE {short_name(model_name)} {set_name}: acc {summ['accuracy_mean']:.4f} macroF1 {summ['f1_macro_mean']:.4f}"
    )
