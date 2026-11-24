"""Summary tables, statistics, t-SNE maps and the results PDF."""

from collections import defaultdict

from .config import CFG, DATA, FEAT, RES, family_of, short_name
from .data import load_index
from .utils import read_json, run_name, write_json


def embedding_analysis(log, out, clean_df):
    """Same real images, two label sets. Silhouette score (cosine) of the frozen features under the CLEAN
    labels, the RAW labels and the SOURCE dataset (= dataset bias), for every model; t-SNE maps for the
    best CNN and the best ViT."""
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    idx = load_index("cleanlp")
    raw = {r["path"]: r["label"] for r in load_index("raw")}
    rng = np.random.RandomState(0)
    by = defaultdict(list)
    for i, r in enumerate(idx):
        by[r["label"]].append(i)
    k = int(CFG["tsne_per_class"])
    sel = []
    for c in sorted(by):
        ids = np.array(by[c])
        rng.shuffle(ids)
        sel += ids[:k].tolist()
    sel = np.array(sorted(sel))
    lab_c = np.array([idx[i]["label"] for i in sel])
    lab_r = np.array([raw.get(idx[i]["path"], "") for i in sel])
    src = np.array([idx[i]["path"].split("/")[1] if idx[i]["path"].count("/") > 1 else "?" for i in sel])
    m = lab_r != ""
    raw_rows = load_index("raw")
    n_raw_total = len({r["label"] for r in raw_rows})
    all_sources = sorted({r["path"].split("/")[1] for r in raw_rows if r["path"].count("/") > 1})
    absent = [s for s in all_sources if s not in set(src)]
    n_src_total = len(all_sources)
    absent_note = f"; not in clean list: {', '.join(absent)}" if absent else ""
    rows, emb = [], {}
    for d in sorted((FEAT / "clean").glob("*")):
        meta = read_json(d / "meta.json", {})
        if not (d / "features.npy").exists() or meta.get("n") != len(idx):
            continue
        X = np.asarray(np.load(d / "features.npy", mmap_mode="r")[sel], dtype=np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        try:
            r = dict(
                model=meta["model"],
                short=short_name(meta["model"]),
                family=family_of(meta["model"]),
                silhouette_clean=float(silhouette_score(X[m], lab_c[m], metric="cosine")),
                silhouette_raw=(
                    float(silhouette_score(X[m], lab_r[m], metric="cosine"))
                    if len(set(lab_r[m])) > 1
                    else None
                ),
                silhouette_source=(
                    float(silhouette_score(X, src, metric="cosine")) if len(set(src)) > 1 else None
                ),
                n_images=int(len(sel)),
                n_with_raw_label=int(m.sum()),
            )
            rows.append(r)
            emb[meta["model"]] = X
        except Exception as e:
            log(f"  silhouette skipped for {d.name}: {e}")
    if not rows:
        return None, []
    sil = pd.DataFrame(rows)
    sil.to_csv(out / "silhouette_clean_vs_raw.csv", index=False)
    figs = []
    best = []
    if len(clean_df):
        for f in ("CNN", "ViT"):
            g = clean_df[clean_df.family == f].sort_values("f1", ascending=False)
            g = g[g.model.isin(list(emb))]
            if len(g):
                best.append(g.iloc[0].model)
    for mname in best:
        try:
            Y = TSNE(
                n_components=2, perplexity=min(30, max(5, len(sel) // 4)), init="pca", random_state=0
            ).fit_transform(emb[mname])
            pd.DataFrame(
                {"x": Y[:, 0], "y": Y[:, 1], "clean_label": lab_c, "raw_label": lab_r, "source": src}
            ).to_csv(out / f"tsne_{short_name(mname)}.csv", index=False)
        except Exception as e:
            log(f"  t-SNE skipped for {mname}: {e}")
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
        for ax, labels, title in (
            (axes[0], lab_c, f"CLEAN labels ({len(set(lab_c))} classes)"),
            (
                axes[1],
                np.where(m, lab_r, "(no raw label)"),
                f"RAW labels ({len(set(lab_r[m]))} of {n_raw_total} classes in this sample)",
            ),
            (axes[2], src, f"SOURCE dataset ({len(set(src))} of {n_src_total}{absent_note})"),
        ):
            u = sorted(set(labels))
            cmap = plt.get_cmap("tab20")
            cols = {l: cmap(i % 20) for i, l in enumerate(u)}
            for l in u:
                q = labels == l
                ax.scatter(Y[q, 0], Y[q, 1], s=3, color=cols[l], label=l if len(u) <= 20 else None)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if len(u) <= 20:
                ax.legend(fontsize=5, markerscale=3, loc="best", frameon=False)
        r = sil[sil.model == mname].iloc[0]
        fig.suptitle(
            f"t-SNE of frozen features - {short_name(mname)} ({family_of(mname)})   silhouette: clean {r.silhouette_clean:.3f} | "
            f"raw {r.silhouette_raw if r.silhouette_raw is None else round(r.silhouette_raw, 3)} | source "
            f"{r.silhouette_source if r.silhouette_source is None else round(r.silhouette_source, 3)}",
            fontsize=10,
        )
        fig.tight_layout()
        png = out / f"tsne_{short_name(mname)}.png"
        fig.savefig(png, dpi=300)
        figs.append(fig)
    log(f"  embedding analysis: silhouette for {len(sil)} models, t-SNE for {len(figs)}")
    return sil, figs


def make_report(log):
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out = RES / "report"
    out.mkdir(exist_ok=True)
    rows = []
    for d in RES.glob("lp_*__*"):
        r = read_json(d / "result.json")
        if r:
            s = r["summary"]
            rows.append(
                dict(
                    set=r["set"],
                    model=r["model"],
                    short=short_name(r["model"]),
                    family=r["family"],
                    acc=s["accuracy_mean"],
                    acc_sd=s["accuracy_std"],
                    f1=s["f1_macro_mean"],
                    f1_sd=s["f1_macro_std"],
                    params_m=r.get("params_m"),
                    input=r.get("input_size"),
                    gpu_img_s=r.get("gpu_images_per_s"),
                    dim=r.get("dim"),
                )
            )
    lp = pd.DataFrame(rows)
    ft_rows = []
    for d in RES.glob("ft_clean__*"):
        r = read_json(d / "result.json")
        if r:
            ft_rows.append(
                dict(
                    model=r["model"],
                    short=short_name(r["model"]),
                    family=r["family"],
                    seed=r["seed"],
                    acc=r["test"]["accuracy"],
                    f1=r["test"]["f1_macro"],
                )
            )
    ft = pd.DataFrame(ft_rows)
    if len(lp):
        lp.sort_values(["set", "f1"], ascending=[True, False]).to_csv(
            out / "linear_probe_summary.csv", index=False
        )
    if len(ft):
        ft.to_csv(out / "finetune_runs.csv", index=False)
        ft.groupby(["model", "family"]).agg(
            acc_mean=("acc", "mean"),
            acc_sd=("acc", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            seeds=("seed", "count"),
        ).reset_index().to_csv(out / "finetune_summary.csv", index=False)
    stats = {}
    clean = lp[lp.set == "clean"] if len(lp) else lp
    raw = lp[lp.set == "raw"] if len(lp) else lp
    if len(clean) >= 2:
        from scipy.stats import kendalltau, mannwhitneyu

        c, v = clean[clean.family == "CNN"].f1, clean[clean.family == "ViT"].f1
        if len(c) and len(v):
            stats["cnn_vs_vit_macroF1_mannwhitney_p"] = float(mannwhitneyu(c, v).pvalue)
            stats["cnn_mean_f1"], stats["vit_mean_f1"] = float(c.mean()), float(v.mean())
        both = clean.merge(raw, on="model", suffixes=("_clean", "_raw"))
        if len(both) >= 3:
            tau = kendalltau(both.f1_clean, both.f1_raw)
            stats["kendall_tau_clean_vs_raw"] = float(tau.statistic if hasattr(tau, "statistic") else tau[0])
            stats["kendall_p"] = float(tau.pvalue)
            both[["model", "family_clean", "f1_clean", "f1_raw", "acc_clean", "acc_raw"]].to_csv(
                out / "clean_vs_raw.csv", index=False
            )
    sil, tsne_figs = None, []
    try:
        sil, tsne_figs = embedding_analysis(log, out, clean)
        if sil is not None and sil.silhouette_raw.notna().any():
            stats["silhouette_clean_mean"] = float(sil.silhouette_clean.mean())
            stats["silhouette_raw_mean"] = float(sil.silhouette_raw.mean())
            stats["models_where_clean_labels_cluster_better"] = int(
                (sil.silhouette_clean > sil.silhouette_raw).sum()
            )
            stats["models_with_silhouette"] = int(sil.silhouette_raw.notna().sum())
    except Exception as e:
        log(f"embedding analysis failed (report continues): {e}")
    write_json(out / "stats.json", stats)
    col = {"CNN": "#3b82f6", "ViT": "#f97316"}
    with PdfPages(out / "results.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.95, f"Satellite benchmark - {run_name()}", fontsize=16, weight="bold")
        info = read_json(DATA / "ready.json", {})
        lines = [
            f"Clean list: {info.get('clean_images')} images, {info.get('clean_classes')} classes "
            f"({info.get('clean_augmented')} augmented copies used for training only)",
            f"Raw list: {info.get('raw_images')} images, {info.get('raw_classes')} classes",
            f"Linear probe models finished (clean): {len(clean)}   raw: {len(raw)}   fine-tune runs: {len(ft)}",
            "",
        ]
        lines += [f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()]
        if len(clean):
            lines += ["", "Top 10 (clean, macro-F1):"]
            for r in clean.sort_values("f1", ascending=False).head(10).itertuples():
                lines.append(f"  {r.short:34s} {r.family:3s}  F1 {r.f1:.4f}±{r.f1_sd:.4f}  acc {r.acc:.4f}")
        fig.text(0.08, 0.9, "\n".join(lines), fontsize=9, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        if len(clean):
            d = clean.sort_values("f1")
            fig, ax = plt.subplots(figsize=(8.27, max(4, 0.22 * len(d) + 1)))
            ax.barh(d.short, d.f1, xerr=d.f1_sd, color=[col.get(f, "gray") for f in d.family])
            ax.set_xlabel("macro-F1 (clean list, linear probe, mean ± SD)")
            ax.set_xlim(max(0, d.f1.min() - 0.05), 1)
            ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            dd = clean.dropna(subset=["gpu_img_s"])
            if len(dd):
                fig, ax = plt.subplots(figsize=(8.27, 5.5))
                for f, g in dd.groupby("family"):
                    ax.scatter(g.gpu_img_s, g.f1, c=col.get(f, "gray"), label=f)
                    for r in g.itertuples():
                        ax.annotate(r.short, (r.gpu_img_s, r.f1), fontsize=5)
                ax.set_xscale("log")
                ax.set_xlabel("GPU speed (images/s, log)")
                ax.set_ylabel("macro-F1")
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        if len(raw):
            b = clean.merge(raw, on="model", suffixes=("_clean", "_raw"))
            fig, ax = plt.subplots(figsize=(8.27, 5.5))
            ax.scatter(b.f1_raw, b.f1_clean, c=[col.get(f, "gray") for f in b.family_clean])
            for r in b.itertuples():
                ax.annotate(r.short_clean, (r.f1_raw, r.f1_clean), fontsize=6)
            ax.set_xlabel("macro-F1 on RAW labels")
            ax.set_ylabel("macro-F1 on CLEAN labels")
            ax.set_title("Same models, raw vs clean labels")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        if len(ft):
            g = ft.groupby(["short", "family"]).f1.agg(["mean", "std"]).reset_index()
            lpm = clean.set_index("short").f1 if len(clean) else {}
            fig, ax = plt.subplots(figsize=(8.27, 4.5))
            x = np.arange(len(g))
            ax.bar(x - 0.2, [lpm.get(s, np.nan) for s in g.short], 0.4, label="linear probe")
            ax.bar(x + 0.2, g["mean"], 0.4, yerr=g["std"].fillna(0), label="fine-tuned")
            ax.set_xticks(x)
            ax.set_xticklabels(g.short, fontsize=7)
            ax.set_ylabel("macro-F1 (clean)")
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        if sil is not None and len(sil) and sil.silhouette_raw.notna().any():
            d = sil.dropna(subset=["silhouette_raw"]).sort_values("silhouette_clean")
            fig, ax = plt.subplots(figsize=(8.27, max(4, 0.22 * len(d) + 1)))
            yy = np.arange(len(d))
            ax.barh(yy - 0.2, d.silhouette_raw, 0.4, label="raw labels", color="#f59e0b")
            ax.barh(yy + 0.2, d.silhouette_clean, 0.4, label="clean labels", color="#10b981")
            ax.set_yticks(yy)
            ax.set_yticklabels(d.short, fontsize=7)
            ax.axvline(0, color="gray", lw=0.6)
            ax.set_xlabel("silhouette score of frozen features (same real images, cosine)")
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        for fig in tsne_figs:
            pdf.savefig(fig)
            plt.close(fig)
    log(f"report written: {out / 'results.pdf'}")
