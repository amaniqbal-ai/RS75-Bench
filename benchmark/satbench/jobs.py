"""The list of jobs, their completion state and the dispatcher used by worker processes."""

import os
import traceback

from .config import CFG, HEAVY_FIRST, LOGS, RES, STATE, models_by_family, short_name
from .finetune import finetune_job
from .linear_probe import linear_probe_job
from .report import make_report
from .smoke import smoke_job
from .utils import Log, is_oom, read_json, write_json


def job_list(selection=None):
    fams = models_by_family()
    allm = fams["CNN"] + fams["ViT"]
    order = [m for m in HEAVY_FIRST if m in allm] + [m for m in allm if m not in HEAVY_FIRST]
    jobs = ["smoke"] + [f"lp_clean__{m}" for m in order]
    if selection:
        jobs += [f"lp_raw__{m}" for m in selection["raw"]]
        jobs += [f"ft_clean__{m}__s{s}" for m in selection["ft"] for s in range(int(CFG["ft_seeds"]))]
    return jobs


def job_done(j):
    return (RES / j / "result.json").exists()


def job_weight(j):
    if j.startswith("ft_"):
        return 6.0
    return 1.0


def make_selection(log):
    p = STATE / "selection.json"
    if p.exists():
        return read_json(p)
    fams = models_by_family()
    scores = {}
    for m in fams["CNN"] + fams["ViT"]:
        r = read_json(RES / f"lp_clean__{m}" / "result.json")
        if r:
            scores[m] = (
                r["summary"]["f1_macro_mean"],
                r["summary"]["accuracy_mean"],
                r.get("input_size", 224),
            )
    sel = {"raw": [], "ft": [], "rule": "ranked by mean macro-F1 (then accuracy) of the clean linear probe"}
    ntop, nft, maxin = int(CFG["raw_top_per_family"]), int(CFG["ft_top_per_family"]), int(CFG["ft_max_input"])
    for f in ("CNN", "ViT"):
        ranked = sorted([m for m in fams[f] if m in scores], key=lambda m: scores[m][:2], reverse=True)
        sel["raw"] += ranked[:ntop]
        sel["ft"] += [m for m in ranked if scores[m][2] <= maxin][:nft]
    write_json(p, sel)
    log(f"Selected for raw labels: {[short_name(m) for m in sel['raw']]}")
    log(f"Selected for fine-tuning: {[short_name(m) for m in sel['ft']]}")
    return sel


def run_job(job):
    log = Log(LOGS / "jobs" / f"{job}.log")
    log(f"=== job {job} started (pid {os.getpid()}) ===")
    try:
        if os.environ.get("BENCH_TEST_FAIL") and os.environ["BENCH_TEST_FAIL"] in job:
            raise RuntimeError("simulated failure for testing")
        if job == "smoke":
            smoke_job(log)
        elif job.startswith("lp_"):
            set_name, model = job[3:].split("__", 1)
            linear_probe_job(model, set_name, log)
        elif job.startswith("ft_clean__"):
            _, model, s = job.split("__")
            finetune_job(model, int(s[1:]), log)
        elif job == "report":
            make_report(log)
        else:
            raise ValueError(f"unknown job {job}")
        log(f"=== job {job} finished ===")
        return 0
    except Exception as e:
        log(f"ERROR in {job}: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        return 3 if is_oom(e) else 1
