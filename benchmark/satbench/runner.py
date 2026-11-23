"""The main loop: runs jobs on the available GPUs, handles stop/resume, progress and uploads."""

import os
import subprocess
import sys
import time
import traceback
from collections import Counter

from .config import CFG, LOGS, RES, ROOT, STATE, STOP_FLAG, TEST_MODE, short_name
from .data import stage_data
from .hardware import cuda_ok, gpu_count, gpu_status
from .jobs import job_done, job_list, job_weight, make_selection
from .notify import Notifier
from .upload import Uploader
from .utils import Log, kill_tree, now, read_json, run_name, tail


class Scheduler:
    def __init__(self, log, note, slots):
        self.log, self.note = log, note
        self.slots = list(range(slots)) if slots > 0 else ["cpu0", "cpu1"]
        self.running = {}  # slot -> (job, Popen, start)
        self.attempts = Counter()
        self.failed = set()

    def kill_all(self):
        for slot, (job, p, _) in list(self.running.items()):
            kill_tree(p.pid)
            self.log(f"stopped job {job}")
        self.running.clear()

    def launch(self, job, slot):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        env["CUDA_VISIBLE_DEVICES"] = str(slot) if isinstance(slot, int) else ""
        out = open(LOGS / "jobs" / f"{job}.console.log", "a", encoding="utf-8")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        p = subprocess.Popen(
            [sys.executable, str(ROOT / "main.py"), "job", job],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(ROOT),
            creationflags=flags,
        )
        self.running[slot] = (job, p, time.time())
        self.attempts[job] += 1
        self.log(
            f"started {job} on {'GPU ' + str(slot) if isinstance(slot, int) else 'CPU'} (attempt {self.attempts[job]})"
        )

    def poll(self):
        finished = []
        for slot, (job, p, t0) in list(self.running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del self.running[slot]
            if rc == 0 and job_done(job):
                self.log(f"finished {job} in {(time.time() - t0) / 60:.1f} min")
                finished.append(job)
            else:
                logf = LOGS / "jobs" / f"{job}.log"
                if self.attempts[job] >= 3:
                    self.failed.add(job)
                    self.log(
                        f"FAILED {job} three times - skipping it for now (it will be retried on the next RUN)"
                    )
                    self.note.send(
                        f"Job failed: {short_name(job)}",
                        f"{job} failed 3 times and is skipped for now.\n\n{tail(logf)}",
                        priority="high",
                        tags="x",
                    )
                else:
                    self.log(f"{job} stopped with an error (code {rc}) - will retry")
        return finished


def already_running():
    p = STATE / "main.pid"
    if not p.exists():
        return False
    try:
        pid = int(p.read_text().strip())
        import psutil

        return (
            pid != os.getpid() and psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
        )
    except Exception:
        return False


def orchestrate():
    log = Log(LOGS / "main.log")
    note = Notifier(log)
    if already_running():
        log("The benchmark is ALREADY RUNNING in another window. Use that window, or STOP.bat first.")
        time.sleep(5)
        return
    STOP_FLAG.unlink(missing_ok=True)
    (STATE / "progress").mkdir(exist_ok=True)
    (STATE / "main.pid").write_text(str(os.getpid()))
    name = run_name()
    log("=" * 70)
    log(f"Satellite benchmark '{name}' starting{' (TEST_MODE MODE)' if TEST_MODE else ''}")
    up = Uploader(log, note)
    sched = None
    t_start = time.time()
    try:
        info = stage_data(log, note)
        slots = gpu_count()
        if slots == 0 and not TEST_MODE:
            raise RuntimeError("No NVIDIA GPU found (nvidia-smi). Install the NVIDIA driver and run again.")
        if slots > 0 and not TEST_MODE and not cuda_ok():
            raise RuntimeError(
                "PyTorch cannot use the GPUs (CUDA not available). Update the NVIDIA driver, "
                "delete the folder 'runtime' and double-click RUN.bat again."
            )
        log(f"GPUs: {slots if slots else 'none (CPU test)'}  |  {gpu_status()}")
        note.send(
            "Benchmark started",
            f"{name}\nclean {info['clean_images']} img / {info['clean_classes']} classes, "
            f"raw {info['raw_images']} img / {info['raw_classes']} classes\nGPUs: {slots}",
            tags="rocket",
        )
        up.start()
        sched = Scheduler(log, note, slots)
        # ---- smoke test (one job, first)
        ran_smoke = not job_done("smoke")
        while not job_done("smoke") and sched.attempts["smoke"] < 2:
            sched.launch("smoke", sched.slots[0])
            while sched.running:
                if check_stop(log, note, sched):
                    return
                sched.poll()
                time.sleep(2)
        if ran_smoke:
            if not job_done("smoke"):
                note.send("Smoke test FAILED", tail(LOGS / "jobs" / "smoke.log"), priority="urgent", tags="x")
                raise RuntimeError(
                    "The smoke test failed - see logs/jobs/smoke.log. Nothing else was started."
                )
            sm = read_json(RES / "smoke" / "result.json")
            bad = sm.get("failed_models", [])
            note.send(
                "Smoke test passed",
                f"All checks passed. Models that could not load: {len(bad)}"
                + ("\n" + "\n".join(bad) if bad else ""),
                tags="white_check_mark",
            )
        bad_models = set(read_json(RES / "smoke" / "result.json", {}).get("failed_models", []))
        # ---- main loop
        last_progress = time.time()
        session_done_w = 0.0
        while True:
            if check_stop(log, note, sched):
                return
            for j in sched.poll():
                session_done_w += job_weight(j)
            sel = read_json(STATE / "selection.json")
            jobs = [j for j in job_list(sel) if not any(b in j for b in bad_models)]
            pend_a = [j for j in jobs if j.startswith("lp_clean__") and not job_done(j)]
            if (
                not sel
                and all(j in sched.failed or job_done(j) for j in pend_a)
                and not any(r[0].startswith("lp_clean__") for r in sched.running.values())
            ):
                sel = make_selection(log)
                jobs = [j for j in job_list(sel) if not any(b in j for b in bad_models)]
            running = {r[0] for r in sched.running.values()}
            todo = [j for j in jobs if not job_done(j) and j not in sched.failed and j not in running]
            for slot in sched.slots:
                if slot not in sched.running and todo:
                    sched.launch(todo.pop(0), slot)
            future = (
                0.0
                if sel
                else (
                    2 * int(CFG["raw_top_per_family"]) * 1.0
                    + 2 * int(CFG["ft_top_per_family"]) * int(CFG["ft_seeds"]) * job_weight("ft_")
                )
            )
            total_w = sum(job_weight(j) for j in jobs) + future
            done_w = sum(job_weight(j) for j in jobs if job_done(j))
            if not sched.running and not todo and sel:
                break
            if time.time() - last_progress > float(CFG["progress_every_min"]) * 60:
                last_progress = time.time()
                el = time.time() - t_start
                left = (el / session_done_w) * (total_w - done_w) if session_done_w else None
                msg = (
                    f"{100 * done_w / total_w:.0f}% done ({sum(job_done(j) for j in jobs)}/{len(jobs)} jobs)\n"
                    f"time left: {'%.1f h' % (left / 3600) if left else 'estimating...'}\n"
                    f"running: {', '.join(short_name(r[0].split('__')[1]) if '__' in r[0] else r[0] for r in sched.running.values())}\n"
                    f"{gpu_status()}"
                )
                log("PROGRESS " + msg.replace("\n", " | "))
                note.send("Progress", msg, priority="low", tags="hourglass")
            time.sleep(3)
        # ---- report
        log("All jobs finished - making the report")
        rc = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "job", "report"],
            cwd=str(ROOT),
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        ).returncode
        up.stop.set()
        up.upload_now(timeout=3600)
        pdf = RES / "report" / "results.pdf"
        failed = sorted(sched.failed)
        note.send(
            "Benchmark FINISHED",
            f"{name} finished.{' Report could not be made.' if rc else ''}"
            f"{' Failed jobs: ' + ', '.join(failed) if failed else ''}\nResults: {up.dest}",
            priority="high",
            file=pdf if pdf.exists() else None,
            tags="tada",
        )
        log(f"FINISHED. Results in {RES} and on Google Drive ({up.dest}).")
        (STATE / "FINISHED").write_text(now())
    except Exception as e:
        log(f"STOPPED BY A PROBLEM: {e}")
        log(traceback.format_exc())
        note.send(
            "Benchmark stopped by a problem",
            f"{e}\n\n{tail(LOGS / 'main.log', 15)}",
            priority="urgent",
            tags="x",
        )
        if sched:
            sched.kill_all()
        time.sleep(3)
    finally:
        try:
            (STATE / "main.pid").unlink(missing_ok=True)
        except Exception:
            pass
        time.sleep(2)  # let the last phone message go out


def check_stop(log, note, sched):
    if STOP_FLAG.exists():
        log("STOP requested - stopping running jobs (finished work is kept)")
        sched.kill_all()
        note.send(
            "Benchmark stopped", "Stopped by STOP.bat. Double-click RUN.bat to continue.", tags="stop_sign"
        )
        STOP_FLAG.unlink(missing_ok=True)
        time.sleep(3)
        return True
    return False
