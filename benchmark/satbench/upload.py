"""Periodic upload of results to Google Drive with rclone. Upload problems never stop the run."""

import os
import re
import shutil
import subprocess
import threading
import time
import zipfile

from .config import CFG, LOGS, RES, ROOT, RUNTIME
from .utils import run_name, scrub

RCLONE_VER = "v1.68.2"


def rclone_exe(log):
    win = os.name == "nt"
    name = "rclone.exe" if win else "rclone"
    for p in (ROOT / name, RUNTIME / "rclone" / name):
        if p.exists():
            return p
    plat = "windows-amd64" if win else "linux-amd64"
    zname = f"rclone-{RCLONE_VER}-{plat}.zip"
    urls = [
        f"https://github.com/rclone/rclone/releases/download/{RCLONE_VER}/{zname}",
        f"https://downloads.rclone.org/{RCLONE_VER}/{zname}",
    ]
    dest = RUNTIME / zname
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    import urllib.request

    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            break
        except Exception as e:
            log(f"rclone download failed from {u.split('/')[2]}: {e}")
            try:
                curl = "curl.exe" if win else "curl"
                subprocess.run(
                    [curl, "-L", "-A", ua, "-o", str(dest), u], timeout=300, check=True, capture_output=True
                )
                break
            except Exception as e2:
                log(f"curl also failed: {e2}")
    try:
        with zipfile.ZipFile(dest) as z:
            member = [m for m in z.namelist() if m.endswith("/" + name)][0]
            (RUNTIME / "rclone").mkdir(exist_ok=True)
            with z.open(member) as src, open(RUNTIME / "rclone" / name, "wb") as out:
                shutil.copyfileobj(src, out)
        if not win:
            os.chmod(RUNTIME / "rclone" / name, 0o755)
        return RUNTIME / "rclone" / name
    except Exception as e:
        log(f"Could not get rclone ({e}). Google Drive upload is switched off for now.")
        return None


class Uploader:
    """Copies results to Google Drive every N minutes. Problems are logged + notified, never fatal."""

    def __init__(self, log, notifier):
        self.log = log
        self.note = notifier
        self.exe = None
        self.last_err_note = 0
        self.stop = threading.Event()
        self.dest = f"{CFG['drive_remote']}:{CFG['drive_folder']}/{run_name()}"
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop.wait(1):
            self.upload_now(timeout=1800)
            if self.stop.wait(float(CFG["upload_every_min"]) * 60):
                return

    def upload_now(self, timeout=1800):
        conf = ROOT / "rclone.conf"
        if not conf.exists():
            if self.last_err_note == 0:
                self.log(
                    "No rclone.conf next to RUN.bat - Google Drive upload is off (results stay on this PC)."
                )
                self.last_err_note = 1
            return False
        if self.exe is None:
            self.exe = rclone_exe(self.log)
            if self.exe is None:
                return False
        ok = True
        for src, sub in ((RES, "results"), (LOGS, "logs")):
            cmd = [
                str(self.exe),
                "copy",
                str(src),
                f"{self.dest}/{sub}",
                "--config",
                str(conf),
                "--exclude",
                "*.pt",
                "--exclude",
                "*.tmp",
                "--transfers",
                "4",
                "--retries",
                "2",
                "--low-level-retries",
                "3",
                "--contimeout",
                "30s",
                "--timeout",
                "120s",
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if r.returncode != 0:
                    ok = False
                    self._problem(scrub(r.stderr)[-1500:])
                    break
            except Exception as e:
                ok = False
                self._problem(str(e))
                break
        if ok:
            self.log(f"Uploaded results to Google Drive ({self.dest})")
        return ok

    def _problem(self, err):
        hint = ""
        if re.search(r"401|invalid.?credentials|invalid_grant|token.*(expired|revoked)", err, re.I):
            hint = (
                "\nGoogle refused the login (401 Invalid Credentials). Fix: on any PC run "
                "'rclone config reconnect gdrive:' (or create a new rclone.conf), then replace rclone.conf "
                "next to RUN.bat. No restart needed - the next upload uses the new file."
            )
        elif "didn't find section" in err or "not found in config" in err.lower():
            hint = f"\nrclone.conf has no remote called '{CFG['drive_remote']}'. Name it gdrive or change drive_remote in settings.ini."
        self.log(f"Google Drive upload problem (the benchmark keeps running): {err[-400:]}{hint}")
        if time.time() - self.last_err_note > 3600:
            self.last_err_note = time.time()
            self.note.send("Drive upload problem", f"{err[-600:]}{hint}", priority="default", tags="warning")
