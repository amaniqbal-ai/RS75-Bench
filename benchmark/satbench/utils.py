"""Small helpers shared by all modules (logging, JSON files, process handling)."""

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from .config import CFG, STATE, TEST_MODE


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def __call__(self, msg):
        line = f"{now()}  {msg}"
        with self.lock:
            try:
                print(line, flush=True)
            except Exception:
                pass
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass


def tail(path, n=40, max_chars=2500):
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        return "\n".join(lines)[-max_chars:]
    except Exception:
        return "(no log)"


SECRET_RE = re.compile(r'("?(access_token|refresh_token|client_secret|token)"?\s*[:=]\s*)"?[^",\s}]+', re.I)


def scrub(text):
    return SECRET_RE.sub(r"\1***", text or "")


def run_name():
    p = STATE / "run_name.txt"
    if CFG["run_name"]:
        return CFG["run_name"]
    if p.exists():
        return p.read_text().strip()
    name = "satbench_" + time.strftime("%Y%m%d_%H%M") + ("_TEST" if TEST_MODE else "")
    p.write_text(name)
    return name


def is_oom(e):
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def kill_tree(pid):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, timeout=60)
        else:
            import signal

            try:
                import psutil

                for c in psutil.Process(pid).children(recursive=True):
                    c.kill()
            except Exception:
                pass
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
