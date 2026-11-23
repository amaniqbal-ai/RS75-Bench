"""GPU discovery and status through nvidia-smi."""

import subprocess
import sys

from .config import CFG


def gpu_count():
    if CFG["gpus"] != "auto":
        return int(CFG["gpus"])
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return len([l for l in r.stdout.splitlines() if l.startswith("GPU ")])
    except Exception:
        return 0


def gpu_status():
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = []
        for l in r.stdout.strip().splitlines():
            i, t, u, mu, mt = [x.strip() for x in l.split(",")]
            lines.append(f"GPU{i}: {t}C, {u}% busy, {int(mu) // 1024}/{int(mt) // 1024} GB")
        return "\n".join(lines) or "no GPU info"
    except Exception:
        return "no GPU info (CPU run)"


def cuda_ok():
    code = "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)"
    try:
        return subprocess.run([sys.executable, "-c", code], timeout=300).returncode == 0
    except Exception:
        return False
