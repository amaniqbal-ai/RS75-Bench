"""Phone notifications through ntfy.sh. Failures are logged and never interrupt the run."""

import threading
import time
from pathlib import Path

from .config import ROOT, read_kv


class Notifier:
    """ntfy.sh messages in a background thread. Never raises, never blocks the run."""

    def __init__(self, log):
        self.log = log
        self.q = []
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def send(self, title, msg, priority="default", file=None, tags=""):
        with self.lock:
            self.q.append((title, msg, priority, file, tags))

    def _loop(self):
        while True:
            item = None
            with self.lock:
                if self.q:
                    item = self.q.pop(0)
            if item is None:
                time.sleep(1)
                continue
            try:
                self._post(*item)
            except Exception as e:
                self.log(f"(phone message not sent: {type(e).__name__}: {str(e)[:120]})")

    def _post(self, title, msg, priority, file, tags):
        cfg = read_kv(ROOT / "notify.txt")
        topic = cfg.get("ntfy_topic", "").strip()
        if not topic:
            return
        server = cfg.get("ntfy_server", "https://ntfy.sh").rstrip("/")
        import urllib.request

        head = {
            "Title": title.encode("ascii", "ignore").decode() or "satbench",
            "Priority": priority,
            "User-Agent": "satbench",
        }
        if tags:
            head["Tags"] = tags
        if file and Path(file).exists() and Path(file).stat().st_size < 14_000_000:
            head["Filename"] = Path(file).name
            head["Message"] = msg.replace("\n", " | ").encode("ascii", "ignore").decode()[:1000]
            req = urllib.request.Request(
                f"{server}/{topic}", data=Path(file).read_bytes(), headers=head, method="PUT"
            )
        else:
            req = urllib.request.Request(
                f"{server}/{topic}", data=msg.encode("utf-8")[:3900], headers=head, method="POST"
            )
        urllib.request.urlopen(req, timeout=20).read()
