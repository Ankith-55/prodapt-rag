"""Mirror a script's stdout to output/<name>.txt so run results can be reviewed from the file."""
from __future__ import annotations

import atexit
import sys
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def start(name: str, out_dir: str = "output") -> Path:
    path = Path(out_dir) / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # never crash on odd characters in a legacy console
    f = open(path, "w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _Tee(original, f)

    def _close():
        sys.stdout = original  # restore first so the interpreter's final flush doesn't hit a closed file
        f.close()

    atexit.register(_close)
    return path
