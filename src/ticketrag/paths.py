"""Default data locations: the working data/processed directory once the pipeline has been run locally, otherwise
the committed, ready-to-run artifacts/ folder, so a fresh clone works without building anything."""
from __future__ import annotations

import os
from pathlib import Path


def default_state_dir() -> Path:
    env = os.getenv("TICKETRAG_STATE")
    if env:
        return Path(env)
    working = Path("data/processed")
    return working if (working / "patterns.parquet").exists() else Path("artifacts/state")


def examples_file(role: str, suffix: str = "") -> Path:
    """LLM-written example complaints (role: index or eval): working copy if present, else the committed one."""
    name = f"pattern_examples_{role}{suffix}.jsonl"
    working = Path("data/processed") / name
    return working if working.exists() else Path("artifacts/examples") / name
