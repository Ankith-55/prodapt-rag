"""Ingest a batch of newly closed tickets into a state directory.

  python scripts/ingest.py --state data/processed --csv new_tickets.csv [--from-date 2026-03-08] [--no-llm]
The CSV needs the dataset columns (unique_key, created_date, closed_date, complaint_type, descriptor,
descriptor_2, resolution_description). A running API picks the batch up via POST /ingest or /reload."""
import argparse
import json

import pandas as pd

from ticketrag import runlog
from ticketrag.embed import Embedder
from ticketrag.ingest import apply_batch
from ticketrag.llm import LLM

ap = argparse.ArgumentParser()
ap.add_argument("--state", default="data/processed")
ap.add_argument("--csv", required=True)
ap.add_argument("--from-date", help="only rows created on/after this date (to simulate a later batch)")
ap.add_argument("--no-llm", action="store_true", help="index new classes by label only (no example complaints)")
a = ap.parse_args()
runlog.start("ingest")

raw = pd.read_csv(a.csv, dtype=str)
if a.from_date:
    raw = raw[pd.to_datetime(raw["created_date"]) >= pd.Timestamp(a.from_date)]
report = apply_batch(a.state, raw, Embedder(), None if a.no_llm else LLM())
print(json.dumps(report, indent=2, default=str))
