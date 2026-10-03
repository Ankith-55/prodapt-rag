"""Evolving-data demo (requirement 3). Builds the system from days 1-7 only, then:
  1. asks the telecom complaint -> abstains (no such class exists yet)
  2. ingests days 8-9 of the real data (existing patterns refresh, any new classes appear)
  3. ingests a SYNTHETIC "Broadband Service" class (clearly fabricated demo data, not part of the corpus)
  4. asks the telecom complaint again -> now answered from the new class, without rebuilding anything
Uses data/demo_state/ and never touches data/processed."""
import argparse
import json
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ticketrag import runlog
from ticketrag.embed import Embedder
from ticketrag.ingest import apply_batch, init_state
from ticketrag.llm import LLM
from ticketrag.patterns import clean_tickets
from ticketrag.pipeline import Assistant
from ticketrag.retrieve import PatternRetriever

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="dataset/closed_tickets_rag.csv")
ap.add_argument("--cutoff", default="2026-03-08")
ap.add_argument("--state", default="data/demo_state")
a = ap.parse_args()
runlog.start("demo_evolution")

QUERY = ("My broadband drops every evening around 8 and I've already restarted the router twice. "
         "I work from home and this is costing me.")

SPECS = [
    ("Broadband Service", "Intermittent Connection", "Drops at Peak Hours", [
        (0.35, "The service provider remotely reset the line and confirmed the connection was stable after testing. "
               "If the problem returns, please contact the provider to log a new fault."),
        (0.30, "A technician visit was scheduled to inspect the line and replace faulty equipment at the premises."),
        (0.25, "The provider identified network congestion in the area during evening peak hours and scheduled "
               "a capacity upgrade."),
        (0.10, "The router was found to be faulty and a replacement unit was shipped to the customer.")]),
    ("Broadband Service", "No Connection", "Full Outage", [
        (0.45, "A network outage in the area was confirmed and service was restored remotely."),
        (0.35, "A technician visit was scheduled to repair damaged cabling serving the property."),
        (0.20, "The provider could not reproduce the fault. The customer was asked to submit a new report "
               "if the outage recurs.")]),
]


def synthetic_telecom(n_each: int = 60, seed: int = 0) -> pd.DataFrame:
    rnd, rows, k = random.Random(seed), [], 0
    for ctype, desc, desc2, outcomes in SPECS:
        for _ in range(n_each):
            created = datetime(2026, 3, 9, 8, 0) + timedelta(minutes=rnd.randint(0, 900))
            closed = created + timedelta(hours=rnd.uniform(4, 72))
            text = rnd.choices([t for _, t in outcomes], weights=[w for w, _ in outcomes])[0]
            k += 1
            rows.append({"unique_key": f"SYN-{k:04d}", "created_date": created.isoformat(), "closed_date": closed.isoformat(),
                         "complaint_type": ctype, "descriptor": desc, "descriptor_2": desc2,
                         "resolution_description": text, "status": "Closed"})
    return pd.DataFrame(rows)


def show(title, r):
    print(f"\n--- {title}")
    if r.abstained:
        print(f"ABSTAINED: {r.abstain_reason}")
        print("nearest:", [f"{s:.2f} {l}" for l, s in r.candidates[:3]])
        return
    print(f"ANSWERED: {r.category} / {r.product} | tier={r.tier} | severity={r.severity} sentiment={r.sentiment}")
    print("opening:", r.answer.get("opening"))
    print("summary:", r.answer["summary"]["text"], r.answer["summary"]["citations"])
    for s in r.answer["steps"]:
        print("  -", s["text"], s["citations"])
    for o in r.answer["outcomes"]:
        print("  *", o["text"], o["citations"])
    for n in r.policy_notes:
        print("  NOTE:", n)


raw = pd.read_csv(a.csv, dtype=str)
clean, _ = clean_tickets(raw)
base = clean[clean["created_date"] < pd.Timestamp(a.cutoff)]
delta = raw[pd.to_datetime(raw["created_date"]) >= pd.Timestamp(a.cutoff)]
print(f"base: {len(base)} tickets before {a.cutoff} | real delta: {len(delta)} raw rows")

if Path(a.state).exists():
    shutil.rmtree(a.state)
embedder, llm = Embedder(), LLM()
print("initial state:", init_state(base, a.state, embedder, "data/processed/pattern_examples_index.jsonl"))
retriever = PatternRetriever(Path(a.state) / "index", a.state, embedder=embedder)
assistant = Assistant(retriever, llm)

show("STEP 1: telecom complaint before the class exists", assistant.ask(QUERY))

rep = apply_batch(a.state, delta, embedder, llm)
retriever.reload()
print("\nSTEP 2: ingested real days 8-9:", json.dumps({k: v for k, v in rep.items() if k != "filter"}, indent=1, default=str))

syn = synthetic_telecom()
rep = apply_batch(a.state, syn, embedder, llm)
retriever.reload()
print("\nSTEP 3: ingested SYNTHETIC telecom class:", json.dumps({k: v for k, v in rep.items() if k != "filter"}, indent=1, default=str))

rep2 = apply_batch(a.state, syn, embedder, llm)
print(f"\nIdempotency check (same batch again): ingested={rep2['ingested']} duplicates_skipped={rep2['duplicates_skipped']}")

show("STEP 4: same telecom complaint after ingestion", assistant.ask(QUERY))
