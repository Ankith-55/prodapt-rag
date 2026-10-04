"""Temporal backtest on REAL later tickets: do historical resolution distributions predict what actually happened?

For each new closed ticket we look up its pattern in the existing table (no LLM, no embeddings: this tests the
premise of the whole system) and measure how much probability the pattern's historical distribution gave to the
ticket's real resolution, against two baselines (complaint_type-level and global distributions).

  python scripts/backtest_temporal.py --csv dataset/next_days.csv --tag backtest_mar10_12
Never ingest the batch into --state before running this (tickets already in the state are skipped and reported)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ticketrag import runlog
from ticketrag.patterns import clean_tickets

ap = argparse.ArgumentParser()
ap.add_argument("--state", default="data/processed")
ap.add_argument("--csv", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()
runlog.start(f"eval_{a.tag}")

state = Path(a.state)
train = pd.read_parquet(state / "tickets.parquet")
pat = pd.read_parquet(state / "patterns.parquet").set_index("pattern_id")
new, filt = clean_tickets(pd.read_csv(a.csv, dtype=str))
overlap = new["unique_key"].isin(set(train["unique_key"]))
if overlap.any():
    print(f"WARNING: {int(overlap.sum())} tickets are already in the state; skipped to avoid leakage")
new = new[~overlap].reset_index(drop=True)
print(f"train tickets={len(train)}  test tickets={len(new)}  date range {new.created_date.min()} -> {new.created_date.max()}")
print("cleaning:", filt)

dist = {pid: {d["resolution_id"]: d["share"] for d in row["resolution_dist"]} for pid, row in pat.iterrows()}
order = {pid: [d["resolution_id"] for d in row["resolution_dist"]] for pid, row in pat.iterrows()}
glob = train["resolution_id"].value_counts(normalize=True).to_dict()
by_type = {t: g["resolution_id"].value_counts(normalize=True).to_dict() for t, g in train.groupby("complaint_type")}

rows = []
for r in new.itertuples():
    seen = r.pattern_id in dist
    p_pat = dist[r.pattern_id].get(r.resolution_id, 0.0) if seen else np.nan
    rows.append({
        "seen": seen,
        "type_seen": r.complaint_type in by_type,
        "res_seen": r.resolution_id in glob,
        "p_pattern": p_pat,
        "p_type": by_type.get(r.complaint_type, {}).get(r.resolution_id, 0.0),
        "p_global": glob.get(r.resolution_id, 0.0),
        "top1": (order[r.pattern_id][0] == r.resolution_id) if seen else np.nan,
        "top3": (r.resolution_id in order[r.pattern_id][:3]) if seen else np.nan,
        "tier": pat.loc[r.pattern_id, "tier"] if seen else "new_pattern",
        "train_n": int(pat.loc[r.pattern_id, "total_cases"]) if seen else 0,
        "pred_top1_share": float(pat.loc[r.pattern_id, "top1_share"]) if seen else np.nan,
    })
df = pd.DataFrame(rows)
s = df[df["seen"]]


def summarize(d: pd.DataFrame) -> dict:
    return {"n": int(len(d)), "mean_p_pattern": round(float(d.p_pattern.mean()), 4),
            "mean_p_type": round(float(d.p_type.mean()), 4), "mean_p_global": round(float(d.p_global.mean()), 4),
            "top1_acc": round(float(d.top1.astype(float).mean()), 4), "top3_cov": round(float(d.top3.astype(float).mean()), 4),
            "mean_predicted_top1_share": round(float(d.pred_top1_share.mean()), 4)}


rep = {
    "tag": a.tag, "train_tickets": int(len(train)), "test_tickets": int(len(df)),
    "pattern_seen_pct": round(100 * df.seen.mean(), 2),
    "complaint_type_seen_pct": round(100 * df.type_seen.mean(), 2),
    "resolution_template_seen_pct": round(100 * df.res_seen.mean(), 2),
    "new_patterns": int(new.loc[~new.pattern_id.isin(dist), "pattern_id"].nunique()),
    "overall_seen_patterns": summarize(s),
    "by_tier": {t: summarize(g) for t, g in s.groupby("tier")},
    "by_train_support": {},
}
for name, lo, hi in (("n>=100", 100, 10**9), ("30<=n<100", 30, 100), ("5<=n<30", 5, 30), ("n<5", 0, 5)):
    g = s[(s.train_n >= lo) & (s.train_n < hi)]
    if len(g):
        rep["by_train_support"][name] = summarize(g)

print(json.dumps(rep, indent=2))
print("\nHow to read: mean_p_* = average probability each distribution gave to the ticket's REAL resolution")
print("(higher is better; pattern vs type vs global shows what retrieval adds). For fast_lookup patterns,")
print("top1_acc should be close to mean_predicted_top1_share if the tier threshold is well calibrated.")
Path("eval/reports/backtest").mkdir(parents=True, exist_ok=True)
Path(f"eval/reports/backtest/{a.tag}.json").write_text(json.dumps(rep, indent=2))
print(f"\nreport -> eval/reports/backtest/{a.tag}.json")
