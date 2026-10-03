"""Build data/processed/{tickets,patterns,resolutions}.parquet from the raw CSV."""
import argparse
import json
from pathlib import Path

import pandas as pd

from ticketrag import runlog
from ticketrag.patterns import build_pattern_table, clean_tickets

runlog.start("build_patterns")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="dataset/closed_tickets_rag.csv")
ap.add_argument("--out", default="data/processed")
a = ap.parse_args()

clean, report = clean_tickets(pd.read_csv(a.csv, dtype=str))
pat, res = build_pattern_table(clean)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
clean.to_parquet(out / "tickets.parquet", index=False)
pat.to_parquet(out / "patterns.parquet", index=False)
res.to_parquet(out / "resolutions.parquet", index=False)

report |= {
    "patterns": len(pat), "unique_resolutions": len(res),
    "tier_patterns": pat["tier"].value_counts().to_dict(),
    "tier_ticket_volume": pat.groupby("tier")["total_cases"].sum().to_dict(),
    "volume_share_top1_lt_0.75": round(float(pat.loc[pat.top1_share < 0.75, "total_cases"].sum() / len(clean)), 4),
    "patterns_top1_ge_0.9": int((pat.top1_share >= 0.9).sum()),
    "date_range": [str(clean.created_date.min()), str(clean.created_date.max())],
}
(out / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
