"""Offline pattern table: collapse closed tickets into (complaint_type, descriptor, descriptor_2)
patterns with their resolution distribution and a routing tier."""
from __future__ import annotations

import hashlib

import pandas as pd

KEY_COLS = ["complaint_type", "descriptor", "descriptor_2"]
FAST_MIN_SHARE, FAST_MIN_CASES, LIGHT_MIN_SHARE = 0.9, 30, 0.5
_MISSING = {"", "n/a", "na", "nan", "none", "null"}


def _hash(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:10]


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 that was decoded as latin-1/cp1252 and re-encoded (e.g. 'â\x80\x99' -> apostrophe)."""
    if "â" not in text and "Ã" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clean_tickets(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep closed tickets with a real resolution. Returns (clean_df, filter_report)."""
    df = raw.copy()
    for c in KEY_COLS + ["resolution_description"]:
        df[c] = df[c].fillna("").astype(str).map(fix_mojibake).str.strip()
    df["descriptor_2"] = df["descriptor_2"].where(~df["descriptor_2"].str.lower().isin(_MISSING), "")
    report = {"raw_rows": len(raw)}
    if "status" in df.columns:
        df = df[df["status"] == "Closed"]
    report["closed"] = len(df)
    df = df[~df["resolution_description"].str.lower().isin(_MISSING)]
    report["with_resolution"] = len(df)
    df = df[(df["complaint_type"] != "") & (df["descriptor"] != "")]
    report["usable"] = len(df)
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
    df["close_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    df["pattern_id"] = [_hash(*r) for r in df[KEY_COLS].itertuples(index=False)]
    df["resolution_id"] = df["resolution_description"].map(lambda t: _hash(t))
    return df.reset_index(drop=True), report


def assign_tier(top1_share: float, total: int) -> str:
    if top1_share >= FAST_MIN_SHARE and total >= FAST_MIN_CASES:
        return "fast_lookup"
    if top1_share >= LIGHT_MIN_SHARE:
        return "rag_light"
    return "rag_core"


def build_pattern_table(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (patterns, resolutions). `resolutions` is the deduped resolution-template catalogue."""
    res = (clean.groupby(["resolution_id", "resolution_description"]).size()
           .rename("total_count").reset_index().sort_values("total_count", ascending=False))
    rows = []
    for pid, g in clean.groupby("pattern_id", sort=False):
        counts = g["resolution_id"].value_counts()
        total = len(g)
        top1 = counts.iloc[0] / total
        first = g.iloc[0]
        rows.append({
            "pattern_id": pid,
            **{c: first[c] for c in KEY_COLS},
            "total_cases": total,
            "n_resolutions": len(counts),
            "top1_share": round(float(top1), 4),
            "tier": assign_tier(top1, total),
            "median_close_hours": round(float(g["close_hours"].median()), 2),
            "p90_close_hours": round(float(g["close_hours"].quantile(0.9)), 2),
            "resolution_dist": [{"resolution_id": r, "count": int(n), "share": round(n / total, 4)}
                                for r, n in counts.items()],
        })
    pat = pd.DataFrame(rows).sort_values("total_cases", ascending=False).reset_index(drop=True)
    return pat, res.reset_index(drop=True)
