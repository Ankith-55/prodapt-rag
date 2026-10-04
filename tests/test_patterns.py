import pandas as pd
import pytest

from ticketrag.patterns import assign_tier, build_pattern_table, clean_tickets, fix_mojibake


def test_fix_mojibake_repairs_double_encoded_utf8():
    broken = "Transportationâ\u0080\u0099s website"
    assert fix_mojibake(broken) == "Transportation’s website"
    assert fix_mojibake("plain ascii") == "plain ascii"


@pytest.mark.parametrize("share,n,tier", [
    (0.95, 30, "fast_lookup"), (0.95, 29, "rag_light"),  # not enough cases to trust a high share
    (0.50, 10, "rag_light"), (0.49, 1000, "rag_core"),
])
def test_assign_tier_thresholds(share, n, tier):
    assert assign_tier(share, n) == tier


def test_clean_tickets_filters_and_normalises():
    raw = pd.DataFrame([
        {"unique_key": "1", "created_date": "2026-03-01T00:00:00", "closed_date": "2026-03-01T01:00:00", "status": "Closed",
         "complaint_type": "Noise", "descriptor": "Loud", "descriptor_2": "N/A", "resolution_description": "Done."},
        {"unique_key": "2", "created_date": "2026-03-01T00:00:00", "closed_date": "", "status": "Open",
         "complaint_type": "Noise", "descriptor": "Loud", "descriptor_2": "", "resolution_description": "Done."},
        {"unique_key": "3", "created_date": "2026-03-01T00:00:00", "closed_date": "2026-03-01T01:00:00", "status": "Closed",
         "complaint_type": "Noise", "descriptor": "Loud", "descriptor_2": "", "resolution_description": "N/A"},
        {"unique_key": "4", "created_date": "2026-03-01T00:00:00", "closed_date": "2026-03-01T01:00:00", "status": "Closed",
         "complaint_type": "", "descriptor": "Loud", "descriptor_2": "", "resolution_description": "Done."},
    ])
    clean, report = clean_tickets(raw)
    assert list(clean["unique_key"]) == ["1"]  # open, N/A resolution and empty complaint_type rows are dropped
    assert clean.loc[0, "descriptor_2"] == ""  # "N/A" normalised to empty
    assert report["usable"] == 1 and report["closed"] == 3
    assert clean.loc[0, "close_hours"] == pytest.approx(1.0)


def test_pattern_table_distribution_sums_to_one(state):
    pat = pd.read_parquet(state / "patterns.parquet")
    assert len(pat) == 3
    for _, row in pat.iterrows():
        assert sum(d["share"] for d in row["resolution_dist"]) == pytest.approx(1.0, abs=1e-3)
    tiers = dict(zip(pat["descriptor"], pat["tier"]))
    assert tiers == {"Loud Music/Party": "rag_core", "No Heat": "rag_light", "Pothole": "fast_lookup"}
