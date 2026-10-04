"""Build a tiny state with the REAL embedder and check that retrieval finds the expected pattern.

Used by CI for two things: (1) a retrieval gate (fails the build if the embedder/index path regresses) and
(2) a state to mount into the container for the smoke test.

  python scripts/make_fixture_state.py --out /tmp/fixture_state --check"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from helpers import make_raw  # noqa: E402
from ticketrag.embed import Embedder  # noqa: E402
from ticketrag.ingest import init_state  # noqa: E402
from ticketrag.patterns import clean_tickets  # noqa: E402
from ticketrag.retrieve import PatternRetriever  # noqa: E402

EXPECTED = [
    ("loud music and a party next door at night", "Noise - Residential"),
    ("there is no heat in my apartment", "Heat/Hot Water"),
    ("a huge pothole on my street", "Street Condition"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--check", action="store_true", help="fail (exit 1) if retrieval returns an unexpected pattern")
a = ap.parse_args()

embedder = Embedder()
clean, _ = clean_tickets(make_raw())
print("state:", init_state(clean, a.out, embedder))

if a.check:
    retriever = PatternRetriever(Path(a.out) / "index", a.out, embedder=embedder)
    failed = 0
    for query, expected in EXPECTED:
        top = retriever.search(query, k=1)[0]
        ok = top.complaint_type == expected
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {query!r} -> {top.complaint_type} (cosine {top.score:.3f}, expected {expected})")
    sys.exit(1 if failed else 0)
