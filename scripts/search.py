"""Retrieval smoke test: python scripts/search.py [--q "complaint text"] [-k 5]"""
import argparse
import textwrap

from ticketrag import runlog
from ticketrag.retrieve import PatternRetriever

runlog.start("search")

SAMPLES = [
    "My neighbours upstairs are blasting music again at 3am, I can't sleep and I have work in the morning",
    "There's no heat in my apartment for 3 days now and it's freezing, I have a baby at home",
    "Some car has been parked across my driveway all day, I can't get out to go to work",
    "A huge pothole on my street destroyed my tyre yesterday",
    "I keep seeing rats near the garbage bins behind my building",
    "My broadband drops every evening around 8 and I've already restarted the router twice. "
    "I work from home and this is costing me.",  # out-of-domain on purpose: watch the scores
]

ap = argparse.ArgumentParser()
ap.add_argument("--q", action="append", help="complaint text (repeatable); defaults to built-in samples")
ap.add_argument("-k", type=int, default=5)
a = ap.parse_args()

retriever = PatternRetriever()
for q in a.q or SAMPLES:
    print("\n" + "=" * 100)
    print("QUERY:", q)
    hits = retriever.search(q, k=a.k)
    for h in hits:
        label = " / ".join(x for x in (h.complaint_type, h.descriptor, h.descriptor_2) if x)
        print(f"  {h.score:.3f}  [{h.tier:<11}] n={h.total_cases:<5} top1={h.top1_share:.2f}  {label}")
    for r in hits[0].resolutions[:2]:
        print(f"      {r['share']:.0%}: " + textwrap.shorten(r["text"], 110))
