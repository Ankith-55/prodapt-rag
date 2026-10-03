"""Ask the assistant: python scripts/ask.py "complaint text"   (no args = built-in demo complaints)"""
import argparse
import json
from dataclasses import asdict

from ticketrag import runlog
from ticketrag.llm import LLM
from ticketrag.pipeline import Assistant
from ticketrag.retrieve import PatternRetriever

DEMO = [
    "My neighbours upstairs are blasting music again at 3am, I can't sleep and I have work in the morning",
    "There's no heat in my apartment for 3 days now and it's freezing, I have a baby at home",
    "A huge pothole on my street destroyed my tyre yesterday",
    "My broadband drops every evening around 8 and I've already restarted the router twice. "
    "I work from home and this is costing me.",
]

ap = argparse.ArgumentParser()
ap.add_argument("complaint", nargs="*")
ap.add_argument("--index", default="data/processed/index")
ap.add_argument("--gate", type=float, default=0.74)
a = ap.parse_args()
runlog.start("ask")

assistant = Assistant(PatternRetriever(a.index), LLM(), gate=a.gate)
results = []
for q in a.complaint or DEMO:
    r = assistant.ask(q)
    results.append(asdict(r))
    print("\n" + "=" * 100)
    print("COMPLAINT:", q)
    if r.abstained:
        print(f"ABSTAINED: {r.abstain_reason}. Route to a human agent.")
        print("Nearest (low confidence):", *[f"  {s:.3f} {l}" for l, s in r.candidates[:3]], sep="\n")
    else:
        print(f"PARSED: category={r.category} | product={r.product} | severity={r.severity}/5 | "
              f"sentiment={r.sentiment} | tier={r.tier} | cosine={r.top_score:.3f}")
        if r.answer.get("opening"):
            print("\nOPENING:", r.answer["opening"])
        print("SUMMARY:", r.answer["summary"]["text"], r.answer["summary"]["citations"])
        print("STEPS:")
        for i, s in enumerate(r.answer["steps"], 1):
            print(f"  {i}. {s['text']} {s['citations']}")
        print("OUTCOMES:")
        for o in r.answer["outcomes"]:
            print(f"  - {o['text']} {o['citations']}")
        for n in r.policy_notes:
            print("NOTE:", n)
        print("SOURCES:", *[f"  [{k}] " + (v["label"] if v["kind"] == "pattern" else v["text"][:90] + "...")
                            for k, v in r.sources.items()], sep="\n")
        print("GROUNDING:", r.grounding)
    u = r.llm_usage
    cost = (u.get("prompt_tokens", 0) * 0.15 + u.get("completion_tokens", 0) * 0.60) / 1e6  # gpt-4o-mini list price
    print(f"TIMING ms: {r.timing_ms} | LLM calls={u.get('calls')} cache_hits={u.get('cache_hits')} "
          f"tokens={u.get('prompt_tokens')}+{u.get('completion_tokens')} (~${cost:.5f} uncached)")

with open("output/ask_last.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
