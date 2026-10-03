"""End-to-end assistant: retrieve -> (abstain?) -> parse -> route by tier -> grounded answer."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ticketrag.generate import GenReport, build_sources, generate_answer, render_template
from ticketrag.llm import LLM
from ticketrag.parse import candidate_label, parse_complaint
from ticketrag.retrieve import PatternRetriever, _tidy

EMERGENCY_NOTE = ("If there is immediate danger to life or safety, tell the customer to call 911. "
                  "(Standard desk policy, not drawn from historical tickets.)")


@dataclass
class Answer:
    complaint: str
    abstained: bool
    abstain_reason: str | None = None
    top_score: float = 0.0
    category: str | None = None
    product: str | None = None
    severity: int | None = None
    sentiment: str | None = None
    tier: str | None = None
    method: str | None = None  # "llm" | "template" | None when abstained
    answer: dict | None = None
    sources: dict = field(default_factory=dict)
    source_block: str = ""  # exact text the LLM saw (used by the faithfulness judge)
    policy_notes: list = field(default_factory=list)
    candidates: list = field(default_factory=list)  # [(label, cosine)]
    grounding: dict = field(default_factory=dict)
    timing_ms: dict = field(default_factory=dict)
    llm_usage: dict = field(default_factory=dict)


class Assistant:
    def __init__(self, retriever: PatternRetriever, llm: LLM, gate: float = 0.74, k: int = 5):
        self.retriever, self.llm, self.gate, self.k = retriever, llm, gate, k

    def ask(self, complaint: str) -> Answer:
        usage0 = dict(self.llm.usage)
        t0 = time.perf_counter()
        hits = self.retriever.search(complaint, k=self.k, top_resolutions=5)
        t1 = time.perf_counter()
        res = Answer(complaint, abstained=True, top_score=hits[0].score if hits else 0.0,
                     candidates=[(candidate_label(h), round(h.score, 3)) for h in hits],
                     timing_ms={"retrieve": round((t1 - t0) * 1000)})

        def finish() -> Answer:
            res.timing_ms["total"] = round((time.perf_counter() - t0) * 1000)
            res.llm_usage = {k: self.llm.usage[k] - usage0[k] for k in usage0}
            return res

        if not hits or hits[0].score < self.gate:
            res.abstain_reason = f"low_similarity (top cosine {res.top_score:.3f} < {self.gate})"
            return finish()

        parsed = parse_complaint(self.llm, complaint, hits)
        t2 = time.perf_counter()
        res.timing_ms["parse"] = round((t2 - t1) * 1000)
        res.severity, res.sentiment = parsed.severity, parsed.sentiment
        if parsed.hit is None:
            res.abstain_reason = "llm_no_matching_candidate"
            return finish()

        main = parsed.hit
        res.abstained, res.tier = False, main.tier
        res.category = _tidy(parsed.category)
        res.product = " / ".join(_tidy(x) for x in (main.descriptor, main.descriptor_2) if x)
        siblings = [h for h in hits if h.pattern_id != main.pattern_id][:2] if main.tier == "rag_core" else []
        src = build_sources(main, siblings, n_res=5 if main.tier == "rag_core" else 3)
        res.sources, res.source_block = src.items, src.block

        if main.tier == "fast_lookup":
            res.answer, report = render_template(main, src)
        else:
            res.answer, report = generate_answer(self.llm, complaint, parsed, src)
            if not res.answer["summary"] or not res.answer["steps"]:  # nothing usable survived validation
                res.answer, fallback = render_template(main, src)
                report.method = fallback.method
        res.method = report.method
        res.grounding = {"method": report.method, "attempts": report.attempts,
                         "first_attempt_violations": report.first_attempt_violations,
                         "remaining_violations": report.violations}
        res.timing_ms["generate"] = round((time.perf_counter() - t2) * 1000)

        if main.total_cases < 30:
            res.policy_notes.append(f"Low evidence: this ticket type has only {main.total_cases} closed tickets, "
                                    "so the shares and times above are unreliable.")
        if parsed.severity >= 4:
            res.policy_notes.append(f"High severity ({parsed.severity}/5): prioritise this ticket.")
        if parsed.severity >= 5:
            res.policy_notes.append(EMERGENCY_NOTE)
        return finish()
