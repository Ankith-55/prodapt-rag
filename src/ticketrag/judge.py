"""LLM faithfulness judge: is each claim in a generated answer supported by the sources the generator saw?
Run with a different model than the generator to reduce self-preference bias."""
from __future__ import annotations

from ticketrag.llm import LLM

SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"},
                       "verdict": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                       "reason": {"type": "string", "description": "max 15 words"}},
        "required": ["id", "verdict", "reason"], "additionalProperties": False}}},
    "required": ["verdicts"],
    "additionalProperties": False,
}

SYSTEM = """You are a strict fact-checker. You receive SOURCES (statistics and recorded resolution texts from past
support tickets) and numbered CLAIMS written for a support agent. For each claim decide whether the SOURCES support it:
- supported: every factual statement in the claim is stated in, or directly implied by, the sources.
- partial: some of it is supported, some is not stated in the sources.
- unsupported: not stated in the sources, interpretive speculation, or contradicts them.
Brief politeness (e.g. "I understand your frustration") is ignorable: judge only the factual remainder.
Promises of actions by the agent ("I will create a complaint", "I will update you") are unsupported unless the
sources describe them. Return one verdict per claim id."""


def claims_from_answer(answer: dict) -> list[tuple[str, str]]:
    # Outcomes are rendered verbatim from the resolution table by code, and the opening is empathy only,
    # so only the LLM-written summary and steps are fact-checked.
    return [("summary", answer["summary"]["text"])] + [("step", s["text"]) for s in answer["steps"]]


def judge_answer(judge: LLM, source_block: str, answer: dict) -> list[dict]:
    claims = claims_from_answer(answer)
    listing = "\n".join(f"{i}. {text}" for i, (_, text) in enumerate(claims, 1))
    out = judge.json(SYSTEM, f"SOURCES:\n{source_block}\n\nCLAIMS:\n{listing}", SCHEMA, name="verdicts",
                     temperature=0.0, max_tokens=900)
    by_id = {v["id"]: v for v in out["verdicts"]}
    return [{"kind": kind, "text": text, "verdict": by_id.get(i, {}).get("verdict", "unsupported"),
             "reason": by_id.get(i, {}).get("reason", "no verdict returned")}
            for i, (kind, text) in enumerate(claims, 1)]
