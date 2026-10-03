"""LLM-written customer complaints for a ticket pattern (the corpus has labels, not prose).

  role "index": clear complaints, embedded next to the label to improve retrieval
  role "eval" : messy complaints from an independent prompt, used as a held-out eval set"""
from __future__ import annotations

from ticketrag.llm import LLM

STYLES = {
    "index": ("Write clear, natural first-person complaints a resident would phone in. Vary length "
              "(1-3 sentences), tone and the details mentioned (time, duration, impact)."),
    "eval": ("Write MESSY, realistic complaints as people actually type or say them: typos, slang, run-ons, "
             "emotion, irrelevant side details, sometimes vague. Vary length from one line to a short rant."),
}
SCHEMA = {
    "type": "object",
    "properties": {"complaints": {"type": "array", "items": {"type": "string"}}},
    "required": ["complaints"],
    "additionalProperties": False,
}


def _system(role: str) -> str:
    return ("You write realistic complaints that residents of New York City submit to the 311 hotline. "
            f"{STYLES[role]} Never mention '311', ticket categories, or agency names unless a real person "
            "naturally would. Do not copy the category label verbatim; describe the situation in your own "
            "words. Each complaint must be about the described problem only.")


def generate_complaints(llm: LLM, label: str, role: str = "index", n: int = 4) -> list[str]:
    user = f"Problem category (hidden from the resident): {label}\nWrite {n} different complaints."
    res = llm.json(_system(role), user, SCHEMA, name="complaints", temperature=0.9, max_tokens=1500)
    return res["complaints"][:n]
