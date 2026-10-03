"""Verify the OpenAI key + model work with schema-constrained output, and show token usage."""
from ticketrag import runlog
from ticketrag.llm import LLM

runlog.start("check_llm")
llm = LLM()
print("model:", llm.model)
schema = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ["angry", "frustrated", "neutral", "positive"]},
                   "severity": {"type": "integer"}},
    "required": ["sentiment", "severity"],
    "additionalProperties": False,
}
out = llm.json("Classify the complaint. Severity is 1 (minor) to 5 (emergency).",
               "No heat for 3 days and I have a baby at home!", schema, name="probe")
print("response:", out)
print("usage:", llm.usage)
try:
    ids = sorted(m.id for m in llm.client.models.list() if m.id.startswith(("gpt-4", "gpt-5", "o4")))
    print("available gpt models:", ids)
except Exception as e:  # listing is optional
    print("could not list models:", e)
