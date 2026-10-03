"""Generate a larger out-of-domain query set, split into truly-outside-311 and borderline (arguably in scope),
so abstention can be measured honestly. Writes eval/ood_outside.txt and eval/ood_borderline.txt."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ticketrag import runlog
from ticketrag.llm import LLM

OUTSIDE = ["home broadband or wifi outages", "mobile phone plan, roaming or billing", "bank account or credit card problems",
           "airline delays, lost luggage or refunds", "online shopping orders that never arrived", "software or app bugs",
           "streaming or cable TV subscriptions", "health insurance claim denied", "employer or payroll disputes",
           "university admissions or tuition", "cryptocurrency or investment app losses", "video game accounts or purchases",
           "appliance warranty claims", "tax software or tax refund status", "hotel booking in another country",
           "car insurance claim disputes"]
BORDERLINE = ["landlord refusing to return a security deposit", "restaurant food quality or food poisoning",
              "store refusing a refund or selling expired goods", "contractor who took a deposit and disappeared",
              "noisy neighbour dispute with no city agency involved", "lost wallet or phone in a taxi"]
SCHEMA = {"type": "object", "properties": {"complaints": {"type": "array", "items": {"type": "string"}}},
          "required": ["complaints"], "additionalProperties": False}
SYSTEM = ("You write realistic, messy first-person customer complaints (typos, emotion, side details, 1-3 sentences). "
          "Each complaint must be clearly about the given topic. Never mention 311 or city agencies.")

runlog.start("make_ood_set")
llm = LLM()


def gen(topic):
    return llm.json(SYSTEM, f"Topic: {topic}\nWrite 5 different complaints.", SCHEMA, name="ood", temperature=0.9,
                    max_tokens=900)["complaints"][:5]


for name, topics in (("ood_outside", OUTSIDE), ("ood_borderline", BORDERLINE)):
    with ThreadPoolExecutor(6) as ex:
        qs = [" ".join(q.split()) for batch in ex.map(gen, topics) for q in batch]
    Path(f"eval/{name}.txt").write_text("\n".join(qs) + "\n", encoding="utf-8")
    print(f"{name}: {len(qs)} queries -> eval/{name}.txt\n  e.g. {qs[0]}\n  e.g. {qs[-1]}")
print("usage:", llm.usage)
