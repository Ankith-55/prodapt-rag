"""Create eval/handlabel.csv: 40 random eval complaints to label by hand for severity (1-5) and sentiment.
Label them yourself WITHOUT looking at the model's answers, then we compare."""
import csv
import json
import random

from ticketrag.paths import examples_file

rows = [json.loads(line) for line in examples_file("eval").read_text(encoding="utf-8").splitlines()]
pool = [c for r in rows for c in r["complaints"]]
random.Random(7).shuffle(pool)
with open("eval/handlabel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "complaint", "severity_1to5", "sentiment (angry/frustrated/anxious/neutral/positive)"])
    for i, c in enumerate(pool[:40], 1):
        w.writerow([i, c, "", ""])
print("wrote eval/handlabel.csv (40 rows). Fill the last two columns, save, and tell me.")
