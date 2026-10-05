"""Download closed NYC 311 tickets from the official open-data API into the CSV the pipeline expects.

  python scripts/fetch_data.py --start 2026-03-01 --end 2026-03-10 --out dataset/closed_tickets_rag.csv
  python scripts/fetch_data.py --start 2026-03-10 --end 2026-03-13 --out dataset/next_days.csv     # a later batch

Source: NYC OpenData "311 Service Requests from 2020 to Present" (dataset erm2-nwe9, Socrata API). Only closed tickets
with a recorded resolution are requested, and only the 7 columns the pipeline uses. Set SOCRATA_APP_TOKEN for higher
rate limits (optional). Then: python scripts/build_patterns.py --csv <out>."""
import argparse
import io
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
COLUMNS = ["unique_key", "created_date", "closed_date", "complaint_type", "descriptor", "descriptor_2",
           "resolution_description"]

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True, help="first created_date, e.g. 2026-03-01")
ap.add_argument("--end", required=True, help="created_date strictly before this day, e.g. 2026-03-10")
ap.add_argument("--out", required=True)
ap.add_argument("--max-rows", type=int, default=200_000)
ap.add_argument("--page", type=int, default=50_000)
a = ap.parse_args()

where = (f"status = 'Closed' AND resolution_description IS NOT NULL "
         f"AND created_date >= '{a.start}T00:00:00' AND created_date < '{a.end}T00:00:00'")
headers = {"User-Agent": "ticketrag-fetch/1.0"}
if os.getenv("SOCRATA_APP_TOKEN"):
    headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]

frames, offset = [], 0
while offset < a.max_rows:
    query = urllib.parse.urlencode({"$select": ",".join(COLUMNS), "$where": where, "$order": "created_date",
                                    "$limit": min(a.page, a.max_rows - offset), "$offset": offset})
    request = urllib.request.Request(f"{ENDPOINT}?{query}", headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                chunk = pd.read_csv(io.StringIO(resp.read().decode("utf-8")), dtype=str)
            break
        except Exception as exc:  # transient network / rate limit: back off and retry
            if attempt == 3:
                sys.exit(f"download failed after 4 attempts: {exc}")
            time.sleep(5 * (attempt + 1))
    frames.append(chunk)
    offset += len(chunk)
    print(f"  fetched {offset} rows")
    if len(chunk) < a.page:
        break

if not frames or sum(len(f) for f in frames) == 0:
    sys.exit("no rows returned: check the date range")
df = pd.concat(frames, ignore_index=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)
print(f"wrote {len(df)} closed tickets ({df['created_date'].min()} -> {df['created_date'].max()}) to {a.out}")
