"""Download the IBM Plex web fonts (Latin subset, SIL Open Font License) into the UI's static folder.

  python scripts/get_fonts.py

Source: the Fontsource distribution of IBM Plex on jsDelivr. About 150 KB in total. Optional: the UI falls back to
system fonts if these files are absent."""
import sys
import urllib.request
from pathlib import Path

OUT = Path("src/ticketrag/static/fonts")
BASE = "https://cdn.jsdelivr.net/fontsource/fonts/{family}@latest/latin-{weight}-{style}.woff2"
FILES = {  # local name -> (fontsource family, weight, style)
    "ibm-plex-sans-400.woff2": ("ibm-plex-sans", 400, "normal"),
    "ibm-plex-sans-600.woff2": ("ibm-plex-sans", 600, "normal"),
    "ibm-plex-serif-400.woff2": ("ibm-plex-serif", 400, "normal"),
    "ibm-plex-serif-400-italic.woff2": ("ibm-plex-serif", 400, "italic"),
    "ibm-plex-serif-600.woff2": ("ibm-plex-serif", 600, "normal"),
    "ibm-plex-mono-400.woff2": ("ibm-plex-mono", 400, "normal"),
    "ibm-plex-mono-500.woff2": ("ibm-plex-mono", 500, "normal"),
}

OUT.mkdir(parents=True, exist_ok=True)
failed = 0
for name, (family, weight, style) in FILES.items():
    url = BASE.format(family=family, weight=weight, style=style)
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ticketrag-fonts/1.0"}), timeout=60).read()
        (OUT / name).write_bytes(data)
        print(f"ok   {name} ({len(data) // 1024} KB)")
    except Exception as exc:
        failed += 1
        print(f"FAIL {name}: {exc}")
sys.exit(1 if failed else 0)
