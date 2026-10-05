# Fonts

The UI uses the IBM Plex family (Sans, Serif, Mono), self-hosted because the page's Content-Security-Policy
only allows resources from its own origin. Licensed under the SIL Open Font License 1.1
(https://github.com/IBM/plex/blob/master/LICENSE.txt), which permits redistribution.

The `.woff2` files are not committed by default. Fetch them once with:

    python scripts/get_fonts.py

If they are missing the page still works: it falls back to Georgia / Segoe UI / Consolas.
