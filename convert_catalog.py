"""
convert_catalog.py
Downloads the official SHL catalog JSON, cleans control characters,
normalises field names, and saves to data/catalog.json.

Run once locally before deploying, OR let main.py call this on startup.
"""

import json
import urllib.request
from pathlib import Path

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_PATH = Path("data/catalog.json")


def fetch_and_clean(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw_bytes = bytearray(response.read())

    # Fix control characters inside JSON strings
    inside_string = False
    escaped = False
    result = bytearray()
    for b in raw_bytes:
        if escaped:
            result.append(b)
            escaped = False
        elif b == ord("\\") and inside_string:
            result.append(b)
            escaped = True
        elif b == ord('"'):
            inside_string = not inside_string
            result.append(b)
        elif inside_string and b < 0x20:
            result.append(ord(" "))  # replace control chars with space
        else:
            result.append(b)

    raw = json.loads(result.decode("utf-8", errors="ignore"))
    print(f"Parsed {len(raw)} items from SHL catalog.")
    return raw


def normalise(raw: list[dict]) -> list[dict]:
    catalog = []
    for item in raw:
        catalog.append({
            "name": item.get("name", "").strip(),
            "url": item.get("link", ""),
            "test_types": item.get("keys", []),
            "remote_testing": item.get("remote", "no") == "yes",
            "adaptive": item.get("adaptive", "no") == "yes",
            "description": (
                item.get("description", "")
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            ),
            "job_levels": item.get("job_levels", []),
            "duration": item.get("duration", ""),
            "languages": item.get("languages", []),
        })
    return catalog


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = fetch_and_clean(CATALOG_URL)
    catalog = normalise(raw)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(catalog)} assessments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
