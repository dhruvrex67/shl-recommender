import json

# Using the catalog directly from the API response
import urllib.request

url = 'https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    raw_bytes = bytearray(response.read())

# Fix ALL control characters including the newline inside strings
# Strategy: parse byte by byte, tracking if we're inside a JSON string
inside_string = False
escaped = False
result = bytearray()

i = 0
while i < len(raw_bytes):
    b = raw_bytes[i]
    if escaped:
        result.append(b)
        escaped = False
    elif b == ord('\\') and inside_string:
        result.append(b)
        escaped = True
    elif b == ord('"'):
        inside_string = not inside_string
        result.append(b)
    elif inside_string and b < 0x20:
        # Replace control chars inside strings with space
        result.append(ord(' '))
    else:
        result.append(b)
    i += 1

raw = json.loads(result.decode('utf-8', errors='ignore'))
print(f"Parsed {len(raw)} items successfully!")

catalog = []
for item in raw:
    catalog.append({
        "name": item.get("name", "").strip(),
        "url": item.get("link", ""),
        "test_types": item.get("keys", []),
        "remote_testing": item.get("remote", "no") == "yes",
        "adaptive": item.get("adaptive", "no") == "yes",
        "description": item.get("description", "").replace('\r', ' ').replace('\n', ' ').strip(),
        "job_levels": item.get("job_levels", []),
        "duration": item.get("duration", ""),
        "languages": item.get("languages", [])
    })

with open('data/catalog.json', 'w', encoding='utf-8') as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)

print(f'Saved {len(catalog)} assessments to data/catalog.json')
