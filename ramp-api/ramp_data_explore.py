"""
Explore the Ramp Data public REST API:
  - Ramp Rate: https://api.ramp.com/v1/public/ramp-rate
  - AI Index:  https://api.ramp.com/v1/public/ai-index
"""

import json
from urllib.parse import quote
import requests

RAMP_RATE_BASE = "https://api.ramp.com/v1/public/ramp-rate"
AI_INDEX_BASE = "https://api.ramp.com/v1/public/ai-index"


def pp(label, data):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    print(json.dumps(data, indent=2))


def get(url, params=None):
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Ramp Rate ─────────────────────────────────────────────────────────────────

# 1. List all categories
categories_resp = get(f"{RAMP_RATE_BASE}/categories")
pp("GET /ramp-rate/categories", categories_resp)

# Pull the first category slug to use in subsequent calls
# Response shape: {"categories": [{"software_category": "..."}, ...]}
raw_cats = categories_resp if isinstance(categories_resp, list) else categories_resp.get("categories", categories_resp.get("data", []))
first_category = raw_cats[0].get("software_category") if raw_cats and isinstance(raw_cats[0], dict) else (raw_cats[0] if raw_cats else None)
print(f"\n>>> Using first category: {first_category}")

# Try a few well-populated categories until we get a 200
TEST_CATEGORIES = ["generative AI", "Communication", "CRM", "DevOps", "Accounting"]
working_category = None
summary = None
vendors_in_cat = None

for cat in TEST_CATEGORIES:
    cat_encoded = quote(cat, safe="")
    try:
        summary = get(f"{RAMP_RATE_BASE}/categories/{cat_encoded}/summary")
        working_category = cat
        break
    except Exception as e:
        print(f"  skipping '{cat}': {e}")

if working_category:
    cat_encoded = quote(working_category, safe="")
    # 2. Category summary
    pp(f"GET /ramp-rate/categories/{working_category}/summary", summary)

    # 3. Category vendors leaderboard
    vendors_in_cat = get(f"{RAMP_RATE_BASE}/categories/{cat_encoded}/vendors")
    pp(f"GET /ramp-rate/categories/{working_category}/vendors", vendors_in_cat)

    # Pull the first vendor slug from the leaderboard
    vendor_list = vendors_in_cat if isinstance(vendors_in_cat, list) else vendors_in_cat.get("vendors", vendors_in_cat.get("data", []))
    first_vendor_slug = None
    if vendor_list:
        first_item = vendor_list[0]
        first_vendor_slug = (
            first_item.get("vendor_slug")
            or first_item.get("slug")
            or first_item.get("vendor")
        )
    print(f"\n>>> Using first vendor slug: {first_vendor_slug}")

# 4. Resolve a vendor name to its canonical slug
resolved = get(f"{RAMP_RATE_BASE}/vendors/resolve", params={"vendor_name": "Slack"})
pp("GET /ramp-rate/vendors/resolve?vendor_name=Slack", resolved)

slack_slug = (
    resolved.get("vendor_slug")
    or resolved.get("slug")
    or (resolved.get("data") or {}).get("vendor_slug")
    or "slack"
)

# 5. Vendor profile
profile = get(f"{RAMP_RATE_BASE}/vendors/{slack_slug}/profile")
pp(f"GET /ramp-rate/vendors/{slack_slug}/profile", profile)

# 6. Compare vendors (Slack vs Zoom)
zoom_resolved = get(f"{RAMP_RATE_BASE}/vendors/resolve", params={"vendor_name": "Zoom"})
zoom_slug = (
    zoom_resolved.get("vendor_slug")
    or zoom_resolved.get("slug")
    or (zoom_resolved.get("data") or {}).get("vendor_slug")
    or "zoom"
)
comparison = get(
    f"{RAMP_RATE_BASE}/vendors/compare",
    params={"vendor_slugs": [slack_slug, zoom_slug]},
)
pp(f"GET /ramp-rate/vendors/compare ({slack_slug} vs {zoom_slug})", comparison)


# ── AI Index ──────────────────────────────────────────────────────────────────

# 7. Overall AI adoption (latest month)
adoption = get(f"{AI_INDEX_BASE}/adoption")
pp("GET /ai-index/adoption (months=1)", adoption)

# 8. Overall AI adoption — last 12 months
adoption_12 = get(f"{AI_INDEX_BASE}/adoption", params={"months": 12})
pp("GET /ai-index/adoption (months=12)", adoption_12)

# 9. AI adoption by sector
sectors = get(f"{AI_INDEX_BASE}/adoption/sectors", params={"months": 3})
pp("GET /ai-index/adoption/sectors (months=3)", sectors)

# 10. AI adoption by company size
sizes = get(f"{AI_INDEX_BASE}/adoption/sizes", params={"months": 3})
pp("GET /ai-index/adoption/sizes (months=3)", sizes)

print("\n\nDone — all Ramp Data endpoints exercised.")
