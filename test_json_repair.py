"""
Quick unit test for the JSON repair logic in parse_and_validate_response.
Tests two failure modes:
  1. Missing commas between top-level fields (structural)
  2. Truncated output (max_tokens cut mid-JSON)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from app.meditron import parse_and_validate_response

RETRIEVED_PAGES = ["Page 22", "Page 24"]
EXPECTED_DISEASE = "Anaemia"
LOCATION = {
    "selected_level": "upazila",
    "selected_location": "Hijla",
    "division": "Barishal",
    "district": "Barishal",
    "upazila": "Hijla",
    "union": "Hizla Gaurabdi",
}

# --- Case 1: Missing commas between top-level fields (original bug) ---
MISSING_COMMAS_JSON = """{
  "disease": "Anaemia",
  "location": {
    "upazila": "Hijla"
  }
  "summary": "In Hijla, Anaemia count is 24 cases now and 12 next week."
  "priority_public_health_measures": ["Provide iron supplementation"]
  "health_facility_readiness": ["Ensure iron tablets are stocked"]
  "caregiver_and_community_communication": ["Advise caregivers on iron-rich foods"]
  "booklet_references": ["Page 22"]
  "requires_expert_review": false
  "expert_review_reason": ""
}"""

# --- Case 2: Truncated JSON (max_tokens cut mid-output) ---
TRUNCATED_JSON = """{
  "disease": "Anaemia",
  "location": {
    "selected_level": "upazila",
    "selected_location": "Hijla",
    "division": "Barishal",
    "district": "Barishal",
    "upazila": "Hijla",
    "union": "Hizla Gaurabdi"
  }"""

DISEASE_COUNTS = {
    "count_now": 24,
    "count_next_week": 12,
    "current_week_total": 48,
    "next_week_total": 36,
    "trend": "decreasing",
}

PASS = 0
FAIL = 0

def run(label, content):
    global PASS, FAIL
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    try:
        result = parse_and_validate_response(
            content=content,
            retrieved_pages=RETRIEVED_PAGES,
            expected_disease=EXPECTED_DISEASE,
            location=LOCATION,
            disease_counts=DISEASE_COUNTS,
        )
        print(f"  disease   : {result.get('disease')}")
        print(f"  summary   : {result.get('summary', '')[:120]}...")
        print(f"  ph_measures : {result.get('priority_public_health_measures', [])[:1]}")
        print(f"  references: {result.get('booklet_references')}")
        print(f"  PASS ✓")
        PASS += 1
    except Exception as e:
        print(f"  FAIL ✗  — {e}")
        FAIL += 1

run("Case 1 — Missing commas", MISSING_COMMAS_JSON)
run("Case 2 — Truncated JSON", TRUNCATED_JSON)

print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
