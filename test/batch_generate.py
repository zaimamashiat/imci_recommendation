
import json
import time
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/generate"

JSON_FILE = Path("imci_all_138_locations_2027-05-11.json")
RESULT_FILE = Path("batch_results.json")

# Load original locations
with JSON_FILE.open("r", encoding="utf-8") as f:
    locations = json.load(f)

# Load previous results
with RESULT_FILE.open("r", encoding="utf-8") as f:
    results = json.load(f)

# Identify only Read timed out errors
timeout_keys = {
    item["location_key"]
    for item in results
    if "error" in item
    and "Read timed out" in item["error"]
}

# Match failed locations with original payloads
retry_locations = [
    location for location in locations
    if location["location_key"] in timeout_keys
]

print(f"Total locations: {len(locations)}")
print(f"Timed-out locations: {len(retry_locations)}")

if not retry_locations:
    print("No timed-out locations found.")
    raise SystemExit

with requests.Session() as session:

    for index, location in enumerate(retry_locations, start=1):

        key = location["location_key"]

        print(
            f"[Retry {index}/{len(retry_locations)}] "
            f"{location['location_type']} - {key}",
            flush=True
        )

        try:
            response = session.post(
                API_URL,
                json=location,
                timeout=1000
            )

            try:
                data = response.json()
            except ValueError:
                data = {"response": response.text[:2000]}

            new_result = {
                "location_key": key,
                "status_code": response.status_code,
                "result": data
            }

        except requests.RequestException as exc:

            new_result = {
                "location_key": key,
                "error": str(exc)
            }

        # Replace the old failed result instead of duplicating it
        for i, old_result in enumerate(results):
            if old_result["location_key"] == key:
                results[i] = new_result
                break

        # Save progress after each retry
        RESULT_FILE.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(new_result, flush=True)

        time.sleep(0.2)

print("\nRetry completed!")
print(f"Locations retried: {len(retry_locations)}")
print(f"Results saved to {RESULT_FILE}")
