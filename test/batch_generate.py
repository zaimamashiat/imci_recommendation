
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

API_URL = "http://127.0.0.1:8000/generate/weekly-monthly"

RESULT_FILE = Path("weekly_monthly_results.json")

TIMEOUT = (15, 7200)

PAYLOAD = {
    "persist": True
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def save_results(data):
    RESULT_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ==========================================
# MAIN GENERATION
# ==========================================

def main():

    overall_start = time.perf_counter()

    print("=" * 70)
    print("WEEKLY & MONTHLY RECOMMENDATION GENERATION")
    print("=" * 70)

    print("Weekly period : 26 Nov - 28 Nov 2026")
    print("Monthly period: 26 Nov - 26 Dec 2026")
    print("Database      : meditron_recommendtaion.weekly_monthly_recommendation")
    print("Started at    :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("=" * 70)
    print("\nConnecting to recommendation API...\n", flush=True)

    try:

        with requests.Session() as session:

            response = session.post(
                API_URL,
                json=PAYLOAD,
                timeout=TIMEOUT
            )

            print("HTTP Status:", response.status_code)

            response.raise_for_status()

            data = response.json()

        # ======================================
        # PROCESS RESPONSE
        # ======================================

        results = data.get("results", [])

        if not results:
            print("\nWARNING: No recommendation results returned.")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            save_results(data)
            return

        total = len(results)
        completed = 0
        failed = 0

        print("\n" + "=" * 70)
        print("LOCATION-WISE GENERATION RESULTS")
        print("=" * 70)

        for index, item in enumerate(results, start=1):

            status = item.get("status", "completed")

            period = item.get("period_type", "unknown").upper()

            location_name = item.get(
                "location_name",
                item.get("location_key", "Unknown location")
            )

            location_key = item.get("location_key", "N/A")

            disease = item.get("disease_name", "N/A")

            elapsed = item.get("elapsed_seconds", 0)

            print(
                f"\n[{index}/{total}] "
                f"{period} | {location_name}",
                flush=True
            )

            print(f"Location Key : {location_key}")
            print(f"Disease      : {disease}")
            print(f"Status       : {status}")
            print(f"Time Taken   : {elapsed} seconds")

            if status == "completed":
                completed += 1
            else:
                failed += 1

        # ======================================
        # SAVE RESULTS
        # ======================================

        save_results(data)

        total_elapsed = time.perf_counter() - overall_start

        # ======================================
        # FINAL SUMMARY
        # ======================================

        print("\n")
        print("=" * 70)
        print("GENERATION SUMMARY")
        print("=" * 70)

        print(f"Total Results       : {total}")
        print(f"Completed           : {completed}")
        print(f"Failed              : {failed}")

        print(
            "Server Processing  :",
            format_time(data.get("elapsed_seconds", 0))
        )

        print(
            "Client Total Time  :",
            format_time(total_elapsed)
        )

        print(f"Database Saved      : {data.get('saved', False)}")

        print(f"Results File        : {RESULT_FILE.resolve()}")

        print("=" * 70)
        print("BATCH GENERATION FINISHED!")
        print("=" * 70)

    except requests.Timeout:

        print("\nERROR: API request timed out.")
        print("Check the FastAPI server logs and database before retrying.")

    except requests.ConnectionError:

        print("\nERROR: Cannot connect to FastAPI.")
        print("Make sure your main Python application is running.")

    except requests.HTTPError as exc:

        print("\nHTTP ERROR:", exc)

        if exc.response is not None:
            print(exc.response.text[:5000])

    except ValueError as exc:

        print("\nINVALID API RESPONSE:", exc)

    except Exception as exc:

        print("\nUNEXPECTED ERROR:", str(exc))


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    main()
