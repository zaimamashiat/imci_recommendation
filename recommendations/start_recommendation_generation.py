"""
Start IMCI recommendations for current and upcoming weekly + monthly forecasts.

Run from the same directory as app.py and .env:
    python start_recommendation_generation.py

Optional:
    python start_recommendation_generation.py --period week
    python start_recommendation_generation.py --period month
    python start_recommendation_generation.py --limit 100
    python start_recommendation_generation.py --dry-run

The script selects aggregated forecast periods that overlap today through
the next six months, skips source forecast IDs already in the recommendation
table, and calls the existing FastAPI POST /generate endpoint one row at a time.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
DB_SCHEMA = os.getenv("DB_SCHEMA", "meditron_recommendtaion")
SOURCE_TABLE = os.getenv("SOURCE_TABLE", "weekly_monthly_forecasts")
TARGET_TABLE = SOURCE_TABLE  # Recommendations are written into forecast rows.
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT = int(os.getenv("REQUEST_BUDGET_SECONDS", "720"))
MIN_INTERVAL = float(os.getenv("MEDITRON_MIN_INTERVAL_SECONDS", "0"))

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def qualified_table(schema: str, table: str) -> str:
    if not IDENTIFIER.fullmatch(schema) or not IDENTIFIER.fullmatch(table):
        raise ValueError(f"Invalid SQL identifier: {schema}.{table}")
    return f'"{schema}"."{table}"'


SOURCE_SQL = qualified_table(DB_SCHEMA, SOURCE_TABLE)
TARGET_SQL = qualified_table(DB_SCHEMA, TARGET_TABLE)


def get_pending_forecasts(period: str | None, limit: int | None):
    """
    Fetch pending forecast rows, then schedule whole forecast periods as:
        4 chronological weekly periods -> 1 chronological monthly period
    Repeat until all available periods have been scheduled.

    All location/disease rows belonging to a period stay together.
    """
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing from .env")

    period_filter = ""
    params = []
    if period:
        period_filter = "AND f.period_type = %s"
        params.append(period)

    query = f"""
        SELECT
            f.id,
            f.period_type,
            f.period_start,
            f.period_end,
            f.hierarchy_level,
            f.location_key,
            f.division_name,
            f.district_name,
            f.upazila_name,
            f.union_name,
            f.disease_code,
            f.disease_name
        FROM {SOURCE_SQL} AS f
        WHERE f.period_type IN ('week', 'month')
          AND f.period_end >= CURRENT_DATE
          AND f.period_start <= (CURRENT_DATE + INTERVAL '6 months')::date
          AND f.generated_at IS NULL
          {period_filter}
        ORDER BY f.period_start, f.id
    """

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(query, params).fetchall()
        columns = [
            "id", "period_type", "period_start", "period_end",
            "hierarchy_level", "location_key", "division_name",
            "district_name", "upazila_name", "union_name",
            "disease_code", "disease_name"
        ]
        forecasts = [dict(zip(columns, row)) for row in rows]

    if period:
        scheduled = sorted(
            forecasts,
            key=lambda x: (x["period_start"], x["id"])
        )
        return scheduled[:limit] if limit else scheduled

    # Group all location/disease rows for the same forecast period.
    weekly_groups = {}
    monthly_groups = {}

    for row in forecasts:
        period_start = row["period_start"]
        if row["period_type"] == "week":
            weekly_groups.setdefault(period_start, []).append(row)
        elif row["period_type"] == "month":
            monthly_groups.setdefault(period_start, []).append(row)

    weekly_dates = sorted(weekly_groups)
    monthly_dates = sorted(monthly_groups)

    # Four whole weekly periods, then one whole monthly period; repeat.
    scheduled = []
    week_position = 0
    month_position = 0

    while week_position < len(weekly_dates) or month_position < len(monthly_dates):
        for _ in range(4):
            if week_position < len(weekly_dates):
                period_date = weekly_dates[week_position]
                scheduled.extend(sorted(
                    weekly_groups[period_date],
                    key=lambda x: (
                        x.get("hierarchy_level") or "",
                        x.get("location_key") or "",
                        x.get("disease_code") or "",
                        x["id"]
                    )
                ))
                week_position += 1

        if month_position < len(monthly_dates):
            period_date = monthly_dates[month_position]
            scheduled.extend(sorted(
                monthly_groups[period_date],
                key=lambda x: (
                    x.get("hierarchy_level") or "",
                    x.get("location_key") or "",
                    x.get("disease_code") or "",
                    x["id"]
                )
            ))
            month_position += 1

    return scheduled[:limit] if limit else scheduled

def display_location(row: dict) -> str:
    return (
        row.get("union_name")
        or row.get("upazila_name")
        or row.get("district_name")
        or row.get("division_name")
        or row.get("location_key")
        or "Unknown location"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--period",
        choices=["week", "month"],
        default=None,
        help="Run only weekly or monthly forecasts. Default: both."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of forecast rows to process."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected forecasts without generating recommendations."
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")

    print("=" * 90)
    print("IMCI WEEKLY + MONTHLY RECOMMENDATION GENERATION")
    print(f"Started at: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"API: {API_BASE_URL}")
    print(f"Source: {DB_SCHEMA}.{SOURCE_TABLE}")
    print(f"Destination: {DB_SCHEMA}.{TARGET_TABLE}")
    print("Date window: CURRENT_DATE through CURRENT_DATE + 6 months")
    print("Schedule: 4 weekly forecast periods, then 1 monthly period; repeat")
    print("=" * 90)

    # Confirm the API is running before selecting and processing records.
    try:
        health = requests.get(
            f"{API_BASE_URL}/health",
            timeout=15
        )
        health.raise_for_status()
        print("API health:", health.json())
    except requests.RequestException as exc:
        print(f"ERROR: API is not reachable at {API_BASE_URL}: {exc}")
        print("Start the FastAPI app first, then run this script again.")
        sys.exit(1)

    try:
        forecasts = get_pending_forecasts(args.period, args.limit)
    except Exception as exc:
        print(f"ERROR selecting forecasts from PostgreSQL: {exc}")
        sys.exit(1)

    total = len(forecasts)
    weekly = sum(1 for x in forecasts if x["period_type"] == "week")
    monthly = sum(1 for x in forecasts if x["period_type"] == "month")

    print(f"\nPending forecast rows: {total}")
    print(f"Weekly rows:            {weekly}")
    print(f"Monthly rows:           {monthly}")

    if not forecasts:
        print(
            "\nNo pending forecast rows found in the current-to-six-month "
            "date window. Check period_start, period_end and existing "
            "recommendation source_forecast_id values."
        )
        return

    if args.dry_run:
        print("\nDRY RUN: no recommendations will be generated.")
        for i, row in enumerate(forecasts, 1):
            print(
                f"[{i}/{total}] {row['period_type']} | "
                f"{row['period_start']} to {row['period_end']} | "
                f"{display_location(row)} | {row['disease_name']} | "
                f"source_id={row['id']}"
            )
        return

    session = requests.Session()
    generated = 0
    failed = 0
    overall_start = time.monotonic()
    last_request_at = None

    for index, row in enumerate(forecasts, start=1):
        # Optional pacing between calls to the API.
        if last_request_at is not None and MIN_INTERVAL > 0:
            wait = MIN_INTERVAL - (time.monotonic() - last_request_at)
            if wait > 0:
                print(f"Rate-limit pause: {wait:.1f}s")
                time.sleep(wait)

        started = time.monotonic()
        location = display_location(row)
        period = row["period_type"]
        disease = row.get("disease_name") or row.get("disease_code")

        print(
            f"\n[{index}/{total}] START | period={period} | "
            f"location={location} | disease={disease} | "
            f"dates={row['period_start']} to {row['period_end']} | "
            f"source_id={row['id']}",
            flush=True
        )

        try:
            response = session.post(
                f"{API_BASE_URL}/generate",
                json={"source_forecast_id": int(row["id"])},
                timeout=API_TIMEOUT
            )
            last_request_at = time.monotonic()
            response.raise_for_status()
            result = response.json()

            generated += 1
            elapsed = time.monotonic() - started

            print(
                f"[{index}/{total}] COMPLETED | period={period} | "
                f"location={location} | disease={disease} | "
                f"recommendation_id={result.get('recommendation_id')} | "
                f"elapsed={elapsed:.2f}s",
                flush=True
            )

        except requests.RequestException as exc:
            last_request_at = time.monotonic()
            failed += 1
            body = ""
            if getattr(exc, "response", None) is not None:
                body = exc.response.text[:2000]

            print(
                f"[{index}/{total}] FAILED | period={period} | "
                f"location={location} | disease={disease} | "
                f"source_id={row['id']} | error={exc} | response={body}",
                flush=True
            )

        except Exception as exc:
            failed += 1
            print(
                f"[{index}/{total}] FAILED | source_id={row['id']} | "
                f"error={exc}",
                flush=True
            )

    elapsed_total = time.monotonic() - overall_start
    print("\n" + "=" * 90)
    print("GENERATION FINISHED")
    print(f"Selected:  {total}")
    print(f"Generated: {generated}")
    print(f"Failed:    {failed}")
    print(f"Elapsed:   {elapsed_total:.2f} seconds ({elapsed_total / 60:.2f} minutes)")
    print("=" * 90)


if __name__ == "__main__":
    main()
