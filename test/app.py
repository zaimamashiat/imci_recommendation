"""Weekly and monthly IMCI recommendation generator.

Reads daily forecast rows and writes aggregated week/month recommendations
into meditron_recommendtaion.weekly_monthly_recommendation.

Configured periods:
  week:  2026-11-26 through 2026-11-28
  month: 2026-11-26 through 2026-12-26
"""
import json
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from psycopg.rows import dict_row

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("weekly-monthly-recommendations")
app = FastAPI(title="Weekly and Monthly IMCI Recommendation Generator")

DB_SCHEMA = os.getenv("DB_SCHEMA", "meditron_recommendtaion")
DB_TABLE = os.getenv("DB_TABLE", "weekly_monthly_recommendation")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", DB_SCHEMA):
    raise ValueError("Invalid DB_SCHEMA")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", DB_TABLE):
    raise ValueError("Invalid DB_TABLE")

WEEK_START, WEEK_END = date(2026, 11, 26), date(2026, 11, 28)
MONTH_START, MONTH_END = date(2026, 11, 26), date(2026, 12, 26)
PERIODS = (("week", WEEK_START, WEEK_END), ("month", MONTH_START, MONTH_END))
MEDITRON_LOCK = threading.Lock()


class GenerateRequest(BaseModel):
    # Leave both unset to process every location in the forecast table.
    location_type: str | None = None
    location_key: str | None = None
    persist: bool = True


def connect():
    return psycopg.connect(os.environ["DATABASE_URL"])


def ensure_schema():
    """Ensure output fields and period-aware uniqueness exist."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"""
            ALTER TABLE {DB_SCHEMA}.{DB_TABLE}
              ADD COLUMN IF NOT EXISTS priority_public_health_measure TEXT,
              ADD COLUMN IF NOT EXISTS health_facility_readiness TEXT,
              ADD COLUMN IF NOT EXISTS community_communication TEXT,
              ADD COLUMN IF NOT EXISTS imci_references JSONB,
              ADD COLUMN IF NOT EXISTS model_name VARCHAR,
              ADD COLUMN IF NOT EXISTS generated_at TIMESTAMPTZ,
              ADD COLUMN IF NOT EXISTS period_type TEXT NOT NULL DEFAULT 'daily',
              ADD COLUMN IF NOT EXISTS period_start DATE,
              ADD COLUMN IF NOT EXISTS period_end DATE,
              ADD COLUMN IF NOT EXISTS recommendation_text JSONB
        """)
        cur.execute(f"""
            UPDATE {DB_SCHEMA}.{DB_TABLE}
            SET period_type = 'daily'
            WHERE period_type IS NULL OR period_type = ''
        """)
        # Remove the old three-column UNIQUE constraint, if it exists.
        cur.execute("""
            SELECT c.conname
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = %s AND t.relname = %s
              AND c.contype = 'u'
              AND pg_get_constraintdef(c.oid) =
                  'UNIQUE (location_key, disease_code, forecast_date)'
        """, (DB_SCHEMA, DB_TABLE))
        for (constraint_name,) in cur.fetchall():
            # Constraint names come from PostgreSQL catalog, quote safely.
            cur.execute(
                psycopg.sql.SQL("ALTER TABLE {}.{} DROP CONSTRAINT {}").format(
                    psycopg.sql.Identifier(DB_SCHEMA),
                    psycopg.sql.Identifier(DB_TABLE),
                    psycopg.sql.Identifier(constraint_name),
                )
            )
        cur.execute(psycopg.sql.SQL("""
            CREATE UNIQUE INDEX IF NOT EXISTS weekly_monthly_recommendation_period_uq
            ON {}.{} (location_key, disease_code, forecast_date, period_type)
        """).format(
            psycopg.sql.Identifier(DB_SCHEMA),
            psycopg.sql.Identifier(DB_TABLE),
        ))
    log.info("DATABASE | table and period-aware index are ready")


def load_pdf_context(query_text: str) -> list[dict]:
    """Retrieve a few relevant text pages from the configured IMCI PDF."""
    pdf_path = Path(os.environ["IMCI_PDF_PATH"])
    if not pdf_path.is_file():
        raise FileNotFoundError(f"IMCI PDF not found: {pdf_path}")
    terms = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", query_text)}
    ranked = []
    reader = PdfReader(str(pdf_path))
    for page_no, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if not text:
            continue
        words = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))
        score = len(terms & words)
        if score:
            ranked.append((score, page_no, text[:1800]))
    ranked.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return [{"pdf_page": page, "text": text} for _, page, text in ranked[:3]]


def fetch_period_rows(period_start: date, period_end: date,
                      location_type: str | None = None,
                      location_key: str | None = None) -> list[dict]:
    """Aggregate daily inputs only; generated week/month rows are excluded."""
    query = f"""
        SELECT
            location_type,
            location_key,
            MAX(division_name) AS division_name,
            MAX(district_name) AS district_name,
            MAX(upazila_name) AS upazila_name,
            MAX(union_name) AS union_name,
            MAX(division_id) AS division_id,
            MAX(district_id) AS district_id,
            MAX(upazila_id) AS upazila_id,
            MAX(union_id) AS union_id,
            disease_code,
            MAX(disease_name) AS disease_name,
            SUM(COALESCE(forecasted_count, 0)) AS forecasted_count,
            SUM(COALESCE(severe_forecast, 0)) AS severe_forecast,
            SUM(COALESCE(at_risk_forecast, 0)) AS at_risk_forecast,
            SUM(COALESCE(total_forecast, forecasted_count, 0)) AS total_forecast,
            CASE
              WHEN SUM(COALESCE(forecasted_count, 0)) > 0 THEN
                SUM(COALESCE(severity_score, 0) * COALESCE(forecasted_count, 0))
                / SUM(COALESCE(forecasted_count, 0))
              ELSE AVG(severity_score)
            END AS severity_score,
            (ARRAY_AGG(risk_level ORDER BY
                CASE LOWER(COALESCE(risk_level, ''))
                  WHEN 'critical' THEN 5 WHEN 'severe' THEN 5
                  WHEN 'high' THEN 4 WHEN 'moderate' THEN 3
                  WHEN 'low' THEN 2 ELSE 1 END DESC
             ) FILTER (WHERE risk_level IS NOT NULL))[1] AS risk_level,
            (ARRAY_AGG(risk_trend_label ORDER BY forecast_date DESC)
                FILTER (WHERE risk_trend_label IS NOT NULL))[1] AS risk_trend_label,
            (ARRAY_AGG(trend ORDER BY forecast_date DESC)
                FILTER (WHERE trend IS NOT NULL))[1] AS trend,
            MAX(priority_level) AS priority_level,
            COUNT(DISTINCT forecast_date) AS days_aggregated
        FROM {DB_SCHEMA}.{DB_TABLE}
        WHERE period_type = 'daily'
          AND forecast_date BETWEEN %s AND %s
          AND (%s::text IS NULL OR location_type = %s::text)
          AND (%s::text IS NULL OR location_key = %s::text)
        GROUP BY location_type, location_key, disease_code
        ORDER BY location_type, location_key,
                 SUM(COALESCE(severe_forecast, 0)) DESC,
                 SUM(COALESCE(at_risk_forecast, 0)) DESC,
                 SUM(COALESCE(forecasted_count, 0)) DESC,
                 disease_code
    """
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (
            period_start, period_end,
            location_type, location_type,
            location_key, location_key,
        ))
        return cur.fetchall()


def parse_model_json(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        if start < 0:
            raise ValueError("Meditron response did not contain a JSON object")
        result, _ = json.JSONDecoder().raw_decode(content[start:])
    if not isinstance(result, dict):
        raise ValueError("Meditron response must be a JSON object")
    return result


def call_meditron(period_type: str, start: date, end: date,
                  row: dict) -> tuple[dict, list[dict]]:
    query_text = (
        f"{row['disease_name']} {row['disease_code']} IMCI classification "
        "danger signs referral treatment prevention caregiver communication"
    )
    passages = load_pdf_context(query_text)
    evidence = passages or [{
        "pdf_page": None,
        "text": "No matching extractable IMCI passage was found. Do not invent clinical instructions.",
    }]

    forecast = {
        "period_type": period_type,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "location_type": row["location_type"],
        "location_key": row["location_key"],
        "division_name": row.get("division_name"),
        "district_name": row.get("district_name"),
        "upazila_name": row.get("upazila_name"),
        "union_name": row.get("union_name"),
        "disease_code": row["disease_code"],
        "disease_name": row["disease_name"],
        "aggregated_forecasted_count": int(row["forecasted_count"] or 0),
        "aggregated_total_forecast": int(row["total_forecast"] or 0),
        "aggregated_severe_forecast": int(row["severe_forecast"] or 0),
        "aggregated_at_risk_forecast": int(row["at_risk_forecast"] or 0),
        "severity_score": float(row["severity_score"] or 0),
        "risk_level": row.get("risk_level"),
        "risk_trend_label": row.get("risk_trend_label"),
        "days_aggregated": int(row["days_aggregated"] or 0),
        "imci_evidence": evidence,
    }

    system_prompt = """You are an expert public-health planning assistant supporting IMCI-based disease preparedness.

The supplied forecast is an AGGREGATED prediction for the complete stated WEEK or MONTH, not confirmed cases and not a daily forecast. Generate three distinct, specific, practical recommendations for that whole period.

Return ONLY valid JSON with exactly these three string keys:
{
  "priority_public_health_measure": "...",
  "health_facility_readiness": "...",
  "community_communication": "..."
}

REQUIREMENTS:
1. priority_public_health_measure: concrete public-health actions proportionate to the aggregated forecast, risk and disease. Consider surveillance, prevention, preparedness, resource prioritisation and escalation where justified.
2. health_facility_readiness: practical facility actions such as staffing, essential supplies, equipment, triage, referral readiness and service capacity, as appropriate to the disease and evidence.
3. community_communication: actionable community/caregiver communication, prevention, early warning signs and timely care-seeking messages, as appropriate.
4. Make each recommendation distinct, concise and actionable. Do not merely repeat counts or provide generic filler.
5. Use the location, disease, period, aggregated counts and risk information supplied. Do not make daily recommendations.
6. Use ONLY supplied IMCI excerpts for clinical claims. Never invent treatment, medicine doses, treatment durations, classifications or referral criteria.
7. When a recommendation uses a supplied evidence passage, cite its actual page as (PDF page N). Never invent a page number. If no supplied passage supports a clinical claim, avoid that claim and state that relevant IMCI evidence was insufficient where necessary.
8. Do not add markdown fences, explanations, or extra JSON keys."""

    body = {
        "model": os.getenv("MEDITRON_MODEL", "meditron-14b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(forecast, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "max_tokens": int(os.getenv("MEDITRON_MAX_TOKENS", "1200")),
    }
    headers = {
        "Authorization": f"{os.getenv('MEDITRON_AUTH_SCHEME', 'Bearer')} {os.environ['MEDITRON_API_KEY']}",
        "Content-Type": "application/json",
    }
    verify_setting = os.getenv("MEDITRON_VERIFY_SSL", "true").lower()
    verify = verify_setting == "true"
    ca_bundle = os.getenv("MEDITRON_CA_BUNDLE", "").strip()
    if ca_bundle:
        verify = ca_bundle

    with MEDITRON_LOCK:
        response = requests.post(
            os.environ["MEDITRON_URL"],
            json=body,
            headers=headers,
            timeout=float(os.getenv("MEDITRON_TIMEOUT", "300")),
            verify=verify,
        )
    response.raise_for_status()
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected Meditron API response: {response.text[:1000]}") from exc

    recommendation = parse_model_json(content)
    required = (
        "priority_public_health_measure",
        "health_facility_readiness",
        "community_communication",
    )
    for field in required:
        value = recommendation.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Meditron returned missing/empty field: {field}")
        recommendation[field] = value.strip()

    # Keep only the three expected recommendation fields in the saved JSON.
    recommendation = {field: recommendation[field] for field in required}
    return recommendation, passages


def save_recommendation(period_type: str, start: date, end: date,
                        row: dict, recommendation: dict,
                        passages: list[dict], persist: bool):
    if not persist:
        return

    references = [
        {"pdf_page": p.get("pdf_page"), "excerpt": p["text"][:300]}
        for p in passages
    ]
    generated_at = datetime.now(timezone.utc)
    model_name = os.getenv("MEDITRON_MODEL", "meditron-14b")
    # forecast_date is the period end; period_type/start/end identify the aggregate row.
    values = {
        "location_type": row["location_type"],
        "location_key": row["location_key"],
        "division_name": row.get("division_name"),
        "district_name": row.get("district_name"),
        "upazila_name": row.get("upazila_name"),
        "union_name": row.get("union_name"),
        "disease_code": row["disease_code"],
        "disease_name": row["disease_name"],
        "forecast_date": end,
        "forecasted_count": row["forecasted_count"],
        "risk_level": row.get("risk_level"),
        "trend": row.get("trend") or row.get("risk_trend_label"),
        "priority": row.get("priority_level"),
        "imci_references": json.dumps(references, ensure_ascii=False),
        "model_name": model_name,
        "generated_at": generated_at,
        "priority_level": row.get("priority_level"),
        "hierarchy_level": row["location_type"],
        "division_id": row.get("division_id"),
        "district_id": row.get("district_id"),
        "upazila_id": row.get("upazila_id"),
        "union_id": row.get("union_id"),
        "total_forecast": row.get("total_forecast"),
        "severe_forecast": row.get("severe_forecast"),
        "at_risk_forecast": row.get("at_risk_forecast"),
        "severity_score": row.get("severity_score"),
        "risk_trend_label": row.get("risk_trend_label"),
        "period_type": period_type,
        "period_start": start,
        "period_end": end,
        "priority_public_health_measure": recommendation["priority_public_health_measure"],
        "health_facility_readiness": recommendation["health_facility_readiness"],
        "community_communication": recommendation["community_communication"],
        "recommendation_text": json.dumps(recommendation, ensure_ascii=False),
    }

    stmt = f"""
        INSERT INTO {DB_SCHEMA}.{DB_TABLE} (
            location_type, location_key, division_name, district_name,
            upazila_name, union_name, disease_code, disease_name, forecast_date,
            forecasted_count, risk_level, trend, priority, imci_references,
            model_name, generated_at, priority_level, hierarchy_level,
            division_id, district_id, upazila_id, union_id, total_forecast,
            severe_forecast, at_risk_forecast, severity_score, risk_trend_label,
            period_type, period_start, period_end,
            priority_public_health_measure, health_facility_readiness,
            community_communication, recommendation_text
        ) VALUES (
            %(location_type)s, %(location_key)s, %(division_name)s,
            %(district_name)s, %(upazila_name)s, %(union_name)s,
            %(disease_code)s, %(disease_name)s, %(forecast_date)s,
            %(forecasted_count)s, %(risk_level)s, %(trend)s, %(priority)s,
            %(imci_references)s::jsonb, %(model_name)s, %(generated_at)s,
            %(priority_level)s, %(hierarchy_level)s, %(division_id)s,
            %(district_id)s, %(upazila_id)s, %(union_id)s, %(total_forecast)s,
            %(severe_forecast)s, %(at_risk_forecast)s, %(severity_score)s,
            %(risk_trend_label)s, %(period_type)s, %(period_start)s,
            %(period_end)s, %(priority_public_health_measure)s,
            %(health_facility_readiness)s, %(community_communication)s,
            %(recommendation_text)s::jsonb
        )
        ON CONFLICT (location_key, disease_code, forecast_date, period_type)
        DO UPDATE SET
            location_type = EXCLUDED.location_type,
            division_name = EXCLUDED.division_name,
            district_name = EXCLUDED.district_name,
            upazila_name = EXCLUDED.upazila_name,
            union_name = EXCLUDED.union_name,
            disease_name = EXCLUDED.disease_name,
            forecasted_count = EXCLUDED.forecasted_count,
            risk_level = EXCLUDED.risk_level,
            trend = EXCLUDED.trend,
            priority = EXCLUDED.priority,
            imci_references = EXCLUDED.imci_references,
            model_name = EXCLUDED.model_name,
            generated_at = EXCLUDED.generated_at,
            priority_level = EXCLUDED.priority_level,
            hierarchy_level = EXCLUDED.hierarchy_level,
            division_id = EXCLUDED.division_id,
            district_id = EXCLUDED.district_id,
            upazila_id = EXCLUDED.upazila_id,
            union_id = EXCLUDED.union_id,
            total_forecast = EXCLUDED.total_forecast,
            severe_forecast = EXCLUDED.severe_forecast,
            at_risk_forecast = EXCLUDED.at_risk_forecast,
            severity_score = EXCLUDED.severity_score,
            risk_trend_label = EXCLUDED.risk_trend_label,
            period_start = EXCLUDED.period_start,
            period_end = EXCLUDED.period_end,
            priority_public_health_measure = EXCLUDED.priority_public_health_measure,
            health_facility_readiness = EXCLUDED.health_facility_readiness,
            community_communication = EXCLUDED.community_communication,
            recommendation_text = EXCLUDED.recommendation_text
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(stmt, values)


def generate(request: GenerateRequest):
    ensure_schema()
    results = []
    overall_start = time.perf_counter()

    for period_type, start, end in PERIODS:
        rows = fetch_period_rows(start, end, request.location_type, request.location_key)
        log.info(
            "========== %s | %s to %s | %d location/disease groups ==========",
            period_type.upper(), start, end, len(rows),
        )
        for index, row in enumerate(rows, start=1):
            item_start = time.perf_counter()
            location_name = (
                row.get("union_name") or row.get("upazila_name")
                or row.get("district_name") or row.get("division_name")
                or row["location_key"]
            )
            log.info(
                "[%d/%d] START | period=%s | location=%s | key=%s | disease=%s",
                index, len(rows), period_type, location_name,
                row["location_key"], row["disease_name"],
            )
            try:
                recommendation, passages = call_meditron(period_type, start, end, row)
                save_recommendation(
                    period_type, start, end, row,
                    recommendation, passages, request.persist,
                )
                elapsed = round(time.perf_counter() - item_start, 2)
                results.append({
                    "period_type": period_type,
                    "period_start": start.isoformat(),
                    "period_end": end.isoformat(),
                    "location_type": row["location_type"],
                    "location_key": row["location_key"],
                    "location_name": location_name,
                    "disease_code": row["disease_code"],
                    "disease_name": row["disease_name"],
                    "forecasted_count": int(row["forecasted_count"] or 0),
                    "severe_forecast": int(row["severe_forecast"] or 0),
                    "at_risk_forecast": int(row["at_risk_forecast"] or 0),
                    "elapsed_seconds": elapsed,
                    "status": "completed",
                    "recommendation": recommendation,
                    "saved": request.persist,
                })
                log.info(
                    "[%d/%d] COMPLETED | period=%s | location=%s | disease=%s | elapsed=%.2fs",
                    index, len(rows), period_type, location_name,
                    row["disease_name"], elapsed,
                )
            except Exception:
                elapsed = round(time.perf_counter() - item_start, 2)
                log.exception(
                    "[%d/%d] FAILED | period=%s | location=%s | disease=%s | elapsed=%.2fs",
                    index, len(rows), period_type, location_name,
                    row["disease_name"], elapsed,
                )
                raise

    total_elapsed = round(time.perf_counter() - overall_start, 2)
    log.info("ALL PERIODS COMPLETED | generated=%d | elapsed=%.2fs", len(results), total_elapsed)
    return {
        "status": "completed",
        "periods": [
            {"period_type": p, "start": s.isoformat(), "end": e.isoformat()}
            for p, s, e in PERIODS
        ],
        "generated": len(results),
        "saved": request.persist,
        "elapsed_seconds": total_elapsed,
        "results": results,
    }


@app.post("/generate/weekly-monthly")
def generate_weekly_monthly(request: GenerateRequest):
    try:
        return generate(request)
    except Exception as exc:
        log.exception("Weekly/monthly recommendation generation failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
