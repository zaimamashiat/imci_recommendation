import logging
import re
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json

from app import config

logger = logging.getLogger("imci_db")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def identifier(name: str):
    if not _IDENTIFIER.fullmatch(name or ""):
        raise ValueError(f"Unsafe SQL identifier in configuration: {name!r}")
    return sql.Identifier(name)


def connect():
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing from .env")
    return psycopg2.connect(config.DATABASE_URL)


def get_source_columns() -> List[Dict[str, Any]]:
    query = """
        SELECT ordinal_position, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (config.DB_SCHEMA, config.DB_TABLE))
        return [
            {"ordinal_position": r[0], "column_name": r[1], "data_type": r[2]}
            for r in cur.fetchall()
        ]


def ensure_recommendation_table():
    """Create the recommendation output table if it does not already exist."""
    ddl = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {}.{} (
            recommendation_id BIGSERIAL PRIMARY KEY,
            location_type VARCHAR(50),
            location_key VARCHAR(255) NOT NULL,
            division_name VARCHAR(150),
            district_name VARCHAR(150),
            upazila_name VARCHAR(150),
            union_name VARCHAR(150),
            disease_code VARCHAR(100) NOT NULL,
            disease_name VARCHAR(150),
            forecast_date DATE NOT NULL,
            forecasted_count NUMERIC,
            risk_level VARCHAR(50),
            trend VARCHAR(50),
            priority VARCHAR(50),
            data_summary TEXT,
            priority_public_health_measure TEXT,
            health_facility_readiness TEXT,
            community_communication TEXT,
            imci_references JSONB NOT NULL DEFAULT '[]'::jsonb,
            model_name VARCHAR(100) NOT NULL DEFAULT 'meditron-27b',
            generated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (location_key, disease_code, forecast_date)
        )
    """).format(identifier(config.DB_SCHEMA), identifier(config.RECOMMENDATION_TABLE))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(ddl)


# -----------------------------------------------------------------------------
# LOCATION & DISEASE CONTEXT (STRICTLY NO ANALYTICS / COUNTS)
# -----------------------------------------------------------------------------

def get_available_diseases() -> List[str]:
    """
    Retrieve unique available disease names from the database for contextual validation.
    No forecast counts or risk metrics are queried.
    """
    disease_col = config.COLUMN.get("disease_name", "disease_name")
    query = sql.SQL("""
        SELECT DISTINCT {}
        FROM {}.{}
        WHERE {} IS NOT NULL
        ORDER BY {}
    """).format(
        identifier(disease_col),
        identifier(config.DB_SCHEMA),
        identifier(config.DB_TABLE),
        identifier(disease_col),
        identifier(disease_col),
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query)
        diseases = [r[0].strip() for r in cur.fetchall() if r[0]]
        logger.info("Found %d distinct diseases in database: %s", len(diseases), diseases)
        return diseases


def get_location_context(location_level: str, location_value: str) -> Dict[str, Any]:
    """
    Retrieve location hierarchy (division, district, upazila, union) for the given location.
    Strictly queries geographical columns only. Does NOT fetch forecast counts,
    predicted cases, risk levels, or any analytics.
    """
    level_to_column = {
        "division": config.COLUMN["division"],
        "district": config.COLUMN["district"],
        "upazila": config.COLUMN["upazila"],
        "union": config.COLUMN["union"],
    }
    target_level = location_level.strip().lower()
    if target_level not in level_to_column:
        raise ValueError(f"location_level must be one of: {list(level_to_column.keys())}")

    match_col = level_to_column[target_level]

    query = sql.SQL("""
        SELECT division_name, district_name, upazila_name, union_name
        FROM {}.{}
        WHERE {} ILIKE %s
        LIMIT 1
    """).format(
        identifier(config.DB_SCHEMA),
        identifier(config.DB_TABLE),
        identifier(match_col),
    )

    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (location_value,))
        row = cur.fetchone()

        location_info = {
            "selected_level": target_level,
            "selected_location": location_value,
        }

        if row:
            div, dist, upz, un = row
            if div:
                location_info["division"] = div
            if dist:
                location_info["district"] = dist
            if upz:
                location_info["upazila"] = upz
            if un:
                location_info["union"] = un
        else:
            logger.warning("No location row found for %s=%s in %s.%s", target_level, location_value, config.DB_SCHEMA, config.DB_TABLE)

        return location_info


def get_disease_counts_now_and_next_week(
    location_level: str,
    location_value: str,
    disease_name: str,
) -> Dict[str, Any]:
    """
    Query the disease case counts for 'now' (day 1 / current week) and 'next week' (day 7 / next week window)
    for the specified location and disease from location_disease_forecasts.
    """
    level_to_column = {
        "division": config.COLUMN["division"],
        "district": config.COLUMN["district"],
        "upazila": config.COLUMN["upazila"],
        "union": config.COLUMN["union"],
    }
    target_level = location_level.strip().lower()
    if target_level not in level_to_column:
        target_level = "upazila"
    match_col = level_to_column[target_level]
    count_col = config.FORECAST_COUNT_COLUMN or "predicted_total_assessments"
    disease_col = config.COLUMN.get("disease_name", "disease_name")

    query = sql.SQL("""
        SELECT 
            SUM(CASE WHEN forecast_horizon_day = 1 THEN {} ELSE 0 END) as day_now,
            SUM(CASE WHEN forecast_horizon_day = 7 THEN {} ELSE 0 END) as day_next_week,
            SUM(CASE WHEN forecast_horizon_day BETWEEN 1 AND 7 THEN {} ELSE 0 END) as week_now_total,
            SUM(CASE WHEN forecast_horizon_day BETWEEN 8 AND 14 THEN {} ELSE 0 END) as week_next_total
        FROM {}.{}
        WHERE {} ILIKE %s AND {} ILIKE %s
    """).format(
        identifier(count_col),
        identifier(count_col),
        identifier(count_col),
        identifier(count_col),
        identifier(config.DB_SCHEMA),
        identifier(config.DB_TABLE),
        identifier(disease_col),
        identifier(match_col),
    )

    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (f"%{disease_name}%", f"%{location_value}%"))
        row = cur.fetchone()
        if not row or (row[0] is None and row[1] is None and row[2] is None):
            return {
                "count_now": 0,
                "count_next_week": 0,
                "current_week_total": 0,
                "next_week_total": 0,
                "trend": "stable",
            }

        day_now = int(row[0] or 0)
        day_next_week = int(row[1] or 0)
        week_now = int(row[2] or 0)
        week_next = int(row[3] or 0)

        if week_next > week_now or day_next_week > day_now:
            trend = "increasing"
        elif week_next < week_now or day_next_week < day_now:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "count_now": day_now,
            "count_next_week": day_next_week,
            "current_week_total": week_now,
            "next_week_total": week_next,
            "trend": trend,
        }


def fetch_forecasts(location_level: str, location_value: str, disease: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch forecast records for the specified location.
    Retained for backwards compatibility with test.py and existing endpoints.
    """
    level_to_column = {
        "division": config.COLUMN["division"],
        "district": config.COLUMN["district"],
        "upazila": config.COLUMN["upazila"],
        "union": config.COLUMN["union"],
    }
    if location_level not in level_to_column:
        raise ValueError("location_level must be division, district, upazila, or union")

    where_parts = [sql.SQL("{} ILIKE %s").format(identifier(level_to_column[location_level]))]
    params = [location_value]
    if disease:
        where_parts.append(sql.SQL("({} ILIKE %s OR {} ILIKE %s)").format(
            identifier(config.COLUMN["disease_name"]),
            identifier(config.COLUMN["disease_code"]),
        ))
        params.extend([disease, disease])

    query = sql.SQL("SELECT * FROM {}.{} WHERE {} ORDER BY {} DESC LIMIT %s").format(
        identifier(config.DB_SCHEMA),
        identifier(config.DB_TABLE),
        sql.SQL(" AND ").join(where_parts),
        identifier(config.COLUMN["forecast_date"]),
    )
    params.append(config.MAX_SOURCE_ROWS)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]


def save_recommendation(record: dict):
    columns = [
        "location_type", "location_key", "division_name", "district_name",
        "upazila_name", "union_name", "disease_code", "disease_name",
        "forecast_date", "forecasted_count", "risk_level", "trend", "priority",
        "data_summary", "priority_public_health_measure", "health_facility_readiness",
        "community_communication", "imci_references", "model_name"
    ]
    values = [record.get(c) for c in columns]
    values[17] = Json(record.get("imci_references") or [])
    insert_cols = sql.SQL(", ").join(map(identifier, columns))
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
    updates = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(identifier(c), identifier(c))
        for c in columns if c not in {"location_key", "disease_code", "forecast_date"}
    )
    query = sql.SQL("""
        INSERT INTO {}.{} ({}) VALUES ({})
        ON CONFLICT (location_key, disease_code, forecast_date)
        DO UPDATE SET {}, generated_at = CURRENT_TIMESTAMP
    """).format(
        identifier(config.DB_SCHEMA), identifier(config.RECOMMENDATION_TABLE),
        insert_cols, placeholders, updates
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, values)


def get_saved_recommendations(location_level: str, location_value: str, disease: Optional[str] = None):
    level_to_column = {
        "division": "division_name",
        "district": "district_name",
        "upazila": "upazila_name",
        "union": "union_name",
    }
    if location_level not in level_to_column:
        raise ValueError("location_level must be division, district, upazila, or union")
    where = [sql.SQL("{} ILIKE %s").format(identifier(level_to_column[location_level]))]
    params = [location_value]
    if disease:
        where.append(sql.SQL("(disease_name ILIKE %s OR disease_code ILIKE %s)"))
        params.extend([disease, disease])
    query = sql.SQL("SELECT * FROM {}.{} WHERE {} ORDER BY forecast_date DESC, generated_at DESC").format(
        identifier(config.DB_SCHEMA), identifier(config.RECOMMENDATION_TABLE),
        sql.SQL(" AND ").join(where)
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]
