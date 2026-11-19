import json
import logging
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from app import config
from app.database import (
    get_available_diseases,
    get_disease_counts_now_and_next_week,
    get_location_context,
    save_recommendation,
)
from app.meditron import build_prompts, call_meditron, parse_and_validate_response
from app.profiler import PipelineProfiler
from app.rag import IMCIRAG

logger = logging.getLogger("imci_service")


def json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def generate_disease_recommendation(
    disease: str,
    location_level: str,
    location_value: str,
    rag: Optional[IMCIRAG] = None,
) -> Dict[str, Any]:
    """
    You are a public-health planning assistant for the IMCI programme in Bangladesh.
You write recommendations for upazila and union health managers, NOT treatment
instructions for individual children.

Rules:
1. Use ONLY the IMCI booklet passages provided. Cite the page for every
   clinical statement. If the passages do not support a statement, leave it out.
2. Base every recommendation on the forecast and recent-case numbers given.
   Mention the expected number of cases and the period.
3. Never give doses, drug schedules or treatment decisions for an individual
   child. Say "per IMCI protocol (page X)" instead.
4. Severe classifications always mean urgent referral. Never suggest home or
   clinic treatment for a severe classification.
5. If the recent trend is marked "unknown", say the data is incomplete; do not
   call it a rise or a fall.
6. Write short, practical actions a manager can assign to staff this week.
7. Always set "requires_expert_review": true.
8. Return ONLY valid JSON in the format below. No text outside the JSON.
 
    """
    clean_disease = disease.strip()
    clean_level = location_level.strip().lower()
    clean_val = location_value.strip()

    logger.info("Generating recommendation for %s at %s=%s", clean_disease, clean_level, clean_val)

    profiler = PipelineProfiler(f"{clean_disease} @ {clean_val}")

    # 1. Fetch Location Hierarchy from Database
    with profiler.measure("db_location_lookup"):
        try:
            location_context = get_location_context(clean_level, clean_val)
        except Exception as exc:
            logger.warning("Could not fetch location context from DB (%s); using direct inputs.", exc)
            location_context = {
                "selected_level": clean_level,
                "selected_location": clean_val,
            }

    # 2. Fetch Disease Caseload Counts for Now and Next Week
    disease_counts = None
    with profiler.measure("db_disease_counts"):
        try:
            disease_counts = get_disease_counts_now_and_next_week(clean_level, clean_val, clean_disease)
            logger.info("Retrieved disease caseload for %s: %s", clean_disease, disease_counts)
        except Exception as exc:
            logger.warning("Could not fetch disease counts from DB (%s)", exc)

    # 3. Semantic Retrieval from IMCI Booklet
    if rag is None:
        rag = IMCIRAG()
        rag.build_index()

    with profiler.measure("rag_retrieval"):
        retrieval_data = rag.get_context(clean_disease)
    passages = retrieval_data.get("passages", [])
    retrieved_pages = retrieval_data.get("pages", [])

    if not passages:
        logger.error("No relevant IMCI passages retrieved for disease: %s", clean_disease)
        return {
            "disease": clean_disease,
            "location": location_context,
            "disease_counts": disease_counts,
            "summary": f"No relevant clinical guidance found in the IMCI Chart Booklet for {clean_disease}.",
            "government_public_health_interventions": [],
            "health_system_readiness": [],
            "community_engagement": [],
            "government_planning_and_monitoring": [],
            "booklet_references": [],
            "requires_expert_review": True,
            "expert_review_reason": f"Semantic retrieval returned no relevant passages for {clean_disease} in the indexed IMCI booklet.",
            "pipeline_timing": profiler.finish(),
        }

    logger.info(
        "Retrieved %d relevant passages across pages: %s",
        len(passages),
        [p.get("page") for p in passages],
    )

    # 4. Build Grounded Prompts (with caseload counts now & next week)
    with profiler.measure("prompt_build"):
        system_prompt, user_prompt = build_prompts(
            disease=clean_disease,
            location=location_context,
            passages=passages,
            disease_counts=disease_counts,
        )

    logger.info(
        "Prompt sizes: system=%d, user=%d, total=%d chars (~%d tokens). "
        "Context: %d passages, %d chars.",
        len(system_prompt), len(user_prompt),
        len(system_prompt) + len(user_prompt),
        (len(system_prompt) + len(user_prompt) + 3) // 4,
        len(passages),
        retrieval_data.get("total_context_chars", 0),
    )

    try:
        # 5. Invoke Meditron-27B
        with profiler.measure("meditron_api"):
            raw_content = call_meditron(system_prompt=system_prompt, user_prompt=user_prompt)

        # 6. Parse and Validate Response
        with profiler.measure("response_parsing"):
            recommendation = parse_and_validate_response(
                content=raw_content,
                retrieved_pages=retrieved_pages,
                expected_disease=clean_disease,
                location=location_context,
                disease_counts=disease_counts,
            )
    except Exception:
        profiler.finish()
        logger.exception("Recommendation generation failed for %s at %s", clean_disease, clean_val)
        raise

    # Attach retrieved passages metadata for auditing and inspection
    recommendation["retrieved_passages"] = [
        {
            "page": p.get("page") or f"Page {p.get('page_number')}",
            "similarity": p.get("similarity"),
        }
        for p in passages
    ]

    # Attach pipeline timing to the recommendation output
    recommendation["pipeline_timing"] = profiler.finish()

    return recommendation


def generate_and_save(
    location_level: str,
    location_value: str,
    disease: Optional[str] = None,
    rag: Optional[IMCIRAG] = None,
) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper for the FastAPI `/recommendations/generate` endpoint.
    Generates RAG recommendations and optionally persists them to PostgreSQL.
    """
    target_disease = disease or "Anaemia"
    rec = generate_disease_recommendation(
        disease=target_disease,
        location_level=location_level,
        location_value=location_value,
        rag=rag,
    )

    # Attempt saving to database if table is available
    save_started = time.perf_counter()
    try:
        loc = rec.get("location", {})
        public_health_text = "; ".join(rec.get("government_public_health_interventions", []))
        facility_text = "; ".join(rec.get("health_system_readiness", []))
        comm_text = "; ".join(rec.get("community_engagement", []))

        db_record = {
            "location_type": loc.get("selected_level", location_level),
            "location_key": f"{loc.get('division', '')}_{loc.get('district', '')}_{loc.get('upazila', '')}_{loc.get('union', '')}".strip("_") or location_value,
            "division_name": loc.get("division"),
            "district_name": loc.get("district"),
            "upazila_name": loc.get("upazila"),
            "union_name": loc.get("union"),
            "disease_code": target_disease.upper().replace(" ", "_"),
            "disease_name": target_disease,
            "forecast_date": date.today(),
            "forecasted_count": None,
            "risk_level": "Standard IMCI Guidance",
            "trend": "N/A",
            "priority": "High" if rec.get("requires_expert_review") else "Normal",
            "data_summary": rec.get("summary", ""),
            "priority_public_health_measure": public_health_text,
            "health_facility_readiness": facility_text,
            "community_communication": comm_text,
            "imci_references": rec.get("booklet_references", []),
            "model_name": config.MEDITRON_MODEL,
        }
        save_recommendation(db_record)
        rec["saved_to_database"] = True
    except Exception as exc:
        logger.warning("Could not persist recommendation to PostgreSQL: %s", exc)
        rec["saved_to_database"] = False
    finally:
        save_elapsed = time.perf_counter() - save_started
        timing = rec.get("pipeline_timing")
        if timing:
            timing["stage_durations_seconds"]["db_save"] = round(save_elapsed, 4)
            timing["formatted_stages"]["db_save"] = (
                f"{save_elapsed * 1000:.2f} ms"
                if save_elapsed < 1.0
                else f"{save_elapsed:.2f} s"
            )
            timing["total_duration_seconds"] = round(
                timing["total_duration_seconds"] + save_elapsed,
                4,
            )
        logger.info("Stage 'db_save' completed in %.4f s", save_elapsed)

    return rec
