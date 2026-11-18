"""
Standalone RAG Testing Script for IMCI Recommendation System.
Tests the complete end-to-end pipeline specifically for 'Anaemia':
1. Database location & disease context retrieval.
2. PDF text extraction, chunking, and cached vector indexing.
3. Disease-specific semantic retrieval for Anaemia.
4. Meditron-27B evidence-grounded prompt execution.
5. Response parsing, citation validation, and saving to outputs/anaemia_test.json.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root and app folder to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "app"))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("imci_test")

from app import config
from app.database import (
    get_available_diseases,
    get_disease_counts_now_and_next_week,
    get_location_context,
)
from app.meditron import build_prompts, call_meditron, parse_and_validate_response
from app.profiler import PipelineProfiler
from app.rag import IMCIRAG

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_DISEASE = "Anaemia"
LOCATION_LEVEL = os.getenv("LOCATION_LEVEL", "upazila")
LOCATION_VALUE = os.getenv("LOCATION_VALUE", "Hizla")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "anaemia_test.json"


def validate_environment():
    """Verify essential environment variables before running."""
    if not config.MEDITRON_API_KEY:
        raise ValueError("MEDITRON_API_KEY is missing from .env. Please configure your API key.")
    if not config.MEDITRON_URL:
        raise ValueError("MEDITRON_URL is missing from .env.")
    if not Path(config.IMCI_PDF_PATH).is_file():
        raise FileNotFoundError(f"IMCI PDF not found at: {config.IMCI_PDF_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def step1_database_context():
    """
    Query PostgreSQL strictly for:
    - Available disease names
    - Selected location and available hierarchy (division, district, upazila, union)
    STRICTLY NO forecast counts or analytics are retrieved.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATABASE CONTEXT (Location & Available Diseases)")
    print("=" * 70)

    # 1. Fetch available diseases
    available_diseases = []
    try:
        available_diseases = get_available_diseases()
        print(f"Available diseases in database ({len(available_diseases)}): {available_diseases}")
    except Exception as exc:
        print(f"[NOTE] Could not query diseases from database: {exc}")

    # 2. Fetch location hierarchy
    location_info = {}
    try:
        location_info = get_location_context(LOCATION_LEVEL, LOCATION_VALUE)
        # If Hizla/Hijla did not find rows, try alternate spelling or fallback
        if not location_info.get("division") and LOCATION_VALUE.lower() == "hizla":
            location_info = get_location_context(LOCATION_LEVEL, "Hijla")
        if not location_info.get("division"):
            # Try union Memania
            location_info = get_location_context("union", "Memania")
    except Exception as exc:
        print(f"[NOTE] Could not query location from database: {exc}")

    if not location_info:
        location_info = {
            "selected_level": LOCATION_LEVEL,
            "selected_location": LOCATION_VALUE,
            "district": "Barishal",
            "division": "Barishal",
        }

    print(f"Target Disease for RAG Test: {TEST_DISEASE}")
    print(f"Retrieved Location Context: {json.dumps(location_info, ensure_ascii=False)}")

    # 3. Fetch disease counts for now and next week
    disease_counts = None
    try:
        target_loc_val = location_info.get("upazila") or location_info.get("union") or LOCATION_VALUE
        target_loc_lvl = "upazila" if location_info.get("upazila") else (location_info.get("selected_level") or LOCATION_LEVEL)
        disease_counts = get_disease_counts_now_and_next_week(target_loc_lvl, target_loc_val, TEST_DISEASE)
        print(f"Disease Counts (Now vs Next Week): {json.dumps(disease_counts, ensure_ascii=False)}")
    except Exception as exc:
        print(f"[NOTE] Could not fetch disease counts: {exc}")

    return location_info, available_diseases, disease_counts


def step2_rag_indexing():
    """
    Initialize IMCI RAG pipeline.
    Builds or loads cached vector index from disk.
    """
    print("\n" + "=" * 70)
    print("STEP 2: RAG INDEXING & CACHE VERIFICATION")
    print("=" * 70)

    rag = IMCIRAG()
    start_time = time.time()
    rag.build_index(force_rebuild=False)
    elapsed = time.time() - start_time

    print(f"RAG Index Ready in {elapsed:.2f} seconds.")
    print(f"Total Chunks Indexed: {len(rag.chunks)}")
    print(f"Metadata: {rag.metadata}")
    return rag


def step3_disease_retrieval(rag: IMCIRAG):
    """
    Retrieve semantic passages for Anaemia from the IMCI booklet.
    """
    print("\n" + "=" * 70)
    print(f"STEP 3: DISEASE-SPECIFIC RETRIEVAL FOR '{TEST_DISEASE}'")
    print("=" * 70)

    retrieval_data = rag.get_context(TEST_DISEASE, top_k=config.RETRIEVAL_TOP_K)
    passages = retrieval_data.get("passages", [])
    pages = retrieval_data.get("pages", [])

    print(f"Retrieved {len(passages)} passages across pages: {pages}\n")
    for i, p in enumerate(passages, start=1):
        page_str = p.get("page") or f"Page {p.get('page_number')}"
        sim = p.get("similarity", 0)
        snippet = p.get("text", "")[:180].replace("\n", " ")
        print(f"  [{i}] {page_str} (similarity: {sim:.4f})")
        print(f"      \"{snippet}...\"\n")

    if not passages:
        raise ValueError(f"No passages retrieved for {TEST_DISEASE}. Retrieval failed.")

    return retrieval_data


def step4_generate_recommendation(location_info, retrieval_data, disease_counts=None):
    """
    Send grounded prompt to Meditron-27B and parse the structured output.
    """
    print("\n" + "=" * 70)
    print("STEP 4: MEDITRON-27B GENERATION")
    print("=" * 70)

    passages = retrieval_data.get("passages", [])
    retrieved_pages = retrieval_data.get("pages", [])

    system_prompt, user_prompt = build_prompts(
        disease=TEST_DISEASE,
        location=location_info,
        passages=passages,
        disease_counts=disease_counts,
    )

    print("Sending grounded request to Meditron-27B...")
    start_t = time.time()
    raw_response = call_meditron(system_prompt=system_prompt, user_prompt=user_prompt)
    elapsed = time.time() - start_t
    print(f"Meditron response received in {elapsed:.2f} seconds.")

    print("\nParsing and validating structured JSON response...")
    recommendation = parse_and_validate_response(
        content=raw_response,
        retrieved_pages=retrieved_pages,
        expected_disease=TEST_DISEASE,
        location=location_info,
        disease_counts=disease_counts,
    )

    # Attach retrieval audit trail
    recommendation["retrieved_passages"] = [
        {
            "page": p.get("page") or f"Page {p.get('page_number')}",
            "similarity": p.get("similarity"),
        }
        for p in passages
    ]

    return recommendation


def step5_save_output(recommendation):
    """
    Save the structured recommendation to outputs/anaemia_test.json.
    """
    print("\n" + "=" * 70)
    print("STEP 5: SAVING TEST RESULT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(recommendation, f, ensure_ascii=False, indent=2)

    print(f"Test output saved successfully to:\n{OUTPUT_FILE.resolve()}")
    print("\n" + "-" * 70)
    print("GENERATED JSON RESULT:")
    print("-" * 70)
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))


def main():
    print("======================================================================")
    print("       IMCI RAG RECOMMENDATION PIPELINE TEST (ANAEMIA ONLY)")
    print("======================================================================")

    validate_environment()

    profiler = PipelineProfiler("Anaemia E2E Test")

    # Step 1: Database context (location + available diseases + disease counts now & next week)
    with profiler.measure("step1_db_context"):
        location_info, _, disease_counts = step1_database_context()

    # Step 2: RAG index
    with profiler.measure("step2_rag_index"):
        rag = step2_rag_indexing()

    # Step 3: Retrieval for Anaemia
    with profiler.measure("step3_retrieval"):
        retrieval_data = step3_disease_retrieval(rag)

    # Step 4: Meditron generation (prompted with counts now and next week)
    with profiler.measure("step4_meditron_generation"):
        recommendation = step4_generate_recommendation(location_info, retrieval_data, disease_counts)

    # Step 5: Save output
    with profiler.measure("step5_save_output"):
        step5_save_output(recommendation)

    timing = profiler.finish()

    print("\n" + "=" * 70)
    print("PIPELINE TIMING SUMMARY")
    print("=" * 70)
    for stage, fmt_time in timing["formatted_stages"].items():
        print(f"  {stage:<35} {fmt_time}")
    print(f"  {'TOTAL':<35} {timing['total_duration_seconds']:.2f} s")
    print("=" * 70)

    # Attach timing to saved output
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        saved["pipeline_timing"] = timing
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        print(f"Timing data appended to: {OUTPUT_FILE.resolve()}")
    except Exception as exc:
        logger.warning("Could not append timing to output file: %s", exc)

    print("\n" + "=" * 70)
    print("SUCCESS: Anaemia RAG test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
