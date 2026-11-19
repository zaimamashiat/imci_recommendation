from contextlib import asynccontextmanager
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.database import (
    ensure_recommendation_table,
    get_available_diseases,
    get_saved_recommendations,
    get_source_columns,
)
from app.meditron import MeditronAPIError
from app.rag import IMCIRAG
from app.service import generate_and_save, generate_disease_recommendation

logger = logging.getLogger("imci_api")
rag: Optional[IMCIRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    try:
        ensure_recommendation_table()
    except Exception as exc:
        print(f"[WARNING] Database initialization skipped: {exc}")

    # Build or load cached RAG index
    rag = IMCIRAG()
    rag.build_index()
    yield


app = FastAPI(
    title="IMCI Evidence-Grounded Recommendation API",
    version="2.0.0",
    description="RAG-based public-health and clinical recommendation engine using the IMCI Chart Booklet and Meditron-27B.",
    lifespan=lifespan,
)


def _meditron_http_exception(exc: MeditronAPIError) -> HTTPException:
    unavailable_categories = {
        "configuration_error",
        "connection_error",
        "rate_limit",
        "server_error",
        "timeout",
        "tls_error",
    }
    status_code = 503 if exc.category in unavailable_categories else 502
    error_code = "meditron_unavailable" if status_code == 503 else "meditron_upstream_error"
    logger.error(
        "Meditron request failed (category=%s, provider_status=%s): %s",
        exc.category,
        exc.status_code,
        exc,
    )
    detail: Dict[str, Any] = {
        "error": error_code,
        "category": exc.category,
        "message": (
            "The recommendation model is temporarily unavailable."
            if status_code == 503
            else "The recommendation model returned an invalid or rejected response."
        ),
    }
    if exc.status_code is not None:
        detail["provider_status"] = exc.status_code
    return HTTPException(status_code=status_code, detail=detail)


class GenerateRequest(BaseModel):
    location_level: str = Field(..., examples=["upazila"])
    location_value: str = Field(..., examples=["Hijla"])
    disease: Optional[str] = Field("Anaemia", examples=["Anaemia"])


class ReadRequest(BaseModel):
    location_level: str
    location_value: str
    disease: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_index_loaded": rag is not None and rag.vectors is not None,
        "rag_metadata": rag.metadata if rag else {},
    }


@app.get("/schema")
def schema():
    try:
        return {
            "database": "configured DATABASE_URL",
            "schema": config.DB_SCHEMA,
            "table": config.DB_TABLE,
            "columns": get_source_columns(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/diseases")
def diseases():
    try:
        return {"diseases": get_available_diseases()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/recommendations/generate")
def generate(request: GenerateRequest):
    try:
        target_disease = request.disease or "Anaemia"
        return generate_and_save(
            location_level=request.location_level,
            location_value=request.location_value,
            disease=target_disease,
            rag=rag,
        )
    except MeditronAPIError as exc:
        raise _meditron_http_exception(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/recommendations/rag")
def generate_rag_direct(request: GenerateRequest):
    """
    Direct RAG endpoint returning evidence-grounded clinical recommendations
    without requiring forecast counts or database analytics.
    """
    try:
        target_disease = request.disease or "Anaemia"
        return generate_disease_recommendation(
            disease=target_disease,
            location_level=request.location_level,
            location_value=request.location_value,
            rag=rag,
        )
    except MeditronAPIError as exc:
        raise _meditron_http_exception(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/recommendations/read")
def read_saved(request: ReadRequest):
    try:
        records = get_saved_recommendations(
            request.location_level,
            request.location_value,
            request.disease,
        )
        return {"count": len(records), "recommendations": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
