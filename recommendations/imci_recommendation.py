import os, re, json, time, logging
from pathlib import Path
from datetime import date, datetime
from decimal import Decimal
import math
from typing import Optional, Any

import fitz
import psycopg
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

SCHEMA = os.getenv("DB_SCHEMA", "meditron_recommendtaion")
SOURCE = os.getenv("SOURCE_TABLE", "weekly_monthly_forecasts")
TARGET = SOURCE  # Recommendations are written back to the forecast rows.
DB_URL = os.getenv("DATABASE_URL")
PDF_PATH = Path(os.getenv("IMCI_PDF_PATH", str(BASE / "IMCI_CHART_BOOKLET.pdf")))

API_URL = os.getenv("MEDITRON_URL", "https://meditron-api.scibd.info/v1/chat/completions")
API_KEY = os.getenv("MEDITRON_API_KEY")
MODEL = os.getenv("MEDITRON_MODEL", "meditron-27b")
AUTH_SCHEME = os.getenv("MEDITRON_AUTH_SCHEME", "Bearer")
VERIFY_SSL = os.getenv("MEDITRON_VERIFY_SSL", "true").lower() == "true"
TIMEOUT = int(os.getenv("MEDITRON_TIMEOUT", "300"))
MAX_TOKENS = int(os.getenv("MEDITRON_MAX_TOKENS", "1200"))
MAX_RETRIES = int(os.getenv("MEDITRON_MAX_RETRIES", "2"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "10"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("imci")
app = FastAPI(title="IMCI Forecast Recommendation API", version="1.0.0")

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
def qtable(schema, table):
    if not IDENT.fullmatch(schema) or not IDENT.fullmatch(table):
        raise ValueError("Invalid SQL identifier in schema/table configuration")
    return f'"{schema}"."{table}"'

SRC = qtable(SCHEMA, SOURCE)
DST = SRC  # Same physical table: forecast rows are updated in place.

embedder = None
reranker = None
pdf_chunks = []
pdf_vectors = None
faiss_global_index = None
disease_faiss_indexes = {}
disease_aliases = {
    "anaemia": ["anaemia", "anemia", "palmar pallor", "severe pallor", "some pallor"],
    "pneumonia": ["pneumonia", "cough", "difficult breathing", "fast breathing", "chest indrawing"],
    "diarrhea_dehydration": ["diarrhoea", "diarrhea", "dehydration", "loose stool", "persistent diarrhoea"],
    "diarrhea": ["diarrhoea", "diarrhea", "dehydration", "loose stool", "persistent diarrhoea"],
    "fever_malaria": ["malaria", "fever", "very severe febrile disease", "antimalarial"],
    "malaria": ["malaria", "fever", "antimalarial"],
    "measles": ["measles", "mouth ulcers", "corneal", "clouding of cornea"],
    "malnutrition": ["malnutrition", "wasting", "oedema", "edema", "mid-upper arm circumference", "muac"]
}

def db():
    if not DB_URL:
        raise RuntimeError("DATABASE_URL is missing from .env")
    return psycopg.connect(DB_URL)

def safe(v):
    """Convert PostgreSQL/numpy values into JSON-safe Python values."""
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, Decimal):
        # Preserve whole-number values as integers; otherwise use float.
        return int(v) if v == v.to_integral_value() else float(v)
    # Handle numpy scalar values without requiring every caller to know
    # their concrete numpy type.
    if hasattr(v, "item") and callable(v.item):
        try:
            v = v.item()
        except (ValueError, TypeError):
            pass
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): safe(value) for k, value in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(value) for value in v]
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)

def ensure_target():
    """
    Add recommendation output columns to the existing forecast table.
    Existing forecast rows and forecast metrics are preserved.
    """
    sql = f"""
    ALTER TABLE {SRC}
        ADD COLUMN IF NOT EXISTS recommendation_id BIGINT,
        ADD COLUMN IF NOT EXISTS priority_public_health_measure TEXT,
        ADD COLUMN IF NOT EXISTS health_facility_readiness TEXT,
        ADD COLUMN IF NOT EXISTS community_communication TEXT,
        ADD COLUMN IF NOT EXISTS imci_references JSONB,
        ADD COLUMN IF NOT EXISTS model_name TEXT,
        ADD COLUMN IF NOT EXISTS generated_at TIMESTAMPTZ;

    -- Use the existing forecast row ID as its recommendation ID.
    -- Only generated rows receive this value.
    UPDATE {SRC}
       SET recommendation_id = id
     WHERE generated_at IS NOT NULL
       AND recommendation_id IS NULL;
    """
    with db() as conn:
        conn.execute(sql)
        conn.commit()
    log.info(
        "Recommendation columns verified on source table %s; no separate table is used.",
        SRC
    )

def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def disease_tags_for_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    tags = []
    for disease, aliases in disease_aliases.items():
        if any(alias in normalized for alias in aliases):
            tags.append(disease)
    # Avoid duplicate conceptual tags where both canonical codes exist.
    return sorted(set(tags))


def split_into_booklet_sections(page_text: str) -> list[tuple[str, str]]:
    """
    Split a PDF page using visible booklet headings and paragraph boundaries.
    Heading lines are retained as section context. Page numbers are stored
    separately as the original one-based PDF page number.
    """
    lines = [line.strip() for line in page_text.splitlines()]
    sections = []
    current_heading = "Page content"
    current_lines = []

    heading_pattern = re.compile(
        r"^(?:"
        r"(?:[0-9]+(?:\.[0-9]+)*\s+)?"
        r"(?:CHECK|ASSESS|LOOK|CLASSIFY|TREAT|GIVE|ADVISE|COUNSEL|FOLLOW|"
        r"REFER|URGENT|RETURN|FEED|EVALUATE|DETERMINE|IDENTIFY|"
        r"IMCI|ANNEX|CHART|TABLE)\b.*"
        r"|[A-Z][A-Z0-9 /(),:;'-]{5,}$"
        r")$"
    )

    def flush():
        nonlocal current_lines, current_heading
        body = "\n".join(x for x in current_lines if x).strip()
        if body:
            sections.append((current_heading, body))
        current_lines = []

    for line in lines:
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        is_heading = bool(heading_pattern.match(line)) and len(line) <= 150
        # A short all-caps line is likely a chart/section heading.
        if is_heading and current_lines:
            flush()
            current_heading = line
            current_lines.append(line)
        elif is_heading and not current_lines:
            current_heading = line
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush()

    if not sections:
        return [("Page content", page_text.strip())]
    return sections


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split long section text at paragraph/sentence/word boundaries."""
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = paragraph
            continue

        if current:
            chunks.append(current)
            current = ""

        # Long paragraph: split on sentence boundaries, then words if needed.
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            if len(sentence) > max_chars:
                words = sentence.split()
                piece = ""
                for word in words:
                    candidate = f"{piece} {word}".strip()
                    if len(candidate) > max_chars and piece:
                        chunks.append(piece)
                        piece = piece[-overlap:] if overlap else ""
                        piece = f"{piece} {word}".strip()
                    else:
                        piece = candidate
                if piece:
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(piece)
                continue

            candidate = f"{current} {sentence}".strip()
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = sentence
            else:
                current = candidate

    if current:
        chunks.append(current)

    # Remove accidental duplicates/empty fragments.
    return list(dict.fromkeys(x.strip() for x in chunks if x.strip()))


def make_faiss_index(chunk_ids: list[int], vectors: np.ndarray):
    if not chunk_ids:
        return None
    matrix = np.asarray(vectors[chunk_ids], dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def load_pdf():
    global embedder, reranker, pdf_chunks, pdf_vectors
    global faiss_global_index, disease_faiss_indexes

    if faiss_global_index is not None:
        return

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"IMCI booklet not found: {PDF_PATH}")

    log.info("Loading embedding model: %s", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

    log.info("Loading reranker model: %s", RERANKER_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)

    chunks = []
    with fitz.open(PDF_PATH) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue

            sections = split_into_booklet_sections(page_text)
            for heading, section_text in sections:
                section_chunks = split_long_text(
                    section_text,
                    max_chars=CHUNK_MAX_CHARS,
                    overlap=CHUNK_OVERLAP_CHARS
                )

                for chunk_text in section_chunks:
                    combined = f"{heading}\n{chunk_text}".strip()
                    tags = disease_tags_for_text(f"{heading}\n{chunk_text}")
                    chunks.append({
                        "pdf_page": page_number,
                        "section_heading": heading,
                        "excerpt": combined,
                        "disease_tags": tags,
                        "is_general": len(tags) == 0
                    })

    if not chunks:
        raise RuntimeError(
            "No selectable text was extracted from the IMCI PDF. "
            "If it is scanned, OCR is required before indexing."
        )

    pdf_chunks = chunks
    texts = [item["excerpt"] for item in pdf_chunks]

    log.info("Embedding %s booklet chunks", len(texts))
    embeddings = embedder.encode(
        texts,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    pdf_vectors = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(pdf_vectors)

    all_ids = list(range(len(pdf_chunks)))
    faiss_global_index = make_faiss_index(all_ids, pdf_vectors)

    # Disease-specific indexes include disease-matched chunks plus general
    # IMCI guidance chunks. This keeps disease sections in their own retrieval
    # pool while allowing general assessment/referral guidance to be retrieved.
    disease_faiss_indexes = {}
    canonical_diseases = sorted(set(disease_aliases.keys()))

    for disease in canonical_diseases:
        disease_ids = [
            i for i, chunk in enumerate(pdf_chunks)
            if disease in chunk["disease_tags"] or chunk["is_general"]
        ]
        if disease_ids:
            disease_faiss_indexes[disease] = {
                "index": make_faiss_index(disease_ids, pdf_vectors),
                "chunk_ids": disease_ids,
                "disease_specific_count": sum(
                    1 for i in disease_ids
                    if disease in pdf_chunks[i]["disease_tags"]
                )
            }

    log.info(
        "IMCI FAISS ready: chunks=%s, disease_indexes=%s, dimension=%s",
        len(pdf_chunks), len(disease_faiss_indexes), pdf_vectors.shape[1]
    )
    for disease, info in disease_faiss_indexes.items():
        log.info(
            "Disease index | %s | disease_chunks=%s | indexed_chunks=%s",
            disease, info["disease_specific_count"], len(info["chunk_ids"])
        )

def forecast_context(f):
    keys = [
        "period_type", "period_start", "period_end", "hierarchy_level",
        "location_key", "division_name", "district_name", "upazila_name",
        "union_name", "disease_code", "disease_name",
        "total_forecast_records", "forecast_days",
        "predicted_infant_total", "predicted_infant_severe_count",
        "predicted_infant_warning_count", "predicted_infant_moderate_count",
        "predicted_infant_safe_count", "predicted_infant_at_risk_count",
        "predicted_infant_at_risk_density",
        "predicted_child_total", "predicted_child_severe_count",
        "predicted_child_warning_count", "predicted_child_moderate_count",
        "predicted_child_safe_count", "predicted_child_at_risk_count",
        "predicted_child_at_risk_density",
        "predicted_total_assessments", "predicted_severe_count",
        "predicted_warning_count", "predicted_moderate_count",
        "predicted_safe_count", "predicted_at_risk_count",
        "predicted_severe_density", "predicted_warning_density",
        "predicted_moderate_density", "predicted_safe_density",
        "predicted_at_risk_density", "predicted_severity_score",
        "predicted_severity_label", "risk_trend_score", "risk_trend_label"
    ]
    return {k: safe(f.get(k)) for k in keys if k in f}

def canonical_disease_code(forecast: dict[str, Any]) -> str:
    raw = str(forecast.get("disease_code") or "").strip().lower()
    aliases = {
        "anaemia": "anaemia",
        "anemia": "anaemia",
        "pneumonia": "pneumonia",
        "diarrhea_dehydration": "diarrhea_dehydration",
        "diarrhea": "diarrhea",
        "diarrhoea": "diarrhea",
        "fever_malaria": "fever_malaria",
        "malaria": "malaria",
        "measles": "measles",
        "malnutrition": "malnutrition"
    }
    return aliases.get(raw, raw)


def retrieve_references(f):
    """Disease-aware FAISS retrieval, followed by cross-encoder reranking."""
    load_pdf()
    c = forecast_context(f)
    disease = canonical_disease_code(f)

    query = (
        f"IMCI booklet guidance specifically for {c.get('disease_name')} "
        f"({c.get('disease_code')}). Find the disease-specific assessment, "
        "danger signs, classification, treatment, referral, follow-up, "
        "facility readiness and caregiver communication guidance. "
        f"Forecast period: {c.get('period_type')} "
        f"{c.get('period_start')} to {c.get('period_end')}. "
        f"Forecast severity: {c.get('predicted_severity_label')}. "
        f"Risk trend: {c.get('risk_trend_label')}."
    )

    # Prefer the matching disease index. If no disease-specific chunk was
    # identified from the booklet, use the global FAISS index transparently.
    disease_index_info = disease_faiss_indexes.get(disease)
    if (
        disease_index_info
        and disease_index_info["disease_specific_count"] > 0
    ):
        search_index = disease_index_info["index"]
        search_chunk_ids = disease_index_info["chunk_ids"]
        log.info(
            "Using disease-specific FAISS index: disease=%s specific_chunks=%s",
            disease, disease_index_info["disease_specific_count"]
        )
    else:
        search_index = faiss_global_index
        search_chunk_ids = list(range(len(pdf_chunks)))
        log.warning(
            "No disease-specific booklet section detected for '%s'; "
            "using global FAISS index.",
            disease
        )

    query_vector = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    faiss.normalize_L2(query_vector)

    candidate_count = min(CANDIDATE_K, len(search_chunk_ids))
    scores, local_indices = search_index.search(query_vector, candidate_count)

    candidate_ids = [
        search_chunk_ids[int(local_i)]
        for local_i in local_indices[0]
        if int(local_i) >= 0
    ]

    if not candidate_ids:
        raise RuntimeError(
            f"FAISS returned no IMCI passages for disease '{disease}'."
        )

    # Rerank candidates by query/passage relevance.
    pairs = [[query, pdf_chunks[i]["excerpt"]] for i in candidate_ids]
    rerank_scores = reranker.predict(
        pairs,
        batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "16")),
        show_progress_bar=False
    )

    ranked = sorted(
        zip(candidate_ids, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    refs = []
    used_chars = 0
    for chunk_id, score in ranked[:min(TOP_K, len(ranked))]:
        item = pdf_chunks[chunk_id]
        remaining = MAX_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break

        excerpt = item["excerpt"][:remaining]
        refs.append({
            "pdf_page": int(item["pdf_page"]),
            "section_heading": item["section_heading"],
            "excerpt": excerpt,
            "disease_tags": item["disease_tags"],
            "reranker_score": round(float(score), 5)
        })
        used_chars += len(excerpt)

    log.info(
        "IMCI RAG | disease=%s | FAISS candidates=%s | reranked=%s | selected=%s",
        disease, candidate_count, len(ranked), len(refs)
    )
    return refs

def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("Model response did not contain a JSON object")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Model response JSON must be an object")
    return obj

def call_model(f, refs):
    if not API_KEY:
        raise RuntimeError("MEDITRON_API_KEY is missing from .env")
    system = """
You are an IMCI-informed public-health recommendation assistant.
Use the supplied IMCI booklet excerpts as the clinical evidence and use
the supplied aggregated forecast metrics to tailor the recommendations.
Return valid JSON only with exactly these string keys:
priority_public_health_measure, health_facility_readiness,
community_communication.
Give concise, specific, actionable recommendations. Do not invent drug
doses, clinical thresholds, referral rules, age groups or follow-up
intervals. Do not cite a PDF page unless the supplied excerpt supports it.
If the supplied excerpts do not support a clinical detail, omit it.
"""
    user = {
        "aggregated_forecast": forecast_context(f),
        "imci_booklet_references": refs,
        "task": "Generate the three recommendation fields for this forecast."
    }
    headers = {
        "Authorization": f"{AUTH_SCHEME} {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = safe({
        "model": MODEL,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(safe(user), ensure_ascii=False, allow_nan=False)}
        ]
    })
    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(
                API_URL, headers=headers, json=payload,
                timeout=TIMEOUT, verify=VERIFY_SSL
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            result = parse_json(content)
            for key in (
                "priority_public_health_measure",
                "health_facility_readiness",
                "community_communication"
            ):
                if not isinstance(result.get(key), str) or not result[key].strip():
                    raise ValueError(f"Missing/empty model output field: {key}")
                result[key] = result[key].strip()
            return result
        except Exception as exc:
            last = exc
            log.exception("Meditron attempt %s failed", attempt + 1)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Meditron generation failed: {last}")

def save_recommendation(f, rec, refs):
    """
    Update the existing weekly_monthly_forecasts row.
    No forecast rows are inserted, deleted, or duplicated.
    """
    forecast_id = int(f["id"])
    sql = f"""
    UPDATE {SRC}
       SET recommendation_id = id,
           priority_public_health_measure = %s,
           health_facility_readiness = %s,
           community_communication = %s,
           imci_references = %s::jsonb,
           model_name = %s,
           generated_at = NOW()
     WHERE id = %s
    RETURNING recommendation_id
    """
    vals = (
        rec["priority_public_health_measure"],
        rec["health_facility_readiness"],
        rec["community_communication"],
        json.dumps(safe(refs), ensure_ascii=False, allow_nan=False),
        MODEL,
        forecast_id
    )
    with db() as conn:
        row = conn.execute(sql, vals).fetchone()
        if row is None:
            raise RuntimeError(
                f"Forecast row id={forecast_id} disappeared before recommendation update."
            )
        conn.commit()
    return int(row[0])

def generate(f):
    refs = retrieve_references(f)
    if not refs:
        raise RuntimeError("No IMCI references retrieved")
    rec = call_model(f, refs)
    rec_id = save_recommendation(f, rec, refs)
    return safe({
        "recommendation_id": rec_id,
        "source_forecast_id": f["id"],
        "period_type": f.get("period_type"),
        "location_key": f.get("location_key"),
        "disease_name": f.get("disease_name"),
        **rec,
        "imci_references": refs,
        "model_name": MODEL
    })

def get_forecast(source_id):
    with db() as conn:
        cur = conn.execute(f"SELECT * FROM {SRC} WHERE id=%s", (source_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            d.name: safe(value)
            for d, value in zip(cur.description, row)
        }

class OneRequest(BaseModel):
    source_forecast_id: int = Field(gt=0)

class BatchRequest(BaseModel):
    period_type: Optional[str] = Field(default=None, pattern="^(week|month)$")
    limit: int = Field(default=100, ge=1, le=5000)

@app.on_event("startup")
def startup():
    ensure_target()
    log.info("Ready. Source=%s Target=%s Model=%s", SRC, DST, MODEL)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "source": f"{SCHEMA}.{SOURCE}",
        "target": f"{SCHEMA}.{SOURCE}",
        "write_mode": "UPDATE existing forecast rows in place",
        "model": MODEL,
        "embedding_model": EMBED_MODEL,
        "reranker_model": RERANKER_MODEL,
        "vector_index": "FAISS IndexFlatIP",
        "chunking": "booklet-heading-and-paragraph, disease-aware",
        "pdf_exists": PDF_PATH.exists()
    }

@app.post("/generate")
def generate_one(req: OneRequest):
    try:
        f = get_forecast(req.source_forecast_id)
        if not f:
            raise HTTPException(status_code=404, detail="Forecast ID not found")
        return generate(f)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Single generation failed")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/generate-all")
def generate_all(req: BatchRequest):
    clauses = [
        "f.period_type IN ('week', 'month')",
        "f.period_end >= CURRENT_DATE",
        "f.period_start <= (CURRENT_DATE + INTERVAL '6 months')::date",
        "f.generated_at IS NULL"
    ]
    params = []

    if req.period_type:
        clauses.append("f.period_type = %s")
        params.append(req.period_type)

    params.append(req.limit)
    sql = f"""
        SELECT f.*
          FROM {SRC} AS f
         WHERE {" AND ".join(clauses)}
         ORDER BY
            CASE f.period_type WHEN 'week' THEN 1 ELSE 2 END,
            f.period_start,
            f.hierarchy_level,
            f.location_key,
            f.disease_code
         LIMIT %s
    """
    try:
        with db() as conn:
            cur = conn.execute(sql, params)
            forecasts = [
                dict(zip([d.name for d in cur.description], row))
                for row in cur.fetchall()
            ]

        results, failures = [], []
        total = len(forecasts)

        for i, f in enumerate(forecasts, 1):
            start = time.monotonic()
            location = (
                f.get("union_name") or f.get("upazila_name")
                or f.get("district_name") or f.get("division_name")
                or f.get("location_key")
            )
            log.info(
                "[%s/%s] START | period=%s | location=%s | disease=%s | source_id=%s",
                i, total, f.get("period_type"), location,
                f.get("disease_name"), f.get("id")
            )
            try:
                result = generate(f)
                results.append(result)
                log.info(
                    "[%s/%s] COMPLETED | location=%s | disease=%s "
                    "| recommendation_id=%s | elapsed=%.2fs",
                    i, total, location, f.get("disease_name"),
                    result["recommendation_id"], time.monotonic() - start
                )
            except Exception as exc:
                failures.append({
                    "source_forecast_id": f.get("id"),
                    "period_type": f.get("period_type"),
                    "location": location,
                    "disease": f.get("disease_name"),
                    "error": str(exc)
                })
                log.exception(
                    "[%s/%s] FAILED | location=%s | disease=%s",
                    i, total, location, f.get("disease_name")
                )

        return {
            "total_selected": total,
            "generated": len(results),
            "failed": len(failures),
            "results": results,
            "failures": failures
        }
    except Exception as exc:
        log.exception("Batch generation failed")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/recommendations")
def recommendations(period_type: Optional[str] = None,
                    location_key: Optional[str] = None,
                    limit: int = 100):
    if period_type and period_type not in ("week", "month"):
        raise HTTPException(status_code=400, detail="period_type must be week or month")
    clauses, params = [], []
    if period_type:
        clauses.append("period_type=%s")
        params.append(period_type)
    if location_key:
        clauses.append("location_key=%s")
        params.append(location_key)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(limit, 1000)))
    sql = f"""SELECT * FROM {SRC} {where + (" AND " if where else "WHERE ")}generated_at IS NOT NULL
              ORDER BY period_start, hierarchy_level, location_key, disease_code
              LIMIT %s"""
    with db() as conn:
        cur = conn.execute(sql, params)
        rows = [dict(zip([d.name for d in cur.description], row))
                for row in cur.fetchall()]
    return [{k: safe(v) for k, v in row.items()} for row in rows]
