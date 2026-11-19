"""Location-specific, PDF-grounded disease recommendation generator."""
import json
import logging
import os
import re
import time
import hashlib
import random
import threading
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import numpy as np
from datetime import date
from functools import lru_cache
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S", force=True)
log = logging.getLogger(__name__)
app = FastAPI(title='IMCI Forecast RAG')

@contextmanager
def progress_timer(stage):
    start = time.monotonic()
    log.info("START | %s", stage)
    try:
        yield
    except Exception:
        log.exception("FAILED | %s | elapsed %.1fs", stage, time.monotonic()-start)
        raise
    else:
        log.info("DONE  | %s | elapsed %.1fs", stage, time.monotonic()-start)


def wait_with_progress(seconds, reason):
    seconds = max(0.0, seconds)
    if seconds == 0:
        return
    log.warning("WAIT  | %s | %.1fs remaining initially", reason, seconds)
    deadline = time.monotonic() + seconds
    interval = max(1, int(os.getenv("PROGRESS_INTERVAL_SECONDS", "10")))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
        remaining = max(0, deadline-time.monotonic())
        if remaining > 0:
            log.info("WAIT  | %s | %.0fs remaining", reason, remaining)
    log.info("WAIT  | %s | finished", reason)


def request_with_heartbeat(url, **kwargs):
    done = threading.Event()
    interval = max(1, int(os.getenv("PROGRESS_INTERVAL_SECONDS", "10")))
    started = time.monotonic()
    def heartbeat():
        while not done.wait(interval):
            log.info("MEDITRON | awaiting API response | elapsed %.0fs (remaining time unknown)", time.monotonic()-started)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return requests.post(url, **kwargs)
    finally:
        done.set()
        thread.join(timeout=1)
        log.info("MEDITRON | HTTP call ended | elapsed %.1fs", time.monotonic()-started)

SCHEMA = os.getenv('DB_SCHEMA', 'ml_analytics_v5')
SOURCE = os.getenv('DB_TABLE', 'location_disease_forecasts')
DEST = os.getenv('RECOMMENDATION_TABLE', 'location_disease_recommendations')
COUNT = os.getenv('FORECAST_COUNT_COLUMN', 'predicted_total_assessments')
ALLOWED_COUNTS = {'predicted_total_assessments', 'predicted_at_risk_count', 'predicted_severe_count'}
if COUNT not in ALLOWED_COUNTS:
    raise ValueError('FORECAST_COUNT_COLUMN must be a verified count column')
for name in (SCHEMA, SOURCE, DEST):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        raise ValueError('Invalid SQL identifier')

FIELDS = ['priority_public_health_measure', 'health_facility_readiness', 'community_communication']
# Only one inference call at a time in this process. Use ONE uvicorn worker.
meditron_lock = threading.Lock()
_last_request = 0.0

class MeditronRateLimit(Exception):
    def __init__(self, wait_seconds=None, message='Meditron rate limit exceeded'):
        super().__init__(message)
        self.wait_seconds = wait_seconds


def retry_after_seconds(response):
    value = response.headers.get('Retry-After')
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def parse_model_json(content):
    content = content.strip()
    content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = content.find('{')
        if start < 0:
            raise ValueError('Meditron response contains no JSON object')
        obj, _ = decoder.raw_decode(content[start:])
        return obj


class RequestBody(BaseModel):
    location_type: str
    location_key: str
    forecast_date: date
    persist: bool = True

@lru_cache(maxsize=1)
def kb():
    log.info("PDF | loading IMCI document and sentence-transformer index")
    path = Path(os.environ['IMCI_PDF_PATH'])
    if not path.is_file():
        raise FileNotFoundError(f'IMCI PDF not found: {path}')
    reader = PdfReader(str(path))
    chunks = []
    # Keep actual PDF page numbers, including any front matter.
    for page_no, page in enumerate(reader.pages, 1):
        text = re.sub(r'\s+', ' ', page.extract_text() or '').strip()
        for start in range(0, len(text), 950):
            part = text[start:start + 1150]
            if len(part) >= 60:
                chunks.append({'page': page_no, 'text': part})
    if not chunks:
        raise ValueError('PDF has no extractable text; OCR is required')
    model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer(model_name, device=os.getenv('EMBEDDING_DEVICE', 'cpu'))
    # Cache document vectors on disk: only recompute if PDF/model/chunking changes.
    cache_dir = Path(os.getenv('EMBEDDING_CACHE_DIR', '.embedding_cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        (str(path.resolve()) + str(path.stat().st_size) + str(path.stat().st_mtime_ns)
         + model_name + 'chunk950step1150-v1').encode()
    ).hexdigest()[:24]
    cache_file = cache_dir / f'imci_{fingerprint}.npz'
    embeddings = None
    if cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                candidate = data['embeddings']
            if candidate.shape[0] == len(chunks):
                embeddings = candidate
                log.info('Loaded cached PDF embeddings from %s', cache_file)
        except (OSError, ValueError, KeyError):
            log.warning('Embedding cache invalid; rebuilding')
    if embeddings is None:
        log.info("PDF | building embeddings for %d chunks (first run may take longer)", len(chunks))
        embeddings = model.encode(
            [x['text'] for x in chunks],
            batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', '32')),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        np.savez_compressed(cache_file, embeddings=embeddings)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    log.info('Indexed %d PDF chunks using %s', len(chunks), model_name)
    return chunks, model, embeddings

def retrieve_many(rows):
    """Embed all disease queries in ONE batch; cosine via normalized dot product."""
    log.info("RETRIEVAL | embedding queries for %d disease(s)", len(rows))
    chunks, model, embeddings = kb()
    queries = [f"{r['disease_name']} {str(r['disease_code']).replace('_', ' ')} "
               'IMCI classification treatment referral follow-up caregiver danger signs'
               for r in rows]
    query_vectors = model.encode(queries, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
    scores = np.asarray(query_vectors, dtype=np.float32) @ embeddings.T
    top_k = max(1, min(int(os.getenv('RETRIEVAL_TOP_K', '2')), len(chunks)))
    max_chars = int(os.getenv('MAX_CONTEXT_CHARS', '1500'))
    results = []
    for row_scores in scores:
        ranked = np.argsort(-row_scores)[:top_k]
        selected, remaining = [], max_chars
        for idx in ranked:
            if remaining <= 0:
                break
            chunk = chunks[int(idx)]
            excerpt = chunk['text'][:remaining]
            if excerpt:
                selected.append({'page': chunk['page'], 'text': excerpt,
                                 'similarity': round(float(row_scores[idx]), 4)})
                remaining -= len(excerpt)
        if not selected:
            raise ValueError('No relevant IMCI passage found')
        results.append(selected)
    log.info("RETRIEVAL | selected %s passages", [len(x) for x in results])
    return results

def fetch_top_three(req):
    if req.location_type not in ('district', 'upazila', 'union'):
        raise ValueError('location_type must be district, upazila, or union')
    # Use the most recently updated forecast for each disease, avoiding duplicate
    # model runs/horizons being incorrectly added together.
    stmt = sql.SQL("""
        WITH latest AS (
          SELECT DISTINCT ON (disease_code)
            hierarchy_level AS location_type, location_key,
            division_name, district_name, upazila_name, union_name,
            disease_code, disease_name, forecast_date,
            {count} AS forecasted_count,
            predicted_severe_count AS severe_count,
            predicted_at_risk_count AS at_risk_count,
            predicted_severity_score AS severity_score,
            predicted_severity_label AS risk_level,
            risk_trend_label AS trend
          FROM {source}
          WHERE hierarchy_level = %s AND location_key = %s AND forecast_date = %s
          ORDER BY disease_code, updated_at DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
        )
        SELECT * FROM latest
        ORDER BY COALESCE(severe_count,0) DESC,
                 COALESCE(at_risk_count,0) DESC,
                 COALESCE(severity_score,0) DESC,
                 COALESCE(forecasted_count,0) DESC, disease_code
        LIMIT 3
    """).format(count=sql.Identifier(COUNT), source=sql.Identifier(SCHEMA, SOURCE))
    with psycopg.connect(os.environ['DATABASE_URL'], row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(stmt, (req.location_type, req.location_key, req.forecast_date))
            return cur.fetchall()

def generate_batch(rows, passages_list, deadline=None):
    """ONE non-concurrent Meditron request for all diseases, with 429 backoff."""
    inputs = []
    for row, passages in zip(rows, passages_list):
        inputs.append({
            'disease_code': str(row['disease_code']),
            'disease_name': row['disease_name'],
            'forecasted_assessments': row['forecasted_count'],
            'predicted_severe_count': row['severe_count'],
            'predicted_at_risk_count': row['at_risk_count'],
            'severity_label': row['risk_level'],
            'trend': row['trend'],
            'imci_evidence': [{'pdf_page': p['page'], 'text': p['text']} for p in passages],
        })

    # THE EDITABLE PROMPT: update these two variables to change LLM instructions.
    system_prompt = (
        'You support public-health planning using IMCI guidelines. '
        'The supplied forecasts are PREDICTIONS, not confirmed patient cases. '
        'Use ONLY supplied IMCI excerpts for clinical advice; do not invent doses, '
        'treatment durations, classifications, or page references. '
        'Never advise oral medication for an actively convulsing or unconscious child. '
        'Emphasize urgent clinical assessment/referral for danger signs when supported. '
        'For EACH supplied disease, output exactly three SHORT, single-sentence '
        'recommendations: priority_public_health_measure, health_facility_readiness, '
        'community_communication. Cite each supported sentence as (PDF page N), '
        'where N is a page actually supplied for THAT disease. '
        'If a category lacks relevant evidence, use exactly: '
        'Insufficient IMCI evidence in retrieved passages. '
        'Return only valid JSON in this shape: '
        '{"recommendations":[{"disease_code":"...",'
        '"priority_public_health_measure":"...",'
        '"health_facility_readiness":"...",'
        '"community_communication":"..."}]}. '
        'Include exactly one object per supplied disease, in the same order.'
    )
    user_prompt = 'DISEASE FORECASTS AND IMCI EVIDENCE:\n' + json.dumps(
        inputs, ensure_ascii=False, default=str, separators=(',', ':'))

    payload = {
        'model': os.getenv('MEDITRON_MODEL', 'meditron-27b'),
        'messages': [{'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': user_prompt}],
        'temperature': 0.1,
        'max_tokens': int(os.getenv('MEDITRON_MAX_TOKENS', '1100')),
    }
    headers = {
        'Authorization': f"{os.getenv('MEDITRON_AUTH_SCHEME', 'Bearer')} {os.environ['MEDITRON_API_KEY']}",
        'Content-Type': 'application/json',
    }
    # Set MEDITRON_CA_BUNDLE to your network CA PEM to retain TLS validation.
    ca_bundle = os.getenv('MEDITRON_CA_BUNDLE', '').strip()
    verify = ca_bundle or os.getenv('MEDITRON_VERIFY_SSL', 'true').lower() == 'true'
    if verify is False:
        log.warning('TLS verification disabled; configure a trusted CA for production')
    attempts = max(1, int(os.getenv('MEDITRON_MAX_RETRIES', '2')))
    max_wait = max(0, int(os.getenv('MEDITRON_MAX_RETRY_WAIT', '90')))
    min_interval = max(0.0, float(os.getenv('MEDITRON_MIN_INTERVAL_SECONDS', '15')))
    global _last_request

    log.info("MEDITRON | prompt prepared for %d diseases | approx %d characters | max output tokens %d", len(rows), len(system_prompt)+len(user_prompt), payload["max_tokens"])
    if not meditron_lock.acquire(blocking=False):
        log.warning("MEDITRON | another generation is active; waiting for process lock")
        meditron_lock.acquire()
    try:
        for attempt in range(attempts):
            elapsed = time.monotonic() - _last_request
            if _last_request and elapsed < min_interval:
                cooldown = min_interval - elapsed
                if deadline is not None and time.monotonic() + cooldown + 1 >= deadline:
                    raise TimeoutError(f'Meditron cooldown requires {cooldown:.1f}s; cannot finish within request deadline')
                wait_with_progress(cooldown, "minimum interval between Meditron calls")
            remaining = (deadline - time.monotonic()) if deadline is not None else float('inf')
            if remaining <= 2:
                raise TimeoutError('Insufficient time remaining for Meditron request')
            _last_request = time.monotonic()
            log.info("MEDITRON | attempt %d/%d | sending request", attempt+1, attempts)
            try:
                response = request_with_heartbeat(
                    os.environ['MEDITRON_URL'], json=payload, headers=headers,
                    timeout=(min(3, max(1, remaining - 1)), min(float(os.getenv('MEDITRON_TIMEOUT', '25')), max(1, remaining - 1))), verify=verify,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == attempts - 1:
                    raise RuntimeError('Meditron connection failed after retries') from exc
                wait_with_progress(min(max_wait, 5 * (2 ** attempt) + random.random()), "network/server error backoff")
                continue

            log.info("MEDITRON | HTTP status %d", response.status_code)
            if response.status_code == 429:
                wait = retry_after_seconds(response)
                if wait is None:
                    wait = min(60, 10 * (2 ** attempt)) + random.uniform(0, 1)
                log.warning('Meditron 429 (attempt %d/%d); retry after %.1fs',
                            attempt + 1, attempts, wait)
                if attempt == attempts - 1 or wait > max_wait:
                    raise MeditronRateLimit(wait_seconds=wait)
                wait_with_progress(wait, "Meditron HTTP 429 cooldown")
                continue
            if response.status_code in (500, 502, 503, 504) and attempt < attempts - 1:
                wait_with_progress(min(max_wait, 5 * (2 ** attempt) + random.random()), "network/server error backoff")
                continue
            if response.status_code >= 400:
                log.error('Meditron returned HTTP %s: %s', response.status_code,
                          response.text[:300])
                response.raise_for_status()

            try:
                content = response.json()['choices'][0]['message']['content']
                output = parse_model_json(content)
                items = output.get('recommendations')
                if not isinstance(items, list) or len(items) != len(rows):
                    raise ValueError('Model did not return one recommendation per disease')
                validated = []
                for row, passages, item in zip(rows, passages_list, items):
                    if not isinstance(item, dict) or str(item.get('disease_code')) != str(row['disease_code']):
                        raise ValueError('Model disease code/order mismatch')
                    valid_pages = {int(p['page']) for p in passages}
                    result = {}
                    for key in FIELDS:
                        sentence = item.get(key)
                        if not isinstance(sentence, str) or not sentence.strip():
                            raise ValueError(f'Missing or empty {key}')
                        sentence = sentence.strip()
                        cited = [int(n) for n in re.findall(r'PDF page\s+(\d+)', sentence, re.I)]
                        if cited and not set(cited).issubset(valid_pages):
                            raise ValueError(f'Invalid page citation: {key}')
                        if not cited and sentence != 'Insufficient IMCI evidence in retrieved passages':
                            raise ValueError(f'Missing PDF citation: {key}')
                        result[key] = sentence
                    validated.append(result)
                log.info("MEDITRON | validated %d disease recommendation(s)", len(validated))
                return validated
            except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                # Do not retry malformed output automatically: repeated large prompts
                # often make rate limits worse. Return a clear error instead.
                raise ValueError(f'Invalid Meditron JSON or recommendations: {exc}') from exc
    finally:
        meditron_lock.release()
    raise RuntimeError('Meditron request failed')

def priority(row):
    severe, risk = float(row['severe_count'] or 0), float(row['at_risk_count'] or 0)
    return 'critical' if severe > 0 else ('high' if risk > 0 else 'routine')

def store(records):
    stmt = sql.SQL('''
      INSERT INTO {dest} (location_type, location_key, division_name, district_name,
        upazila_name, union_name, disease_code, disease_name, forecast_date,
        forecasted_count, risk_level, trend, priority, data_summary,
        priority_public_health_measure, health_facility_readiness,
        community_communication, imci_references, model_name)
      VALUES (%(location_type)s, %(location_key)s, %(division_name)s, %(district_name)s,
        %(upazila_name)s, %(union_name)s, %(disease_code)s, %(disease_name)s,
        %(forecast_date)s, %(forecasted_count)s, %(risk_level)s, %(trend)s,
        %(priority)s, %(data_summary)s, %(priority_public_health_measure)s,
        %(health_facility_readiness)s, %(community_communication)s,
        %(imci_references)s::jsonb, %(model_name)s)
      RETURNING recommendation_id
    ''').format(dest=sql.Identifier(SCHEMA, DEST))
    with psycopg.connect(os.environ['DATABASE_URL']) as conn:
        with conn.cursor() as cur:
            # Atomic replacement for the same location/date/disease, preventing
            # duplicates on retries. A unique index is recommended for concurrency.
            delete_stmt = sql.SQL("""DELETE FROM {dest}
              WHERE location_type = %s AND location_key = %s
                AND forecast_date = %s AND disease_code = %s""").format(
                dest=sql.Identifier(SCHEMA, DEST))
            for record in records:
                cur.execute(delete_stmt, (record['location_type'], record['location_key'],
                                         record['forecast_date'], record['disease_code']))
                cur.execute(stmt, record)
                record['recommendation_id'] = cur.fetchone()[0]

def run(req):
    started = time.perf_counter()
    deadline = time.monotonic() + float(os.getenv('REQUEST_BUDGET_SECONDS', '30'))
    log.info("REQUEST | location_type=%s location_key=%s date=%s persist=%s", req.location_type, req.location_key, req.forecast_date, req.persist)
    with progress_timer("PostgreSQL forecast lookup and risk ranking"):
        rows = fetch_top_three(req)
    log.info("FORECAST | %d disease(s) selected: %s", len(rows), ", ".join(str(r['disease_name']) for r in rows))
    if not rows:
        raise LookupError('No forecast records found for this location and date')
    with progress_timer("IMCI PDF semantic retrieval"):
        passages_list = retrieve_many(rows)
    log.info('BUDGET | %.1fs remain before Meditron call', max(0, deadline-time.monotonic()))

    # Exactly one Meditron request for the complete top-three batch.
    with progress_timer("Meditron single-batch generation and validation"):
        generated_list = generate_batch(rows, passages_list, deadline=deadline)

    records = []
    for row, passages, generated in zip(rows, passages_list, generated_list):
        summary = (f"For {row['location_type']} {row['location_key']} on {row['forecast_date']}, "
                   f"{row['disease_name']} has {row['forecasted_count']} forecast assessments, "
                   f"including {row['severe_count']} severe and {row['at_risk_count']} at-risk; "
                   f"reported severity label: {row['risk_level']}; trend: {row['trend']}.")
        record = {k: row[k] for k in ('location_type', 'location_key', 'division_name',
            'district_name', 'upazila_name', 'union_name', 'disease_code', 'disease_name',
            'forecast_date', 'forecasted_count', 'risk_level', 'trend')}
        record.update(priority=priority(row), data_summary=summary, **generated,
                      imci_references=json.dumps([{'pdf_page': p['page'], 'excerpt': p['text'][:250]}
                                                  for p in passages]),
                      model_name=os.getenv('MEDITRON_MODEL', 'meditron-27b'))
        records.append(record)
    if req.persist:
        with progress_timer("PostgreSQL recommendation persistence"):
            store(records)
        log.info("DATABASE | saved %d recommendation record(s)", len(records))
    else:
        log.info("DATABASE | persist=false; skipping write")
    for record in records:
        record['imci_references'] = json.loads(record['imci_references'])
    elapsed = round(time.perf_counter() - started, 2)
    log.info('Generated %d recommendations in %.2f seconds', len(records), elapsed)
    return {'location_type': req.location_type, 'location_key': req.location_key,
            'forecast_date': req.forecast_date.isoformat(), 'generated': len(records),
            'saved': req.persist, 'elapsed_seconds': elapsed, 'recommendations': records}

@app.post('/generate')
def generate_endpoint(request: RequestBody):
    try:
        return run(request)
    except TimeoutError as exc:
        log.warning('30-second budget exceeded: %s', exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MeditronRateLimit as exc:
        headers = {'Retry-After': str(max(1, int(exc.wait_seconds or 30)))}
        raise HTTPException(status_code=429, detail='Meditron rate limited this request; retry later', headers=headers) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.HTTPError as exc:
        log.error('Meditron HTTP failure: %s', exc)
        raise HTTPException(status_code=502, detail='Meditron upstream HTTP error') from exc
    except RuntimeError as exc:
        log.exception('Generation failed')
        raise HTTPException(status_code=504 if 'connection failed after retries' in str(exc) else 502, detail=str(exc)) from exc
    except Exception:
        log.exception('Generation failed')
        raise HTTPException(status_code=502, detail='Recommendation generation failed; see server logs')

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/recommendations')
def read_recommendations(location_type: str = Query(..., pattern='^(district|upazila|union)$'),
                         location_key: str = Query(...), forecast_date: date = Query(...)):
    statement = sql.SQL("""SELECT * FROM {dest}
       WHERE location_type = %s AND location_key = %s AND forecast_date = %s
       ORDER BY generated_at DESC, recommendation_id DESC""").format(
       dest=sql.Identifier(SCHEMA, DEST))
    try:
        with psycopg.connect(os.environ['DATABASE_URL'], row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (location_type, location_key, forecast_date))
                return {'recommendations': cur.fetchall()}
    except Exception:
        log.exception('Reading recommendations failed')
        raise HTTPException(status_code=502, detail='Database read failed')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
