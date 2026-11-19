import hashlib
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import Future
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3

from app import config

logger = logging.getLogger("imci_meditron")

if config.MEDITRON_VERIFY_SSL is False:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_JSON_MISSING_COMMA_RX = re.compile(r'([\}\]"0-9]|true|false|null)(\s*\n\s*")')
_JSON_TRAILING_COMMA_RX = re.compile(r',(\s*[\}\]])')
_CITATION_PAGE_RX = re.compile(r"\d+")
_LIST_ITEM_PREFIX_RX = re.compile(r"^[-*•\d.]+\s*")
_MEDICATION_RXS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\bmg/kg\b",
        r"\bper kg\b",
        r"\boral dose\b",
        r"\bgive \w+ mg\b",
        r"\bdose of \d",
        r"\btablets? of\b",
        r"\bsyrup \d",
        r"\bfirst dose\b",
        r"\bsecond dose\b",
        r"\bprescri(?:be|ption)\b",
    )
)
_SESSION_LOCAL = threading.local()
_IN_FLIGHT_REQUESTS: Dict[str, Future] = {}
_IN_FLIGHT_LOCK = threading.Lock()


def _get_http_session() -> requests.Session:
    """Keep a reusable connection pool per worker thread."""
    session = getattr(_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _SESSION_LOCAL.session = session
    return session


# ---------------------------------------------------------------------------
# PROMPT BUILDER — ultra-compact for speed
# ---------------------------------------------------------------------------

def build_prompts(
    disease: str,
    location: Dict[str, Any],
    passages: List[Dict[str, Any]],
    disease_counts: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Build minimal system+user prompts for Meditron-27B.

    Design philosophy: every extra token in the prompt costs ~10-20 ms on
    Meditron-27B. Keep the instructions telegraphic; move all formatting,
    validation and fallback enrichment into parse_and_validate_response().
    """
    loc_name = (
        location.get("upazila") or location.get("union")
        or location.get("district") or location.get("selected_location")
        or "the location"
    )

    system_prompt = (
        "You are an IMCI public-health planning assistant. "
        "Output JSON only — no markdown, no text outside the object.\n"
        "Use only the supplied booklet evidence; cite page numbers in each "
        "recommendation; do not infer unsupported medical advice or give "
        "individual treatment, medicines, or doses. Give 3 concise, actionable "
        "recommendations per category, one sentence each; flag expert review.\n"
        '{"disease":"…","summary":"…",'
        '"government_public_health_interventions":["…"],'
        '"health_system_readiness":["…"],'
        '"community_engagement":["…"],'
        '"government_planning_and_monitoring":["…"],'
        '"booklet_references":["Page N"],'
        '"requires_expert_review":false,"expert_review_reason":""}'
    )

    parts = []
    for p in passages:
        pn = p.get("page_number") or "?"
        txt = p.get("text", "").strip()[:config.RAG_MAX_PASSAGE_CHARS]
        parts.append(f"[p{pn}] {txt}")
    passage_block = "\n".join(parts)

    counts_line = ""
    if disease_counts:
        counts_line = (
            f"Cases: now={disease_counts.get('count_now',0)}, "
            f"next_week={disease_counts.get('count_next_week',0)}."
        )

    user_prompt = (
        f"{disease} in {loc_name}. {counts_line}\n"
        f"Passages:\n{passage_block}\n"
        f"Return JSON."
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# RESPONSE PARSER — robust, handles malformed / truncated JSON
# ---------------------------------------------------------------------------

def parse_and_validate_response(
    content: str,
    retrieved_pages: List[str],
    expected_disease: str,
    location: Dict[str, Any],
    disease_counts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Parse Meditron's JSON output with repair for missing commas and truncation.

    Post-processing:
    - Normalises old 3-category keys → new 4-category schema.
    - Caps each category at 5 items.
    - Rejects items containing medication/dosage language.
    - Validates page citations against retrieved passages.
    - Guarantees summary opens with disease counts.
    """
    content = content.strip()

    # Strip markdown fences
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # --- JSON repair helper ---
    def _repair(text: str) -> str:
        # Insert missing commas between adjacent fields
        text = _JSON_MISSING_COMMA_RX.sub(r'\1,\2', text)
        # Strip trailing commas
        text = _JSON_TRAILING_COMMA_RX.sub(r'\1', text)
        # Close unclosed structures (truncation)
        stack: List[str] = []
        in_str = esc = False
        for ch in text:
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch in ('}', ']') and stack:
                stack.pop()
        if in_str:
            text = text.rstrip() + '"'
        if stack:
            text = text.rstrip().rstrip(',') + ''.join(reversed(stack))
        return text

    # --- Parse with fallback chain ---
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        if start == -1:
            raise ValueError(f"No JSON object in LLM response: {content[:200]}")
        end = content.rfind("}")
        candidates = []
        if end != -1 and end > start:
            candidates.append(content[start:end + 1])
        candidates.append(content[start:])

        data = None
        last_exc: Optional[Exception] = None
        for cand in candidates:
            for attempt in (cand, _repair(cand)):
                try:
                    data = json.loads(attempt)
                    if attempt != cand:
                        logger.info("JSON repaired successfully.")
                    break
                except json.JSONDecodeError as e:
                    last_exc = e
            if data is not None:
                break
        if data is None:
            raise ValueError(
                f"JSON parse failed after repair: {(candidates[0] if candidates else content)[:200]}"
            ) from last_exc

    # --- Helpers ---
    def as_list(val: Any, cap: int = 5) -> List[str]:
        if isinstance(val, list):
            items = []
            for value in val:
                item = str(value).strip()
                if item:
                    items.append(item)
        elif isinstance(val, str) and val.strip():
            items = [_LIST_ITEM_PREFIX_RX.sub("", ln).strip() for ln in val.split("\n")]
            items = [i for i in items if i]
        else:
            items = []
        return items[:cap]

    def clean_med(items: List[str]) -> Tuple[List[str], bool]:
        clean, found = [], False
        for i in items:
            if any(pattern.search(i) for pattern in _MEDICATION_RXS):
                found = True
            else:
                clean.append(i)
        return clean, found

    # --- Extract fields with alias fallback ---
    disease = data.get("disease") or data.get("disease_name") or expected_disease
    summary = str(data.get("summary") or "").strip()

    gov, med1 = clean_med(as_list(
        data.get("government_public_health_interventions")
        or data.get("priority_public_health_measures")
        or data.get("priority_public_health_measure")
    ))
    health, med2 = clean_med(as_list(
        data.get("health_system_readiness")
        or data.get("health_facility_readiness")
    ))
    comm, med3 = clean_med(as_list(
        data.get("community_engagement")
        or data.get("caregiver_and_community_communication")
        or data.get("community_communication")
    ))
    plan, med4 = clean_med(as_list(
        data.get("government_planning_and_monitoring") or []
    ))
    medication_found = med1 or med2 or med3 or med4

    # --- Citation validation ---
    raw_refs = as_list(data.get("booklet_references") or data.get("imci_references"), cap=20)
    retrieved_nums = {
        match.group()
        for page in retrieved_pages
        if (match := _CITATION_PAGE_RX.search(page))
    }
    validated_refs: List[str] = []
    unverified = False
    for ref in raw_refs:
        m = _CITATION_PAGE_RX.search(ref)
        if m:
            pn = m.group()
            if pn in retrieved_nums:
                validated_refs.append(f"Page {pn}")
            else:
                validated_refs.append(f"Page {pn} (unverified)")
                unverified = True
        elif ref:
            validated_refs.append(ref)
    if not validated_refs and retrieved_pages:
        validated_refs = sorted(set(retrieved_pages))

    # --- Expert review ---
    requires_review = bool(
        data.get("requires_expert_review")
        if "requires_expert_review" in data
        else data.get("expert_review_required", False)
    )
    review_reason = str(data.get("expert_review_reason") or "").strip()
    if unverified:
        requires_review = True
        review_reason = f"{review_reason} Unverified page citations found.".strip()
    if medication_found:
        requires_review = True
        review_reason = f"{review_reason} Medication language removed.".strip()

    # --- Summary: guarantee it opens with counts ---
    if disease_counts and disease_counts.get("count_now") is not None:
        c_now = disease_counts.get("count_now", 0)
        c_next = disease_counts.get("count_next_week", 0)
        w_now = disease_counts.get("current_week_total", 0)
        w_next = disease_counts.get("next_week_total", 0)
        loc_name = (
            location.get("upazila") or location.get("union")
            or location.get("district") or location.get("selected_location")
            or "the selected location"
        )
        has_now = str(c_now) in summary or "currently" in summary.lower()
        has_next = str(c_next) in summary or "next week" in summary.lower()
        if not (has_now and has_next):
            summary = (
                f"In {loc_name}, the forecasted {disease} caseload is {c_now} cases now "
                f"({w_now} this week), projected at {c_next} next week ({w_next}). "
            ) + summary

    # --- Build result ---
    res: Dict[str, Any] = {
        "disease": disease,
        "location": location,
        "summary": summary,
        "government_public_health_interventions": gov,
        "health_system_readiness": health,
        "community_engagement": comm,
        "government_planning_and_monitoring": plan,
        "booklet_references": sorted(dict.fromkeys(validated_refs)),
        "requires_expert_review": requires_review,
        "expert_review_reason": review_reason,
    }
    if disease_counts:
        res["disease_counts"] = disease_counts
    return res


# ---------------------------------------------------------------------------
# MEDITRON API CALLER — pooled connections, bounded retries, and timing
# ---------------------------------------------------------------------------

class MeditronAPIError(RuntimeError):
    """A classified failure calling the configured Meditron endpoint."""

    def __init__(
        self,
        message: str,
        category: str,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.category = category
        self.status_code = status_code


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _retry_delay(retry_number: int, retry_after: Optional[str] = None) -> float:
    max_delay = max(0.0, config.MEDITRON_RETRY_MAX_DELAY)
    if retry_after is not None:
        requested_delay = _retry_after_seconds(retry_after)
        if requested_delay is not None:
            delay = min(max_delay, requested_delay)
            jitter = min(
                max_delay - delay,
                delay * max(0.0, config.MEDITRON_RETRY_JITTER),
            )
            return delay + random.uniform(0.0, jitter)

    exponential_cap = min(
        max_delay,
        max(0.0, config.MEDITRON_RETRY_BASE_DELAY) * (2 ** (retry_number - 1)),
    )
    return random.uniform(0.0, exponential_cap)


def _request_meditron(
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: Tuple[float, float],
    retries: int,
) -> Tuple[str, float]:
    request_started = time.perf_counter()
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        attempt_started = time.perf_counter()
        try:
            response = _get_http_session().post(
                config.MEDITRON_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=config.MEDITRON_VERIFY_SSL,
            )
        except requests.exceptions.SSLError as exc:
            elapsed = time.perf_counter() - attempt_started
            logger.error("Meditron TLS error after %.2fs: %s", elapsed, exc)
            raise MeditronAPIError(
                f"Meditron TLS connection failed: {exc}",
                category="tls_error",
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            elapsed = time.perf_counter() - attempt_started
            logger.warning(
                "Meditron connection error on attempt %d/%d after %.2fs: %s",
                attempt, attempts, elapsed, exc,
            )
            if attempt < attempts:
                delay = _retry_delay(attempt)
                logger.info("Retrying Meditron connection in %.2fs", delay)
                time.sleep(delay)
                continue
            raise MeditronAPIError(
                f"Meditron connection failed after {attempts} attempts: {exc}",
                category="connection_error",
            ) from exc
        except requests.exceptions.Timeout as exc:
            elapsed = time.perf_counter() - attempt_started
            logger.error(
                "Meditron response timed out after %.2fs; not retrying to avoid "
                "duplicating an inference that may still be running: %s",
                elapsed,
                exc,
            )
            raise MeditronAPIError(
                f"Meditron response timed out: {exc}",
                category="timeout",
            ) from exc
        except requests.exceptions.RequestException as exc:
            elapsed = time.perf_counter() - attempt_started
            logger.error("Non-retryable Meditron request error after %.2fs: %s", elapsed, exc)
            raise MeditronAPIError(
                f"Meditron request failed: {exc}",
                category="request_error",
            ) from exc

        elapsed = time.perf_counter() - attempt_started
        status_code = response.status_code
        if status_code == 429 or 500 <= status_code < 600:
            category = "rate_limit" if status_code == 429 else "server_error"
            logger.warning(
                "Meditron HTTP %d (%s) on attempt %d/%d after %.2fs",
                status_code, category, attempt, attempts, elapsed,
            )
            if attempt < attempts:
                retry_after = response.headers.get("Retry-After") if status_code == 429 else None
                delay = _retry_delay(attempt, retry_after)
                logger.info(
                    "Retrying Meditron HTTP %d in %.2fs%s",
                    status_code,
                    delay,
                    " (Retry-After)" if retry_after else "",
                )
                time.sleep(delay)
                continue
            raise MeditronAPIError(
                f"Meditron HTTP {status_code} after {attempts} attempts: {response.text[:300]}",
                category=category,
                status_code=status_code,
            )

        if not response.ok:
            logger.error("Non-retryable Meditron HTTP %d after %.2fs", status_code, elapsed)
            raise MeditronAPIError(
                f"Meditron HTTP {status_code}: {response.text[:300]}",
                category="client_error",
                status_code=status_code,
            )

        try:
            result = response.json()
        except ValueError as exc:
            raise MeditronAPIError(
                "Meditron returned invalid JSON.",
                category="invalid_response",
                status_code=status_code,
            ) from exc

        choices = result.get("choices") if isinstance(result, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices else None
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise MeditronAPIError(
                "Meditron response did not contain choices[0].message.content.",
                category="invalid_response",
                status_code=status_code,
            )

        total_elapsed = time.perf_counter() - request_started
        usage = result.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        logger.info(
            "Meditron response attempt_latency=%.2fs total_latency=%.2fs "
            "(attempt %d/%d, HTTP %d) — "
            "prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
            elapsed,
            total_elapsed,
            attempt,
            attempts,
            status_code,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
            usage.get("total_tokens", "?"),
        )
        return content.strip(), total_elapsed

    raise MeditronAPIError("Meditron request exhausted retries.", category="request_error")


def call_meditron(
    system_prompt: str,
    user_prompt: str,
    max_retries: Optional[int] = None,
    timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
    return_latency: bool = False,
) -> Any:
    """
    Send chat-completion request to Meditron-27B. max_retries counts retries
    after the initial request; identical concurrent requests are coalesced.

    Returns the raw content string (or (content, elapsed) if return_latency=True).
    Uses a pooled HTTP session and logs each attempt, prompt size, and token usage.
    """
    if not config.MEDITRON_API_KEY:
        raise MeditronAPIError(
            "MEDITRON_API_KEY is not configured.",
            category="configuration_error",
        )

    timeout_val = config.MEDITRON_TIMEOUT if timeout is None else timeout
    max_tokens_val = config.MEDITRON_MAX_TOKENS if max_tokens is None else max_tokens
    retries = config.MEDITRON_MAX_RETRIES if max_retries is None else max_retries
    if retries < 0:
        raise ValueError("max_retries must be zero or greater.")
    if timeout_val <= 0 or config.MEDITRON_CONNECT_TIMEOUT <= 0:
        raise ValueError("Meditron connection and read timeouts must be greater than zero.")
    if max_tokens_val <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    prompt_chars = len(system_prompt) + len(user_prompt)
    logger.info(
        "Prompt sizes: system=%d chars, user=%d chars, total=%d chars "
        "(estimated_tokens≈%d), max_tokens=%d",
        len(system_prompt), len(user_prompt), prompt_chars, (prompt_chars + 3) // 4,
        max_tokens_val,
    )

    headers = {
        "Authorization": f"{config.MEDITRON_AUTH_SCHEME} {config.MEDITRON_API_KEY}".strip(),
        "Content-Type": "application/json",
    }

    payload = {
        "model": config.MEDITRON_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens_val,
        "stream": False,
    }
    request_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "url": config.MEDITRON_URL,
                "payload": payload,
                "timeout": [config.MEDITRON_CONNECT_TIMEOUT, timeout_val],
                "retries": retries,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with _IN_FLIGHT_LOCK:
        future = _IN_FLIGHT_REQUESTS.get(request_fingerprint)
        is_owner = future is None
        if is_owner:
            future = Future()
            _IN_FLIGHT_REQUESTS[request_fingerprint] = future
    assert future is not None

    if not is_owner:
        logger.info("Coalescing an identical in-flight Meditron request.")
        content, elapsed = future.result()
        return (content, elapsed) if return_latency else content

    try:
        content, elapsed = _request_meditron(
            payload=payload,
            headers=headers,
            timeout=(config.MEDITRON_CONNECT_TIMEOUT, float(timeout_val)),
            retries=retries,
        )
        future.set_result((content, elapsed))
        return (content, elapsed) if return_latency else content
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        with _IN_FLIGHT_LOCK:
            if _IN_FLIGHT_REQUESTS.get(request_fingerprint) is future:
                del _IN_FLIGHT_REQUESTS[request_fingerprint]
