import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3

from app import config

logger = logging.getLogger("imci_meditron")

if config.MEDITRON_VERIFY_SSL is False:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def build_prompts(
    disease: str,
    location: Dict[str, Any],
    passages: List[Dict[str, Any]],
    disease_counts: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Build concise system and user prompts for Meditron-27B.

    Schema (4 categories, max 5 items each, 1–2 sentences per item):
      government_public_health_interventions
      health_system_readiness
      community_engagement
      government_planning_and_monitoring

    Rules enforced in prompt:
    - Government/public-health planning level only (not individual patient treatment).
    - No medication names, dosages, prescriptions, or individual treatment instructions.
    - No disease counts, forecasts, trends, or risk scores in recommendations.
    - Sole clinical evidence source: supplied IMCI booklet passages.
    - Summary opens with disease count now and next week (injected by post-processor if LLM omits).
    """
    # Build the location label used in instructions
    loc_name = (
        location.get("upazila")
        or location.get("union")
        or location.get("district")
        or location.get("selected_location")
        or "the selected location"
    )

    # Counts context — only for the summary sentence, NOT for recommendations
    counts_line = ""
    if disease_counts:
        c_now = disease_counts.get("count_now", 0)
        c_next = disease_counts.get("count_next_week", 0)
        w_now = disease_counts.get("current_week_total", 0)
        w_next = disease_counts.get("next_week_total", 0)
        counts_line = (
            f"CASELOAD (for summary only — do NOT use in recommendations): "
            f"Day-1={c_now} cases (week total={w_now}); "
            f"Day-7={c_next} cases (week total={w_next})."
        )

    system_prompt = (
        f"You are a government public-health planning assistant for the IMCI programme.\n"
        f"Your output helps district and upazila health managers plan this week's response — "
        f"not clinicians treating individual children.\n\n"
        f"RULES (follow strictly):\n"
        f"1. Use ONLY the IMCI booklet passages provided as evidence. Cite each page used.\n"
        f"2. Generate location-specific, actionable interventions for {loc_name}.\n"
        f"3. NEVER include medication names, dosages, prescriptions, or individual patient treatment.\n"
        f"4. NEVER include disease counts, forecasts, trends, or risk scores inside recommendations.\n"
        f"5. Each category: 3–5 items, one or two sentences each. No lists within items.\n"
        f"6. Set requires_expert_review=true if passages do not support a full recommendation set.\n"
        f"7. Return ONLY valid JSON. No markdown. No text outside the JSON object.\n\n"
        f"JSON SCHEMA (exact field names required):\n"
        f'{{\n'
        f'  "disease": "{disease}",\n'
        f'  "location": <location object>,\n'
        f'  "summary": "<one sentence: count now, count next week, why action is needed>",\n'
        f'  "government_public_health_interventions": ["<3–5 items>"],\n'
        f'  "health_system_readiness": ["<3–5 items>"],\n'
        f'  "community_engagement": ["<3–5 items>"],\n'
        f'  "government_planning_and_monitoring": ["<3–5 items>"],\n'
        f'  "booklet_references": ["Page N"],\n'
        f'  "requires_expert_review": false,\n'
        f'  "expert_review_reason": ""\n'
        f'}}'
    )

    # Format passages with page labels
    formatted_passages = []
    for p in passages:
        page_label = p.get("page") or f"Page {p.get('page_number', '?')}"
        formatted_passages.append(f"[{page_label}]\n{p.get('text', '').strip()}")
    passages_text = "\n\n".join(formatted_passages)

    user_prompt = (
        f"Disease: {disease}\n"
        f"Location: {loc_name} "
        f"(division={location.get('division','')}, district={location.get('district','')}, "
        f"upazila={location.get('upazila','')}, union={location.get('union','')})\n"
        f"{counts_line}\n\n"
        f"IMCI BOOKLET PASSAGES:\n{passages_text}\n\n"
        f"Generate the JSON recommendation object now."
    )

    return system_prompt, user_prompt


def parse_and_validate_response(
    content: str,
    retrieved_pages: List[str],
    expected_disease: str,
    location: Dict[str, Any],
    disease_counts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Parse Meditron's structured JSON response and validate/sanitise all fields.

    New 4-category schema:
      government_public_health_interventions
      health_system_readiness
      community_engagement
      government_planning_and_monitoring

    For backwards compatibility the old 3-category keys are also accepted as fallback
    aliases and migrated to the new schema.

    Validation:
    - 5-item cap per category.
    - Medication / prescription language detection → triggers expert_review flag.
    - Booklet page citation verification against retrieved passages.
    - Summary guaranteed to open with disease count now and next week.
    """
    content = content.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # -----------------------------------------------------------------
    # JSON REPAIR HELPER (handles missing commas + truncated output)
    # -----------------------------------------------------------------
    def _repair_json(text: str) -> str:
        # Pass 1: insert missing commas between adjacent JSON fields
        text = re.sub(r'([\}\]"0-9]|true|false|null)(\s*\n\s*")', r'\1,\2', text)
        # Pass 1b: strip trailing commas before ] or }
        text = re.sub(r',(\s*[\}\]])', r'\1', text)

        # Pass 2: close unclosed structures left by truncation
        stack: List[str] = []
        in_string = False
        escape_next = False
        for ch in text:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch in ('}', ']') and stack:
                stack.pop()
        if in_string:
            text = text.rstrip() + '"'
        if stack:
            text = text.rstrip().rstrip(',') + ''.join(reversed(stack))
        return text

    # -----------------------------------------------------------------
    # PARSE: try raw → slice → truncated; each with and without repair
    # -----------------------------------------------------------------
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in LLM response: {content[:300]}")
        end = content.rfind("}")
        candidates = []
        if end != -1 and end > start:
            candidates.append(content[start : end + 1])
        candidates.append(content[start:])  # truncated fallback

        last_exc: Optional[Exception] = None
        data = None
        for candidate in candidates:
            for attempt in (candidate, _repair_json(candidate)):
                try:
                    data = json.loads(attempt)
                    if attempt != candidate:
                        logger.info("JSON repaired successfully (missing commas / truncation).")
                    break
                except json.JSONDecodeError as exc:
                    last_exc = exc
            if data is not None:
                break

        if data is None:
            snippet = (candidates[0] if candidates else content)[:300]
            logger.warning("JSON repair failed: %s. Snippet: %s", last_exc, snippet)
            raise ValueError(
                f"Could not parse valid JSON from LLM response even after repair: {snippet}"
            ) from last_exc

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------
    def as_list(val: Any, cap: int = 5) -> List[str]:
        """Convert a value to a list of non-empty strings, capped at `cap` items."""
        if isinstance(val, list):
            items = [str(item).strip() for item in val if str(item).strip()]
        elif isinstance(val, str) and val.strip():
            items = [re.sub(r"^[-*•\d.]+\s*", "", ln).strip() for ln in val.split("\n")]
            items = [i for i in items if i]
        else:
            items = []
        return items[:cap]

    # Medication / prescription detection patterns
    MEDICATION_SIGNALS = [
        r"\bmg/kg\b", r"\bper kg\b", r"\boral dose\b", r"\bim dose\b",
        r"\bgive \w+ mg\b", r"\bdose of \d",
        r"\btablets? of\b", r"\bsyrup \d",
        r"\bfirst dose\b", r"\bsecond dose\b",
        r"\bcomplete (?:the )?treatment\b",
        r"\btreatment [abc]:\b",
        r"\bplan [abc]:\b",
        r"\bprescri(?:be|ption)\b",
    ]
    _med_patterns = [re.compile(p, re.IGNORECASE) for p in MEDICATION_SIGNALS]

    def has_medication_language(items: List[str]) -> bool:
        for item in items:
            if any(pat.search(item) for pat in _med_patterns):
                return True
        return False

    def reject_medication_items(items: List[str]) -> List[str]:
        """Remove individual-prescription items; flag if any were removed."""
        clean = [i for i in items if not any(pat.search(i) for pat in _med_patterns)]
        return clean

    # -----------------------------------------------------------------
    # EXTRACT FIELDS — new 4-category schema with old-schema aliases
    # -----------------------------------------------------------------
    disease = data.get("disease") or data.get("disease_name") or expected_disease
    summary = str(data.get("summary") or "").strip()

    # New schema keys (preferred)
    gov_interventions = as_list(
        data.get("government_public_health_interventions")
        # old alias fallback
        or data.get("priority_public_health_measures")
        or data.get("priority_public_health_measure")
    )
    health_system = as_list(
        data.get("health_system_readiness")
        or data.get("health_facility_readiness")
    )
    community = as_list(
        data.get("community_engagement")
        or data.get("caregiver_and_community_communication")
        or data.get("community_communication")
    )
    planning = as_list(
        data.get("government_planning_and_monitoring")
        or []
    )

    # Strip medication language from every category
    medication_found = False
    for cat_items in (gov_interventions, health_system, community, planning):
        if has_medication_language(cat_items):
            medication_found = True
            break

    gov_interventions = reject_medication_items(gov_interventions)
    health_system = reject_medication_items(health_system)
    community = reject_medication_items(community)
    planning = reject_medication_items(planning)

    # -----------------------------------------------------------------
    # CITATIONS: validate against retrieved pages
    # -----------------------------------------------------------------
    raw_refs = as_list(data.get("booklet_references") or data.get("imci_references"), cap=20)
    retrieved_page_numbers = {
        re.search(r"\d+", p).group() for p in retrieved_pages if re.search(r"\d+", p)
    }

    validated_refs: List[str] = []
    unverified_found = False
    for ref in raw_refs:
        m = re.search(r"\d+", ref)
        if m:
            pn = m.group()
            if pn in retrieved_page_numbers:
                validated_refs.append(f"Page {pn}")
            else:
                logger.warning("LLM cited page %s not in retrieved set %s", pn, retrieved_page_numbers)
                validated_refs.append(f"Page {pn} (unverified)")
                unverified_found = True
        elif ref:
            validated_refs.append(ref)

    if not validated_refs and retrieved_pages:
        validated_refs = sorted(set(retrieved_pages))

    # -----------------------------------------------------------------
    # EXPERT REVIEW FLAG
    # -----------------------------------------------------------------
    requires_expert_review = bool(
        data.get("requires_expert_review")
        if "requires_expert_review" in data
        else data.get("expert_review_required", False)
    )
    expert_review_reason = str(data.get("expert_review_reason") or "").strip()

    if unverified_found and not requires_expert_review:
        requires_expert_review = True
        expert_review_reason = (expert_review_reason + " " if expert_review_reason else "") + \
            "Cited booklet references could not all be verified against retrieved passages."

    if medication_found and not requires_expert_review:
        requires_expert_review = True
        expert_review_reason = (expert_review_reason + " " if expert_review_reason else "") + \
            "Response contained medication/dosage language that was removed from recommendations."

    # -----------------------------------------------------------------
    # SUMMARY: ensure it opens with disease count now and next week
    # -----------------------------------------------------------------
    if disease_counts and disease_counts.get("count_now") is not None:
        c_now  = disease_counts.get("count_now", 0)
        c_next = disease_counts.get("count_next_week", 0)
        w_now  = disease_counts.get("current_week_total", 0)
        w_next = disease_counts.get("next_week_total", 0)
        loc_name = (
            location.get("upazila") or location.get("union")
            or location.get("district") or location.get("selected_location")
            or "the selected location"
        )
        has_now  = str(c_now) in summary or "currently" in summary.lower() or "cases now" in summary.lower()
        has_next = str(c_next) in summary or "next week" in summary.lower()

        if not (has_now and has_next):
            count_lead = (
                f"In {loc_name}, the forecasted {disease} caseload is currently {c_now} cases "
                f"({w_now} this week) and is projected at {c_next} cases next week "
                f"({w_next} over the next 7 days). "
                f"Given this burden, the following public-health interventions are recommended: "
            )
            summary = count_lead + summary

    # -----------------------------------------------------------------
    # BUILD OUTPUT
    # -----------------------------------------------------------------
    res: Dict[str, Any] = {
        "disease": disease,
        "location": location,
        "summary": summary,
        # New 4-category schema
        "government_public_health_interventions": gov_interventions,
        "health_system_readiness": health_system,
        "community_engagement": community,
        "government_planning_and_monitoring": planning,
        "booklet_references": sorted(list(dict.fromkeys(validated_refs))),
        "requires_expert_review": requires_expert_review,
        "expert_review_reason": expert_review_reason.strip(),
    }
    if disease_counts:
        res["disease_counts"] = disease_counts

    return res

    return res


def call_meditron(
    system_prompt: str,
    user_prompt: str,
    max_retries: Optional[int] = None,
    timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
    return_latency: bool = False,
) -> Any:
    """
    Send chat completion request to Meditron-27B with retry backoff for rate limits (429).
    Uses configurable timeout and max_tokens to prevent excessive generation delays.
    """
    if not config.MEDITRON_API_KEY:
        raise ValueError("MEDITRON_API_KEY is not configured in .env")

    timeout_val = timeout or config.MEDITRON_TIMEOUT
    max_tokens_val = max_tokens or config.MEDITRON_MAX_TOKENS
    retries = max_retries or config.MEDITRON_MAX_RETRIES

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

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        logger.info(
            "Calling Meditron API at %s (attempt %d/%d, max_tokens=%d, timeout=%ds)...",
            config.MEDITRON_URL,
            attempt,
            retries,
            max_tokens_val,
            timeout_val,
        )
        try:
            start_t = time.perf_counter()
            response = requests.post(
                config.MEDITRON_URL,
                headers=headers,
                json=payload,
                timeout=(10, timeout_val),
                verify=config.MEDITRON_VERIFY_SSL,
            )
            elapsed = time.perf_counter() - start_t
            logger.info("Meditron response received in %.2f seconds (status=%d)", elapsed, response.status_code)

            if response.status_code == 429:
                wait_sec = 20
                try:
                    retry_hdr = response.headers.get("Retry-After")
                    if retry_hdr:
                        wait_sec = max(15, int(retry_hdr))
                except (ValueError, TypeError):
                    pass
                logger.warning(
                    "Meditron busy or rate limited (HTTP 429: %s). Waiting %d seconds before attempt %d/%d...",
                    response.text.strip()[:100],
                    wait_sec,
                    attempt + 1,
                    retries,
                )
                if attempt < retries:
                    time.sleep(wait_sec)
                    continue
                else:
                    raise RuntimeError(
                        f"Meditron API rate limit / busy (HTTP 429): {response.text[:300]}. "
                        "Please wait a moment before retrying."
                    )

            if not response.ok:
                raise RuntimeError(
                    f"Meditron API error (HTTP {response.status_code}): {response.text[:400]}"
                )

            result = response.json()
            choices = result.get("choices")
            if not choices or not choices[0].get("message", {}).get("content"):
                raise ValueError(f"Malformed response structure from Meditron: {result}")

            content = choices[0]["message"]["content"].strip()
            if return_latency:
                return content, elapsed
            return content

        except requests.exceptions.RequestException as exc:
            last_error = exc
            logger.warning("Request error on attempt %d: %s", attempt, exc)
            if attempt < retries:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"Meditron API connection failed after {retries} attempts: {exc}") from exc

    raise RuntimeError(f"Failed to call Meditron API: {last_error}")
