
import os
import time
import json
import requests
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(
    r"C:\PROJECTS\IMCI\forecast_recommendation\.env"
)

MEDITRON_URL = os.getenv("MEDITRON_URL")
MEDITRON_MODEL = os.getenv("MEDITRON_MODEL", "meditron-27b")
MEDITRON_API_KEY = os.getenv("MEDITRON_API_KEY")
AUTH_SCHEME = os.getenv("MEDITRON_AUTH_SCHEME", "Bearer")
VERIFY_SSL = os.getenv("MEDITRON_VERIFY_SSL", "true").lower() == "true"

if not MEDITRON_URL or not MEDITRON_API_KEY:
    raise RuntimeError(
        "MEDITRON_URL or MEDITRON_API_KEY missing from .env"
    )

headers = {
    "Authorization": f"{AUTH_SCHEME} {MEDITRON_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MEDITRON_MODEL,
    "messages": [
        {
            "role": "system",
            "content": "You are a concise public-health assistant."
        },
        {
            "role": "user",
            "content": (
                "Write exactly five short sentences about "
                "the importance of clean drinking water. "
                "Use plain English and no bullet points."
            )
        }
    ],
    "temperature": 0,
    "max_tokens": 100,
    "stream": False
}

print("=" * 60)
print("MEDITRON-27B PERFORMANCE TEST")
print("=" * 60)
print("Model:", MEDITRON_MODEL)
print("Max output tokens:", payload["max_tokens"])
print("Starting API request...")
print()

start = time.perf_counter()

try:
    response = requests.post(
        MEDITRON_URL,
        headers=headers,
        json=payload,
        timeout=(5, 120),
        verify=VERIFY_SSL
    )

    elapsed = time.perf_counter() - start

    print("HTTP status:", response.status_code)
    print("Total elapsed:", round(elapsed, 2), "seconds")
    print()

    response.raise_for_status()
    data = response.json()

    choices = data.get("choices", [])
    if choices:
        choice = choices[0]
        message = choice.get("message", {})

        print("GENERATED RESPONSE:")
        print(message.get("content", ""))
        print()
        print("Finish reason:", choice.get("finish_reason"))

    usage = data.get("usage", {})
    timings = data.get("timings", {})

    print()
    print("TOKEN USAGE:")
    print(json.dumps(usage, indent=2))

    print()
    print("SERVER TIMINGS:")
    print(json.dumps(timings, indent=2))

    completion_tokens = usage.get("completion_tokens", 0)

    if completion_tokens and elapsed > 0:
        print()
        print("OVERALL THROUGHPUT:")
        print(
            round(completion_tokens / elapsed, 2),
            "completion tokens/second (including overhead)"
        )

    predicted_per_second = timings.get("predicted_per_second")

    if predicted_per_second:
        print(
            "Server-reported generation speed:",
            round(float(predicted_per_second), 2),
            "tokens/second"
        )

    print()
    print("=" * 60)

    if elapsed <= 30:
        print("RESULT: Completed within 30 seconds")
    else:
        print("RESULT: Exceeded the 30-second target")

    print("=" * 60)

except requests.exceptions.Timeout:
    elapsed = time.perf_counter() - start
    print("ERROR: Meditron request timed out")
    print("Elapsed:", round(elapsed, 2), "seconds")

except requests.exceptions.RequestException as exc:
    elapsed = time.perf_counter() - start
    print("ERROR:", type(exc).__name__)
    print("Elapsed:", round(elapsed, 2), "seconds")
    print(str(exc))

except ValueError as exc:
    print("ERROR: Invalid JSON response")
    print(str(exc))
