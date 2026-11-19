import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_SCHEMA = os.getenv("DB_SCHEMA", "gold")
DB_TABLE = os.getenv("DB_TABLE", "location_disease_forecasts")
RECOMMENDATION_TABLE = os.getenv("RECOMMENDATION_TABLE", "location_disease_recommendations")

# Meditron API configuration
MEDITRON_URL = (
    os.getenv("MEDITRON_URL")
    or os.getenv("MEDITRON_API_URL")
    or "https://meditron-api.scibd.info/v1/chat/completions"
)
MEDITRON_MODEL = os.getenv("MEDITRON_MODEL", "meditron-27b")
MEDITRON_API_KEY = os.getenv("MEDITRON_API_KEY", "")
MEDITRON_AUTH_SCHEME = os.getenv("MEDITRON_AUTH_SCHEME", "Bearer")
MEDITRON_TIMEOUT = int(os.getenv("MEDITRON_TIMEOUT", "180"))
MEDITRON_CONNECT_TIMEOUT = float(os.getenv("MEDITRON_CONNECT_TIMEOUT", "10"))
MEDITRON_MAX_TOKENS = int(os.getenv("MEDITRON_MAX_TOKENS", "400"))
MEDITRON_MAX_RETRIES = int(os.getenv("MEDITRON_MAX_RETRIES", "3"))
MEDITRON_RETRY_BASE_DELAY = float(os.getenv("MEDITRON_RETRY_BASE_DELAY", "1"))
MEDITRON_RETRY_MAX_DELAY = float(os.getenv("MEDITRON_RETRY_MAX_DELAY", "20"))
MEDITRON_RETRY_JITTER = float(os.getenv("MEDITRON_RETRY_JITTER", "0.25"))

_verify_ssl_env = os.getenv("MEDITRON_VERIFY_SSL", "true").strip()
if _verify_ssl_env.lower() in ("false", "0", "no"):
    MEDITRON_VERIFY_SSL = False
elif _verify_ssl_env.lower() in ("true", "1", "yes"):
    MEDITRON_VERIFY_SSL = True
else:
    MEDITRON_VERIFY_SSL = _verify_ssl_env

# PDF knowledge base settings
IMCI_PDF_PATH = os.getenv(
    "IMCI_PDF_PATH",
    str(BASE_DIR / "IMCI_CHART_BOOKLET.pdf")
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", os.getenv("RETRIEVAL_TOP_K", "5")))
RETRIEVAL_TOP_K = RAG_TOP_K
RAG_MAX_PASSAGE_CHARS = int(os.getenv("RAG_MAX_PASSAGE_CHARS", "450"))
MAX_CONTEXT_CHARS = int(
    os.getenv(
        "MAX_CONTEXT_CHARS",
        str(RAG_TOP_K * (RAG_MAX_PASSAGE_CHARS + 32)),
    )
)

# Persistence cache for the vector index
INDEX_CACHE_PATH = os.getenv(
    "INDEX_CACHE_PATH",
    str(BASE_DIR / ".cache" / "imci_rag_index.pkl")
)

# Database column mappings
FORECAST_COUNT_COLUMN = os.getenv("FORECAST_COUNT_COLUMN", "").strip()

COLUMN = {
    "location_type": os.getenv("LOCATION_TYPE_COLUMN", "hierarchy_level"),
    "location_key": os.getenv("LOCATION_KEY_COLUMN", "location_key"),
    "division": os.getenv("DIVISION_COLUMN", "division_name"),
    "district": os.getenv("DISTRICT_COLUMN", "district_name"),
    "upazila": os.getenv("UPAZILA_COLUMN", "upazila_name"),
    "union": os.getenv("UNION_COLUMN", "union_name"),
    "disease_code": os.getenv("DISEASE_CODE_COLUMN", "disease_code"),
    "disease_name": os.getenv("DISEASE_NAME_COLUMN", "disease_name"),
    "forecast_date": os.getenv("FORECAST_DATE_COLUMN", "forecast_date"),
    "risk": os.getenv("RISK_COLUMN", "predicted_severity_label"),
    "trend": os.getenv("TREND_COLUMN", "risk_trend_label"),
}

MAX_SOURCE_ROWS = int(os.getenv("MAX_SOURCE_ROWS", "10000"))
