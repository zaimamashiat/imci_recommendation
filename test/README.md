# IMCI Forecast RAG — single-file application

All application logic is in `app.py`. Dependencies and environment settings are separate configuration files.

## Install and run (Windows PowerShell)

```powershell
cd C:\PROJECTS\IMCI\forecast_recommendation\imci_single_codebase
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env with valid database password, rotated API key, and local PDF path
python app.py
```

Open http://127.0.0.1:8000/docs.

### POST /generate
```json
{"location_type":"union","location_key":"union:10|06|36|352","forecast_date":"2027-03-02","persist":true}
```
`location_type` accepts `division`, `district`, `upazila`, and `union`.

To generate for several hierarchy levels in one API call, use
`POST /generate/batch` (the same batch body is also accepted by `POST /generate`):

```json
{
  "forecast_date": "2027-03-02",
  "persist": true,
  "locations": [
    {"location_type": "division", "location_key": "division:10"},
    {"location_type": "district", "location_key": "district:10|06"},
    {"location_type": "upazila", "location_key": "upazila:10|06|36"},
    {"location_type": "union", "location_key": "union:10|06|36|352"}
  ]
}
```

Replace the example keys with the exact `location_key` values in your forecast
table. Each location is processed independently; the response reports success
or an error for each requested hierarchy.

`/generate` retrieves passages from the configured IMCI PDF and asks Meditron
to create recommendations for the three highest-priority forecast diseases.
It validates that the output includes each disease and only cites pages
retrieved for that disease, then optionally saves the recommendations. It does
not require entries in the guidance approval cache. The response identifies
`source: "imci-pdf-rag-meditron"` and sets `clinical_review_required` to `true`;
have a qualified clinician review generated guidance before using it for care.
Each `data_summary` uses the matching hierarchy name from the forecast row (for
example, `For union Hijla on ...`) instead of the coded `location_key`.

The request includes PDF retrieval and model generation, so configure
`REQUEST_BUDGET_SECONDS` to exceed the model timeout and any configured
`MEDITRON_MIN_INTERVAL_SECONDS` cooldown. By default, the budget is derived
from those settings with an additional 30 seconds for retrieval and overhead.

### GET /recommendations
`/recommendations?location_type=union&location_key=union%3A10%7C06%7C36%7C352&forecast_date=2027-03-02`

### Notes
- Source and destination tables must already exist.
- Only the latest forecast row for each disease at the requested level, location and date is used.
- Clinical guidance is generated from extracted IMCI PDF passages; scanned PDFs need OCR beforehand.
- Source PDF page numbers are physical PDF page positions and may differ from printed booklet numbers.
- The program checks cited page membership, not clinical correctness; human clinical review is required before use.
- It uses sentence-transformer embeddings for semantic retrieval from the booklet.
- Repeated writes are replaced in a transaction; concurrent requests need a database uniqueness constraint for stronger deduplication.
- Do not disable SSL certificate verification in production.
