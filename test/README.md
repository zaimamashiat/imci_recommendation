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
`location_type` accepts `district`, `upazila`, `union`.

### GET /recommendations
`/recommendations?location_type=union&location_key=union%3A10%7C06%7C36%7C352&forecast_date=2027-03-02`

### Notes
- Source and destination tables must already exist.
- Only the latest forecast row for each disease at the requested level, location and date is used.
- Clinical guidance is generated from extracted IMCI PDF passages; scanned PDFs need OCR beforehand.
- Source PDF page numbers are physical PDF page positions and may differ from printed booklet numbers.
- The program checks cited page membership, not clinical correctness; human clinical review is required before use.
- It uses TF-IDF retrieval, which is lexical RAG rather than embedding-based semantic retrieval.
- Repeated writes are replaced in a transaction; concurrent requests need a database uniqueness constraint for stronger deduplication.
- Do not disable SSL certificate verification in production.
