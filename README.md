# IMCI Location-based Recommendation Engine

This starter backend generates a location-specific recommendation report from the existing PostgreSQL forecast table, retrieves relevant passages from `IMCI_CHART_BOOKLET.pdf` using RAG, calls the Meditron-27B chat-completions API, and saves the generated recommendations into a new table:

`ml_analytics_v5.gold.location_disease_recommendations`

The existing dashboard analytics are not changed. The dashboard can call the generate endpoint when recommendations need refreshing, then read the saved output from PostgreSQL.

## 1. Architecture

1. Read forecast rows from `gold.location_disease_forecasts`.
2. Aggregate the explicitly configured forecast-count column by disease and forecast date.
3. Retrieve relevant IMCI booklet passages using local sentence-transformer embeddings.
4. Send the selected location, aggregated forecast data and retrieved passages to Meditron-27B.
5. Ask Meditron for one short data-summary paragraph and disease-wise recommendations in three categories.
6. Save the generated text and source references in `gold.location_disease_recommendations`.
7. The dashboard reads the saved recommendation records.

## 2. Requirements

- Python 3.10 or newer (64-bit recommended).
- PostgreSQL access to the `ml_analytics_v5` database.
- The IMCI PDF available on the computer/server running this API.
- Network access to the Meditron API and to download the embedding model the first time it runs.
- A Meditron API key only if the API provider requires one.

## 3. Windows installation

Open Command Prompt and run:

```bat
cd /d C:\PROJECTS\IMCI\forecast_recommendation
```

Extract this ZIP there, so the folder looks like:

```text
C:\PROJECTS\IMCI\forecast_recommendation\imci_recommendation_backend\
```

Then:

```bat
cd /d C:\PROJECTS\IMCI\forecast_recommendation\imci_recommendation_backend
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` in Notepad or VS Code.

## 4. Configure `.env`

Set your PostgreSQL connection. Example:

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ml_analytics_v5
```

Use the actual PostgreSQL password. Do not commit `.env` or share it because it contains credentials.

Set the PDF path:

```env
IMCI_PDF_PATH=C:/PROJECTS/IMCI/forecast_recommendation/IMCI_CHART_BOOKLET.pdf
```

Set the Meditron endpoint and model:

```env
MEDITRON_URL=https://meditron-api.scibd.info/v1/chat/completions
MEDITRON_MODEL=meditron-27b
MEDITRON_API_KEY=
```

Only fill `MEDITRON_API_KEY` if the API provider gave you a key. The code uses Bearer authentication by default; confirm this with the API provider if authentication is required.

### Important: identify the forecast count column

The source table contains several numeric fields. Do not assume any of them is the disease count. Start the API as described below and open:

`http://127.0.0.1:8000/schema`

Or run this SQL in the `ml_analytics_v5` database:

```sql
SELECT ordinal_position, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'gold'
  AND table_name = 'location_disease_forecasts'
ORDER BY ordinal_position;
```

After confirming which field actually stores the forecasted case count, set it in `.env`, for example:

```env
FORECAST_COUNT_COLUMN=actual_forecast_count_column
```

Replace the example with the real column name. If the forecast is represented by multiple horizon columns, the aggregation logic in `app/service.py` must be adjusted to select the intended horizon(s). Do not sum unrelated metrics or multiple horizons together.

Also verify the source column mappings in `.env`. The defaults are common names such as `division_name`, `disease_name`, `risk_level`, and `trend`; change them if your table uses different names.

## 5. Start the API

From the project directory, with the virtual environment active:

```bat
uvicorn app.main:app --reload
```

The first startup downloads/loads the sentence-transformer embedding model and indexes the PDF in memory. This can take a few minutes. The PDF must contain extractable text; scanned pages may need OCR first.

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

Health check:

`http://127.0.0.1:8000/health`

## 6. Generate and save recommendations

Use `POST /recommendations/generate`.

Example request for an upazila:

```json
{
  "location_level": "upazila",
  "location_value": "Hijla"
}
```

Example request for a union and a single disease:

```json
{
  "location_level": "union",
  "location_value": "Memania",
  "disease": "Malnutrition"
}
```

Valid `location_level` values are `division`, `district`, `upazila`, and `union`.

On startup, the API creates `gold.location_disease_recommendations` if it does not exist. The generate endpoint then:

- reads source forecast records for the selected location;
- calculates totals using only `FORECAST_COUNT_COLUMN`;
- retrieves relevant IMCI text;
- calls Meditron;
- saves the generated summary and recommendation categories.

The output table has one record per location key, disease code and forecast date. Re-running generation for the same combination updates the existing record rather than inserting a duplicate.

## 7. Read saved recommendations for the dashboard

Use `POST /recommendations/read`.

Request:

```json
{
  "location_level": "upazila",
  "location_value": "Hijla"
}
```

The response includes saved fields such as:

- `data_summary`
- `disease_name`
- `forecasted_count`
- `risk_level`
- `trend`
- `priority`
- `priority_public_health_measure`
- `health_facility_readiness`
- `community_communication`
- `imci_references`

Your dashboard should display the summary paragraph first, followed by a disease heading and the three recommendation sections.

## 8. Example dashboard JavaScript

Generate/update recommendations when an authorised user chooses to refresh them:

```javascript
const generated = await fetch("http://127.0.0.1:8000/recommendations/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    location_level: "upazila",
    location_value: "Hijla"
  })
});

if (!generated.ok) throw new Error(await generated.text());
const generationResult = await generated.json();

const saved = await fetch("http://127.0.0.1:8000/recommendations/read", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    location_level: "upazila",
    location_value: "Hijla"
  })
});

if (!saved.ok) throw new Error(await saved.text());
const dashboardRecommendations = await saved.json();
console.log(dashboardRecommendations.recommendations);
```

In a production dashboard, it is usually better to generate recommendations through a scheduled/background job or an authorised refresh action, not on every page load. The page should normally read the already-saved recommendations.

## 9. Verify the new table in PostgreSQL

```sql
SELECT *
FROM gold.location_disease_recommendations
ORDER BY generated_at DESC
LIMIT 100;
```

## 10. Important limitations and production safeguards

- **Count semantics:** The configured count column must genuinely represent the forecasted count. If source rows contain repeated horizons, the aggregation must be changed to avoid double-counting.
- **Forecast vs observed:** The prompt instructs the model to describe these values as forecasts, not observed cases.
- **Clinical safety:** AI-generated content is a draft. Qualified IMCI/public-health professionals must validate it before operational use. Do not rely on the model to invent or determine clinical thresholds, medicine doses or treatment protocols.
- **RAG citations:** The service asks the model to return PDF page references from retrieved passages. Validate those references against the booklet before publication.
- **RAG persistence:** This starter builds embeddings in memory at startup. For a production deployment, persist the embeddings in pgvector or another vector store and index the booklet once.
- **Security:** Add authentication/authorization, HTTPS, CORS restrictions, rate limiting, logging, and secrets management before exposing the API beyond localhost or a trusted network.
- **Database access:** Use a dedicated PostgreSQL service account with only the permissions needed to read the forecast table and create/update the recommendation table.
- **API compatibility:** The code expects an OpenAI-compatible response with `choices[0].message.content`. Confirm the provider's exact authentication and response format.
