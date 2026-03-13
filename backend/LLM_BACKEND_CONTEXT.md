# Aravalli Intelligence — Backend System Context for LLMs

## Target Audience
Another Large Language Model (LLM) acting as a coding assistant or reviewer.

## Purpose
Provide total, exhaustive context of the Python FastAPI Backend architecture, the 4-stage Machine Learning pipeline, data models, tunable parameters, and the reasoning behind each implementation choice.

---

## 1. System Overview & Tech Stack
The backend is a production-ready REST API designed for **Geospatial Ecological Monitoring** of the Aravalli Range. It detects deforestation, encroachment, and illegal mining using 84 months of satellite derived index data (NDVI, NDBI, BSI, Nightlight).

**Core Tech Stack (from `requirements.txt`):**
- **Framework:** `FastAPI` + `Uvicorn` (runs on port 8000).
- **Data Processing:** `pandas` (heavy use of vectorized groupby/rolling operations) + `numpy`.
- **Machine Learning:** `scikit-learn` (Ensemble: IsolationForest, DBSCAN, LOF, KMeans).
- **Geospatial/Stats:** `libpysal` / `esda` (Moran's I for spatial autocorrelation).
- **Time-series:** `ruptures` (Pelt algorithm for changepoint detection).
- **LLM Integration:** `requests` / `groq` (OpenAI-compatible endpoints for LLM field report generation).

---

## 2. Configuration (`config.yaml` & `config_loader.py`)
The system is entirely configurable via `config.yaml`, which contains **84 tunable parameters** across 10 groups (Data, Baseline, Ensemble, Drift, Filters, Classification, Normalization, Spatial, Changepoint, Reports).

### Key Concepts:
1. **Single Source of Truth:** `config.yaml` is never modified at runtime by the backend.
2. **Overrides:** The POST `/api/analyze` endpoint accepts `overrides`. The `config_loader.deep_merge` deeply merges these over the base config.
3. **Auto-Correction & Validation:** `validate_config` parses the config before a pipeline run.
   - **Corrections:** Normalize weights to sum to 1.0, clamp contamination percentages, ensure DSR transition > normal.
   - **Warnings:** Flags single-detector setups, missing filters, or low smoothing windows.
   - **Rejections (HTTP 422):** e.g., if all ensemble methods are disabled.

---

## 3. The ML Pipeline (`pipeline/` directory)
The ML logic is strictly separated into 4 sequential stages, orchestrated synchronously when a user calls `POST /api/analyze`. It runs under an `asyncio.Lock()` to ensure only 1 pipeline executes at once to save memory. 

### Stage 1: Ingestion (`ingest.py`)
Generates/loads 84 months (Jan 2019 - Dec 2025) of data for 1000 zones.
- **Modes:** 
  1. `real_file` (Offline CSV matching)
  2. `synthetic` (Generates perfectly controlled data for testing using `numpy.random`. Injects mathematically pure degradation events for deforestation, encroachment, and mining.)
  3. `gee_live` (Stub for Google Earth Engine fetch).
- **Output:** `raw_data.csv` (84,000 rows × 9 columns).

### Stage 2: Feature Engineering (`features.py`)
Extracts 19 dense features from the timeseries per zone (reduces 84,000 rows to 1,000 rows representing the *current* state).
- **Smoothing:** 3-month rolling mean applied to NDVI/NDBI/BSI to strip atmospheric/sensor noise but preserve monsoon peaks.
- **Adaptive Baseline:** Computes a 6-year historical baseline *per calendar month* (to account for seasonality). 
- **Temporal Features:** Computes short slope (last 3 mos), long slope (7 years), acceleration (short - long), volatility ratio, and consecutive declines counting backwards.
- **Spatial Features (KNN):** Uses `sklearn.neighbors.NearestNeighbors` (k=8) to compute `local_anomaly_score` (deviation from neighbors), `regional_health`, and `spatial_gradient`.
- **Changepoint Detection:** Uses `ruptures.Pelt` with RBF model to find structural breaks in the timeseries.
- **Global Moran's I:** Uses `libpysal` (with manual fallback) to compute spatial autocorrelation to see if threats are clustered globally.

### Stage 3: Detection & Classification (`detect.py`)
This is the core decision engine. It follows a rigorous 13-step flow:
1. **DSR (Deviation from Seasonal Referent):** Mathematical proof separating real drops from seasonal ones. Converts delta into a Z-score and p-value.
2. **Drift Score:** Composite severity (1.0 to 10.0) combining NDVI departure (35%), temporal persistence (25%), spatial isolation (20%), and seasonal proof derived from DSR (20%).
3. **ML Ensemble:**
   - **IsolationForest (Weight 35%):** Good at finding path-based unusual multidimensional signatures.
   - **DBSCAN (Weight 30%):** Flags density-based outliers.
   - **LocalOutlierFactor (Weight 25%):** Density vs local neighbors.
   - **KMeans (Weight 10%):** Zones in the smallest cluster are flagged.
   - **Voting:** Weighted sum of boolean votes. If `sum >= min_weighted_score` (default 0.5), it's initially an anomaly.
4. **Temporal Confirmation:** Scales confidence based on consecutive drops (e.g., < 3 drops halves confidence, >= 12 guarantees confidence).
5. **Post-Detection Filters:** 6 explicit filters remove false positives:
   - F1: Drop if DSR is "seasonal_normal".
   - F2: Drop isolated anomalies in highly healthy regions.
   - F3: Cap confidence if the duration is too short.
   - F4: Downgrade severity if `recovery_signal` is positive.
   - F5: Drop anomalies occurring only during monsoon months with positive short slopes.
   - F6: Drop if final confidence < 30%.
6. **Classify:** Hardcoded logic matching spectral signatures:
   - **Encroachment:** NDVI drops, NDBI rises, Nightlight rises.
   - **Mining:** NDVI drops, BSI (Bare Soil) rises, NDBI flat.
   - **Deforestation:** Large NDVI drop + stable soil/urban indexes.
7. **Outputs:** `detections.csv` and `detected_zones.geojson`.

### Stage 4: AI Explanation Layer (`explain.py`)
Provides plain-language (<200 words) field reports for each detected threat based on the data.
- **Fallback Chain:** 
  1. Primary LLM (`openai/gpt-oss-120b` via Groq) - Reasoning model.
  2. Fallback LLM (`llama-3.3-70b` via Groq) - Faster, text model.
  3. Deterministic Template (Guaranteed string generation if APIs fail or timeout).
- **Caching:** Caches responses in `cached_reports.json` so re-requesting an already-explained zone is instantaneous.

---

## 4. FastAPI Endpoints (`main.py`)
11 Endpoints, heavily relying on pandas loaded into memory caches:
- `GET /api/health`: Healthcheck, returns data mode (synthetic vs real).
- `GET /api/zones/geojson`: Serves the detected geometry.
- `GET /api/zones`: Paginated table view of 1000 zones (can filter `detected_only=true`). Uses `_sanitize_dict` to block `NaN`/`Infinity` JSON parsing errors.
- `GET /api/zones/{zone_id}`: Detail view. Merges in the LLM field report from the cache.
- `GET /api/zones/{zone_id}/timeseries`: Feeds frontend Echarts. Pre-computes 84 months of data, baseline mean/std for shading bands, and changepoint indexes.
- `GET /api/zones/{zone_id}/importance`: Top 3 driving features.
- `GET /api/zones/{zone_id}/neighbors`: Returns 8 spatially closest zones (computes Euclidean distance on Lat/Lon dynamically).
- `GET /api/summary`: Powers the "AccuracyBar" Dashboard. Provides precision, recall, drift, and breakdown of threat types.
- `GET /api/accuracy`: JSON accuracy report dumped by `detect.py`.
- `GET /api/config`: Reads `config.yaml`, scrubs the API keys, and sends default frontend config.
- `POST /api/analyze`: Triggers the pipeline. Takes `overrides` dict. Has a config hashing mechanism: if the exact same config is requested, it skips the costly pipeline run and returns cached results.

---

## 5. Key Architecture & Design Decisions (The "Why")
- **Why Synchronous Pipeline?** The ML pipeline relies heavily on Pandas and Scikit-Learn. It runs in `asyncio.get_event_loop().run_in_executor` to avoid blocking the API, but only 1 pipeline can run at once (`_pipeline_lock`) to prevent OOM errors on cheap servers.
- **Why Circular Encoding for Months?** `explain.py` uses `month_sin` and `month_cos`. It preserves the fact that December (12) and January (1) are chronologically adjacent, which a linear scalar (1 vs 12) destroys.
- **Why LLM Fallback Chain?** External APIs rate limit or die. To prevent the frontend from hanging, it falls back to a deterministic string builder function `template_report` if the LLMs time out or HTTP 500 error.

---

## 6. How To Assist
If you (the LLM reading this) are asked to modify the backend:
1. **Never use `pd.DataFrame.iterrows` for mutations.** Use vectorized `.loc` or `.where` (as heavily used in `features.py` and `detect.py`).
2. **Never change config structure arbitrarily.** `config.yaml` maps 1:1 with frontend UI sliders and deep_merge logic.
3. **Handle NaN in JSON.** FastAPI `JSONResponse` crashes on `NaN` or `Infinity`. Always use the `_sanitize_dict` helper found in `main.py`.
4. **Follow the Logger.** The entire pipeline is heavily instrumented with `logger.info()`. Do not use `print()`.
