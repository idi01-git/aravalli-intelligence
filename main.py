"""
Aravalli Intelligence — FastAPI Backend
main.py

Production-ready REST API serving 11 endpoints for geospatial ecological
monitoring. Implements PRD Section 13 exactly.

Framework: FastAPI + Uvicorn
Author:    Team BIOBYTES
PRD Ref:   Section 13 — Backend API
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path as PathlibPath
from typing import Any, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config_loader import deep_merge, load_config, validate_config

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("aravalli_api")

# ─────────────────────────────────────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR           = PathlibPath("output")
DETECTIONS_CSV       = OUTPUT_DIR / "detections.csv"
CACHED_REPORTS_JSON  = OUTPUT_DIR / "cached_reports.json"
ACCURACY_REPORT_JSON = OUTPUT_DIR / "accuracy_report.json"
GEOJSON_PATH         = OUTPUT_DIR / "detected_zones.geojson"
TIMESERIES_DIR       = OUTPUT_DIR / "zone_timeseries"

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline mutex (only one pipeline run at a time)
# ─────────────────────────────────────────────────────────────────────────────
_pipeline_lock    = asyncio.Lock()
_pipeline_running = False
_current_config_hash: str | None = None

# ─────────────────────────────────────────────────────────────────────────────
# In-memory cache for hot data
# ─────────────────────────────────────────────────────────────────────────────
_detections_cache: pd.DataFrame | None = None
_reports_cache:    dict[str, Any] | None = None
_detections_hash: str | None = None


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a hash of the relevant config parameters for caching."""
    import hashlib
    # Only hash relevant parameters that affect results
    relevant_keys = ["data", "ensemble", "drift", "filters", "classification", "dsr", "baseline"]
    relevant_config = {k: config.get(k) for k in relevant_keys if k in config}
    config_str = json.dumps(relevant_config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_detections(force: bool = False) -> pd.DataFrame:
    """Load detections.csv with in-memory caching."""
    global _detections_cache
    if _detections_cache is None or force:
        if not DETECTIONS_CSV.exists():
            raise HTTPException(
                status_code=503,
                detail="detections.csv not found. Run pipeline first.",
            )
        _detections_cache = pd.read_csv(DETECTIONS_CSV)
        logger.info("Loaded detections.csv (%d rows)", len(_detections_cache))
    return _detections_cache.copy()


import math


def _sanitize_value(v):
    """Convert NaN/inf to None for JSON compatibility."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v


def _sanitize_dict(d: dict) -> dict:
    """Sanitize all values in dict for JSON compatibility."""
    return {k: _sanitize_value(v) for k, v in d.items()}


def _load_reports() -> dict[str, Any]:
    """Load cached_reports.json with in-memory caching."""
    global _reports_cache
    if _reports_cache is None:
        if not CACHED_REPORTS_JSON.exists():
            return {}
        with CACHED_REPORTS_JSON.open("r", encoding="utf-8") as fh:
            _reports_cache = json.load(fh)
    return _reports_cache


def _invalidate_caches() -> None:
    """Flush in-memory caches after a pipeline run."""
    global _detections_cache, _reports_cache, _detections_hash, _current_config_hash
    _detections_cache = None
    _reports_cache = None
    _detections_hash = None
    _current_config_hash = None


def _run_pipeline(config: dict[str, Any]) -> None:
    """Execute: ingest → features → detect → explain."""
    import sys
    sys.path.insert(0, str(PathlibPath(__file__).resolve().parent))

    from pipeline.ingest import run_ingestion   # type: ignore[import]
    from pipeline.features import run_features  # type: ignore[import]
    from pipeline.detect import run_detection   # type: ignore[import]
    from pipeline.explain import run_explain    # type: ignore[import]

    logger.info("Pipeline: ingest starting")
    raw_df = run_ingestion(config)
    logger.info("Pipeline: features starting")
    features_df = run_features(raw_df, config)
    logger.info("Pipeline: detect starting")
    run_detection(config)
    logger.info("Pipeline: explain starting")
    run_explain(config)
    logger.info("Pipeline: complete")
    _invalidate_caches()


# ─────────────────────────────────────────────────────────────────────────────
# App lifespan (startup + shutdown)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────────────
    logger.info("Aravalli Intelligence API starting up ...")
    required = [DETECTIONS_CSV, CACHED_REPORTS_JSON]
    for fp in required:
        if fp.exists():
            logger.info("  [OK] %s", fp.name)
        else:
            logger.warning("  [MISSING] %s — run the pipeline first", fp.name)
    yield
    # ── shutdown ─────────────────────────────────────────────────────────────
    logger.info("Aravalli Intelligence API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App initialization
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aravalli Intelligence",
    description="Ecological monitoring system for Aravalli Range",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers — always JSON, never stack traces
# ─────────────────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning("HTTP %s: %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": "HTTP_ERROR"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request model
# ─────────────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    sensitivity: Optional[str] = "medium"
    overrides:   Optional[dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: GET /api/health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health", summary="Health check")
def get_health():
    """Returns API health, version, and pipeline status."""
    cfg = load_config()
    zone_count = 0
    if DETECTIONS_CSV.exists():
        zone_count = len(pd.read_csv(DETECTIONS_CSV))
    logger.info("GET /api/health")
    return {
        "status":       "healthy",
        "version":      "1.0.0",
        "project_name": "Aravalli Intelligence",
        "zone_count":   zone_count,
        "data_mode":    _get_actual_data_mode(cfg),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }


def _get_actual_data_mode(cfg: dict[str, Any]) -> str:
    """Get the actual data mode used - checks output folder for mode tracking."""
    mode_file = OUTPUT_DIR / "data_mode.txt"
    if mode_file.exists():
        with mode_file.open("r") as f:
            return f.read().strip()
    return cfg.get("data", {}).get("mode", "synthetic")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 6: GET /api/zones/geojson  ← must be declared BEFORE /{zone_id}
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/zones/geojson", summary="GeoJSON of detected threat zones")
def get_geojson():
    """Returns all detected threat zones as a GeoJSON FeatureCollection."""
    logger.info("GET /api/zones/geojson")
    if not GEOJSON_PATH.exists():
        raise HTTPException(status_code=404, detail="GeoJSON file not found")
    with GEOJSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: GET /api/zones
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/zones", summary="List all monitoring zones")
def get_zones(
    page:          int  = Query(1,     ge=1,            description="Page number"),
    size:          int  = Query(100,   ge=1,   le=1000, description="Results per page"),
    detected_only: bool = Query(False,                  description="Only confirmed threats"),
):
    """Returns paginated list of all 1,000 monitoring zones."""
    logger.info("GET /api/zones - page=%d size=%d detected_only=%s", page, size, detected_only)
    df = _load_detections()

    if detected_only:
        df = df[df["is_anomaly"] == 1]

    total     = len(df)
    start_idx = (page - 1) * size
    end_idx   = start_idx + size
    paginated = df.iloc[start_idx:end_idx]

    records = paginated.to_dict("records")
    sanitized_records = [_sanitize_dict(record) for record in records]

    return {
        "total":     total,
        "page":      page,
        "page_size": size,
        "pages":     max(1, (total + size - 1) // size),
        "data":      sanitized_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3: GET /api/zones/{zone_id}
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/zones/{zone_id}", summary="Get zone details + AI report")
def get_zone(zone_id: str = Path(..., description="Zone ID, e.g. zone_0001")):
    """Returns full zone data including AI-generated field report."""
    logger.info("GET /api/zones/%s", zone_id)
    df   = _load_detections()
    zone = df[df["zone_id"] == zone_id]
    if zone.empty:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")

    zone_dict = _sanitize_dict(zone.iloc[0].to_dict())

    reports = _load_reports()
    zone_dict["ai_report"]     = reports.get(zone_id, {}).get("text", "")
    zone_dict["report_source"] = reports.get(zone_id, {}).get("source", "unknown")
    return zone_dict


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4: GET /api/zones/{zone_id}/timeseries
# PRD §18 (Trend Analysis chart): smoothed NDVI, baseline, ±1std shading band,
# changepoint markers, event onset vertical line.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/zones/{zone_id}/timeseries", summary="84-month timeseries for zone")
def get_timeseries(zone_id: str = Path(..., description="Zone ID")):
    """
    Returns 84-month spectral index timeseries for frontend ECharts charts.

    PRD §18 mandates: smoothed NDVI line, baseline line, ±1 std shading band,
    red vertical line at anomaly onset, red dots at changepoints.
    The `metadata` block provides all values the chart needs pre-computed.
    """
    logger.info("GET /api/zones/%s/timeseries", zone_id)
    ts_file = TIMESERIES_DIR / f"{zone_id}.csv"
    if not ts_file.exists():
        raise HTTPException(
            status_code=404, detail=f"Timeseries for {zone_id} not found"
        )

    df = pd.read_csv(ts_file)

    # ── Add human-readable month labels and 0-based month index ────────────
    # Input timestamps are ISO strings like "2019-01".
    # ECharts X-axis wants labels like "Jan 2019".
    def _fmt_month(ts: str) -> str:
        try:
            dt = pd.to_datetime(ts, format="%Y-%m")
            return dt.strftime("%b %Y")  # e.g. "Jan 2019"
        except Exception:
            return ts

    df["month_label"] = df["timestamp"].apply(_fmt_month)
    df["month_index"] = range(len(df))  # 0..83 — ECharts uses this for x-axis data index

    # ── Compute NDVI baseline statistics for chart shading ─────────────────
    # PRD §18: "Shaded band: +/- 1 std from baseline"
    # Degrade gracefully: prefer ndvi_baseline > ndvi > any numeric column > zeros
    baseline_mean, baseline_std = 0.0, 0.05
    for _col in ("ndvi_baseline", "ndvi"):
        if _col in df.columns:
            _vals = df[_col].dropna()
            if len(_vals) > 0:
                baseline_mean = float(_vals.mean())
                baseline_std  = float(_vals.std()) if len(_vals) > 1 else 0.05
            break
    healthy_band_min = round(max(0.0, baseline_mean - baseline_std), 4)
    healthy_band_max = round(min(1.0, baseline_mean + baseline_std), 4)

    # ── Derive event onset index from detections.csv ───────────────────────
    # PRD §18: "Red vertical line at detected event start"
    # event_onset = the month index where consecutive declines started.
    # We estimate: onset = total_months - consecutive_declines.
    event_onset_month_index: Optional[int] = None
    changepoint_months: list[int] = []
    try:
        det_df   = _load_detections()
        zone_row = det_df[det_df["zone_id"] == zone_id]
        if not zone_row.empty:
            row = zone_row.iloc[0]
            consec = int(row.get("consecutive_declines", 0) or 0)
            n_cp   = int(row.get("n_changepoints",       0) or 0)
            if consec > 0:
                event_onset_month_index = max(0, len(df) - consec)
            # Distribute changepoints evenly as approximate markers
            # (exact changepoint months are not stored individually in detections.csv)
            if n_cp > 0 and event_onset_month_index is not None:
                step = max(1, consec // (n_cp + 1))
                changepoint_months = [
                    min(len(df) - 1, event_onset_month_index + step * i)
                    for i in range(1, n_cp + 1)
                ]
    except Exception:
        pass  # safe fallback — charts still render without these markers

    return {
        "zone_id":    zone_id,
        "timestamps": len(df),
        "metadata": {
            "ndvi_baseline_mean":      round(baseline_mean, 4),
            "ndvi_baseline_std":       round(baseline_std, 4),
            "healthy_band_min":        healthy_band_min,
            "healthy_band_max":        healthy_band_max,
            "event_onset_month_index": event_onset_month_index,
            "changepoint_months":      changepoint_months,
        },
        "data": df.to_dict("records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5: GET /api/zones/{zone_id}/importance
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/zones/{zone_id}/importance", summary="Top-3 features driving detection")
def get_importance(zone_id: str = Path(..., description="Zone ID")):
    """Returns the top 3 features that drove this zone's anomaly detection."""
    logger.info("GET /api/zones/%s/importance", zone_id)
    df   = _load_detections()
    zone = df[df["zone_id"] == zone_id]
    if zone.empty:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")

    row = zone.iloc[0].to_dict()
    importance = []
    for k in range(1, 4):
        name   = row.get(f"driver_{k}",        row.get(f"feature_{k}_name",        ""))
        score  = row.get(f"driver_{k}_zscore",  row.get(f"feature_{k}_importance",  0))
        detail = row.get(f"driver_{k}_detail",  "")
        importance.append({
            "feature_name":      str(name),
            "importance_score":  round(float(score) if score else 0, 4),
            "detail":            str(detail),
        })

    return {"zone_id": zone_id, "top_features": importance}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 6b: GET /api/zones/{zone_id}/neighbors
# PRD §18 (Zone Popup — Regional Context section)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/zones/{zone_id}/neighbors", summary="8 nearest geographic zones with health data")
def get_neighbors(zone_id: str = Path(..., description="Zone ID")):
    """
    Returns the 8 nearest zones by geographic distance with their health metrics.
    Powers the Regional Context section of the PRD §18 Zone Popup.
    Distance is Euclidean on lat/lon (acceptable at ~3km zone scale).
    """
    logger.info("GET /api/zones/%s/neighbors", zone_id)
    df   = _load_detections()
    zone = df[df["zone_id"] == zone_id]
    if zone.empty:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")

    row = zone.iloc[0]
    lat0, lon0 = float(row["lat"]), float(row["lon"])

    others = df[df["zone_id"] != zone_id].copy()

    # Euclidean distance on lat/lon, then convert to approximate km
    # 1 degree lat ≈ 111 km; 1 degree lon ≈ 111 * cos(lat) km
    import math
    cos_lat = math.cos(math.radians(lat0))
    others["_dist_deg"] = (
        (others["lat"] - lat0) ** 2 +
        ((others["lon"] - lon0) * cos_lat) ** 2
    ) ** 0.5
    others["_dist_km"] = (others["_dist_deg"] * 111.0).round(2)

    nearest = others.nsmallest(8, "_dist_km")

    neighbors = []
    for _, nr in nearest.iterrows():
        neighbors.append({
            "zone_id":        str(nr["zone_id"]),
            "distance_km":    float(nr["_dist_km"]),
            "lat":            float(nr["lat"]),
            "lon":            float(nr["lon"]),
            "ndvi_current":   round(float(nr.get("ndvi_current", 0) or 0), 4),
            "drift_score":    round(float(nr.get("drift_score",  0) or 0), 2),
            "is_anomaly":     int(nr.get("is_anomaly", 0) or 0),
            "threat_type":    str(nr["threat_type"]) if nr.get("threat_type") else None,
            "confidence":     round(float(nr.get("confidence", 0) or 0), 1),
            "severity":       str(nr["drift_severity"]) if nr.get("drift_severity") else "none",
        })

    return {
        "zone_id":   zone_id,
        "neighbors": neighbors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 12: GET /api/summary
# PRD §20 (AccuracyBar) — 6 metric cards: zones, threats, precision, recall, F1, drift
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/summary", summary="Global dashboard summary for AccuracyBar")
def get_summary():
    """
    Returns aggregated statistics powering PRD §20 AccuracyBar's 6 metric cards:
    Zones Analyzed, Threats Detected, Precision, Recall, F1, Avg Drift.
    Also provides threat-type and severity breakdowns for header stats.
    """
    logger.info("GET /api/summary")
    df = _load_detections()

    threats = df[df["is_anomaly"] == 1]

    # Threat type breakdown
    threat_breakdown: dict[str, int] = {}
    if "threat_type" in df.columns:
        counts = threats["threat_type"].value_counts().to_dict()
        threat_breakdown = {str(k): int(v) for k, v in counts.items()}

    # Severity breakdown
    severity_breakdown: dict[str, int] = {}
    if "drift_severity" in df.columns:
        scounts = threats["drift_severity"].value_counts().to_dict()
        severity_breakdown = {str(k): int(v) for k, v in scounts.items()}

    # Averages
    avg_threat_score = round(float(threats["threat_score"].mean()), 1) if len(threats) and "threat_score" in threats else 0.0
    avg_confidence   = round(float(threats["confidence"].mean()),   1) if len(threats) and "confidence"   in threats else 0.0
    avg_drift        = round(float(df["drift_score"].mean()),        2) if "drift_score" in df.columns else 0.0

    # Pipeline accuracy from accuracy_report.json
    pipeline_accuracy: dict[str, Any] = {}
    if ACCURACY_REPORT_JSON.exists():
        try:
            with ACCURACY_REPORT_JSON.open("r", encoding="utf-8") as fh:
                acc = json.load(fh)
            # Try overall_metrics first, fall back to metrics
            pipeline_accuracy = acc.get("overall_metrics") or acc.get("metrics", {})
        except Exception:
            pass

    # Data mode from config
    cfg = load_config()

    return {
        "total_zones":        len(df),
        "total_threats":      len(threats),
        "threat_breakdown":   threat_breakdown,
        "severity_breakdown": severity_breakdown,
        "avg_threat_score":   avg_threat_score,
        "avg_confidence":     avg_confidence,
        "avg_drift_score":    avg_drift,
        "pipeline_accuracy":  pipeline_accuracy,
        "data_mode":          _get_actual_data_mode(cfg),
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 7: GET /api/accuracy
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/accuracy", summary="Accuracy report from latest pipeline run")
def get_accuracy():
    """Returns the full accuracy_report.json from the last pipeline run."""
    logger.info("GET /api/accuracy")
    if not ACCURACY_REPORT_JSON.exists():
        raise HTTPException(status_code=404, detail="Accuracy report not found. Run pipeline first.")
    with ACCURACY_REPORT_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 8: GET /api/config
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/config", summary="Current configuration for frontend initialization")
def get_config():
    """Returns frontend-relevant sections of the current config.yaml."""
    logger.info("GET /api/config")
    cfg = load_config()
    # Exclude secrets: remove LLM API keys
    llm = cfg.get("llm", {})
    for side in ["primary", "fallback"]:
        if side in llm:
            llm[side].pop("api_key", None)

    return {
        "data":           cfg.get("data", {}),
        "ensemble":       cfg.get("ensemble", {}),
        "filters":        cfg.get("filters", {}),
        "drift":          cfg.get("drift", {}),
        "spatial":        cfg.get("spatial", {}),
        "classification": cfg.get("classification", {}),
        "dsr":            cfg.get("dsr", {}),
        "baseline":       cfg.get("baseline", {}),
        "reports":        cfg.get("reports", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 9: POST /api/analyze
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/analyze", summary="Trigger new pipeline run with overrides")
async def analyze(body: AnalyzeRequest):
    """
    Re-run the full ML pipeline with optional parameter overrides.
    Only one pipeline run is allowed at a time — 409 if already running.
    Uses config-based caching: if same config was run before, skip pipeline.
    """
    global _pipeline_running, _current_config_hash

    if _pipeline_running:
        logger.warning("POST /api/analyze — 409, pipeline already running")
        raise HTTPException(
            status_code=409,
            detail="Pipeline already running. Try again after completion.",
        )

    async with _pipeline_lock:
        _pipeline_running = True
        start_time = time.time()
        warnings: list[str] = []
        skipped = False

        try:
            logger.info("POST /api/analyze — pipeline starting")
            cfg = load_config()

            # Apply sensitivity preset
            sensitivity = (body.sensitivity or "medium").lower()
            # Presets are handled by frontend - frontend sends min_weighted_score in overrides
            # This is kept for direct API calls without frontend
            presets = {
                "high":   {"ensemble": {"min_score": 0.3}},   # Less strict = more detections
                "medium": {"ensemble": {"min_score": 0.5}},
                "low":    {"ensemble": {"min_score": 0.7}},   # More strict = fewer detections
            }
            if sensitivity in presets:
                cfg = deep_merge(cfg, presets[sensitivity])

            # Apply user overrides
            if body.overrides:
                cfg = deep_merge(cfg, body.overrides)

            # Validate
            cfg, warns, corrections = validate_config(cfg)
            warnings.extend(warns)
            warnings.extend([f"[CORRECTED] {c}" for c in corrections])

            # Compute config hash to check cache
            config_hash = _compute_config_hash(cfg)
            logger.info("Config hash: %s", config_hash)

            # Check if we already have results for this config
            if _current_config_hash == config_hash and DETECTIONS_CSV.exists():
                logger.info("Using cached results for config hash: %s", config_hash)
                skipped = True
            else:
                # Run pipeline in a thread (sync pipeline code)
                await asyncio.get_event_loop().run_in_executor(
                    None, _run_pipeline, cfg
                )
                # Update the current config hash after successful run
                _current_config_hash = config_hash

            # Load results
            accuracy_report: dict[str, Any] = {}
            if ACCURACY_REPORT_JSON.exists():
                with ACCURACY_REPORT_JSON.open("r", encoding="utf-8") as fh:
                    accuracy_report = json.load(fh)

            elapsed = round(time.time() - start_time, 2)
            logger.info("POST /api/analyze — complete in %.1fs (skipped=%s)", elapsed, skipped)

            # Mask API keys in config_used
            cfg.get("llm", {}).get("primary", {}).pop("api_key", None)
            cfg.get("llm", {}).get("fallback", {}).pop("api_key", None)

            return {
                "status":                    "success",
                "computation_time_seconds":  elapsed,
                "skipped":                   skipped,
                "accuracy_report":           accuracy_report,
                "config_used":               cfg,
                "warnings":                  warnings,
            }

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

        finally:
            _pipeline_running = False


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 10: GET /api/download/geojson
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/download/geojson", summary="Download GeoJSON as a file")
def download_geojson():
    """Streams detected_zones.geojson as a downloadable attachment."""
    logger.info("GET /api/download/geojson")
    if not GEOJSON_PATH.exists():
        raise HTTPException(status_code=404, detail="GeoJSON file not found")
    return FileResponse(
        path=str(GEOJSON_PATH),
        filename="aravalli_threats.geojson",
        media_type="application/geo+json",
        headers={"Content-Disposition": "attachment; filename=aravalli_threats.geojson"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 11: GET /api/download/detections
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/download/detections", summary="Download detections CSV as a file")
def download_detections():
    """Streams detections.csv as a downloadable attachment."""
    logger.info("GET /api/download/detections")
    if not DETECTIONS_CSV.exists():
        raise HTTPException(status_code=404, detail="Detections CSV not found")
    return FileResponse(
        path=str(DETECTIONS_CSV),
        filename="aravalli_detections.csv",
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=aravalli_detections.csv"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Static files — serve React build if present
# ─────────────────────────────────────────────────────────────────────────────
frontend_path = PathlibPath("frontend/dist")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
    logger.info("Serving React frontend from frontend/dist")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg  = load_config()
    host = cfg.get("server", {}).get("host", "0.0.0.0")
    port = int(cfg.get("server", {}).get("port", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)
