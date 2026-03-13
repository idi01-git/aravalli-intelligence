"""
tests/test_main.py

19 pytest tests for the FastAPI backend (main.py).
Uses TestClient — no live server required.

PRD Reference: Section 13 (Backend API), Section 18 (Zone Popup), Section 20 (AccuracyBar)
Author: Team BIOBYTES
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures and mock data
# ─────────────────────────────────────────────────────────────────────────────

MOCK_ZONE_ROW = {
    "zone_id":             "zone_0001",
    "lat":                  25.412,
    "lon":                  73.891,
    "elevation":            820.0,
    "threat_type":          "deforestation",
    "threat_score":         75.5,
    "drift_score":          7.2,
    "drift_severity":       "high",
    "confidence":           62.0,
    "dsr":                  3.5,
    "ndvi_current":         0.28,
    "ndvi_delta":          -0.18,
    "ndbi_delta":          -0.05,
    "bsi_delta":           -0.03,
    "nightlight_delta":     0.0,
    "is_anomaly":           1,
    "severity":             "high",
    "consecutive_declines": 6,
    "n_changepoints":       2,
    "local_anomaly_score":  2.1,
    "is_isolated":          True,
    "regional_health":      0.62,
    "ensemble_votes":       3,
    "dsr_classification":   "confirmed_degradation",
    "slope_short":         -0.012,
    "recovery_signal":     -0.04,
    "volatility_ratio":     1.2,
    "weighted_score":       0.75,
    "driver_1":             "Change exceeds seasonal expectation",
    "driver_1_zscore":      3.4,
    "driver_1_detail":      "3.4x above normal",
    "driver_2":             "Consecutive declining months",
    "driver_2_zscore":      2.9,
    "driver_2_detail":      "2.9x above normal",
    "driver_3":             "Bare soil exposure",
    "driver_3_zscore":      2.1,
    "driver_3_detail":      "2.1x above normal",
}

# Build mock DataFrame: zone_0001 is the threat; zones 2-200 are healthy with
# distinct lat/lon offsets so neighbors distance calculation is meaningful.
MOCK_DF = pd.DataFrame(
    [MOCK_ZONE_ROW]
    + [
        {
            **MOCK_ZONE_ROW,
            "zone_id":      f"zone_{i:04d}",
            "is_anomaly":   0,
            "threat_type":  None,
            "drift_severity": "low",
            "threat_score": 0.0,
            # Spread zones across a grid so distance calculations vary
            "lat":          25.412 + (i % 20) * 0.05,
            "lon":          73.891 + (i // 20) * 0.05,
        }
        for i in range(2, 201)
    ]
)

MOCK_REPORTS = {
    "zone_0001": {
        "text":      "Deforestation confirmed at zone_0001. Action: investigate.",
        "source":    "live_llama",
        "timestamp": "2026-02-26T13:00:00+00:00",
    }
}

MOCK_ACCURACY = {
    "metadata":        {"run_at": "2026-02-26T13:00:00"},
    "overall_metrics": {"precision": 0.91, "recall": 0.88, "f1": 0.89},
    "confusion_matrix": {"TP": 27, "FP": 3, "TN": 967, "FN": 3},
}

MOCK_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [73.891, 25.412]},
            "properties": {"zone_id": "zone_0001", "threat_type": "deforestation"},
        }
    ],
}


@pytest.fixture()
def client(tmp_path):
    """Patch all file I/O and return a TestClient."""
    import main  # Import here after patches are applied

    with (
        patch("main._load_detections", return_value=MOCK_DF.copy()),
        patch("main._load_reports",    return_value=MOCK_REPORTS),
        patch("main.load_config",      return_value={
            "data":   {"source": "synthetic"},
            "server": {"host": "0.0.0.0", "port": 8000},
            "ensemble": {},
            "filters": {},
            "drift": {},
            "spatial": {},
            "classification": {},
            "dsr": {},
            "baseline": {},
            "reports": {},
        }),
        patch("main.DETECTIONS_CSV",       tmp_path / "detections.csv"),
        patch("main.CACHED_REPORTS_JSON",  tmp_path / "cached_reports.json"),
        patch("main.ACCURACY_REPORT_JSON", tmp_path / "accuracy_report.json"),
        patch("main.GEOJSON_PATH",         tmp_path / "detected_zones.geojson"),
    ):
        # Write mock files
        (tmp_path / "detections.csv").write_text(MOCK_DF.to_csv(index=False))
        (tmp_path / "cached_reports.json").write_text(json.dumps(MOCK_REPORTS))
        (tmp_path / "accuracy_report.json").write_text(json.dumps(MOCK_ACCURACY))
        (tmp_path / "detected_zones.geojson").write_text(json.dumps(MOCK_GEOJSON))

        yield TestClient(main.app)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Health endpoint
# ─────────────────────────────────────────────────────────────────────────────
def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "project_name" in data
    assert "timestamp" in data


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Zones list
# ─────────────────────────────────────────────────────────────────────────────
def test_zones_list(client):
    resp = client.get("/api/zones")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "data" in data
    assert isinstance(data["data"], list)
    assert data["total"] == 200  # MOCK_DF has 200 rows


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Zones pagination
# ─────────────────────────────────────────────────────────────────────────────
def test_zones_pagination(client):
    resp = client.get("/api/zones?page=1&size=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 10
    assert data["page"] == 1
    assert data["page_size"] == 10
    # page 2
    resp2 = client.get("/api/zones?page=2&size=10")
    assert resp2.json()["data"][0]["zone_id"] != data["data"][0]["zone_id"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Zone detail with AI report
# ─────────────────────────────────────────────────────────────────────────────
def test_zone_get(client):
    resp = client.get("/api/zones/zone_0001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["zone_id"] == "zone_0001"
    assert "ai_report" in data
    assert "report_source" in data
    assert data["ai_report"] != ""


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Timeseries (404 on missing file)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Timeseries (404 on missing file)
# ─────────────────────────────────────────────────────────────────────────────
def test_zone_timeseries_missing(client, tmp_path):
    """Timeseries endpoint returns 404 when zone's CSV file does not exist."""
    empty_dir = tmp_path / "empty_ts"
    empty_dir.mkdir()
    with patch("main.TIMESERIES_DIR", empty_dir):
        import main
        local_client = TestClient(main.app)
        resp = local_client.get("/api/zones/zone_0001/timeseries")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


def test_zone_timeseries_present(client, tmp_path):
    """Timeseries returns correct structure when file exists."""
    ts_dir = tmp_path / "zone_timeseries"
    ts_dir.mkdir()
    ts_df = pd.DataFrame([{"timestamp": f"2019-{i:02d}", "ndvi_raw": 0.4} for i in range(1, 13)])
    ts_df.to_csv(ts_dir / "zone_0001.csv", index=False)

    with patch("main.TIMESERIES_DIR", ts_dir):
        import main
        local_client = TestClient(main.app)
        resp = local_client.get("/api/zones/zone_0001/timeseries")
        assert resp.status_code == 200
        data = resp.json()
        assert data["zone_id"] == "zone_0001"
        assert data["timestamps"] == 12


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Feature importance
# ─────────────────────────────────────────────────────────────────────────────
def test_zone_importance(client):
    resp = client.get("/api/zones/zone_0001/importance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["zone_id"] == "zone_0001"
    assert "top_features" in data
    assert len(data["top_features"]) == 3
    for feat in data["top_features"]:
        assert "feature_name" in feat
        assert "importance_score" in feat


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — GeoJSON endpoint
# ─────────────────────────────────────────────────────────────────────────────
def test_geojson_endpoint(client, tmp_path):
    geo_path = tmp_path / "detected_zones.geojson"
    geo_path.write_text(json.dumps(MOCK_GEOJSON))
    with patch("main.GEOJSON_PATH", geo_path):
        import main
        local_client = TestClient(main.app)
        resp = local_client.get("/api/zones/geojson")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Accuracy endpoint
# ─────────────────────────────────────────────────────────────────────────────
def test_accuracy_endpoint(client, tmp_path):
    acc_path = tmp_path / "accuracy_report.json"
    acc_path.write_text(json.dumps(MOCK_ACCURACY))
    with patch("main.ACCURACY_REPORT_JSON", acc_path):
        import main
        local_client = TestClient(main.app)
        resp = local_client.get("/api/accuracy")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_metrics" in data
        assert "confusion_matrix" in data


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Config endpoint (no API keys in response)
# ─────────────────────────────────────────────────────────────────────────────
def test_config_endpoint(client):
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "ensemble" in data
    assert "filters" in data
    # API keys must NEVER appear
    config_str = json.dumps(data)
    assert "api_key" not in config_str


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — Analyze endpoint (mocked pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def test_analyze_endpoint(client, tmp_path):
    acc_path = tmp_path / "accuracy_report.json"
    acc_path.write_text(json.dumps(MOCK_ACCURACY))

    with (
        patch("main._run_pipeline"),
        patch("main.ACCURACY_REPORT_JSON", acc_path),
        patch("main.validate_config", return_value=({}, [], [])),
    ):
        import main
        local_client = TestClient(main.app)
        resp = local_client.post("/api/analyze", json={
            "sensitivity": "high",
            "overrides":   {"ensemble": {"min_score": 0.3}},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "computation_time_seconds" in data
        assert "warnings" in data


# ─────────────────────────────────────────────────────────────────────────────
# TEST 11 — Download GeoJSON (file response)
# ─────────────────────────────────────────────────────────────────────────────
def test_download_geojson(client, tmp_path):
    geo_path = tmp_path / "detected_zones.geojson"
    geo_path.write_text(json.dumps(MOCK_GEOJSON))
    with patch("main.GEOJSON_PATH", geo_path):
        import main
        local_client = TestClient(main.app)
        resp = local_client.get("/api/download/geojson")
        assert resp.status_code == 200
        assert "content-disposition" in resp.headers
        assert "aravalli_threats.geojson" in resp.headers["content-disposition"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST 12 — Download detections CSV (file response)
# ─────────────────────────────────────────────────────────────────────────────
def test_download_detections(client, tmp_path):
    csv_path = tmp_path / "detections.csv"
    MOCK_DF.to_csv(csv_path, index=False)
    with patch("main.DETECTIONS_CSV", csv_path):
        import main
        local_client = TestClient(main.app)
        resp = local_client.get("/api/download/detections")
        assert resp.status_code == 200
        assert "content-disposition" in resp.headers
        assert "aravalli_detections.csv" in resp.headers["content-disposition"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST 13 — CORS headers present
# ─────────────────────────────────────────────────────────────────────────────
def test_cors_enabled(client):
    resp = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    # Status 200 or 204 on preflight
    assert resp.status_code in (200, 204)
    assert "access-control-allow-origin" in resp.headers


# ─────────────────────────────────────────────────────────────────────────────
# TEST 14 — Error handling: 404 on missing zone, JSON response
# ─────────────────────────────────────────────────────────────────────────────
def test_error_handling_404(client):
    resp = client.get("/api/zones/zone_NONEXISTENT")
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data
    assert "error_code" in data
    # Must be JSON, never HTML
    assert resp.headers["content-type"].startswith("application/json")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 15 — Pipeline mutex: 409 on concurrent analyze calls
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_mutex(tmp_path):
    """Simulates mutex by setting _pipeline_running flag before request."""
    import main
    main._pipeline_running = True  # Simulate a pipeline already running
    try:
        with (
            patch("main._load_detections", return_value=MOCK_DF.copy()),
            patch("main._load_reports",    return_value=MOCK_REPORTS),
            patch("main.load_config",      return_value={"data": {}, "server": {}}),
        ):
            local_client = TestClient(main.app)
            resp = local_client.post("/api/analyze", json={"sensitivity": "medium"})
            assert resp.status_code == 409
            assert "already running" in resp.json()["detail"].lower()
    finally:
        main._pipeline_running = False  # Always reset


# ─────────────────────────────────────────────────────────────────────────────
# TEST 16 — GET /api/summary (PRD §20 AccuracyBar)
# Verifies: total_zones, total_threats, threat_breakdown, severity_breakdown,
#           avg fields, pipeline_accuracy, data_mode, timestamp
# ─────────────────────────────────────────────────────────────────────────────
def test_summary_endpoint(client, tmp_path):
    """GET /api/summary returns all 6 AccuracyBar metric fields (PRD §20)."""
    acc_path = tmp_path / "accuracy_report.json"
    acc_path.write_text(json.dumps(MOCK_ACCURACY))
    with patch("main.ACCURACY_REPORT_JSON", acc_path):
        import main
        local_client = TestClient(main.app)
        resp = local_client.get("/api/summary")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # PRD §20 AccuracyBar requires these 6 metric card values
    assert "total_zones"   in data, "Missing total_zones"
    assert "total_threats" in data, "Missing total_threats"
    assert "avg_drift_score"     in data, "Missing avg_drift_score"
    assert "threat_breakdown"    in data, "Missing threat_breakdown"
    assert "severity_breakdown"  in data, "Missing severity_breakdown"
    assert "pipeline_accuracy"   in data, "Missing pipeline_accuracy"
    assert "data_mode"           in data, "Missing data_mode"
    assert "timestamp"           in data, "Missing timestamp"
    # MOCK_DF has 1 threat (zone_0001)
    assert data["total_zones"]   == 200
    assert data["total_threats"] == 1
    assert isinstance(data["threat_breakdown"], dict)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 17 — GET /api/zones/{zone_id}/neighbors (PRD §18 Regional Context)
# Verifies: returns 8 neighbors, sorted by distance_km, correct fields.
# ─────────────────────────────────────────────────────────────────────────────
def test_neighbors_endpoint(client):
    """GET /api/zones/{zone_id}/neighbors returns 8 neighbors sorted by distance (PRD §18)."""
    resp = client.get("/api/zones/zone_0001/neighbors")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["zone_id"] == "zone_0001"
    assert "neighbors" in data
    neighbors = data["neighbors"]
    assert len(neighbors) == 8, f"Expected 8 neighbors, got {len(neighbors)}"
    # Verify required fields are present in each neighbor
    required_fields = {"zone_id", "distance_km", "lat", "lon",
                       "ndvi_current", "drift_score", "is_anomaly", "confidence"}
    for nb in neighbors:
        assert required_fields.issubset(nb.keys()), f"Missing fields in neighbor: {nb.keys()}"
    # Verify sorted by distance
    distances = [nb["distance_km"] for nb in neighbors]
    assert distances == sorted(distances), "Neighbors not sorted by distance_km"


def test_neighbors_endpoint_404(client):
    """GET /api/zones/{zone_id}/neighbors returns 404 for unknown zone."""
    resp = client.get("/api/zones/zone_NONEXISTENT/neighbors")
    assert resp.status_code == 404
    assert "detail" in resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 18 — Enriched timeseries: metadata block with PRD §18 chart fields
# Verifies: healthy_band_min/max, ndvi_baseline_std, event_onset_month_index,
#           changepoint_months, month_label per row
# ─────────────────────────────────────────────────────────────────────────────
def test_zone_timeseries_metadata(client, tmp_path):
    """
    PRD §18 Trend Analysis chart requires:
      - metadata.healthy_band_min / healthy_band_max  (±1 std shading band)
      - metadata.event_onset_month_index              (red vertical line)
      - metadata.changepoint_months                   (red dots)
      - data[].month_label                            (X-axis labels)
      - data[].month_index                            (ECharts index)
    """
    ts_dir = tmp_path / "zone_timeseries"
    ts_dir.mkdir()
    # Full 84-month mock with baseline values
    rows = [
        {
            "timestamp":     f"2019-{(i % 12) + 1:02d}",
            "ndvi":          round(0.52 - i * 0.004, 4),
            "ndbi":          0.08,
            "bsi":           0.12,
            "nightlight":    0.10,
            "ndvi_smoothed":  round(0.52 - i * 0.003, 4),
            "ndbi_smoothed":  0.08,
            "bsi_smoothed":   0.12,
            "ndvi_baseline":  0.499,
            "ndvi_delta":    round(-0.01 * i, 4),
        }
        for i in range(84)
    ]
    pd.DataFrame(rows).to_csv(ts_dir / "zone_0001.csv", index=False)

    with patch("main.TIMESERIES_DIR", ts_dir):
        import main
        local_client = TestClient(main.app)
        resp = local_client.get("/api/zones/zone_0001/timeseries")

    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Metadata block must exist with all PRD-required fields
    assert "metadata" in data, "Missing metadata block"
    meta = data["metadata"]
    assert "healthy_band_min"        in meta, "Missing healthy_band_min"
    assert "healthy_band_max"        in meta, "Missing healthy_band_max"
    assert "ndvi_baseline_std"       in meta, "Missing ndvi_baseline_std"
    assert "event_onset_month_index" in meta, "Missing event_onset_month_index"
    assert "changepoint_months"      in meta, "Missing changepoint_months"
    assert isinstance(meta["changepoint_months"], list)

    # Per-row ECharts fields
    first_row = data["data"][0]
    assert "month_label" in first_row, "Missing month_label in data rows"
    assert "month_index" in first_row, "Missing month_index in data rows"
    assert first_row["month_index"] == 0
    # month_label should be human-readable, e.g. "Jan 2019"
    assert len(first_row["month_label"]) > 0
    assert first_row["month_label"] != first_row["timestamp"]  # Reformatted

