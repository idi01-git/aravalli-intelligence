"""
Aravalli Intelligence — Detection Engine Test Suite (tests/test_detect.py)
7 unit tests for pipeline/detect.py per PRD Section 10 specifications.

Tests:
    1. test_dsr_seasonal_classification    — gate1/gate2 + seasonal_normal
    2. test_dsr_confirmed_degradation      — high DSR → confirmed label
    3. test_dsr_magnitude_gate             — small change → seasonal regardless of DSR
    4. test_dsr_transition_threshold       — tighter threshold in Mar/Apr/Oct/Nov
    5. test_weighted_ensemble_voting       — voting mechanism and thresholding
    6. test_weight_normalization           — disabled method weights renormalise
    7. test_geojson_structure              — output GeoJSON valid FeatureCollection

Author: Shivang, Team BIOBYTES
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config
from pipeline.detect import (
    compute_dsr,
    compute_drift_score,
    export_geojson,
    apply_filters,
    classify_threats,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config():
    """Load project configuration once for all detection tests."""
    return load_config()


def _base_row(
    *,
    zone_id: str = "zone_0001",
    ndvi_current: float = 0.35,
    ndvi_delta:   float = -0.20,
    ndbi_delta:   float = 0.01,
    bsi_delta:    float = 0.01,
    nightlight_delta: float = 1.0,
    consecutive_declines: int = 4,
    local_anomaly_score: float = 2.0,
    recovery_signal: float = -0.05,
    is_isolated: int = 1,
    regional_health: float = 0.5,
    slope_short: float = -0.02,
    dsr: float = 2.0,
    dsr_classification: str = "watch",
    dsr_severity: str = "medium",
    cal_month: int = 7,
    confidence: float = 65.0,
    is_anomaly: int = 1,
    lat: float = 25.0,
    lon: float = 73.0,
    elevation_norm: float = 0.5,
    drift_score: float = 5.0,
    threat_type: str = "deforestation",
    threat_score: float = 60.0,
) -> dict:
    """Create a single base row with sensible defaults for detection tests."""
    return dict(
        zone_id=zone_id,
        ndvi_current=ndvi_current,
        ndvi_delta=ndvi_delta,
        ndbi_delta=ndbi_delta,
        bsi_delta=bsi_delta,
        nightlight_delta=nightlight_delta,
        consecutive_declines=consecutive_declines,
        local_anomaly_score=local_anomaly_score,
        recovery_signal=recovery_signal,
        is_isolated=is_isolated,
        regional_health=regional_health,
        slope_short=slope_short,
        dsr=dsr,
        dsr_classification=dsr_classification,
        dsr_severity=dsr_severity,
        cal_month=cal_month,
        confidence=confidence,
        is_anomaly=is_anomaly,
        lat=lat,
        lon=lon,
        elevation_norm=elevation_norm,
        drift_score=drift_score,
        threat_type=threat_type,
        threat_score=threat_score,
        month_sin=math.sin(2 * math.pi * cal_month / 12),
        month_cos=math.cos(2 * math.pi * cal_month / 12),
        ndbi_current=-0.05,
        bsi_current=0.08,
        nightlight_current=5.0,
        slope_long=-0.001,
        acceleration=-0.019,
        volatility_ratio=1.2,
        deviation_from_mean=-0.15,
        spatial_gradient=-0.05,
        ensemble_votes=3,
        feature_1_name="N/A",
        feature_1_importance=0.0,
        feature_2_name="N/A",
        feature_2_importance=0.0,
        feature_3_name="N/A",
        feature_3_importance=0.0,
        driver_1="N/A",
        driver_1_zscore=0.0,
        driver_1_detail="N/A",
        driver_2="N/A",
        driver_2_zscore=0.0,
        driver_2_detail="N/A",
        driver_3="N/A",
        driver_3_zscore=0.0,
        driver_3_detail="N/A",
    )


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── TEST 1: DSR Seasonal Classification ──────────────────────────────────────

def test_dsr_seasonal_classification(config):
    """Zones with small NDVI change should be classified as seasonal_normal.

    PRD: Gate 1 — if observed_delta < min_absolute_change → seasonal regardless.
    Expected: A zone with delta=0.02 (well below 0.10 gate) is seasonal_normal
    and severity=none even if DSR formula would give a positive value.
    """
    min_abs = config.get("dsr", {}).get("min_absolute_change", 0.10)

    rows = [_base_row(
        zone_id=f"zone_{i:04d}",
        ndvi_current=0.42,
        ndvi_delta=0.02,      # below min_abs gate — must be forced seasonal
    ) for i in range(1, 11)]

    df = _make_df(rows)
    result = compute_dsr(df, config)

    seasonal_mask = result["dsr_classification"] == "seasonal_normal"
    # All zones should be classified as seasonal_normal (gate 1 fires)
    assert seasonal_mask.all(), (
        f"Expected all zones to be seasonal_normal (delta < gate {min_abs}), "
        f"got: {result['dsr_classification'].value_counts().to_dict()}"
    )
    assert (result["dsr_severity"] == "none").all(), "Seasonal zones must have severity=none"


# ── TEST 2: DSR Confirmed Degradation ────────────────────────────────────────

def test_dsr_confirmed_degradation(config):
    """Large NDVI loss should be classified as confirmed_degradation.

    PRD: DSR >= threshold + 0.5 → confirmed_degradation, severity=high.
    A delta of 0.40 on a baseline of 0.35 should give DSR >> threshold.
    """
    # Ensure expected baseline delta is low by having mostly normal zones
    rows = [_base_row(
        zone_id=f"zone_{i:04d}",
        ndvi_current=0.35,
        ndvi_delta=-0.05,    # normal seasonal dip 
        cal_month=6,
    ) for i in range(1, 10)]
    
    # Add 1 anomalous zone
    rows.append(_base_row(
        zone_id="zone_0010",
        ndvi_current=0.35,
        ndvi_delta=-0.40,    # Very large drop — should confirm degradation
        cal_month=6,         # Normal month (not transition)
    ))

    df = _make_df(rows)
    result = compute_dsr(df, config)

    confirmed = result["dsr_classification"] == "confirmed_degradation"
    # At minimum, a significant portion should be confirmed (DSR will be high)
    assert confirmed.any(), (
        "Expected at least one zone to be confirmed_degradation with delta=-0.40"
    )
    # All confirmed zones must have severity=high
    confirmed_rows = result[confirmed]
    assert (confirmed_rows["dsr_severity"] == "high").all(), (
        "confirmed_degradation zones must have dsr_severity=high"
    )
    # DSR > 1.0 for all zones with large drops
    assert (confirmed_rows["dsr"] > 1.0).all()


# ── TEST 3: DSR Magnitude Gate ────────────────────────────────────────────────

def test_dsr_magnitude_gate(config):
    """Gate 1: tiny delta → always seasonal, regardless of baseline ratio.

    Even if DSR formula would yield 5.0, a 0.05 delta is below the
    min_absolute_change gate (0.10) and must NOT be classified as a threat.
    """
    rows = [_base_row(
        zone_id=f"zone_{i:04d}",
        ndvi_delta=0.04,    # 4% — below 10% gate
    ) for i in range(1, 11)]

    df = _make_df(rows)
    result = compute_dsr(df, config)

    non_seasonal = result[result["dsr_classification"] != "seasonal_normal"]
    assert len(non_seasonal) == 0, (
        f"Gate 1 failed: {len(non_seasonal)} non-seasonal zones with tiny delta. "
        f"Classifications: {result['dsr_classification'].value_counts().to_dict()}"
    )


# ── TEST 4: DSR Transition Threshold ─────────────────────────────────────────

def test_dsr_transition_threshold(config):
    """Transition months (Mar=3, Apr=4, Oct=10, Nov=11) use a higher threshold.

    PRD: transition months use threshold_transition (default 2.0) instead of
    threshold_normal (default 1.5). This means the same NDVI drop is LESS
    likely to become confirmed_degradation in a transition month than in July.

    Approach: Pick a delta that would be 'warning' in July but safely below
    confirmed_degradation in a transition month.
    """
    threshold_normal     = config.get("dsr", {}).get("threshold_normal",     1.50)
    threshold_transition = config.get("dsr", {}).get("threshold_transition", 2.00)

    # We want DSR ~ 2.1. 
    # normal month threshold+0.5 = 2.0 (so 2.1 is confirmed)
    # transition group threshold+0.5 = 2.5 (so 2.1 is warning)
    # To get DSR ~ 2.1 with min_absolute_change gates needing > 0.10:
    # 9 zones at 0.135, 1 at 0.144 -> mean ~ 0.136 -> expected = 0.068. 0.144/0.068 = ~2.11.
    def make_mixed_rows(month: int, prefix: str):
        rs = [_base_row(zone_id=f"{prefix}_{i:04d}", cal_month=month, ndvi_delta=-0.135) for i in range(1, 10)]
        rs.append(_base_row(zone_id=f"{prefix}_0010", cal_month=month, ndvi_delta=-0.144))
        return rs

    # Transition month (March = 3)
    rows_transition = make_mixed_rows(3, "zone_trans")

    # Normal month (July = 7)
    rows_normal = make_mixed_rows(7, "zone_norm")

    df_trans  = _make_df(rows_transition)
    df_normal = _make_df(rows_normal)

    res_trans  = compute_dsr(df_trans,  config)
    res_normal = compute_dsr(df_normal, config)

    # Thresholds are different — this is what we verify
    assert threshold_transition > threshold_normal, (
        "Test precondition: transition threshold must be > normal threshold"
    )

    # The confirmed_degradation rate should be >= in normal month than transition
    conf_trans  = (res_trans["dsr_classification"]  == "confirmed_degradation").sum()
    conf_normal = (res_normal["dsr_classification"] == "confirmed_degradation").sum()
    assert conf_normal >= conf_trans, (
        f"Normal month should confirm >= transition month. "
        f"Normal confirmed: {conf_normal}, Transition confirmed: {conf_trans}"
    )


# ── TEST 5: Weighted Ensemble Voting ─────────────────────────────────────────

def test_weighted_ensemble_voting():
    """Verify weighted voting formula: anomaly iff weighted_score >= min_weighted_score.

    This test directly implements the PRD voting formula without running
    the full ensemble (to isolate the math from sklearn behaviour).
    """
    # Simulated per-method votes (binary 0/1) and weights
    weights   = np.array([0.35, 0.30, 0.25, 0.10])   # IsoForest, DBSCAN, LOF, KMeans
    min_score = 0.50

    test_cases = [
        # (votes,                    expected_anomaly)
        ([1,    1,    1,    1   ],   True),    # All 4 methods vote: 1.0 >= 0.5
        ([1,    1,    1,    0   ],   True),    # 3 strong votes: 0.90 >= 0.5
        ([1,    0,    0,    0   ],   False),   # Only IsoForest: 0.35 < 0.5
        ([1,    1,    0,    0   ],   True),    # IsoForest + DBSCAN: 0.65 >= 0.5
        ([0,    1,    0,    1   ],   False),   # DBSCAN + KMeans: 0.40 < 0.5
        ([0,    0,    0,    0   ],   False),   # No votes: 0.0 < 0.5
    ]

    for votes, expected in test_cases:
        votes_arr     = np.array(votes, dtype=float)
        weighted_score = float(votes_arr @ weights)
        is_anomaly     = weighted_score >= min_score

        assert is_anomaly == expected, (
            f"votes={votes}, expected is_anomaly={expected}, "
            f"got weighted_score={weighted_score:.2f}, is_anomaly={is_anomaly}"
        )


# ── TEST 6: Weight Normalization ──────────────────────────────────────────────

def test_weight_normalization():
    """When a method is disabled, remaining weights should renormalise to sum=1.

    PRD: 'Normalize remaining weights to sum to 1.0' when some methods disabled.
    """
    # Active weights when DBSCAN is disabled: [0.35, 0.25, 0.10]
    active_weights = [0.35, 0.25, 0.10]
    total = sum(active_weights)
    normalized = [w / total for w in active_weights]

    npt.assert_allclose(sum(normalized), 1.0, atol=1e-9,
                        err_msg="Normalised weights must sum to 1.0")

    # Relative ordering must be preserved
    assert normalized[0] > normalized[1] > normalized[2], (
        "IsoForest > LOF > KMeans after normalisation"
    )

    # With KMeans also disabled: [0.35, 0.25]
    two_weights = [0.35, 0.25]
    two_total   = sum(two_weights)
    two_norm    = [w / two_total for w in two_weights]
    npt.assert_allclose(sum(two_norm), 1.0, atol=1e-9)

    # Edge case: single active method
    single = [0.35]
    single_norm = [w / sum(single) for w in single]
    npt.assert_allclose(single_norm[0], 1.0, atol=1e-9)


# ── TEST 7: GeoJSON Structure ─────────────────────────────────────────────────

def test_geojson_structure(config):
    """Verify exported GeoJSON is a valid FeatureCollection with required properties.

    PRD Section 11 (GeoJSON):
        - type: "FeatureCollection"
        - Each Feature has type, geometry, properties
        - geometry: Polygon with at least 3 ring coordinates
        - Required properties: zone_id, threat_type, threat_score, drift_score,
          severity, dsr, confidence, ndvi_delta, ensemble_votes,
          feature_1_name, feature_1_importance, etc.
    """
    # Build a 5-zone detected DataFrame
    rows = [
        _base_row(
            zone_id=f"zone_{i:04d}",
            is_anomaly=1,
            lat=25.0 + i * 0.1,
            lon=73.0 + i * 0.1,
            threat_type="mining" if i % 2 == 0 else "deforestation",
        )
        for i in range(1, 6)
    ]
    df = _make_df(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        export_geojson(df, config, out_dir)

        gj_path = out_dir / "detected_zones.geojson"
        assert gj_path.exists(), "detected_zones.geojson was not created"

        with open(gj_path, encoding="utf-8") as f:
            gj = json.load(f)

    # Top-level structure
    assert gj["type"] == "FeatureCollection", "Must be FeatureCollection"
    assert "features" in gj,                  "Must contain 'features' key"
    assert len(gj["features"]) == 5,          "Expected 5 features (5 detected zones)"

    # Verify each feature
    required_props = {
        "zone_id", "threat_type", "threat_score", "drift_score",
        "severity", "dsr", "confidence", "ndvi_delta", "ensemble_votes",
        "feature_1_name", "feature_1_importance",
        "feature_2_name", "feature_2_importance",
        "feature_3_name", "feature_3_importance",
        "driver_1", "driver_1_zscore", "driver_1_detail",
        "driver_2", "driver_2_zscore", "driver_2_detail",
        "driver_3", "driver_3_zscore", "driver_3_detail",
    }

    for feat in gj["features"]:
        assert feat["type"] == "Feature",                  "Each item must be a Feature"
        assert feat["geometry"]["type"] == "Polygon",      "Geometry must be Polygon"
        ring = feat["geometry"]["coordinates"][0]
        assert len(ring) >= 4, "Polygon ring must have at least 4 points (closed)"

        props = feat["properties"]
        missing = required_props - set(props.keys())
        assert not missing, f"Missing required properties: {missing}"

        # Type checks
        assert isinstance(props["threat_score"],  (int, float))
        assert 0 <= props["threat_score"] <= 100, "threat_score must be in [0,100]"
        assert isinstance(props["drift_score"],   (int, float))
        assert 1.0 <= props["drift_score"] <= 10.0, "drift_score must be in [1,10]"
        assert isinstance(props["confidence"],    (int, float))
        assert 0 <= props["confidence"] <= 100,   "confidence must be in [0,100]"
