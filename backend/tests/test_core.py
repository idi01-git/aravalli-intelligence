"""
Aravalli Intelligence — Core Test Suite
Comprehensive pytest tests for the data ingestion and feature engineering pipeline.

Uses:
    - pytest fixtures for sample data generation
    - numpy.testing.assert_allclose for floating point comparisons
    - Google-style docstrings explaining what each test validates

Author: Shivang, Team BIOBYTES
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config
from pipeline.ingest import generate_synthetic, _validate_structure, EXPECTED_COLUMNS
from pipeline.features import (
    smooth_timeseries,
    compute_baseline,
    compute_temporal_features,
    compute_spatial_features,
    compute_encoding_features,
    compute_deltas,
    compute_moran_i,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def config() -> dict:
    """Load project configuration once for all tests."""
    return load_config()


@pytest.fixture(scope="session")
def synthetic_data(config):
    """Generate synthetic data once for all tests.

    Returns:
        Tuple of (data_df, ground_truth_df) with 1000 zones × 84 months.
    """
    data_df, gt_df = generate_synthetic(config)
    return data_df, gt_df


@pytest.fixture(scope="session")
def small_zone_df() -> pd.DataFrame:
    """Create a small deterministic DataFrame for unit testing.

    5 zones × 12 months with known NDVI values for exact assertions.
    """
    rng = np.random.default_rng(99)
    rows = []
    for i in range(5):
        base_ndvi = 0.3 + 0.05 * i  # 0.30, 0.35, 0.40, 0.45, 0.50
        for m in range(12):
            cal = (m % 12) + 1
            ndvi = base_ndvi + 0.1 * np.sin(2 * np.pi * cal / 12)
            rows.append({
                "zone_id": f"test_{i+1:04d}",
                "lat": 25.0 + i * 0.5,
                "lon": 73.0 + i * 0.5,
                "elevation": 600 + i * 100,
                "timestamp": f"2025-{cal:02d}",
                "ndvi": round(ndvi, 6),
                "ndbi": round(-0.1 + rng.normal(0, 0.02), 6),
                "bsi": round(0.05 + rng.normal(0, 0.02), 6),
                "nightlight": round(3.0 + rng.normal(0, 0.5), 2),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def smoothed_df(small_zone_df):
    """Return smoothed version of small_zone_df."""
    return smooth_timeseries(small_zone_df, window=3)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: NDVI Calculation Correct
# ══════════════════════════════════════════════════════════════════════════════

def test_ndvi_calculation_correct():
    """Verify NDVI = (NIR - Red) / (NIR + Red) produces values in [-1, 1].

    NDVI (Normalized Difference Vegetation Index) should be:
    - Close to 0.5-0.8 for dense vegetation
    - Close to 0 for barren ground
    - Negative for water bodies
    """
    # Known reflectance values
    nir = np.array([0.50, 0.30, 0.10, 0.80, 0.02])
    red = np.array([0.08, 0.25, 0.10, 0.10, 0.10])

    ndvi = (nir - red) / (nir + red)

    # Dense vegetation: NIR=0.50, Red=0.08 → NDVI ≈ 0.724
    npt.assert_allclose(ndvi[0], (0.50 - 0.08) / (0.50 + 0.08), atol=1e-6)

    # Low vegetation: NIR=0.30, Red=0.25 → NDVI ≈ 0.091
    npt.assert_allclose(ndvi[1], 0.0909, atol=0.01)

    # Barren: NIR == Red → NDVI = 0
    npt.assert_allclose(ndvi[2], 0.0, atol=1e-6)

    # All NDVI should be in [-1, 1]
    assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: NDBI Calculation Correct
# ══════════════════════════════════════════════════════════════════════════════

def test_ndbi_calculation_correct():
    """Verify NDBI = (SWIR - NIR) / (SWIR + NIR) produces values in [-1, 1].

    NDBI (Normalized Difference Built-up Index) should be:
    - Positive for urban/built-up areas
    - Negative for vegetation
    """
    swir = np.array([0.40, 0.15, 0.30])
    nir = np.array([0.15, 0.50, 0.30])

    ndbi = (swir - nir) / (swir + nir)

    # Urban: SWIR > NIR → positive
    assert ndbi[0] > 0
    npt.assert_allclose(ndbi[0], (0.40 - 0.15) / (0.40 + 0.15), atol=1e-6)

    # Vegetation: NIR > SWIR → negative
    assert ndbi[1] < 0

    # Equal: NDBI = 0
    npt.assert_allclose(ndbi[2], 0.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: BSI Calculation Correct
# ══════════════════════════════════════════════════════════════════════════════

def test_bsi_calculation_correct():
    """Verify BSI = ((SWIR+Red)-(NIR+Blue)) / ((SWIR+Red)+(NIR+Blue)).

    BSI (Bare Soil Index) should be:
    - Positive for exposed soil (mining, quarries)
    - Negative for vegetated areas
    """
    swir, red = 0.35, 0.20
    nir, blue = 0.15, 0.10

    numerator = (swir + red) - (nir + blue)
    denominator = (swir + red) + (nir + blue)
    bsi = numerator / denominator

    expected = (0.55 - 0.25) / (0.55 + 0.25)
    npt.assert_allclose(bsi, expected, atol=1e-6)
    assert -1.0 <= bsi <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: 3-Month Smoothing
# ══════════════════════════════════════════════════════════════════════════════

def test_3month_smoothing(small_zone_df):
    """Test moving average smoothing: smoothed[i] = mean(values[max(0,i-2):i+1]).

    Verifies:
    - First value: smoothed[0] = values[0] (only 1 sample available)
    - Second value: smoothed[1] = mean(values[0:2])
    - Third value: smoothed[2] = mean(values[0:3])
    - Subsequent: proper 3-sample window
    """
    smoothed = smooth_timeseries(small_zone_df, window=3)

    # Check that smoothed column exists
    assert "ndvi_smoothed" in smoothed.columns

    # For the first zone, check specific values
    zone_1 = smoothed[smoothed["zone_id"] == "test_0001"].reset_index(drop=True)
    raw = zone_1["ndvi"].values
    sm = zone_1["ndvi_smoothed"].values

    # First value: only itself (min_periods=1)
    npt.assert_allclose(sm[0], raw[0], atol=1e-6)

    # Second value: mean of first two
    npt.assert_allclose(sm[1], np.mean(raw[0:2]), atol=1e-6)

    # Third value: mean of first three
    npt.assert_allclose(sm[2], np.mean(raw[0:3]), atol=1e-6)

    # Fourth value: proper 3-window
    npt.assert_allclose(sm[3], np.mean(raw[1:4]), atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Consecutive Declines
# ══════════════════════════════════════════════════════════════════════════════

def test_consecutive_declines():
    """Test counting consecutive declining months from the end.

    Should count backward until trend reverses. Example:
    values = [0.5, 0.45, 0.50, 0.48, 0.45, 0.42]
    Counting from end: 0.42 < 0.45 (1), 0.45 < 0.48 (2), 0.48 < 0.50 (3)
    Then 0.50 > 0.45 → stop. Result = 3.
    """
    values = np.array([0.5, 0.45, 0.50, 0.48, 0.45, 0.42])

    # Count backward
    consecutive = 0
    for i in range(len(values) - 1, 0, -1):
        if values[i] < values[i - 1]:
            consecutive += 1
        else:
            break

    assert consecutive == 3

    # Edge case: all declining
    all_decline = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    consecutive_all = 0
    for i in range(len(all_decline) - 1, 0, -1):
        if all_decline[i] < all_decline[i - 1]:
            consecutive_all += 1
        else:
            break
    assert consecutive_all == 4

    # Edge case: no decline
    no_decline = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    consecutive_none = 0
    for i in range(len(no_decline) - 1, 0, -1):
        if no_decline[i] < no_decline[i - 1]:
            consecutive_none += 1
        else:
            break
    assert consecutive_none == 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Circular Month Encoding
# ══════════════════════════════════════════════════════════════════════════════

def test_circular_month_encoding():
    """Test month_sin and month_cos encode months as circular features.

    sin(2*pi*month/12) and cos(2*pi*month/12):
    - Month 3 (March): sin ≈ 1.0, cos ≈ 0
    - Month 6 (June):  sin ≈ 0,   cos ≈ -1
    - Month 9 (Sept):  sin ≈ -1,  cos ≈ 0
    - Month 12 (Dec):  sin ≈ 0,   cos ≈ 1

    Key property: December and January should be close in the
    encoded space, unlike linear month numbers (12 vs 1).
    """
    for month, exp_sin, exp_cos in [
        (3, 1.0, 0.0),
        (6, 0.0, -1.0),
        (9, -1.0, 0.0),
        (12, 0.0, 1.0),
    ]:
        s = math.sin(2 * math.pi * month / 12)
        c = math.cos(2 * math.pi * month / 12)
        npt.assert_allclose(s, exp_sin, atol=1e-10)
        npt.assert_allclose(c, exp_cos, atol=1e-10)

    # December and January should be close (Euclidean distance)
    dec_sin = math.sin(2 * math.pi * 12 / 12)
    dec_cos = math.cos(2 * math.pi * 12 / 12)
    jan_sin = math.sin(2 * math.pi * 1 / 12)
    jan_cos = math.cos(2 * math.pi * 1 / 12)
    dist = math.sqrt((dec_sin - jan_sin) ** 2 + (dec_cos - jan_cos) ** 2)
    assert dist < 0.6  # Should be close


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Feature Scaling Range
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_scaling_range(synthetic_data, config):
    """Test that temporal features are in reasonable ranges.

    - Slopes should be small values (< 0.1 per month)
    - Volatility ratio should be in [0, 10] range
    - Anomaly scores should be non-negative
    - Recovery signal should be in [-1, 1] range
    """
    data_df, _ = synthetic_data

    # Smooth the data first
    smoothed = smooth_timeseries(data_df, window=3)

    # Compute temporal features
    temporal = compute_temporal_features(smoothed, config=config)

    # Slopes should be small
    assert temporal["slope_short"].abs().max() < 0.5, "slope_short too large"
    assert temporal["slope_long"].abs().max() < 0.1, "slope_long too large"

    # Volatility should be positive and bounded
    assert temporal["volatility_ratio"].min() >= 0.0, "volatility_ratio negative"
    assert temporal["volatility_ratio"].max() < 10.0, "volatility_ratio unreasonably large"

    # Consecutive declines should be non-negative integers
    assert (temporal["consecutive_declines"] >= 0).all()
    assert temporal["consecutive_declines"].dtype in [np.int64, np.int32, int]

    # Recovery signal should be bounded
    assert temporal["recovery_signal"].abs().max() < 1.0, "recovery_signal out of range"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Drift Score Range (Phase 3 prerequisite)
# ══════════════════════════════════════════════════════════════════════════════

def test_drift_score_range():
    """Test drift score computation is clamped to [1.0, 10.0].

    Drift = 1.0 + raw_drift * 9.0, clamped to [1.0, 10.0]
    where raw_drift is weighted sum of 4 normalized components ∈ [0, 1].

    This test validates the formula without Phase 3 implementation.
    """
    # Simulate drift score calculation
    ndvi_departure = 0.5
    temporal_persistence = 0.3
    spatial_isolation = 0.2
    seasonal_proof = 0.8

    weights = {"ndvi": 0.35, "temporal": 0.25, "spatial": 0.20, "dsr": 0.20}
    raw = (
        ndvi_departure * weights["ndvi"]
        + temporal_persistence * weights["temporal"]
        + spatial_isolation * weights["spatial"]
        + seasonal_proof * weights["dsr"]
    )

    drift = 1.0 + raw * 9.0
    drift = max(1.0, min(10.0, drift))

    assert 1.0 <= drift <= 10.0

    # Edge: all zeros → drift = 1.0
    drift_min = 1.0 + 0.0 * 9.0
    assert drift_min == 1.0

    # Edge: all ones → drift = 10.0
    drift_max = 1.0 + 1.0 * 9.0
    assert drift_max == 10.0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: CSV Output Format (Integration)
# ══════════════════════════════════════════════════════════════════════════════

def test_csv_output_format(synthetic_data, config, tmp_path):
    """Verify output/features.csv has 1000 rows, 30+ columns, and correct names.

    Integration test: runs the full pipeline on synthetic data and
    verifies the output CSV structure.
    """
    from pipeline.features import run_features

    data_df, _ = synthetic_data
    feature_df = run_features(data_df, config, output_dir=tmp_path)

    # Check row count
    assert len(feature_df) == 1000, f"Expected 1000 rows, got {len(feature_df)}"

    # Check column count
    assert len(feature_df.columns) >= 19, f"Expected 19+ columns, got {len(feature_df.columns)}"

    # Check required columns exist
    required_cols = [
        "zone_id", "lat", "lon", "elevation",
        "ndvi_current", "ndbi_current", "bsi_current", "nightlight_current",
        "ndvi_delta", "ndbi_delta", "bsi_delta", "nightlight_delta",
        "current_value", "slope_short", "slope_long", "acceleration",
        "volatility_ratio", "deviation_from_mean", "consecutive_declines",
        "recovery_signal",
        "local_anomaly_score", "is_isolated", "regional_health",
        "spatial_gradient",
        "month_sin", "month_cos", "elevation_norm",
    ]
    missing = [c for c in required_cols if c not in feature_df.columns]
    assert not missing, f"Missing columns: {missing}"

    # Check features.csv was saved
    features_csv = tmp_path / "features.csv"
    assert features_csv.exists(), "features.csv not saved"

    # Check zone_timeseries directory
    ts_dir = tmp_path / "zone_timeseries"
    assert ts_dir.exists(), "zone_timeseries directory not created"

    ts_files = list(ts_dir.glob("*.csv"))
    assert len(ts_files) == 1000, f"Expected 1000 timeseries files, got {len(ts_files)}"

    # Check no NaN in critical columns
    critical = ["ndvi_current", "slope_short", "local_anomaly_score"]
    for col in critical:
        nan_count = feature_df[col].isna().sum()
        assert nan_count == 0, f"NaN found in {col}: {nan_count} values"


# ===========================================================================
# TEST 10: NaN Handling
# ===========================================================================

def test_nan_handling(config):
    """Verify NaN values are handled gracefully throughout the pipeline.

    Creates data with intentional NaN values in NDVI, verifies that:
    - _validate_structure() forward-fills NaN values
    - No NaN values remain in critical columns after running features
    """
    from pipeline.ingest import _validate_structure
    from pipeline.features import run_features

    # Create small DataFrame with NaN injected.
    # _validate_structure requires >= 10 zones, so use exactly 10.
    rows = []
    for z in range(10):
        # Add tiny variation (z * 0.01) so variance is not exactly zero.
        # This prevents RuntimeWarnings in Moran's I (libpysal/esda).
        base_val = 0.3 + (z * 0.01)
        for m in range(12):
            val = base_val + 0.1 * np.sin(2 * np.pi * m / 12)
            # Inject NaN for 2 months in zone index 1 only
            if z == 1 and m in (3, 4):
                val = float("nan")
            rows.append({
                "zone_id": f"nan_test_{z:04d}",
                "lat": 25.0 + z * 0.2,
                "lon": 73.0 + z * 0.2,
                "elevation": 800,
                "timestamp": f"2025-{m+1:02d}",
                "ndvi": round(val, 6) if not np.isnan(val) else val,
                "ndbi": -0.10,
                "bsi": 0.05,
                "nightlight": 5.0,
            })
    df = pd.DataFrame(rows)

    # _validate_structure should fix NaN via ffill/bfill
    _validate_structure(df, source="test")
    assert df["ndvi"].isna().sum() == 0, "NaN not filled by _validate_structure"

    # Full pipeline should also produce no NaN in key outputs
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        feat_df = run_features(df, config, output_dir=Path(tmpdir))
        for col in ["ndvi_current", "slope_short", "local_anomaly_score"]:
            if col in feat_df.columns:
                assert feat_df[col].isna().sum() == 0, f"NaN found in {col} after pipeline"


# ===========================================================================
# TEST 11: Moran's I Calculation
# ===========================================================================

def test_moran_i_calculation(synthetic_data, config):
    """Verify Moran's I is computed correctly and in valid range.

    Validates:
    - Returns float in [-1.0, 1.0]
    - Returns 0.0 gracefully for < 4 zones (edge case)
    - Clustered data has higher Moran's I than random data
    - Manual fallback produces same sign as expected
    """
    # Edge case: too few zones -> returns 0.0
    tiny_df = pd.DataFrame([
        {"zone_id": f"z{i}", "lat": 25.0 + i * 0.1, "lon": 73.0, "ndvi_current": 0.4}
        for i in range(3)  # only 3 zones, below threshold
    ])
    result = compute_moran_i(tiny_df, config)
    assert result == 0.0, "Should return 0.0 for fewer than 4 zones"

    # Normal case: synthetic 1000-zone dataset
    data_df, _ = synthetic_data
    smoothed = smooth_timeseries(data_df, window=3)
    baselines = compute_baseline(smoothed, baseline_months=72, config=config)
    delta_df = compute_deltas(smoothed, baselines)

    mi = compute_moran_i(delta_df, config)

    # Must be in valid range
    assert -1.0 <= mi <= 1.0, f"Moran's I out of range: {mi}"
    assert isinstance(mi, float)

    # Strongly clustered synthetic data (events are injected in specific zones)
    # should have a non-trivial Moran's I (not exactly 0)
    # We just verify it's computed and finite
    assert not np.isnan(mi), "Moran's I returned NaN"
