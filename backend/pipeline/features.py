"""
Aravalli Intelligence — Feature Engineering Module
Computes 19 features per zone from 84-month timeseries data.

Feature Groups:
    A. 8 Temporal features (from NDVI timeseries)
    B. 4 Spatial features (KNN with k=8)
    C. 4 Index deltas (current vs baseline)
    D. 3 Encodings (month_sin, month_cos, elevation_norm)

Also:
    - 3-month moving average smoothing
    - Adaptive per-zone seasonal baseline (6-year history)
    - Changepoint detection via ruptures Pelt

Author: Shivang, Team BIOBYTES
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# ── A. SMOOTHING ────────────────────────────────────────────────────────────

def smooth_timeseries(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Apply moving average smoothing to NDVI, NDBI, BSI timeseries.

    Statistical reasoning: Sentinel-2 has per-pixel noise from atmospheric
    effects and cloud shadow residuals. A 3-month moving average removes
    single-month outliers while preserving seasonal trends (monsoon period
    is 3+ months, so the signal survives).

    Formula: smoothed[i] = mean(values[max(0, i-window+1) : i+1])

    Args:
        df: DataFrame with timeseries data, must have zone_id column
            and ndvi, ndbi, bsi columns.
        window: Window size for moving average (default 3 months).

    Returns:
        DataFrame with added columns: ndvi_smoothed, ndbi_smoothed, bsi_smoothed.
    """
    logger.info("Smoothing timeseries with window=%d", window)
    result = df.copy()

    for col in ["ndvi", "ndbi", "bsi"]:
        smoothed_col = f"{col}_smoothed"
        result[smoothed_col] = (
            result.groupby("zone_id")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )

    logger.info("Smoothing complete. Added columns: ndvi_smoothed, ndbi_smoothed, bsi_smoothed")
    return result


# ── B. ADAPTIVE BASELINE ───────────────────────────────────────────────────

def compute_baseline(
    df: pd.DataFrame,
    baseline_months: int = 72,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Compute adaptive per-zone seasonal baselines from 6-year history.

    Statistical reasoning: Each calendar month (Jan-Dec) has its own
    baseline computed from 6 historical samples. This captures the
    seasonal cycle precisely — a monsoon NDVI of 0.55 is normal,
    but a January NDVI of 0.55 in a dry zone is anomalous.

    Args:
        df: Smoothed DataFrame with zone_id, timestamp, ndvi_smoothed etc.
        baseline_months: Number of months for baseline (default 72 = 6 years).
        config: Optional config for regional fallback values.

    Returns:
        Dict structure: {zone_id: {cal_month: {mean_ndvi, std_ndvi, mean_ndbi, ...}}}
    """
    logger.info("Computing adaptive baselines from months 1-%d", baseline_months)
    baselines: dict[str, dict[int, dict[str, float]]] = {}

    # Regional fallbacks from config
    fallback = {}
    if config:
        fb = config.get("baseline", {}).get("regional_fallback", {})
        fallback = {"ndvi": fb.get("ndvi", 0.35), "ndbi": fb.get("ndbi", -0.10), "bsi": fb.get("bsi", 0.05)}

    for zone_id, zone_df in df.groupby("zone_id"):
        zone_df = zone_df.sort_values("timestamp").reset_index(drop=True)
        baseline_data = zone_df.iloc[:baseline_months]

        zone_baselines: dict[int, dict[str, float]] = {}
        for cal_month in range(1, 13):
            # Extract all readings for this calendar month from baseline period
            mask = (baseline_data.index % 12) + 1 == cal_month
            month_data = baseline_data[mask]

            if len(month_data) < 2:
                # Insufficient history — use regional fallback
                zone_baselines[cal_month] = {
                    "mean_ndvi": fallback.get("ndvi", 0.35),
                    "std_ndvi": 0.05,
                    "mean_ndbi": fallback.get("ndbi", -0.10),
                    "std_ndbi": 0.03,
                    "mean_bsi": fallback.get("bsi", 0.05),
                    "std_bsi": 0.03,
                    "mean_nightlight": 3.0,
                    "std_nightlight": 1.0,
                    "n_samples": len(month_data),
                }
            else:
                zone_baselines[cal_month] = {
                    "mean_ndvi": float(month_data["ndvi_smoothed"].mean()),
                    "std_ndvi": float(month_data["ndvi_smoothed"].std()),
                    "mean_ndbi": float(month_data["ndbi_smoothed"].mean()),
                    "std_ndbi": float(month_data["ndbi_smoothed"].std()),
                    "mean_bsi": float(month_data["bsi_smoothed"].mean()),
                    "std_bsi": float(month_data["bsi_smoothed"].std()),
                    "mean_nightlight": float(month_data["nightlight"].mean()),
                    "std_nightlight": float(month_data["nightlight"].std()),
                    "n_samples": len(month_data),
                }

        baselines[str(zone_id)] = zone_baselines

    logger.info("Baselines computed for %d zones (%d calendar months each)", len(baselines), 12)
    return baselines


# ── C. INDEX DELTAS ─────────────────────────────────────────────────────────

def compute_deltas(
    df: pd.DataFrame,
    baselines: dict[str, dict[int, dict[str, float]]],
) -> pd.DataFrame:
    """Compute current vs baseline deltas for month 84 (December 2025).

    Statistical reasoning: The delta isolates the deviation from the
    expected seasonal value. If delta_ndvi = -0.15, the zone lost 0.15
    NDVI units MORE than normal for that calendar month.

    Args:
        df: DataFrame with smoothed indices.
        baselines: Baseline dict from compute_baseline.

    Returns:
        DataFrame with one row per zone, containing delta columns.
    """
    logger.info("Computing index deltas (current vs baseline)")
    delta_rows: list[dict[str, Any]] = []

    for zone_id, zone_df in df.groupby("zone_id"):
        zone_df = zone_df.sort_values("timestamp").reset_index(drop=True)
        latest_idx = len(zone_df) - 1
        latest = zone_df.iloc[latest_idx]

        # Calendar month of latest observation
        cal_month = (latest_idx % 12) + 1
        bl = baselines.get(str(zone_id), {}).get(cal_month, {})

        # Nightlight delta uses overall mean (not monthly)
        all_nightlight = zone_df["nightlight"].iloc[:72]
        nl_baseline = float(all_nightlight.mean()) if len(all_nightlight) > 0 else 3.0

        delta_rows.append({
            "zone_id": zone_id,
            "lat": latest["lat"],
            "lon": latest["lon"],
            "elevation": latest["elevation"],
            "ndvi_current": float(latest.get("ndvi_smoothed", latest["ndvi"])),
            "ndbi_current": float(latest.get("ndbi_smoothed", latest["ndbi"])),
            "bsi_current": float(latest.get("bsi_smoothed", latest["bsi"])),
            "nightlight_current": float(latest["nightlight"]),
            "ndvi_delta": float(latest.get("ndvi_smoothed", latest["ndvi"])) - bl.get("mean_ndvi", 0.35),
            "ndbi_delta": float(latest.get("ndbi_smoothed", latest["ndbi"])) - bl.get("mean_ndbi", -0.10),
            "bsi_delta": float(latest.get("bsi_smoothed", latest["bsi"])) - bl.get("mean_bsi", 0.05),
            "nightlight_delta": float(latest["nightlight"]) - nl_baseline,
        })

    result = pd.DataFrame(delta_rows)
    logger.info("Deltas computed for %d zones", len(result))
    return result


# ── D. TEMPORAL FEATURES ───────────────────────────────────────────────────

def compute_temporal_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Compute 8 temporal features from each zone's smoothed NDVI timeseries.

    Statistical reasoning:
    - slope_short catches rapid recent changes (last 3 months)
    - slope_long captures the 7-year trend
    - acceleration = short - long: positive means recent deterioration
    - volatility_ratio flags zones that became unstable recently
    - consecutive_declines proves the trend is persistent, not noise
    - recovery_signal distinguishes active degradation from recovered areas

    Args:
        df: Full DataFrame with ndvi_smoothed column.
        config: Optional config for short window size.

    Returns:
        DataFrame with one row per zone with 8 temporal feature columns.
    """
    short_window = 3
    if config:
        short_window = config.get("baseline", {}).get("slope_short_window", 3)

    logger.info("Computing temporal features (short_window=%d)", short_window)
    feature_rows: list[dict[str, Any]] = []

    for zone_id, zone_df in df.groupby("zone_id"):
        zone_df = zone_df.sort_values("timestamp").reset_index(drop=True)
        ndvi = zone_df["ndvi_smoothed"].values if "ndvi_smoothed" in zone_df.columns else zone_df["ndvi"].values
        n = len(ndvi)

        # current_value
        current_value = float(ndvi[-1])

        # slope_short: polyfit slope of last `short_window` values
        if n >= short_window:
            x_short = np.arange(short_window)
            slope_short = float(np.polyfit(x_short, ndvi[-short_window:], 1)[0])
        else:
            slope_short = 0.0

        # slope_long: polyfit slope of all values
        x_long = np.arange(n)
        slope_long = float(np.polyfit(x_long, ndvi, 1)[0])

        # acceleration
        acceleration = slope_short - slope_long

        # volatility_ratio
        recent_std = float(np.std(ndvi[-12:])) if n >= 12 else float(np.std(ndvi))
        total_std = float(np.std(ndvi))
        volatility_ratio = recent_std / (total_std + 1e-10)

        # deviation_from_mean
        deviation_from_mean = current_value - float(np.mean(ndvi))

        # consecutive_declines: count backwards from end
        consecutive_declines = 0
        for i in range(n - 1, 0, -1):
            if ndvi[i] < ndvi[i - 1]:
                consecutive_declines += 1
            else:
                break

        # recovery_signal: mean(second half) - mean(first half)
        half = n // 2
        recovery_signal = float(np.mean(ndvi[half:])) - float(np.mean(ndvi[:half]))

        feature_rows.append({
            "zone_id": zone_id,
            "current_value": current_value,
            "slope_short": slope_short,
            "slope_long": slope_long,
            "acceleration": acceleration,
            "volatility_ratio": volatility_ratio,
            "deviation_from_mean": deviation_from_mean,
            "consecutive_declines": consecutive_declines,
            "recovery_signal": recovery_signal,
        })

    result = pd.DataFrame(feature_rows)
    logger.info("Temporal features computed for %d zones", len(result))
    return result


# ── E. SPATIAL FEATURES ────────────────────────────────────────────────────

def compute_spatial_features(
    delta_df: pd.DataFrame,
    k: int = 8,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute 4 spatial features using KNN with k=8.

    Statistical reasoning:
    - local_anomaly_score: high value = this zone differs from neighbors
    - is_isolated: binary flag for zones declining alone
    - regional_health: context — is this a broadly degraded area?
    - spatial_gradient: does NDVI decline with proximity to this zone?

    Uses sklearn.neighbors.NearestNeighbors on (lat, lon) coordinates.

    Args:
        delta_df: DataFrame with zone_id, lat, lon, ndvi_current columns.
        k: Number of nearest neighbors (default 8).
        config: Optional config.

    Returns:
        DataFrame with spatial feature columns added.
    """
    if config:
        k = config.get("spatial", {}).get("k_neighbors", k)

    health_threshold = 0.30
    if config:
        health_threshold = config.get("baseline", {}).get("ndvi_health_threshold", 0.30)

    logger.info("Computing spatial features (k=%d, health_threshold=%.2f)", k, health_threshold)

    coords = delta_df[["lat", "lon"]].values
    ndvi_vals = delta_df["ndvi_current"].values
    n_zones = len(delta_df)

    # Adjust k if we have fewer zones
    k_actual = min(k, n_zones - 1)
    if k_actual < 1:
        logger.warning("Too few zones for spatial features, using defaults")
        delta_df["local_anomaly_score"] = 0.0
        delta_df["is_isolated"] = 0
        delta_df["regional_health"] = 1.0
        delta_df["spatial_gradient"] = 0.0
        return delta_df

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="haversine")
    # Convert degrees to radians for haversine
    coords_rad = np.radians(coords)
    nn.fit(coords_rad)
    distances, indices = nn.kneighbors(coords_rad)

    anomaly_scores = np.zeros(n_zones)
    is_isolated = np.zeros(n_zones, dtype=int)
    regional_health = np.zeros(n_zones)
    spatial_gradient = np.zeros(n_zones)

    for i in range(n_zones):
        # Exclude self (index 0 in the results)
        neighbor_idx = indices[i, 1:]
        neighbor_dists = distances[i, 1:]
        neighbor_ndvi = ndvi_vals[neighbor_idx]

        n_mean = np.mean(neighbor_ndvi)
        n_std = np.std(neighbor_ndvi)

        # local_anomaly_score
        anomaly_scores[i] = abs(ndvi_vals[i] - n_mean) / (n_std + 1e-10)

        # is_isolated
        is_isolated[i] = 1 if ndvi_vals[i] < (n_mean - n_std) else 0

        # regional_health: fraction of neighbors above threshold
        regional_health[i] = np.mean(neighbor_ndvi > health_threshold)

        # spatial_gradient: slope of NDVI vs distance
        if len(neighbor_dists) >= 2:
            spatial_gradient[i] = float(np.polyfit(neighbor_dists, neighbor_ndvi, 1)[0])
        else:
            spatial_gradient[i] = 0.0

    result = delta_df.copy()
    result["local_anomaly_score"] = anomaly_scores
    result["is_isolated"] = is_isolated
    result["regional_health"] = regional_health
    result["spatial_gradient"] = spatial_gradient

    logger.info(
        "Spatial features: mean_anomaly=%.3f, isolated_count=%d, mean_health=%.2f",
        np.mean(anomaly_scores), np.sum(is_isolated), np.mean(regional_health),
    )
    return result


# ── F. CIRCULAR ENCODING + ELEVATION ──────────────────────────────────────

def compute_encoding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circular month encoding and normalized elevation.

    Statistical reasoning: Circular encoding preserves the fact that
    December (12) and January (1) are adjacent, unlike linear month
    numbers which create an artificial gap.

    Args:
        df: DataFrame with elevation column.

    Returns:
        DataFrame with month_sin, month_cos, elevation_norm columns.
    """
    latest_month = 12  # December 2025 (month 84 = December)
    df = df.copy()
    df["month_sin"] = math.sin(2 * math.pi * latest_month / 12)
    df["month_cos"] = math.cos(2 * math.pi * latest_month / 12)
    df["elevation_norm"] = (df["elevation"] - 500) / (1500 - 500)
    df["elevation_norm"] = df["elevation_norm"].clip(0.0, 1.0)

    logger.info("Encoding features added: month_sin, month_cos, elevation_norm")
    return df


# ── G. CHANGEPOINT DETECTION ──────────────────────────────────────────────

def detect_changepoints(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Detect changepoints in NDVI timeseries using ruptures Pelt.

    Statistical reasoning: Changepoint detection finds structural breaks
    in the timeseries — moments where the statistical properties (mean,
    variance) fundamentally shift. This complements the temporal features
    which measure gradual trends.

    Args:
        df: Full DataFrame with ndvi_smoothed column.
        config: Optional config for model, penalty, min_segment.

    Returns:
        Dict: {zone_id: [{index: int, magnitude: float, direction: str}]}
    """
    model = "rbf"
    penalty = 3.0
    min_size = 3
    if config:
        cp = config.get("changepoint", {})
        model = cp.get("model", "rbf")
        penalty = cp.get("penalty", 3.0)
        min_size = cp.get("min_segment_length", 3)

    logger.info("Detecting changepoints (model=%s, penalty=%.1f)", model, penalty)

    try:
        import ruptures as rpt
    except ImportError:
        logger.warning("ruptures not installed — skipping changepoint detection")
        return {}

    changepoints: dict[str, list[dict[str, Any]]] = {}

    for zone_id, zone_df in df.groupby("zone_id"):
        zone_df = zone_df.sort_values("timestamp").reset_index(drop=True)
        signal = zone_df["ndvi_smoothed"].values if "ndvi_smoothed" in zone_df.columns else zone_df["ndvi"].values

        try:
            algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
            bkps = algo.predict(pen=penalty)
            # Remove the last element (always the signal length)
            bkps = [b for b in bkps if b < len(signal)]

            zone_cps: list[dict[str, Any]] = []
            for bp in bkps:
                if bp > 0 and bp < len(signal):
                    before = float(np.mean(signal[max(0, bp - 3):bp]))
                    after = float(np.mean(signal[bp:min(len(signal), bp + 3)]))
                    magnitude = after - before
                    direction = "decline" if magnitude < 0 else "increase"
                    zone_cps.append({
                        "index": bp,
                        "magnitude": round(magnitude, 4),
                        "direction": direction,
                    })
            changepoints[str(zone_id)] = zone_cps
        except Exception as e:
            logger.debug("Changepoint detection failed for %s: %s", zone_id, e)
            changepoints[str(zone_id)] = []

    total_cps = sum(len(v) for v in changepoints.values())
    logger.info("Changepoints detected: %d total across %d zones", total_cps, len(changepoints))
    return changepoints


# ── H. MORAN'S I ─────────────────────────────────────────────────────

def compute_moran_i(
    feature_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> float:
    """Compute Moran's I for spatial autocorrelation of NDVI across study area.

    PRD Section 9 Specification:
    - Measures spatial clustering: > 0 = clustering, < 0 = dispersion, 0 = random.
    - Uses libpysal distance weights; falls back to manual if unavailable.
    - Config: moran_i_threshold (default 0.30).

    Statistical reasoning:
        Moran's I tells us whether degraded zones cluster together (like
        an illegal mine expanding outward) or appear randomly. A high
        Moran's I indicates coordinated, spatially correlated degradation
        rather than random sensor noise.

    Args:
        feature_df: Feature DataFrame with lat, lon, ndvi_current columns.
        config: Optional config dict.

    Returns:
        Moran's I statistic as float in range [-1, 1].
    """
    threshold = 0.30
    if config:
        threshold = config.get("spatial", {}).get("moran_i_threshold", 0.30)

    ndvi = feature_df["ndvi_current"].values
    lats = feature_df["lat"].values
    lons = feature_df["lon"].values
    n = len(ndvi)

    if n < 4:
        logger.warning("Too few zones for Moran's I calculation")
        return 0.0

    # ── Try libpysal ──────────────────────────────────────────────────
    try:
        from libpysal.weights import DistanceBand
        from esda.moran import Moran

        coords = list(zip(lons, lats))
        # Use 0.5 degree (~55km) as bandwidth for local spatial weights
        w = DistanceBand(coords, threshold=0.5, binary=True, silence_warnings=True)
        w.transform = "r"  # row-standardise
        mi = Moran(ndvi, w)
        result = float(mi.I)
        logger.info(
            "Moran's I (libpysal): %.4f (p=%.4f, threshold=%.2f)",
            result, mi.p_sim, threshold,
        )
        return result

    except ImportError:
        logger.info("libpysal not installed — using manual Moran's I calculation")
    except Exception as e:
        logger.warning("libpysal Moran's I failed (%s) — using manual fallback", e)

    # ── Manual fallback ───────────────────────────────────────────────
    # Build inverse-distance weight matrix (bandwidth = 0.5 degrees)
    x = ndvi - np.mean(ndvi)
    coords_arr = np.column_stack([lats, lons])

    # Compute pairwise distances (degrees, approximate)
    from sklearn.metrics import pairwise_distances
    dist = pairwise_distances(coords_arr, metric="euclidean")

    # Weight: 1/d for d > 0 and d < threshold; 0 otherwise
    bandwidth = 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where((dist > 0) & (dist < bandwidth), 1.0 / dist, 0.0)

    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums

    # Moran's I = (n / S0) * (x @ W @ x) / (x @ x)
    S0 = W.sum()
    if S0 == 0 or np.dot(x, x) == 0:
        return 0.0

    numerator = float(x @ W @ x)
    denominator = float(np.dot(x, x))
    moran_i = (n / S0) * (numerator / denominator)
    moran_i = float(np.clip(moran_i, -1.0, 1.0))

    logger.info(
        "Moran's I (manual): %.4f (threshold=%.2f, n_zones=%d)",
        moran_i, threshold, n,
    )
    return moran_i


# ── SAVE ────────────────────────────────────────────────────────────────────

def save_features(
    feature_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    changepoints: dict[str, list[dict[str, Any]]],
    baselines: dict[str, dict[int, dict[str, float]]],
    output_dir: Path,
    moran_i: float = 0.0,
) -> None:
    """Save features.csv and per-zone timeseries CSVs.

    Saves:
    - output/features.csv (1000 rows, 30+ columns)
    - output/zone_timeseries/zone_XXXX.csv (1000 files, 84 rows each)

    Args:
        feature_df: Feature DataFrame (1 row per zone).
        raw_df: Full smoothed raw DataFrame (84 rows per zone).
        changepoints: Changepoint dict from detect_changepoints.
        baselines: Baseline dict from compute_baseline.
        output_dir: Path to output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main features CSV
    features_path = output_dir / "features.csv"
    feature_df.to_csv(features_path, index=False)
    logger.info("Saved features.csv: %d rows, %d columns", len(feature_df), len(feature_df.columns))

    # Save global metadata (Moran's I + run stats)
    import json
    metadata = {
        "moran_i": round(moran_i, 6),
        "n_zones": len(feature_df),
        "n_features": len(feature_df.columns),
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    meta_path = output_dir / "features_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved features_metadata.json (Moran's I = %.4f)", moran_i)

    # Save per-zone timeseries
    ts_dir = output_dir / "zone_timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    for zone_id, zone_df in raw_df.groupby("zone_id"):
        zone_df = zone_df.sort_values("timestamp").reset_index(drop=True)
        ts_data = zone_df[["timestamp", "ndvi", "ndbi", "bsi", "nightlight"]].copy()

        # Add smoothed columns if available
        for col in ["ndvi_smoothed", "ndbi_smoothed", "bsi_smoothed"]:
            if col in zone_df.columns:
                ts_data[col] = zone_df[col].values

        # Add baseline
        bl = baselines.get(str(zone_id), {})
        baseline_ndvi = []
        for i in range(len(zone_df)):
            cal_month = (i % 12) + 1
            bl_month = bl.get(cal_month, {})
            baseline_ndvi.append(bl_month.get("mean_ndvi", np.nan))
        ts_data["ndvi_baseline"] = baseline_ndvi

        # Add delta
        ts_data["ndvi_delta"] = ts_data.get("ndvi_smoothed", ts_data["ndvi"]) - ts_data["ndvi_baseline"]

        # Save
        zone_path = ts_dir / f"{zone_id}.csv"
        ts_data.to_csv(zone_path, index=False)

    logger.info("Saved %d zone timeseries files to %s", raw_df["zone_id"].nunique(), ts_dir)


# ── ORCHESTRATOR ────────────────────────────────────────────────────────────

def run_features(
    raw_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Args:
        raw_df: Raw ingested DataFrame (84,000 rows).
        config: Configuration dict.
        output_dir: Output directory (default: ./output).

    Returns:
        Feature DataFrame (1000 rows, 30+ columns).
    """
    t0 = time.time()
    out = output_dir or Path("output")

    logger.info("═══ FEATURE ENGINEERING ═══")
    logger.info("Input: %d rows, %d zones", len(raw_df), raw_df["zone_id"].nunique())

    # Step 1: Smooth
    smoothed = smooth_timeseries(raw_df, window=config.get("baseline", {}).get("smoothing_window", 3))

    # Step 2: Baseline
    baselines = compute_baseline(smoothed, baseline_months=72, config=config)

    # Step 3: Deltas
    delta_df = compute_deltas(smoothed, baselines)

    # Step 4: Temporal features
    temporal_df = compute_temporal_features(smoothed, config=config)

    # Step 5: Merge deltas + temporal
    feature_df = delta_df.merge(
        temporal_df.drop(columns=["zone_id"], errors="ignore"),
        left_index=True,
        right_index=True,
    )
    # Re-add zone_id from delta_df if dropped
    if "zone_id" not in feature_df.columns:
        feature_df["zone_id"] = delta_df["zone_id"].values

    # Step 6: Spatial features
    feature_df = compute_spatial_features(feature_df, config=config)

    # Step 7: Encoding
    feature_df = compute_encoding_features(feature_df)

    # Step 8: Changepoints
    changepoints = detect_changepoints(smoothed, config=config)

    # Add changepoint count as a feature
    feature_df["n_changepoints"] = feature_df["zone_id"].map(
        lambda z: len(changepoints.get(str(z), []))
    )

    # Step 9: Moran's I (global spatial autocorrelation)
    moran_i = compute_moran_i(feature_df, config=config)
    feature_df["moran_i"] = moran_i  # stored as constant column for reference

    # Step 10: DSR placeholder (filled by detect module)
    if "dsr" not in feature_df.columns:
        feature_df["dsr"] = 0.0

    # Step 11: Save
    save_features(feature_df, smoothed, changepoints, baselines, out, moran_i=moran_i)

    elapsed = time.time() - t0
    logger.info("Feature engineering complete in %.1fs", elapsed)
    logger.info("Output: %d zones, %d features", len(feature_df), len(feature_df.columns))

    return feature_df


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config_loader import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config()

    # Load raw data
    raw_path = Path("output/raw_data.csv")
    if not raw_path.exists():
        logger.error("output/raw_data.csv not found. Run `python -m pipeline.ingest` first.")
        sys.exit(1)

    raw_df = pd.read_csv(raw_path)
    feature_df = run_features(raw_df, config)

    print(f"\n[OK] Feature engineering complete: {len(feature_df)} zones, {len(feature_df.columns)} features")
    print(f"  Columns: {list(feature_df.columns)}")
    print(f"  Output: output/features.csv + output/zone_timeseries/")
