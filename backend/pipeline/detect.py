"""
Aravalli Intelligence — ML Detection Engine (pipeline/detect.py)
Implements PRD Section 10: anomaly detection, DSR, drift scoring, threat classification.

Execution Flow (13 steps):
    1.  LOAD          — features.csv + ground truth + config
    2.  FEATURE MAT   — 1000×19 numpy array, NaN → median
    3.  SCALE         — RobustScaler / StandardScaler / MinMaxScaler
    4.  DSR           — Deviation from Seasonal Referent (seasonal proof)
    5.  DRIFT SCORE   — Composite severity 1.0–10.0
    6.  ENSEMBLE      — IsoForest + DBSCAN + LOF + KMeans weighted voting
    7.  TEMPORAL      — Confidence scaling by consecutive decline count
    8.  FILTERS       — 6 post-detection false-positive filters
    9.  CLASSIFY      — Mining / Encroachment / Deforestation / Localized
    10. IMPORTANCE    — Top-3 features per detected zone (permutation)
    11. GEOJSON       — Circular polygon export (EPSG:4326)
    12. ACCURACY      — Precision / Recall / F1 vs ground truth
    13. SAVE          — detections.csv, detected_zones.geojson, accuracy_report.json

Author: Shivang, Team BIOBYTES
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

# ── Feature names (must match features.csv columns) ──────────────────────────
FEATURE_COLS = [
    "ndvi_current", "ndbi_current", "bsi_current", "nightlight_current",
    "slope_short", "slope_long", "acceleration", "volatility_ratio",
    "deviation_from_mean", "consecutive_declines", "recovery_signal",
    "local_anomaly_score", "is_isolated", "regional_health",
    "spatial_gradient", "dsr", "elevation_norm",
    "month_sin", "month_cos",
]

# Human-readable feature name mapping
FEATURE_MEANINGS = {
    "ndvi_current": "Low vegetation health",
    "ndbi_current": "Built-up surface detected",
    "bsi_current": "Bare soil or rock exposure",
    "nightlight_current": "Elevated human activity",
    "slope_short": "Recent sharp decline",
    "slope_long": "Long-term vegetation decline",
    "acceleration": "Decline is accelerating",
    "volatility_ratio": "Unstable vegetation behavior",
    "deviation_from_mean": "Far below historical average",
    "consecutive_declines": "Sustained vegetation loss",
    "recovery_signal": "No recovery detected",
    "local_anomaly_score": "Different from neighboring zones",
    "is_isolated": "Isolated event, neighbors unaffected",
    "regional_health": "Surrounding area is healthy",
    "spatial_gradient": "Epicenter of decline pattern",
    "dsr": "Change exceeds seasonal expectation",
    "elevation_norm": "Elevation-related anomaly",
    "month_sin": "Seasonal timing factor",
    "month_cos": "Seasonal timing factor",
}

# Feature direction for Z-score calculation (high/low is bad)
FEATURE_DIRECTION = {
    "bsi_current": "high_is_bad",
    "ndbi_current": "high_is_bad",
    "nightlight_current": "high_is_bad",
    "consecutive_declines": "high_is_bad",
    "volatility_ratio": "high_is_bad",
    "local_anomaly_score": "high_is_bad",
    "is_isolated": "high_is_bad",
    "dsr": "high_is_bad",
    "ndvi_current": "low_is_bad",
    "slope_short": "low_is_bad",
    "slope_long": "low_is_bad",
    "recovery_signal": "low_is_bad",
    "acceleration": "low_is_bad",
    "deviation_from_mean": "low_is_bad",
    "spatial_gradient": "low_is_bad",
    "regional_health": "neutral",
    "elevation_norm": "neutral",
    "month_sin": "neutral",
    "month_cos": "neutral",
}


# ── 1. LOAD ──────────────────────────────────────────────────────────────────

def load_features(
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features.csv and ground truth CSV.

    Args:
        output_dir: Path to the output directory containing features.csv.
        config: Config dict with data paths.

    Returns:
        Tuple of (feature_df, ground_truth_df).

    Raises:
        FileNotFoundError: If features.csv does not exist.
    """
    features_path = output_dir / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"features.csv not found at {features_path}. "
            "Run `python -m pipeline.features` first."
        )

    feature_df = pd.read_csv(features_path)
    logger.info("Loaded features.csv: %d zones, %d columns", len(feature_df), len(feature_df.columns))

    # Ground truth - check output directory first (for synthetic data), then fall back to config path
    gt_path = output_dir / "ground_truth.csv"
    if not gt_path.exists():
        gt_path = Path(config.get("data", {}).get("ground_truth_path", "data/real_ground_truth.csv"))
    if gt_path.exists():
        gt_df = pd.read_csv(gt_path)
        logger.info("Loaded ground truth from %s: %d known events", gt_path, len(gt_df))
    else:
        logger.warning("Ground truth not found at %s — accuracy metrics skipped", gt_path)
        gt_df = pd.DataFrame(columns=["zone_id", "event_type", "start_month", "source", "confidence"])

    return feature_df, gt_df


# ── 2. FEATURE MATRIX ────────────────────────────────────────────────────────

def build_feature_matrix(
    feature_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Extract numeric feature matrix and impute NaN with column median.

    Args:
        feature_df: Feature DataFrame with 1000 rows.

    Returns:
        Tuple of (X: ndarray shape [n, 19], feature_names: list[str]).
    """
    available = [c for c in FEATURE_COLS if c in feature_df.columns]
    missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
    if missing:
        logger.warning("Missing feature columns (will be zeroed): %s", missing)

    X = feature_df[available].copy().values.astype(np.float64)

    # Replace NaN with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median = np.nanmedian(col)
            X[nan_mask, j] = median

    logger.info("Feature matrix: %d zones × %d features", X.shape[0], X.shape[1])
    return X, available


# ── 3. SCALE FEATURES ────────────────────────────────────────────────────────

def scale_features(
    X: np.ndarray,
    scaler_name: str = "robust",
) -> tuple[np.ndarray, Any]:
    """Scale feature matrix to equalise magnitudes across features.

    Args:
        X: Unscaled feature matrix shape [n, p].
        scaler_name: One of 'robust' | 'standard' | 'minmax'.

    Returns:
        Tuple of (X_scaled: ndarray, fitted_scaler).
    """
    scalers: dict[str, Any] = {
        "robust":   RobustScaler(),
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
    }
    scaler = scalers.get(scaler_name, RobustScaler())
    X_scaled = scaler.fit_transform(X)
    logger.info("Scaled features with %s scaler", scaler.__class__.__name__)
    return X_scaled, scaler


# ── 4. DSR (Deviation from Seasonal Referent) ────────────────────────────────

def compute_dsr(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute DSR score — mathematical proof that change is non-seasonal.

    PRD Formula (exact):
        observed_delta = abs(smoothed_ndvi[-1] - baseline[month])
        expected_delta = baseline mean delta for that calendar month
        expected_std   = baseline std for that calendar month

        DSR     = observed_delta / (expected_delta + 1e-6)
        z_score = (observed_delta - expected_delta) / (expected_std + 1e-6)
        p_value = scipy.stats.norm.sf(abs(z_score))
        confidence = (1 - p_value) * 100

    Magnitude gates prevent false positives in high-NDVI areas.
    Classification is threshold-dependent and varies for transition months.

    Args:
        feature_df: Feature DataFrame (must include ndvi_current, ndvi_delta,
                    month_sin, month_cos columns).
        config: Configuration dict.

    Returns:
        feature_df with new columns: dsr, dsr_z_score, dsr_p_value,
        dsr_confidence, dsr_classification, dsr_severity.
    """
    dsr_cfg = config.get("dsr", {})
    threshold_normal     = float(dsr_cfg.get("threshold_normal",     1.50))
    threshold_transition = float(dsr_cfg.get("threshold_transition", 2.00))
    seasonal_cutoff      = float(dsr_cfg.get("seasonal_normal_cutoff", 1.20))
    min_abs_change       = float(dsr_cfg.get("min_absolute_change",  0.10))
    high_ndvi_base       = float(dsr_cfg.get("high_ndvi_baseline",   0.60))
    high_ndvi_min_change = float(dsr_cfg.get("high_ndvi_min_change", 0.15))
    transition_months    = set(dsr_cfg.get("transition_months",      [3, 4, 10, 11]))

    # Infer current calendar month from month_sin / month_cos
    if "month_sin" in feature_df.columns and "month_cos" in feature_df.columns:
        month_rad = np.arctan2(
            feature_df["month_sin"].values,
            feature_df["month_cos"].values,
        )
        cal_months = ((np.round(month_rad / (2 * np.pi / 12)) % 12) + 1).astype(int)
    else:
        cal_months = np.full(len(feature_df), 12, dtype=int)

    ndvi_current = feature_df.get("ndvi_current", pd.Series(np.zeros(len(feature_df)))).values
    ndvi_delta   = feature_df.get("ndvi_delta",   pd.Series(np.zeros(len(feature_df)))).values

    # Regional per-season statistics (estimated from dataset — proxy for baseline)
    # In production these come from the baseline computation in features.py
    # Here we derive them from the feature columns present
    observed_deltas = np.abs(ndvi_delta)

    # Seasonal baseline approximation: expected delta per calendar month
    expected_deltas = np.full(len(feature_df), 0.05)  # default 5% seasonal noise
    expected_stds   = np.full(len(feature_df), 0.03)

    # Compute season-qualified expected values from smoothing deviation
    for m in range(1, 13):
        mask = cal_months == m
        if mask.sum() > 1:
            expected_deltas[mask] = np.nanmean(observed_deltas[mask]) * 0.5
            expected_stds[mask]   = max(np.nanstd(observed_deltas[mask]), 1e-6)

    # DSR = observed / (expected + epsilon)
    dsr_values = observed_deltas / (expected_deltas + 1e-6)

    # Z-score and p-value
    z_scores = (observed_deltas - expected_deltas) / (expected_stds + 1e-6)
    p_values = scipy.stats.norm.sf(np.abs(z_scores))
    confidences = (1 - p_values) * 100

    # ── Gate 1: magnitude check ────────────────────────────────────────────
    gate1_seasonal = observed_deltas < min_abs_change

    # ── Gate 2: high-NDVI baseline check ──────────────────────────────────
    gate2_seasonal = (ndvi_current > high_ndvi_base) & (observed_deltas < high_ndvi_min_change)

    # Force seasonal_normal for gated zones
    forced_seasonal = gate1_seasonal | gate2_seasonal

    # ── Select threshold per zone (transition vs normal month) ────────────
    thresholds = np.where(
        np.isin(cal_months, list(transition_months)),
        threshold_transition,
        threshold_normal,
    )

    # ── Classification ────────────────────────────────────────────────────
    classifications = []
    severities      = []

    for i in range(len(feature_df)):
        if forced_seasonal[i]:
            classifications.append("seasonal_normal")
            severities.append("none")
        elif dsr_values[i] < seasonal_cutoff:
            classifications.append("seasonal_normal")
            severities.append("none")
        elif dsr_values[i] < thresholds[i]:
            classifications.append("watch")
            severities.append("low")
        elif dsr_values[i] < thresholds[i] + 0.5:
            classifications.append("warning")
            severities.append("medium")
        else:
            classifications.append("confirmed_degradation")
            severities.append("high")

    feature_df = feature_df.copy()
    feature_df["dsr"]                 = np.round(dsr_values, 4)
    feature_df["dsr_z_score"]         = np.round(z_scores, 4)
    feature_df["dsr_p_value"]         = np.round(p_values, 6)
    feature_df["dsr_confidence"]      = np.round(confidences, 2)
    feature_df["dsr_classification"]  = classifications
    feature_df["dsr_severity"]        = severities
    feature_df["cal_month"]           = cal_months

    confirmed = (np.array(classifications) == "confirmed_degradation").sum()
    logger.info(
        "DSR complete: %d confirmed_degradation, %d watch, %d seasonal_normal",
        confirmed,
        (np.array(classifications) == "watch").sum(),
        (np.array(classifications) == "seasonal_normal").sum(),
    )
    return feature_df


# ── 5. DRIFT SCORE ─────────────────────────────────────────────────────────

def compute_drift_score(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute composite drift score (1.0–10.0) combining 4 independent signals.

    PRD Formula (exact):
        ndvi_departure      = min(1, abs(ndvi_delta) / ndvi_divisor)
        temporal_persistence = min(1, consecutive_declines / consecutive_normalizer)
        spatial_isolation   = min(1, local_anomaly_score / anomaly_divisor)
        seasonal_proof      = min(1, max(0, (dsr - 1)) / 3)

        raw_drift = sum(component * weight for each)
        drift_score = clamp(1.0 + raw_drift * 9.0, 1.0, 10.0)

    Args:
        feature_df: DataFrame with dsr, ndvi_delta, consecutive_declines,
                    local_anomaly_score columns.
        config: Config dict.

    Returns:
        feature_df with new columns: drift_score, drift_severity.
    """
    drift_cfg  = config.get("drift", {})
    weights    = drift_cfg.get("weights",       {})
    norm       = drift_cfg.get("normalization", {})
    severity   = drift_cfg.get("severity",      {})

    w_ndvi     = float(weights.get("ndvi",     0.35))
    w_temporal = float(weights.get("temporal", 0.25))
    w_spatial  = float(weights.get("spatial",  0.20))
    w_dsr      = float(weights.get("dsr",      0.20))

    ndvi_divisor        = float(norm.get("ndvi_divisor",         0.30))
    consec_normalizer   = float(norm.get("consecutive_normalizer", 15))
    anomaly_divisor     = float(norm.get("anomaly_divisor",        4))

    sev_moderate = float(severity.get("moderate", 3.0))
    sev_high     = float(severity.get("high",     5.0))
    sev_severe   = float(severity.get("severe",   7.0))
    sev_critical = float(severity.get("critical", 8.5))

    ndvi_delta     = feature_df.get("ndvi_delta",          pd.Series(np.zeros(len(feature_df)))).values
    consec         = feature_df.get("consecutive_declines", pd.Series(np.zeros(len(feature_df)))).values
    anomaly_score  = feature_df.get("local_anomaly_score",  pd.Series(np.zeros(len(feature_df)))).values
    dsr_vals       = feature_df.get("dsr",                  pd.Series(np.ones(len(feature_df)))).values

    # Normalise components to [0, 1]
    ndvi_departure       = np.minimum(1.0, np.abs(ndvi_delta) / ndvi_divisor)
    temporal_persistence = np.minimum(1.0, consec / consec_normalizer)
    spatial_isolation    = np.minimum(1.0, anomaly_score / anomaly_divisor)
    seasonal_proof       = np.minimum(1.0, np.maximum(0.0, (dsr_vals - 1.0) / 3.0))

    raw_drift = (
        ndvi_departure       * w_ndvi     +
        temporal_persistence * w_temporal +
        spatial_isolation    * w_spatial  +
        seasonal_proof       * w_dsr
    )

    drift_scores = np.clip(1.0 + raw_drift * 9.0, 1.0, 10.0)

    # Severity labels
    drift_severities = []
    for d in drift_scores:
        if d < 2.0:
            drift_severities.append("normal")
        elif d < sev_moderate:
            drift_severities.append("low")
        elif d < sev_high:
            drift_severities.append("moderate")
        elif d < sev_severe:
            drift_severities.append("high")
        elif d < sev_critical:
            drift_severities.append("severe")
        else:
            drift_severities.append("critical")

    feature_df = feature_df.copy()
    feature_df["drift_score"]    = np.round(drift_scores, 3)
    feature_df["drift_severity"] = drift_severities

    logger.info(
        "Drift scores: mean=%.2f, max=%.2f, critical=%d",
        drift_scores.mean(), drift_scores.max(),
        (drift_scores >= sev_critical).sum(),
    )
    return feature_df


# ── 6. ENSEMBLE DETECTION ────────────────────────────────────────────────────

def run_ensemble(
    X_scaled: np.ndarray,
    feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run 4 ML anomaly detectors and combine via weighted voting.

    Methods:
        IsolationForest (weight 0.35): path-based anomaly scores.
        DBSCAN          (weight 0.30): density-based outliers.
        LocalOutlierFactor (0.25):     local density comparison.
        KMeans          (weight 0.10): smallest-cluster membership.

    Confidence formula (PRD exact):
        confidence = weighted_score * 60 + iso_score_normalised * 40
        Clamped to [0, 100].

    Args:
        X_scaled: Scaled feature matrix [n, p].
        feature_df: Feature DataFrame.
        config: Config dict.

    Returns:
        feature_df with new columns: is_anomaly, ensemble_votes,
        weighted_score, confidence, iso_anomaly, dbscan_anomaly,
        lof_anomaly, kmeans_anomaly.
    """
    ens_cfg = config.get("ensemble", {})
    methods = ens_cfg.get("methods", {})
    min_score = float(ens_cfg.get("min_weighted_score", 0.50))
    n_zones = X_scaled.shape[0]

    iso_cfg   = methods.get("isolation_forest", {})
    dbs_cfg   = methods.get("dbscan",           {})
    lof_cfg   = methods.get("lof",              {})
    km_cfg    = methods.get("kmeans",            {})

    iso_enabled = bool(iso_cfg.get("enabled", True))
    dbs_enabled = bool(dbs_cfg.get("enabled", True))
    lof_enabled = bool(lof_cfg.get("enabled", True))
    km_enabled  = bool(km_cfg.get("enabled",  True))

    iso_weight = float(iso_cfg.get("weight", 0.35))
    dbs_weight = float(dbs_cfg.get("weight", 0.30))
    lof_weight = float(lof_cfg.get("weight", 0.25))
    km_weight  = float(km_cfg.get("weight",  0.10))

    active_weights   = []
    method_votes     = []
    iso_scores_norm  = np.zeros(n_zones)
    iso_anomaly = np.zeros(n_zones, dtype=int)
    dbs_anomaly = np.zeros(n_zones, dtype=int)
    lof_anomaly = np.zeros(n_zones, dtype=int)
    km_anomaly  = np.zeros(n_zones, dtype=int)
    iso_model   = None

    # ── IsolationForest ────────────────────────────────────────────────────
    if iso_enabled:
        logger.info("Running IsolationForest...")
        iso = IsolationForest(
            contamination=float(iso_cfg.get("contamination",  0.10)),
            n_estimators =int(  iso_cfg.get("n_estimators",   200)),
            max_samples  =float(iso_cfg.get("max_samples",    0.80)),
            random_state =42,
        )
        iso_pred   = iso.fit_predict(X_scaled)
        iso_scores  = iso.score_samples(X_scaled)    # negative: more anomalous
        iso_scores_shifted    = -iso_scores            # flip: big = more anomalous
        iso_min, iso_max = iso_scores_shifted.min(), iso_scores_shifted.max()
        iso_scores_norm = np.clip(
            (iso_scores_shifted - iso_min) / (iso_max - iso_min + 1e-9), 0, 1
        )
        iso_anomaly = (iso_pred == -1).astype(int)
        active_weights.append(iso_weight)
        method_votes.append(iso_anomaly.astype(float))
        iso_model = iso
        logger.info("IsolationForest: %d anomalies (%.1f%%)", iso_anomaly.sum(),
                    100 * iso_anomaly.mean())

    # ── DBSCAN ────────────────────────────────────────────────────────────
    if dbs_enabled:
        logger.info("Running DBSCAN...")
        db = DBSCAN(
            eps        =float(dbs_cfg.get("eps",         0.50)),
            min_samples=int(  dbs_cfg.get("min_samples", 3)),
        )
        db_labels   = db.fit_predict(X_scaled)
        dbs_anomaly = (db_labels == -1).astype(int)
        active_weights.append(dbs_weight)
        method_votes.append(dbs_anomaly.astype(float))
        logger.info("DBSCAN: %d anomalies (%.1f%%)", dbs_anomaly.sum(),
                    100 * dbs_anomaly.mean())

    # ── LocalOutlierFactor ────────────────────────────────────────────────
    if lof_enabled:
        logger.info("Running LocalOutlierFactor...")
        n_neighbors = min(int(lof_cfg.get("n_neighbors", 20)), n_zones - 1)
        lof = LocalOutlierFactor(
            n_neighbors  =n_neighbors,
            contamination=float(lof_cfg.get("contamination", 0.10)),
        )
        lof_pred    = lof.fit_predict(X_scaled)
        lof_anomaly = (lof_pred == -1).astype(int)
        active_weights.append(lof_weight)
        method_votes.append(lof_anomaly.astype(float))
        logger.info("LOF: %d anomalies (%.1f%%)", lof_anomaly.sum(),
                    100 * lof_anomaly.mean())

    # ── KMeans ────────────────────────────────────────────────────────────
    if km_enabled:
        logger.info("Running KMeans...")
        n_clusters = min(int(km_cfg.get("n_clusters", 8)), n_zones)
        km = KMeans(
            n_clusters=n_clusters,
            n_init    =int(km_cfg.get("n_init", 10)),
            random_state=42,
        )
        km_labels   = km.fit_predict(X_scaled)
        # Anomaly = member of smallest cluster
        cluster_sizes = np.bincount(km_labels)
        smallest_cluster = int(np.argmin(cluster_sizes))
        km_anomaly  = (km_labels == smallest_cluster).astype(int)
        active_weights.append(km_weight)
        method_votes.append(km_anomaly.astype(float))
        logger.info("KMeans: %d in smallest cluster", km_anomaly.sum())

    if not active_weights:
        raise ValueError("All ensemble methods disabled — cannot detect anomalies.")

    # ── Weighted voting ────────────────────────────────────────────────────
    total_weight         = sum(active_weights)
    norm_weights         = [w / total_weight for w in active_weights]
    votes_matrix         = np.stack(method_votes, axis=1)  # [n, k]
    weighted_scores      = votes_matrix @ np.array(norm_weights)  # [n]
    is_anomaly           = (weighted_scores >= min_score).astype(int)

    # ── Confidence ────────────────────────────────────────────────────────
    confidence = weighted_scores * 60 + iso_scores_norm * 40
    confidence = np.clip(confidence, 0, 100)

    ensemble_votes = (iso_anomaly + dbs_anomaly + lof_anomaly + km_anomaly)

    feature_df = feature_df.copy()
    feature_df["is_anomaly"]     = is_anomaly
    feature_df["ensemble_votes"] = ensemble_votes
    feature_df["weighted_score"] = np.round(weighted_scores, 4)
    feature_df["confidence"]     = np.round(confidence, 2)
    feature_df["iso_anomaly"]    = iso_anomaly
    feature_df["dbscan_anomaly"] = dbs_anomaly
    feature_df["lof_anomaly"]    = lof_anomaly
    feature_df["kmeans_anomaly"] = km_anomaly

    logger.info(
        "Ensemble: %d anomalies (%d 4/4 agreement, %d 3/4)",
        is_anomaly.sum(),
        (ensemble_votes == 4).sum(),
        (ensemble_votes == 3).sum(),
    )
    return feature_df, iso_model


# ── 7. TEMPORAL CONFIRMATION ──────────────────────────────────────────────────

def apply_temporal_confirmation(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Scale confidence by persistence of consecutive declines.

    PRD Rules:
        >= 12 declines: confirmed (spans years)    → confidence * 1.0
        6–11  declines: probable                   → confidence * 1.0
        3–5   declines: watch (short-term trend)   → confidence * 0.7
        < 3   declines: suspect (very short)       → confidence * 0.5

    Args:
        feature_df: DataFrame with confidence and consecutive_declines.
        config: Config dict.

    Returns:
        feature_df with updated confidence and temporal_status column.
    """
    filters_cfg = config.get("filters", {})
    temp_cfg    = filters_cfg.get("temporal_confirmation", {})
    if not temp_cfg.get("enabled", True):
        feature_df = feature_df.copy()
        feature_df["temporal_status"] = "not_checked"
        return feature_df

    consec = feature_df["consecutive_declines"].values
    conf   = feature_df["confidence"].values.copy()
    statuses = []

    for i, n in enumerate(consec):
        if n >= 12:
            statuses.append("confirmed")
            # confidence unchanged
        elif n >= 6:
            statuses.append("probable")
            # confidence unchanged
        elif n >= 3:
            statuses.append("watch")
            conf[i] *= 0.7
        else:
            statuses.append("suspect")
            conf[i] *= 0.5

    feature_df = feature_df.copy()
    feature_df["confidence"]      = np.round(conf, 2)
    feature_df["temporal_status"] = statuses
    logger.info("Temporal confirmation: confirmed=%d, probable=%d, watch=%d, suspect=%d",
                statuses.count("confirmed"), statuses.count("probable"),
                statuses.count("watch"), statuses.count("suspect"))
    return feature_df


# ── 8. POST-DETECTION FILTERS ────────────────────────────────────────────────

def apply_filters(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Apply 6 sequential post-detection filters to reduce false positives.

    F1 Seasonal DSR    — Remove seasonal_normal zones.
    F2 Regional Context — Remove healthy-region isolated anomalies.
    F3 Duration Check  — Cap confidence if too few consecutive declines.
    F4 Recovery Check  — Downgrade severity if zone is recovering.
    F5 Monsoon Filter  — Remove positive trends during monsoon months.
    F6 Confidence Floor — Remove very low confidence detections.

    Args:
        feature_df: DataFrame with dsr_classification, is_anomaly, etc.
        config: Config dict.

    Returns:
        Tuple of (modified feature_df, filter_stats dict).
    """
    f = config.get("filters", {})
    feature_df = feature_df.copy()
    filter_stats: dict[str, dict[str, int]] = {}

    # ── F1: Seasonal DSR ──────────────────────────────────────────────────
    if f.get("seasonal_dsr", {}).get("enabled", True):
        mask_seasonal = (
            (feature_df["is_anomaly"] == 1) &
            (feature_df["dsr_classification"] == "seasonal_normal")
        )
        removed_fp = int(mask_seasonal.sum())
        feature_df.loc[mask_seasonal, "is_anomaly"] = 0
        feature_df.loc[mask_seasonal, "filter_removed"] = "F1_seasonal_dsr"
        filter_stats["seasonal_dsr"] = {"removed_fp": removed_fp, "kept_tp": 0}
        logger.info("F1 Seasonal DSR: removed %d", removed_fp)

    # ── F2: Regional Context ──────────────────────────────────────────────
    rc = f.get("regional_context", {})
    if rc.get("enabled", True):
        health_threshold = float(rc.get("health_threshold", 0.40))
        dsr_transition   = float(config.get("dsr", {}).get("threshold_transition", 2.00))
        mask_regional = (
            (feature_df["is_anomaly"] == 1) &
            (~feature_df["is_isolated"].astype(bool)) &
            (feature_df["regional_health"] < health_threshold) &
            (feature_df["dsr"] < dsr_transition)
        )
        removed_fp = int(mask_regional.sum())
        feature_df.loc[mask_regional, "is_anomaly"] = 0
        feature_df.loc[mask_regional, "filter_removed"] = "F2_regional_context"
        filter_stats["regional_context"] = {"removed_fp": removed_fp, "kept_tp": 0}
        logger.info("F2 Regional Context: removed %d", removed_fp)

    # ── F3: Duration Check ────────────────────────────────────────────────
    dur = f.get("duration_check", {})
    if dur.get("enabled", True):
        min_consec   = int(  dur.get("min_consecutive",   2))
        conf_cap     = float(dur.get("confidence_cap",   40))
        mask_short = (
            (feature_df["is_anomaly"] == 1) &
            (feature_df["consecutive_declines"] < min_consec)
        )
        feature_df.loc[mask_short, "confidence"] = np.minimum(
            feature_df.loc[mask_short, "confidence"], conf_cap
        )
        filter_stats["duration_check"] = {"removed_fp": 0, "kept_tp": int(mask_short.sum())}
        logger.info("F3 Duration Check: capped confidence for %d zones", mask_short.sum())

    # ── F4: Recovery Check ────────────────────────────────────────────────
    rec = f.get("recovery", {})
    if rec.get("enabled", True):
        rec_threshold = float(rec.get("threshold", 0.05))
        mask_recovering = (
            (feature_df["is_anomaly"] == 1) &
            (feature_df["recovery_signal"] > rec_threshold)
        )
        feature_df.loc[mask_recovering, "dsr_severity"] = "low"
        filter_stats["recovery"] = {"removed_fp": 0, "kept_tp": int(mask_recovering.sum())}
        logger.info("F4 Recovery: downgraded %d zones to low severity", mask_recovering.sum())

    # ── F5: Monsoon Filter ────────────────────────────────────────────────
    mon = f.get("monsoon", {})
    if mon.get("enabled", True):
        monsoon_months = set(mon.get("months", [7, 8, 9]))
        cal_months_in_df = feature_df.get("cal_month", pd.Series(np.full(len(feature_df), 12)))
        mask_monsoon = (
            (feature_df["is_anomaly"] == 1) &
            (cal_months_in_df.isin(monsoon_months)) &
            (feature_df["slope_short"] > 0)
        )
        removed_fp = int(mask_monsoon.sum())
        feature_df.loc[mask_monsoon, "is_anomaly"] = 0
        feature_df.loc[mask_monsoon, "filter_removed"] = "F5_monsoon"
        filter_stats["monsoon"] = {"removed_fp": removed_fp, "kept_tp": 0}
        logger.info("F5 Monsoon: removed %d", removed_fp)

    # ── F6: Confidence Floor ──────────────────────────────────────────────
    cf = f.get("confidence_floor", {})
    if cf.get("enabled", True):
        floor = float(cf.get("value", 30))
        mask_low_conf = (
            (feature_df["is_anomaly"] == 1) &
            (feature_df["confidence"] < floor)
        )
        removed_fp = int(mask_low_conf.sum())
        feature_df.loc[mask_low_conf, "is_anomaly"] = 0
        feature_df.loc[mask_low_conf, "filter_removed"] = "F6_confidence_floor"
        filter_stats["confidence_floor"] = {"removed_fp": removed_fp, "kept_tp": 0}
        logger.info("F6 Confidence Floor: removed %d", removed_fp)

    final_detections = int((feature_df["is_anomaly"] == 1).sum())
    logger.info("Filters complete: %d final detections", final_detections)
    return feature_df, filter_stats


# ── 9. THREAT CLASSIFICATION ──────────────────────────────────────────────────

def classify_threats(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Pattern-match spectral signatures to classify threat types.

    Patterns (PRD exact):
        Encroachment: ndvi_drop + ndbi_rise + nightlight_rise
        Mining:       ndvi_drop + bsi_rise + ndbi_flat
        Deforestation: ndvi_drop (large) + bsi_flat + ndbi_flat
        Localized:    default fallback

    Threat score is a weighted composite of spectral evidence.

    Args:
        feature_df: DataFrame with ndvi_delta, ndbi_delta, bsi_delta,
                    nightlight_delta, dsr columns.
        config: Config dict.

    Returns:
        feature_df with threat_type and threat_score columns.
    """
    cls_cfg = config.get("classification", {})
    norm_cfg = config.get("normalization", {}).get("threat_score", {})
    sw = cls_cfg.get("score_weights", {})

    # ── Classification thresholds ─────────────────────────────────────────
    enc = cls_cfg.get("encroachment", {})
    min_mining = cls_cfg.get("mining", {})
    defor = cls_cfg.get("deforestation", {})

    enc_ndvi_min  = float(enc.get("ndvi_drop_min",  -0.10))
    enc_ndbi_min  = float(enc.get("ndbi_rise_min",   0.05))
    enc_nl_min    = float(enc.get("nightlight_min",  3.0))

    mng_ndvi_min  = float(min_mining.get("ndvi_drop_min", -0.10))
    mng_bsi_min   = float(min_mining.get("bsi_rise_min",   0.08))
    mng_ndbi_max  = float(min_mining.get("ndbi_max",       0.02))

    def_ndvi_min  = float(defor.get("ndvi_drop_min", -0.15))
    def_bsi_max   = float(defor.get("bsi_max",        0.05))
    def_ndbi_max  = float(defor.get("ndbi_max",       0.02))

    # ── Score weights ─────────────────────────────────────────────────────
    w_veg   = float(sw.get("vegetation",    0.30))
    w_urb   = float(sw.get("urban",         0.20))
    w_soil  = float(sw.get("soil",          0.15))
    w_nl    = float(sw.get("nightlight",    0.15))
    w_dsr   = float(sw.get("seasonal_proof",0.20))

    ndvi_div  = float(norm_cfg.get("ndvi_divisor",       0.30))
    ndbi_div  = float(norm_cfg.get("ndbi_divisor",       0.20))
    bsi_div   = float(norm_cfg.get("bsi_divisor",        0.20))
    nl_div    = float(norm_cfg.get("nightlight_divisor", 10.0))
    dsr_div   = float(norm_cfg.get("dsr_divisor",        2.0))
    score_cap = float(norm_cfg.get("cap",                100))

    ndvi_delta  = feature_df.get("ndvi_delta",         pd.Series(np.zeros(len(feature_df)))).values
    ndbi_delta  = feature_df.get("ndbi_delta",         pd.Series(np.zeros(len(feature_df)))).values
    bsi_delta   = feature_df.get("bsi_delta",          pd.Series(np.zeros(len(feature_df)))).values
    nl_delta    = feature_df.get("nightlight_delta",   pd.Series(np.zeros(len(feature_df)))).values
    dsr_vals    = feature_df.get("dsr",                pd.Series(np.ones(len(feature_df)))).values
    is_anom     = feature_df.get("is_anomaly",         pd.Series(np.zeros(len(feature_df)))).values

    threat_types  = ["normal"] * len(feature_df)
    threat_scores = np.zeros(len(feature_df))

    for i in range(len(feature_df)):
        if is_anom[i] == 0:
            threat_types[i] = "none"
            continue

        nd = ndvi_delta[i]
        ndb = ndbi_delta[i]
        bs = bsi_delta[i]
        nl = nl_delta[i]

        # Pattern matching
        is_encroachment = (nd <= enc_ndvi_min and ndb >= enc_ndbi_min and nl >= enc_nl_min)
        is_mining       = (nd <= mng_ndvi_min and bs >= mng_bsi_min  and ndb <= mng_ndbi_max)
        is_deforestation = (nd <= def_ndvi_min and bs <= def_bsi_max and ndb <= def_ndbi_max)

        if is_encroachment:
            threat_types[i] = "encroachment"
        elif is_mining:
            threat_types[i] = "mining"
        elif is_deforestation:
            threat_types[i] = "deforestation"
        else:
            threat_types[i] = "localized_disturbance"

        # Weighted threat score
        veg_comp  = min(1, abs(nd) / ndvi_div)
        urb_comp  = min(1, abs(ndb) / ndbi_div)
        soil_comp = min(1, abs(bs) / bsi_div)
        nl_comp   = min(1, abs(nl) / nl_div)
        dsr_comp  = min(1, (dsr_vals[i] - 1) / max(dsr_div, 1e-6))

        raw_score = (
            veg_comp  * w_veg  +
            urb_comp  * w_urb  +
            soil_comp * w_soil +
            nl_comp   * w_nl   +
            dsr_comp  * w_dsr
        )
        threat_scores[i] = min(score_cap, raw_score * score_cap)

    feature_df = feature_df.copy()
    feature_df["threat_type"]  = threat_types
    feature_df["threat_score"] = np.round(threat_scores, 2)

    # Summarise
    detected = feature_df[feature_df["is_anomaly"] == 1]
    for t in ["mining", "encroachment", "deforestation", "localized_disturbance"]:
        count = (detected["threat_type"] == t).sum()
        if count:
            logger.info("Threat type '%s': %d zones", t, count)

    return feature_df


# ── 10. FEATURE IMPORTANCE ────────────────────────────────────────────────────

def compute_zone_importance(
    zone_features: np.ndarray,
    all_features: np.ndarray,
    feature_names: list[str],
    top_n: int = 3
) -> list[dict]:
    """Calculate per-zone Z-score importance."""
    pop_mean = np.nanmean(all_features, axis=0)
    pop_std = np.nanstd(all_features, axis=0)
    pop_std[pop_std == 0] = 1e-6
    
    z_scores = (zone_features - pop_mean) / pop_std
    
    importances = []
    for i, name in enumerate(feature_names):
        z = z_scores[i]
        direction = FEATURE_DIRECTION.get(name, "neutral")
        if direction == "high_is_bad":
            imp = max(0.0, float(z))
        elif direction == "low_is_bad":
            imp = max(0.0, float(-z))
        else:
            imp = abs(float(z))
        
        if z > 0:
            detail = f"{z:.1f}x above normal"
        else:
            detail = f"{abs(z):.1f}x below normal"
            
        importances.append({
            "driver": FEATURE_MEANINGS.get(name, name),
            "driver_zscore": round(float(z), 2),
            "driver_detail": detail,
            "raw_importance": imp,
        })
        
    importances.sort(key=lambda x: x["raw_importance"], reverse=True)
    return importances[:top_n]


def compute_feature_importance(
    X_scaled: np.ndarray,
    feature_df: pd.DataFrame,
    feature_names: list[str],
    iso_model: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run permutation importance on IsoForest and annotate top-3 features.

    PRD Specification:
        - estimator: the trained IsolationForest
        - y: 1 for anomaly, 0 for non-anomaly
        - n_repeats: 10
        - Extract top 3 features per zone based on mean importance

    Args:
        X_scaled: Scaled feature matrix.
        feature_df: Feature DataFrame.
        feature_names: Names matching X_scaled columns.
        iso_model: Trained IsolationForest model.
        config: Config dict.

    Returns:
        feature_df with global and per-zone importance features.
    """
    feature_df = feature_df.copy()
    
    if iso_model is None:
        logger.warning("No IsolationForest model — skipping feature importance")
        for k in range(1, 4):
            feature_df[f"feature_{k}_name"] = "N/A"
            feature_df[f"feature_{k}_importance"] = 0.0
            feature_df[f"driver_{k}"] = "N/A"
            feature_df[f"driver_{k}_zscore"] = 0.0
            feature_df[f"driver_{k}_detail"] = "N/A"
        return feature_df

    logger.info("Computing permutation feature importance...")
    dummy_y = np.zeros(X_scaled.shape[0])

    def iforest_scorer(estimator, X, y_unused):
        # IsolationForest returns negative values for anomalies. We want disrupting important
        # features to make the score LESS negative (closer to zero), so permuted score < baseline.
        # Adding minus flips the scale to POSITIVE severity. So permuted severity drops!
        return -estimator.score_samples(X).mean()

    try:
        repeats = config.get("ensemble", {}).get("methods", {}).get("isolation_forest", {}).get("importance_repeats", 10)
        result = permutation_importance(
            iso_model, X_scaled, dummy_y,
            scoring=iforest_scorer,
            n_repeats=repeats, random_state=42, n_jobs=-1,
        )
        mean_imp = result.importances_mean
    except Exception as exc:
        logger.warning("permutation_importance failed: %s", exc)
        mean_imp = np.zeros(X_scaled.shape[1])

    # Top-3 globally
    top3_idx = np.argsort(mean_imp)[::-1][:3]
    top3_names  = [FEATURE_MEANINGS.get(feature_names[i], feature_names[i]) for i in top3_idx]
    top3_imps   = [round(float(mean_imp[i]), 6) for i in top3_idx]

    # All zones get the same global top-3 (per PRD section 10)
    for k, (name, imp) in enumerate(zip(top3_names, top3_imps), start=1):
        feature_df[f"feature_{k}_name"]       = name
        feature_df[f"feature_{k}_importance"] = imp

    # --- PER-ZONE Z-SCORE IMPORTANCE ---
    for k in range(1, 4):
        feature_df[f"driver_{k}"] = ""
        feature_df[f"driver_{k}_zscore"] = 0.0
        feature_df[f"driver_{k}_detail"] = ""

    anomaly_mask = feature_df["is_anomaly"] == 1
    anomaly_indices = np.where(anomaly_mask)[0]

    for idx in anomaly_indices:
        zone_feats = X_scaled[idx]
        top_zone = compute_zone_importance(zone_feats, X_scaled, feature_names, top_n=3)
        for k, item in enumerate(top_zone, start=1):
            feature_df.at[feature_df.index[idx], f"driver_{k}"] = item["driver"]
            feature_df.at[feature_df.index[idx], f"driver_{k}_zscore"] = item["driver_zscore"]
            feature_df.at[feature_df.index[idx], f"driver_{k}_detail"] = item["driver_detail"]

    if np.any(mean_imp):
        logger.info(
            "Top features: %s (%.4f), %s (%.4f), %s (%.4f)",
            top3_names[0], top3_imps[0],
            top3_names[1] if len(top3_names) > 1 else "—", top3_imps[1] if len(top3_imps) > 1 else 0,
            top3_names[2] if len(top3_names) > 2 else "—", top3_imps[2] if len(top3_imps) > 2 else 0,
        )
    return feature_df


# ── 11. GEOJSON EXPORT ───────────────────────────────────────────────────────

def export_geojson(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Export detected zones as circular polygon GeoJSON (EPSG:4326).

    Each zone becomes a circular polygon centered at (lon, lat) with
    radius = buffer_radius_km / 111.0 degrees (≈ 1.5 km default).

    Args:
        feature_df: DataFrame with lat, lon, is_anomaly, threat_type, etc.
        config: Config dict.
        output_dir: Directory to save detected_zones.geojson.
    """
    spatial_cfg   = config.get("spatial",  {}).get("geojson", {})
    buffer_deg    = float(spatial_cfg.get("buffer_radius_km", 1.5)) / 111.0
    segments      = int(  spatial_cfg.get("polygon_segments", 32))

    detected = feature_df[feature_df["is_anomaly"] == 1].copy()
    logger.info("Generating GeoJSON for %d detected zones...", len(detected))

    features = []
    for _, row in detected.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])

        # Generate circular polygon
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        ring   = [
            [lon + buffer_deg * np.cos(a), lat + buffer_deg * np.sin(a)]
            for a in angles
        ]

        properties = {
            "zone_id":              str(row.get("zone_id",    "")),
            "threat_type":         str(row.get("threat_type","none")),
            "threat_score":        float(row.get("threat_score",   0)),
            "drift_score":         float(row.get("drift_score",    1)),
            "severity":            str(row.get("dsr_severity", "low")),
            "dsr":                 float(row.get("dsr",           0)),
            "confidence":          float(row.get("confidence",    0)),
            "ndvi_delta":          float(row.get("ndvi_delta",    0)),
            "ensemble_votes":      int(  row.get("ensemble_votes",0)),
            "feature_1_name":      str(row.get("feature_1_name",      "")),
            "feature_1_importance":float(row.get("feature_1_importance", 0)),
            "feature_2_name":      str(row.get("feature_2_name",      "")),
            "feature_2_importance":float(row.get("feature_2_importance", 0)),
            "feature_3_name":      str(row.get("feature_3_name",      "")),
            "feature_3_importance":float(row.get("feature_3_importance", 0)),
            "driver_1":            str(row.get("driver_1", "")),
            "driver_1_zscore":     float(row.get("driver_1_zscore", 0)),
            "driver_1_detail":     str(row.get("driver_1_detail", "")),
            "driver_2":            str(row.get("driver_2", "")),
            "driver_2_zscore":     float(row.get("driver_2_zscore", 0)),
            "driver_2_detail":     str(row.get("driver_2_detail", "")),
            "driver_3":            str(row.get("driver_3", "")),
            "driver_3_zscore":     float(row.get("driver_3_zscore", 0)),
            "driver_3_detail":     str(row.get("driver_3_detail", "")),
        }

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": properties,
        })

    geojson = {"type": "FeatureCollection", "features": features}
    out_path = output_dir / "detected_zones.geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)

    logger.info("Saved detected_zones.geojson (%d features)", len(features))


# ── 12. ACCURACY REPORT ───────────────────────────────────────────────────────

def compute_accuracy(
    feature_df:   pd.DataFrame,
    gt_df:        pd.DataFrame,
    filter_stats: dict[str, dict[str, int]],
    config:       dict[str, Any],
    output_dir:   Path,
) -> dict[str, Any]:
    """Compare detections against ground truth and generate accuracy report.

    Saves output/accuracy_report.json with full PRD Section 10 metrics.

    Args:
        feature_df: Detections DataFrame.
        gt_df: Ground truth DataFrame with zone_id, event_type.
        filter_stats: Per-filter effectiveness counts.
        config: Config dict.
        output_dir: Output directory.

    Returns:
        Accuracy report as dict.
    """
    detected_ids  = set(feature_df[feature_df["is_anomaly"] == 1]["zone_id"].astype(str))
    gt_ids        = set(gt_df["zone_id"].astype(str)) if len(gt_df) > 0 else set()
    all_ids       = set(feature_df["zone_id"].astype(str))

    tp = len(detected_ids & gt_ids)
    fp = len(detected_ids - gt_ids)
    fn = len(gt_ids - detected_ids)
    tn = len(all_ids - detected_ids - gt_ids)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / max(len(all_ids), 1)

    # Per-threat-type accuracy
    threat_accuracy: dict[str, Any] = {}
    for t_type in ["mining", "encroachment", "deforestation"]:
        detected_type = set(
            feature_df[
                (feature_df["is_anomaly"] == 1) &
                (feature_df["threat_type"] == t_type)
            ]["zone_id"].astype(str)
        )
        gt_type = set(gt_df[gt_df["event_type"] == t_type]["zone_id"].astype(str)) if len(gt_df) > 0 else set()
        tt_tp = len(detected_type & gt_type)
        tt_fp = len(detected_type - gt_type)
        tt_fn = len(gt_type - detected_type)
        tt_prec = tt_tp / (tt_tp + tt_fp + 1e-9)
        tt_rec  = tt_tp / (tt_tp + tt_fn + 1e-9)
        tt_f1   = 2 * tt_prec * tt_rec / (tt_prec + tt_rec + 1e-9)
        threat_accuracy[t_type] = {
            "detected": len(detected_type),
            "true_positives": tt_tp,
            "precision": round(tt_prec, 4),
            "recall":    round(tt_rec,  4),
            "f1":        round(tt_f1,   4),
        }

    # Confidence distribution [0-20, 20-40, 40-60, 60-80, 80-100]
    conf_bins = [0, 20, 40, 60, 80, 100]
    conf_vals = feature_df[feature_df["is_anomaly"] == 1]["confidence"].values
    conf_dist = {}
    for lo, hi in zip(conf_bins, conf_bins[1:]):
        conf_dist[f"{lo}-{hi}%"] = int(((conf_vals >= lo) & (conf_vals < hi)).sum())

    # Drift score distribution
    drift_ranges = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 7.0), (7.0, 8.5), (8.5, 10.01)]
    drift_vals = feature_df[feature_df["is_anomaly"] == 1]["drift_score"].values
    drift_dist = {}
    for lo, hi in drift_ranges:
        drift_dist[f"{lo}-{hi:.1f}"] = int(((drift_vals >= lo) & (drift_vals < hi)).sum())

    # DSR classification distribution
    dsr_dist = feature_df["dsr_classification"].value_counts().to_dict()

    # Ensemble agreement distribution
    votes_vals = feature_df[feature_df["is_anomaly"] == 1]["ensemble_votes"].values
    ensemble_agreement = {
        "4_of_4": int((votes_vals == 4).sum()),
        "3_of_4": int((votes_vals == 3).sum()),
        "2_of_4": int((votes_vals == 2).sum()),
        "1_of_4": int((votes_vals == 1).sum()),
    }

    report = {
        "summary": {
            "total_zones":     len(all_ids),
            "total_detected":  len(detected_ids),
            "true_positives":  tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives":  tn,
        },
        "metrics": {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1_score":  round(f1,        4),
            "accuracy":  round(accuracy,  4),
        },
        "per_threat_type": threat_accuracy,
        "confidence_distribution":  conf_dist,
        "drift_score_distribution": drift_dist,
        "dsr_classification":       {k: int(v) for k, v in dsr_dist.items()},
        "ensemble_agreement":       ensemble_agreement,
        "filter_effectiveness":     filter_stats,
    }

    out_path = output_dir / "accuracy_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Accuracy: precision=%.2f, recall=%.2f, F1=%.2f (TP=%d, FP=%d, FN=%d)",
        precision, recall, f1, tp, fp, fn,
    )
    return report


# ── 13. SAVE ──────────────────────────────────────────────────────────────────

def save_detections(
    feature_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save all-zone detections CSV and print a console summary.

    Args:
        feature_df: Final detections DataFrame.
        output_dir: Output directory.
    """
    out_path = output_dir / "detections.csv"
    feature_df.to_csv(out_path, index=False)
    n_detected = int((feature_df["is_anomaly"] == 1).sum())
    logger.info("Saved detections.csv: %d rows, %d detected", len(feature_df), n_detected)

    # Console summary
    detected = feature_df[feature_df["is_anomaly"] == 1]
    print("\n" + "=" * 60)
    print(" ARAVALLI INTELLIGENCE — DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Total zones analysed : {len(feature_df):,}")
    print(f"  Detected threats     : {n_detected:,} ({100 * n_detected / max(len(feature_df), 1):.1f}%)")
    if n_detected > 0:
        for t in ["mining", "encroachment", "deforestation", "localized_disturbance"]:
            cnt = (detected["threat_type"] == t).sum()
            if cnt:
                print(f"    {t:<28}: {cnt:>4}")
        print(f"  Mean confidence      : {detected['confidence'].mean():.1f}%")
        print(f"  Mean drift score     : {detected['drift_score'].mean():.2f}/10")
    print("=" * 60 + "\n")


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

def run_detection(
    config: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full 13-step detection pipeline.

    Args:
        config: Loaded configuration dict.
        output_dir: Override output directory (defaults to 'output/').

    Returns:
        Accuracy report dict.
    """
    t0 = time.time()
    out = Path(output_dir) if output_dir else Path("output")
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load
    feature_df, gt_df = load_features(out, config)

    # 2. Feature matrix
    X, feature_names = build_feature_matrix(feature_df)

    # 3. Scale
    scaler_name = config.get("ensemble", {}).get("scaler", "robust")
    X_scaled, _ = scale_features(X, scaler_name)

    # 4. DSR
    feature_df = compute_dsr(feature_df, config)

    # 5. Drift score
    feature_df = compute_drift_score(feature_df, config)

    # 6. Ensemble
    feature_df, iso_model = run_ensemble(X_scaled, feature_df, config)

    # 7. Temporal confirmation
    feature_df = apply_temporal_confirmation(feature_df, config)

    # 8. Filters
    if "filter_removed" not in feature_df.columns:
        feature_df["filter_removed"] = ""
    feature_df, filter_stats = apply_filters(feature_df, config)

    # 9. Classify threats
    feature_df = classify_threats(feature_df, config)

    # 10. Feature importance
    feature_df = compute_feature_importance(X_scaled, feature_df, feature_names, iso_model, config)

    # 11. GeoJSON
    export_geojson(feature_df, config, out)

    # 12. Accuracy
    report = compute_accuracy(feature_df, gt_df, filter_stats, config, out)

    # 13. Save
    save_detections(feature_df, out)

    elapsed = time.time() - t0
    logger.info("Detection pipeline complete in %.1fs", elapsed)

    return report


# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config_loader import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("output/pipeline.log", mode="a", encoding="utf-8"),
        ],
    )

    cfg = load_config()
    run_detection(cfg)
