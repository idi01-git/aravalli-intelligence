"""
Aravalli Intelligence — Data Ingestion Module
Loads satellite data from real CSV, generates synthetic data, or connects to GEE.

Modes:
    real_file  — Read data/real_aravalli_7year.csv (default, offline)
    synthetic  — Generate mathematically controlled data with injected events
    gee_live   — Download from Google Earth Engine (developer only, run once)

Author: Shivang, Team BIOBYTES
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = [
    "zone_id", "lat", "lon", "elevation",
    "timestamp", "ndvi", "ndbi", "bsi", "nightlight",
]
MONTHS = 84  # Jan 2019 — Dec 2025
TIMESTAMPS = pd.date_range("2019-01-01", periods=MONTHS, freq="MS").strftime("%Y-%m").tolist()


# ── Mode 1: Real File ──────────────────────────────────────────────────────

def load_real_file(config: dict[str, Any]) -> pd.DataFrame:
    """Load real satellite data from CSV.

    Args:
        config: Configuration dict from config.yaml.

    Returns:
        DataFrame with columns: zone_id, lat, lon, elevation,
        timestamp, ndvi, ndbi, bsi, nightlight.

    Raises:
        FileNotFoundError: If CSV is missing and fallback is disabled.
        ValueError: If CSV structure is invalid.
    """
    csv_path = Path(config["data"]["real_file_path"])
    if not csv_path.exists():
        if config["data"].get("fallback_to_synthetic", True):
            logger.warning("Real CSV not found at %s — falling back to synthetic", csv_path)
            df, _ = generate_synthetic(config)
            return df
        raise FileNotFoundError(
            f"Real CSV not found: {csv_path}. "
            "Run `python scripts/harvest_gee.py` to download, "
            "or set data.fallback_to_synthetic: true in config.yaml."
        )

    logger.info("Loading real data from %s", csv_path)
    df = pd.read_csv(csv_path)
    _validate_structure(df, source="real_file")
    logger.info("Loaded %d rows from real CSV (%d zones × %d months)", len(df), df["zone_id"].nunique(), MONTHS)
    return df


# ── Mode 2: Synthetic ──────────────────────────────────────────────────────

def generate_synthetic(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic satellite data with injected degradation events.

    Uses numpy.random.default_rng for full reproducibility. Generates
    realistic NDVI seasonal curves, correlated NDBI/BSI/nightlight,
    and injects degradation events per PRD Section 8.

    Args:
        config: Configuration dict with synthetic parameters.

    Returns:
        Tuple of (data_df, ground_truth_df).
        data_df has columns matching EXPECTED_COLUMNS.
        ground_truth_df has columns: zone_id, event_type, start_month, source, confidence.
    """
    synth = config["data"]["synthetic"]
    zone_count: int = synth.get("zone_count", 1000)
    seed: int = synth.get("seed", 42)
    noise: float = synth.get("noise_level", 0.03)
    deforestation_pct: float = synth.get("deforestation_pct", 0.12)
    encroachment_pct: float = synth.get("encroachment_pct", 0.08)
    mining_pct: float = synth.get("mining_pct", 0.05)
    event_onset: int = synth.get("event_onset_months", 3)

    rng = np.random.default_rng(seed)

    lat_range = synth.get("lat_range", [24.5, 27.5])
    lon_range = synth.get("lon_range", [72.5, 75.5])
    elev_range = synth.get("elevation_range", [500, 1500])

    logger.info("Generating synthetic data: %d zones, seed=%d", zone_count, seed)

    # ── Zone metadata ───────────────────────────────────────────────────
    lats = rng.uniform(lat_range[0], lat_range[1], zone_count)
    lons = rng.uniform(lon_range[0], lon_range[1], zone_count)
    elevations = rng.uniform(elev_range[0], elev_range[1], zone_count).astype(int)

    zone_ids = [f"zone_{i+1:04d}" for i in range(zone_count)]

    # ── Assign events (no zone gets multiple) ──────────────────────────
    n_deforest = int(zone_count * deforestation_pct)
    n_encroach = int(zone_count * encroachment_pct)
    n_mining = int(zone_count * mining_pct)
    total_events = n_deforest + n_encroach + n_mining

    if total_events > zone_count:
        scale = zone_count / total_events
        n_deforest = int(n_deforest * scale)
        n_encroach = int(n_encroach * scale)
        n_mining = int(n_mining * scale)
        logger.warning("Event percentages scaled to fit zone count")

    indices = rng.permutation(zone_count)
    event_map: dict[int, tuple[str, int]] = {}
    ground_truth_rows: list[dict[str, Any]] = []
    ptr = 0

    for event_type, count in [
        ("deforestation", n_deforest),
        ("encroachment", n_encroach),
        ("mining", n_mining),
    ]:
        for _ in range(count):
            zone_idx = int(indices[ptr])
            start_month = int(rng.integers(36, 73))  # month 36-72
            event_map[zone_idx] = (event_type, start_month)
            ground_truth_rows.append({
                "zone_id": zone_ids[zone_idx],
                "event_type": event_type,
                "start_month": start_month,
                "source": "synthetic_injection",
                "confidence": "high",
            })
            ptr += 1

    # ── Generate timeseries ────────────────────────────────────────────
    all_rows: list[dict[str, Any]] = []
    months_array = np.arange(MONTHS)

    for zone_idx in range(zone_count):
        elev = elevations[zone_idx]

        # ── PRD Section 8: Config-driven elevation tier NDVI curves ────
        elev_threshold = synth.get("elevation_tier_threshold", 900)
        curves = synth.get("curves", {})

        if elev < elev_threshold:
            monthly_curve = curves.get("low_elevation", [
                0.22, 0.20, 0.18, 0.16, 0.18, 0.28,
                0.42, 0.50, 0.48, 0.38, 0.30, 0.25,
            ])
        else:
            monthly_curve = curves.get("high_elevation", [
                0.32, 0.30, 0.28, 0.26, 0.28, 0.38,
                0.52, 0.62, 0.60, 0.48, 0.40, 0.35,
            ])

        # Repeat 12-month curve across 84 months, then add noise
        curve_arr = np.array(monthly_curve * 7)[:MONTHS]  # 12*7=84
        ndvi = curve_arr + rng.normal(0, noise, MONTHS)
        ndvi = np.clip(ndvi, 0.05, 0.80)

        # NDBI: anti-correlated with NDVI (urban vs vegetation)
        ndbi = -0.15 + 0.1 * (1 - ndvi) + rng.normal(0, noise * 0.5, MONTHS)
        ndbi = np.clip(ndbi, -0.40, 0.40)

        # BSI: higher when less vegetation
        bsi = 0.05 + 0.15 * (1 - ndvi) + rng.normal(0, noise * 0.5, MONTHS)
        bsi = np.clip(bsi, -0.20, 0.50)

        # Nightlight: PRD Section 8 — fixed rural baseline (default 5) + noise
        nl_base = synth.get("nightlight_rural_baseline", 5)
        nightlight = nl_base + rng.normal(0, 0.5, MONTHS)
        nightlight = np.clip(nightlight, 0.5, 63.0)

        # ── Inject degradation events ──────────────────────────────────
        if zone_idx in event_map:
            event_type, start_m = event_map[zone_idx]
            onset = event_onset

            if event_type == "deforestation":
                drop_mag = rng.uniform(0.15, 0.25)
                for m in range(start_m, MONTHS):
                    progress = min(1.0, (m - start_m) / onset)
                    ndvi[m] -= drop_mag * progress

            elif event_type == "encroachment":
                ndvi_drop = rng.uniform(0.10, 0.15)
                ndbi_rise = rng.uniform(0.10, 0.20)
                nl_rise = rng.uniform(5.0, 15.0)
                for m in range(start_m, MONTHS):
                    progress = min(1.0, (m - start_m) / onset)
                    ndvi[m] -= ndvi_drop * progress
                    ndbi[m] += ndbi_rise * progress
                    nightlight[m] += nl_rise * progress

            elif event_type == "mining":
                ndvi_drop = rng.uniform(0.10, 0.20)
                bsi_rise = rng.uniform(0.10, 0.20)
                nl_rise = rng.uniform(2.0, 5.0)
                for m in range(start_m, MONTHS):
                    progress = min(1.0, (m - start_m) / onset)
                    ndvi[m] -= ndvi_drop * progress
                    bsi[m] += bsi_rise * progress
                    nightlight[m] += nl_rise * progress

        # Reclip after injection
        ndvi = np.clip(ndvi, 0.01, 0.80)
        ndbi = np.clip(ndbi, -0.40, 0.50)
        bsi = np.clip(bsi, -0.20, 0.60)
        nightlight = np.clip(nightlight, 0.5, 63.0)

        for m in range(MONTHS):
            all_rows.append({
                "zone_id": zone_ids[zone_idx],
                "lat": round(float(lats[zone_idx]), 6),
                "lon": round(float(lons[zone_idx]), 6),
                "elevation": int(elev),
                "timestamp": TIMESTAMPS[m],
                "ndvi": round(float(ndvi[m]), 6),
                "ndbi": round(float(ndbi[m]), 6),
                "bsi": round(float(bsi[m]), 6),
                "nightlight": round(float(nightlight[m]), 2),
            })

    data_df = pd.DataFrame(all_rows)
    ground_truth_df = pd.DataFrame(ground_truth_rows)

    logger.info(
        "Synthetic data generated: %d rows, %d events (%d deforestation, %d encroachment, %d mining)",
        len(data_df), total_events, n_deforest, n_encroach, n_mining,
    )
    return data_df, ground_truth_df


# ── Mode 3: GEE Live (Stub — implemented in scripts/harvest_gee.py) ─────

def load_gee_live(config: dict[str, Any]) -> pd.DataFrame:
    """Download data from Google Earth Engine.

    This is a stub. The actual GEE harvesting logic lives in
    scripts/harvest_gee.py and is run once to create the real CSV.

    Args:
        config: Configuration dict.

    Returns:
        DataFrame with standard columns.

    Raises:
        NotImplementedError: Always, directing user to harvest script.
    """
    raise NotImplementedError(
        "GEE Live mode is implemented via `python scripts/harvest_gee.py`. "
        "Run the harvester once, then switch to mode: real_file."
    )


# ── Validation ──────────────────────────────────────────────────────────────

def _validate_structure(df: pd.DataFrame, source: str = "unknown") -> None:
    """Validate DataFrame has expected columns and reasonable shape.

    Args:
        df: DataFrame to validate.
        source: Label for error messages.

    Raises:
        ValueError: If structure is invalid.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"[{source}] Missing columns: {missing}. Expected: {EXPECTED_COLUMNS}")

    n_zones = df["zone_id"].nunique()
    if n_zones < 10:
        raise ValueError(f"[{source}] Only {n_zones} zones found — expected at least 10.")

    n_months = df.groupby("zone_id").size()
    if n_months.min() < 12:
        raise ValueError(
            f"[{source}] Some zones have fewer than 12 months of data. "
            f"Min: {n_months.min()}, Zone: {n_months.idxmin()}"
        )

    # Check for NaN in critical columns
    nan_counts = df[["ndvi", "ndbi", "bsi"]].isna().sum()
    if nan_counts.any():
        logger.warning("[%s] NaN values found: %s — will be forward-filled", source, nan_counts.to_dict())
        df[["ndvi", "ndbi", "bsi"]] = df.groupby("zone_id")[["ndvi", "ndbi", "bsi"]].ffill().bfill()

    logger.info("[%s] Validation passed: %d zones, %d-%d months/zone", source, n_zones, n_months.min(), n_months.max())


# ── Save ────────────────────────────────────────────────────────────────────

def save_raw_data(df: pd.DataFrame, path: Path) -> None:
    """Save raw data to CSV.

    Args:
        df: DataFrame to save.
        path: Output file path (e.g., output/raw_data.csv).

    Raises:
        IOError: If file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Saved raw data to %s (%.1f MB, %d rows)", path, size_mb, len(df))


# ── Entry Point ─────────────────────────────────────────────────────────────

def run_ingestion(config: dict[str, Any], output_dir: Path | None = None) -> pd.DataFrame:
    """Run the full ingestion pipeline based on config mode.

    Args:
        config: Configuration dict from config.yaml.
        output_dir: Output directory (default: ./output).

    Returns:
        Raw data DataFrame ready for feature engineering.
    """
    mode = config["data"]["mode"]
    out = output_dir or Path("output")

    logger.info("═══ DATA INGESTION [mode=%s] ═══", mode)

    if mode == "real_file":
        df = load_real_file(config)
        # Copy ground truth if available
        gt_path = Path(config["data"].get("ground_truth_path", "data/real_ground_truth.csv"))
        if gt_path.exists():
            shutil.copy2(gt_path, out / "ground_truth.csv")
            logger.info("Copied ground truth to output/")

    elif mode == "synthetic":
        df, gt_df = generate_synthetic(config)
        gt_df.to_csv(out / "ground_truth.csv", index=False)
        logger.info("Saved synthetic ground truth: %d events", len(gt_df))

    elif mode == "gee_live":
        df = load_gee_live(config)

    else:
        raise ValueError(f"Unknown data.mode: '{mode}'. Use: real_file, synthetic, or gee_live.")

    # Save the actual data mode used for UI display
    with open(out / "data_mode.txt", "w") as f:
        f.write(mode)

    save_raw_data(df, out / "raw_data.csv")
    return df


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
    df = run_ingestion(config)
    print(f"\n✓ Ingestion complete: {len(df):,} rows, {df['zone_id'].nunique()} zones")
    print(f"  Columns: {list(df.columns)}")
    print(f"  NDVI range: [{df['ndvi'].min():.4f}, {df['ndvi'].max():.4f}]")
    print(f"  Output: output/raw_data.csv")
