"""
Aravalli Intelligence — GEE Harvester
One-time script to download 7-year Sentinel-2 + VIIRS data from Google Earth Engine.

Run ONCE on developer machine:
    python scripts/harvest_gee.py

Prerequisites:
    pip install earthengine-api
    earthengine authenticate          <- run this in terminal first

What this does:
    1. Creates a 40×25 grid (1000 points) across Aravalli bounding box
    2. For each month (Jan 2019 – Dec 2025, 84 months):
       - Sentinel-2 L2A: computes NDVI, NDBI, BSI with cloud masking
       - VIIRS: computes monthly nightlight composite
    3. Saves data/real_aravalli_7year.csv  (84,000 rows)
    4. Saves data/real_ground_truth.csv    (empty — no known events in real data)

Author: Shivang, Team BIOBYTES
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# ── Logging setup (ASCII only — Windows cp1252 safe) ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [harvester] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/harvest.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("harvester")

# ── Constants ────────────────────────────────────────────────────────────────
ARAVALLI_BOUNDS = {
    "lat_min": 24.5, "lat_max": 27.5,
    "lon_min": 72.5, "lon_max": 75.5,
}
GRID_COLS = 40  # 40 × 25 = 1000 points
GRID_ROWS = 25
ZONE_COUNT = GRID_COLS * GRID_ROWS  # 1000

START_YEAR, START_MONTH = 2019, 1
END_YEAR,   END_MONTH   = 2025, 12
MONTHS_TOTAL = 84

OUTPUT_CSV  = Path("data/real_aravalli_7year.csv")
GT_CSV      = Path("data/real_ground_truth.csv")
CHECKPOINT  = Path("output/harvest_checkpoint.json")


# ── Zone Grid ────────────────────────────────────────────────────────────────

def build_zone_grid() -> list[dict]:
    """Create a 40×25 regular grid of lat/lon points across Aravalli range.

    Returns:
        List of dicts with zone_id, lat, lon, elevation (placeholder).
    """
    import numpy as np
    lats = np.linspace(ARAVALLI_BOUNDS["lat_min"], ARAVALLI_BOUNDS["lat_max"], GRID_ROWS)
    lons = np.linspace(ARAVALLI_BOUNDS["lon_min"], ARAVALLI_BOUNDS["lon_max"], GRID_COLS)
    zones = []
    idx = 1
    for lat in lats:
        for lon in lons:
            zones.append({
                "zone_id": f"zone_{idx:04d}",
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
                "elevation": 800,  # Will be overridden by SRTM below
            })
            idx += 1
    logger.info("Built zone grid: %d zones", len(zones))
    return zones


# ── GEE Functions ────────────────────────────────────────────────────────────

def _cloud_mask_s2(image):
    """Apply Sentinel-2 SCL cloud mask (Scene Classification Layer).

    Args:
        image: ee.Image Sentinel-2 L2A image.

    Returns:
        Masked image with cloud and cloud-shadow pixels removed.
    """
    import ee
    scl = image.select("SCL")
    # PRD Section 8 — SCL values to EXCLUDE (8 classes):
    # 0=no data, 1=saturated, 2=dark area, 3=cloud shadow,
    # 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus, 11=snow/ice
    # KEEP: 4=vegetation, 5=bare soil, 6=water, 7=unclassified (borderline but allowed)
    cloud_free = (
        scl.neq(0).And(scl.neq(1)).And(scl.neq(2)).And(scl.neq(3))
           .And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    )
    return image.updateMask(cloud_free)


def _compute_indices(image):
    """Compute NDVI, NDBI, BSI from Sentinel-2 L2A bands.

    Sentinel-2 L2A Band Reference:
        B2  = Blue  (490nm)
        B4  = Red   (665nm)
        B8  = NIR   (842nm)
        B11 = SWIR1 (1610nm)

    Args:
        image: ee.Image with S2 bands scaled to [0, 1].

    Returns:
        Image with added bands: ndvi, ndbi, bsi.
    """
    import ee
    # Scale factor: S2 L2A values are 0-10000
    blue  = image.select("B2").divide(10000)
    red   = image.select("B4").divide(10000)
    nir   = image.select("B8").divide(10000)
    swir  = image.select("B11").divide(10000)

    ndvi = nir.subtract(red).divide(nir.add(red).add(1e-10)).rename("ndvi")
    ndbi = swir.subtract(nir).divide(swir.add(nir).add(1e-10)).rename("ndbi")

    # BSI = ((SWIR+Red) - (NIR+Blue)) / ((SWIR+Red) + (NIR+Blue))
    bsi_num = swir.add(red).subtract(nir.add(blue))
    bsi_den = swir.add(red).add(nir.add(blue)).add(1e-10)
    bsi = bsi_num.divide(bsi_den).rename("bsi")

    return image.addBands([ndvi, ndbi, bsi])


def fetch_month(
    ee_module,
    zones: list[dict],
    year: int,
    month: int,
    checkpoint_data: dict,
) -> list[dict]:
    """Fetch Sentinel-2 and VIIRS data for a single month and all zones.

    Args:
        ee_module: The imported ee module.
        zones: List of zone dicts with lat, lon.
        year: Year to fetch (2019-2025).
        month: Month to fetch (1-12).
        checkpoint_data: Dict tracking already-fetched months.

    Returns:
        List of data rows for this month.
    """
    ee = ee_module
    timestamp = f"{year}-{month:02d}"

    if timestamp in checkpoint_data:
        logger.info("  [SKIP] %s already fetched", timestamp)
        return checkpoint_data[timestamp]

    logger.info("  Fetching %s ...", timestamp)
    t0 = time.time()

    # Date range for this month
    start = f"{year}-{month:02d}-01"
    if month == 12:
        end = f"{year+1}-01-01"
    else:
        end = f"{year}-{month+1:02d}-01"

    rows = []

    try:
        # ── Sentinel-2 L2A ─────────────────────────────────────────────
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # PRD: < 20%
            .map(_cloud_mask_s2)
            .map(_compute_indices)
            .select(["ndvi", "ndbi", "bsi"])
        )

        # Monthly median composite (reduces cloud noise further)
        composite_s2 = s2.median()

        # ── VIIRS Nightlight — PRD: VCMSLCFG (stray-light corrected) ────
        viirs = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")  # PRD: not VCMCFG
            .filterDate(start, end)
            .select("avg_rad")
            .median()
        )

        # ── Sample all zones ────────────────────────────────────────────
        # Build feature collection from zone points
        features = [
            ee.Feature(
                ee.Geometry.Point([z["lon"], z["lat"]]),
                {"zone_id": z["zone_id"], "lat": z["lat"], "lon": z["lon"], "elevation": z["elevation"]}
            )
            for z in zones
        ]
        fc = ee.FeatureCollection(features)

        # Sample S2 composite at zone locations
        sampled_s2 = composite_s2.sampleRegions(
            collection=fc,
            scale=10,          # Sentinel-2 native resolution (10m)
            projection="EPSG:4326",
            geometries=True,
        )

        # Sample VIIRS
        sampled_viirs = viirs.sampleRegions(
            collection=fc,
            scale=500,         # VIIRS native resolution (~500m)
            projection="EPSG:4326",
            geometries=True,
        )

        # ── Retrieve as JSON ────────────────────────────────────────────
        s2_data = sampled_s2.getInfo()
        viirs_data = sampled_viirs.getInfo()

        # Build zone→nightlight lookup
        nl_lookup: dict[str, float] = {}
        for f in viirs_data.get("features", []):
            props = f.get("properties", {})
            nl_lookup[props.get("zone_id", "")] = props.get("avg_rad", 0.0) or 0.0

        # Process S2 results
        for f in s2_data.get("features", []):
            props = f.get("properties", {})
            zone_id = props.get("zone_id", "")
            zone_meta = next((z for z in zones if z["zone_id"] == zone_id), {})

            rows.append({
                "zone_id": zone_id,
                "lat": props.get("lat", zone_meta.get("lat", 0)),
                "lon": props.get("lon", zone_meta.get("lon", 0)),
                "elevation": props.get("elevation", zone_meta.get("elevation", 800)),
                "timestamp": timestamp,
                "ndvi": round(float(props.get("ndvi") or 0.0), 6),
                "ndbi": round(float(props.get("ndbi") or 0.0), 6),
                "bsi": round(float(props.get("bsi") or 0.0), 6),
                "nightlight": round(float(nl_lookup.get(zone_id, 0.0)), 2),
            })

        # Handle zones with no data (cloud cover 100%)
        received_ids = {r["zone_id"] for r in rows}
        for zone in zones:
            if zone["zone_id"] not in received_ids:
                rows.append({
                    "zone_id": zone["zone_id"],
                    "lat": zone["lat"],
                    "lon": zone["lon"],
                    "elevation": zone["elevation"],
                    "timestamp": timestamp,
                    "ndvi": float("nan"),
                    "ndbi": float("nan"),
                    "bsi": float("nan"),
                    "nightlight": float("nan"),
                })

        elapsed = time.time() - t0
        logger.info("  ✓ %s: %d zones fetched in %.1fs", timestamp, len(rows), elapsed)

    except Exception as e:
        logger.error("  ✗ %s: GEE error: %s — using NaN row", timestamp, e)
        for zone in zones:
            rows.append({
                "zone_id": zone["zone_id"],
                "lat": zone["lat"],
                "lon": zone["lon"],
                "elevation": zone["elevation"],
                "timestamp": timestamp,
                "ndvi": float("nan"),
                "ndbi": float("nan"),
                "bsi": float("nan"),
                "nightlight": float("nan"),
            })

    return rows


def fetch_elevation(ee_module, zones: list[dict]) -> list[dict]:
    """Fetch real elevation from SRTM DEM for all zone centroids.

    Args:
        ee_module: The imported ee module.
        zones: List of zone dicts to update in place.

    Returns:
        Updated zones list with real elevation values.
    """
    ee = ee_module
    logger.info("Fetching elevation from SRTM...")

    try:
        srtm = ee.Image("USGS/SRTMGL1_003")
        features = [
            ee.Feature(ee.Geometry.Point([z["lon"], z["lat"]]), {"zone_id": z["zone_id"]})
            for z in zones
        ]
        fc = ee.FeatureCollection(features)
        sampled = srtm.sampleRegions(collection=fc, scale=30, geometries=False)
        data = sampled.getInfo()

        elev_lookup: dict[str, int] = {}
        for f in data.get("features", []):
            props = f.get("properties", {})
            zid = props.get("zone_id", "")
            elev_val = props.get("elevation", 800)
            if elev_val is None:
                elev_val = 800
            elev_lookup[zid] = max(0, int(elev_val))

        for z in zones:
            z["elevation"] = elev_lookup.get(z["zone_id"], 800)

        logger.info("Elevation fetched for %d zones", len(elev_lookup))
    except Exception as e:
        logger.warning("Elevation fetch failed (%s) — using 800m default", e)

    return zones


# ── Main Harvest Orchestrator ─────────────────────────────────────────────

def main() -> None:
    """Run the full GEE harvest pipeline.

    Flow:
        1. Authenticate with GEE
        2. Build 1000-zone grid
        3. Fetch SRTM elevation
        4. For each of 84 months: fetch S2 + VIIRS
        5. Impute NaN values (forward-fill per zone)
        6. Save to CSV
    """
    import ee
    import numpy as np
    import pandas as pd

    Path("output").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # ── Auth ───────────────────────────────────────────────────────────
    logger.info("=" * 52)
    logger.info("  ARAVALLI GEE HARVESTER - Starting")
    logger.info("=" * 52)
    logger.info("Initializing GEE...")

    # Read project from config.yaml
    import yaml
    cfg_path = Path("config.yaml")
    gee_project = None
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        gee_project = cfg.get("data", {}).get("gee_project")
        if gee_project:
            logger.info("Using GEE project from config: %s", gee_project)

    # Try initialize with project, then without, then authenticate
    init_ok = False
    if gee_project:
        try:
            ee.Initialize(project=gee_project)
            init_ok = True
        except Exception as e1:
            logger.warning("Project init failed (%s), trying without project...", e1)

    if not init_ok:
        try:
            ee.Initialize()
            init_ok = True
        except Exception as e2:
            logger.warning("No-project init failed (%s), re-authenticating...", e2)

    if not init_ok:
        try:
            ee.Authenticate()
            ee.Initialize()
            init_ok = True
        except Exception as e3:
            logger.error("GEE authentication failed: %s", e3)
            logger.error("Steps to fix:")
            logger.error("  1. earthengine authenticate")
            logger.error("  2. Create a GEE Cloud project at: https://code.earthengine.google.com/")
            logger.error("  3. Set data.gee_project in config.yaml to your project ID")
            sys.exit(1)

    logger.info("GEE initialized OK")

    # ── Build grid ─────────────────────────────────────────────────────
    zones = build_zone_grid()
    zones = fetch_elevation(ee, zones)

    # ── Load checkpoint ────────────────────────────────────────────────
    checkpoint_data: dict = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            checkpoint_data = json.load(f)
        logger.info("Checkpoint found: %d months already done", len(checkpoint_data))

    # ── Fetch 84 months ────────────────────────────────────────────────
    all_rows: list[dict] = []
    month_count = 0
    year, month = START_YEAR, START_MONTH
    t_total = time.time()

    while (year, month) <= (END_YEAR, END_MONTH):
        logger.info(
            "[%d/%d] Year %d, Month %02d ...",
            month_count + 1, MONTHS_TOTAL, year, month
        )
        rows = fetch_month(ee, zones, year, month, checkpoint_data)
        all_rows.extend(rows)

        # Save checkpoint after each month
        timestamp = f"{year}-{month:02d}"
        checkpoint_data[timestamp] = rows
        with open(CHECKPOINT, "w") as f:
            json.dump(checkpoint_data, f)

        month_count += 1
        month += 1
        if month > 12:
            month = 1
            year += 1

        # Rate limiting — GEE allows ~1 request/second burst
        time.sleep(0.5)

    # ── Assemble DataFrame ──────────────────────────────────────────────
    logger.info("Assembling DataFrame...")
    df = pd.DataFrame(all_rows)

    # Forward-fill NaN values per zone (satellite gaps due to clouds)
    for col in ["ndvi", "ndbi", "bsi", "nightlight"]:
        df[col] = df.groupby("zone_id")[col].transform(
            lambda x: x.ffill().bfill()
        )

    # Final fallback: regional median
    for col in ["ndvi", "ndbi", "bsi", "nightlight"]:
        df[col] = df[col].fillna(df[col].median())

    # Clip to valid ranges
    df["ndvi"] = df["ndvi"].clip(-1.0, 1.0)
    df["ndbi"] = df["ndbi"].clip(-1.0, 1.0)
    df["bsi"]  = df["bsi"].clip(-1.0, 1.0)
    df["nightlight"] = df["nightlight"].clip(0.0, 63.0)

    # ── Save ────────────────────────────────────────────────────────────
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # Ground truth (empty for real data — no confirmed events)
    gt_df = pd.DataFrame(columns=["zone_id", "event_type", "start_month", "source", "confidence"])
    gt_df.to_csv(GT_CSV, index=False)

    elapsed = time.time() - t_total
    size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)

    logger.info("=" * 52)
    logger.info("  HARVEST COMPLETE")
    logger.info("  Rows:     %d", len(df))
    logger.info("  Zones:    %d", df["zone_id"].nunique())
    logger.info("  Months:   %d", month_count)
    logger.info("  File:     data/real_aravalli_7year.csv")
    logger.info("  Size:     %.1f MB", size_mb)
    logger.info("  Time:     %.1f minutes", elapsed / 60)
    logger.info("=" * 52)

    logger.info("")
    logger.info("Next step: Set config.yaml `data.mode: real_file` and run the pipeline.")

    # Clean up checkpoint
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        logger.info("Checkpoint file removed.")


if __name__ == "__main__":
    main()
