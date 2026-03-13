"""Quick verification script for Phase 2 pipeline."""
import sys
sys.stdout.reconfigure(line_buffering=True)
print("Starting verification...", flush=True)

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("[1/5] Importing modules...", flush=True)
from config_loader import load_config
from pipeline.ingest import generate_synthetic, _validate_structure
from pipeline.features import smooth_timeseries, compute_baseline, compute_temporal_features

print("[2/5] Loading config...", flush=True)
config = load_config()
print(f"  Config loaded: {config['project']['name']} v{config['project']['version']}", flush=True)

print("[3/5] Generating synthetic data (1000 zones x 84 months)...", flush=True)
df, gt = generate_synthetic(config)
print(f"  Data: {len(df)} rows, {df['zone_id'].nunique()} zones", flush=True)
print(f"  Ground truth: {len(gt)} events", flush=True)
print(f"  NDVI range: [{df['ndvi'].min():.4f}, {df['ndvi'].max():.4f}]", flush=True)

print("[4/5] Validating structure...", flush=True)
_validate_structure(df, source="synthetic")
print("  Validation passed!", flush=True)

print("[5/5] Testing smoothing...", flush=True)
smoothed = smooth_timeseries(df, window=3)
print(f"  Smoothed columns: {[c for c in smoothed.columns if 'smoothed' in c]}", flush=True)

# Save outputs
Path("output").mkdir(exist_ok=True)
df.to_csv("output/raw_data.csv", index=False)
gt.to_csv("output/ground_truth.csv", index=False)
print(f"  Saved output/raw_data.csv ({Path('output/raw_data.csv').stat().st_size / 1024:.0f} KB)", flush=True)

print("\n=== PHASE 2 VERIFICATION PASSED ===", flush=True)
