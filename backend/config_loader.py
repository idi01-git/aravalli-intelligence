"""
Aravalli Intelligence — Configuration Loader & Validator
Loads config.yaml, validates parameters, deep-merges runtime overrides.

Author: Shivang, Team BIOBYTES
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file and inject .env secrets.

    Args:
        path: Path to config file. Defaults to project root config.yaml.

    Returns:
        Configuration dictionary containing API keys.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML is malformed.
    """
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Load .env file overrides silently
    load_dotenv()
    
    # Inject LLM keys into config if they exist
    if "llm" in config:
        if "primary" in config["llm"]:
            primary_key = os.environ.get("PRIMARY_LLM_API_KEY")
            if primary_key:
                config["llm"]["primary"]["api_key"] = primary_key
                
        if "fallback" in config["llm"]:
            fallback_key = os.environ.get("FALLBACK_LLM_API_KEY")
            if fallback_key:
                config["llm"]["fallback"]["api_key"] = fallback_key

    logger.info("Configuration loaded from %s", config_path)
    return config


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep merge overrides into a copy of base config.

    Args:
        base: Base configuration dictionary.
        overrides: Runtime override parameters from API.

    Returns:
        New merged dictionary (base is not mutated).
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate and auto-correct configuration parameters.

    Three-tier response:
        - Corrections: silently auto-applied
        - Warnings: included in response, pipeline runs
        - Rejections: HTTP 422, pipeline does NOT run

    Args:
        config: Configuration dictionary to validate.

    Returns:
        Tuple of (corrected_config, warnings, corrections).

    Raises:
        ValueError: If config has rejection-level issues.
    """
    config = copy.deepcopy(config)
    warnings: list[str] = []
    corrections: list[str] = []

    # ── REJECTIONS ──────────────────────────────────────────────────────
    methods = config.get("ensemble", {}).get("methods", {})
    enabled_methods = [m for m, v in methods.items() if v.get("enabled", False)]
    all_weights = [methods[m].get("weight", 0) for m in enabled_methods]

    if not enabled_methods:
        raise ValueError("BLOCK-1: All ensemble methods are disabled. Enable at least one.")

    # ── CORRECTIONS ─────────────────────────────────────────────────────

    # Normalize ensemble weights
    weight_sum = sum(all_weights)
    if enabled_methods and abs(weight_sum - 1.0) > 0.02:
        for m in enabled_methods:
            old = methods[m]["weight"]
            methods[m]["weight"] = round(old / weight_sum, 4)
        corrections.append(f"Ensemble weights normalized from sum={weight_sum:.2f} to 1.0")

    # Normalize score weights
    score_weights = config.get("classification", {}).get("score_weights", {})
    if score_weights:
        sw_sum = sum(score_weights.values())
        if abs(sw_sum - 1.0) > 0.02:
            for k in score_weights:
                score_weights[k] = round(score_weights[k] / sw_sum, 4)
            corrections.append(f"Score weights normalized from sum={sw_sum:.2f} to 1.0")

    # Normalize drift weights
    drift_weights = config.get("drift", {}).get("weights", {})
    if drift_weights:
        dw_sum = sum(drift_weights.values())
        if abs(dw_sum - 1.0) > 0.02:
            for k in drift_weights:
                drift_weights[k] = round(drift_weights[k] / dw_sum, 4)
            corrections.append(f"Drift weights normalized from sum={dw_sum:.2f} to 1.0")

    # DSR transition >= normal
    dsr = config.get("dsr", {})
    normal = dsr.get("threshold_normal", 1.5)
    transition = dsr.get("threshold_transition", 2.0)
    if transition < normal:
        dsr["threshold_transition"] = round(normal + 0.1, 2)
        corrections.append(f"DSR transition threshold raised from {transition} to {dsr['threshold_transition']}")

    # Clamp contamination
    for m_name in ["isolation_forest", "lof"]:
        m = methods.get(m_name, {})
        cont = m.get("contamination", 0.10)
        if cont < 0.01:
            m["contamination"] = 0.01
            corrections.append(f"{m_name} contamination clamped from {cont} to 0.01")
        elif cont > 0.30:
            m["contamination"] = 0.30
            corrections.append(f"{m_name} contamination clamped from {cont} to 0.30")

    # Synthetic event percentages
    synth = config.get("data", {}).get("synthetic", {})
    event_sum = synth.get("deforestation_pct", 0) + synth.get("encroachment_pct", 0) + synth.get("mining_pct", 0)
    if event_sum > 1.0:
        scale = 1.0 / event_sum
        for k in ["deforestation_pct", "encroachment_pct", "mining_pct"]:
            synth[k] = round(synth.get(k, 0) * scale, 4)
        corrections.append(f"Synthetic event percentages scaled down from {event_sum:.2f} to 1.0")

    # ── WARNINGS ────────────────────────────────────────────────────────

    if len(enabled_methods) == 1:
        warnings.append(f"ACK-1: Single detection method active ({enabled_methods[0]}). Higher false positive risk.")

    filters_cfg = config.get("filters", {})
    all_disabled = all(not filters_cfg.get(f, {}).get("enabled", True) for f in filters_cfg if isinstance(filters_cfg.get(f), dict))
    if all_disabled:
        warnings.append("ACK-4: All post-detection filters disabled. Results will be unfiltered.")

    smoothing = config.get("baseline", {}).get("smoothing_window", 3)
    if smoothing == 1:
        warnings.append("ACK-5: Smoothing window set to 1 (no smoothing). False positives may increase 20-30%.")

    logger.info("Config validated: %d corrections, %d warnings", len(corrections), len(warnings))
    return config, warnings, corrections


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    cfg = load_config()
    validated, warns, corrs = validate_config(cfg)
    print(f"Config loaded successfully. Corrections: {len(corrs)}, Warnings: {len(warns)}")
    for c in corrs:
        print(f"  [CORRECTED] {c}")
    for w in warns:
        print(f"  [WARNING] {w}")
