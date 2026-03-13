"""
Aravalli Intelligence — AI Explanation Layer
pipeline/explain.py

Transforms ML detections into plain-language field reports using a 3-tier
LLM chain: GPT OSS 120B (Reasoning) → Llama 3.3 70B (Text) → Template.

Author: Team BIOBYTES
PRD Reference: Section 12 — AI Explanation Layer
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt  (RTF + RODES framework — prompt-engineer skill)
# Optimised for both reasoning (GPT OSS 120B) and text-to-text (Llama 3.3 70B)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an experienced ecological analyst specialising in the \
Aravalli Range, Rajasthan, India.

Your role: provide field officers with clear, actionable threat assessments.

For every detected threat you MUST address:
1. What type of threat? (mining, encroachment, or deforestation)
2. What evidence supports this? (cite the specific index values you are given)
3. Is this definitely NOT seasonal? (refer to the DSR value explicitly)
4. How confident are we? (state the confidence percentage)
5. What is the spatial context? (isolated or regional pattern?)
6. What action should the officer take?
7. How urgent is this? (low / medium / high / critical)

Hard rules:
- Maximum 200 words. No exceptions.
- Plain language — suitable for field rangers, not PhDs.
- No technical jargon. No emoji.
- Be direct and actionable. Cite numbers, do not speculate."""

# ─────────────────────────────────────────────────────────────────────────────
# User prompt template
# ─────────────────────────────────────────────────────────────────────────────
USER_PROMPT_TEMPLATE = """\
THREAT DETECTED: {threat_type} (Confidence: {confidence:.0f}%)

ZONE INFORMATION:
  Zone ID:   {zone_id}
  Location:  {lat:.4f}, {lon:.4f}
  Elevation: {elevation:.0f}m

SPECTRAL EVIDENCE:
  NDVI change:       {ndvi_delta:+.3f}  (current: {ndvi_current:.3f})
  NDBI change:       {ndbi_delta:+.3f}
  BSI change:        {bsi_delta:+.3f}
  Nightlight change: {nightlight_delta:+.1f}

TEMPORAL EVIDENCE:
  Consecutive decline months: {consecutive_declines}
  Short-term slope:           {slope_short:+.4f}
  Recovery signal:            {recovery_signal:+.3f}
  Volatility ratio:           {volatility_ratio:.2f}

SPATIAL EVIDENCE:
  Anomaly score:    {local_anomaly_score:.2f}
  Is isolated:      {is_isolated}
  Regional health:  {regional_health:.2f}
  Ensemble votes:   {ensemble_votes}/4

STATISTICAL PROOF:
  DSR value:          {dsr:.2f}
  DSR classification: {dsr_classification}
  Drift score:        {drift_score:.2f}/10

THREAT ASSESSMENT:
  Threat score: {threat_score:.1f}
  Severity:     {severity}
  Top drivers:
    1. {driver_1} (z-score: {driver_1_zscore:.1f})
    2. {driver_2} (z-score: {driver_2_zscore:.1f})
    3. {driver_3} (z-score: {driver_3_zscore:.1f})

Please provide a concise field assessment (≤200 words)."""


# ─────────────────────────────────────────────────────────────────────────────
# Template fallback — GUARANTEED output, never fails
# ─────────────────────────────────────────────────────────────────────────────
def template_report(zone: dict[str, Any]) -> str:
    """Generate a deterministic field report from zone data (no LLM required).

    Args:
        zone: Dictionary of zone data values from detections.csv.

    Returns:
        Plain-text report under 200 words.
    """
    threat = str(zone.get("threat_type", "unclassified")).upper()
    zone_id = zone.get("zone_id", "UNKNOWN")
    lat = float(zone.get("lat", 0))
    lon = float(zone.get("lon", 0))
    elevation = float(zone.get("elevation", 0))
    ndvi_delta = float(zone.get("ndvi_delta", 0))
    ndvi_current = float(zone.get("ndvi_current", 0))
    dsr = float(zone.get("dsr", 0))
    confidence = float(zone.get("confidence", 0))
    consecutive_declines = int(zone.get("consecutive_declines", 0))
    local_anomaly_score = float(zone.get("local_anomaly_score", 0))
    severity = str(zone.get("severity", "low"))
    ndbi_delta = float(zone.get("ndbi_delta", 0))
    bsi_delta = float(zone.get("bsi_delta", 0))
    nightlight_delta = float(zone.get("nightlight_delta", 0))

    # Threat-specific evidence sentence
    threat_lower = str(zone.get("threat_type", "")).lower()
    if "encroachment" in threat_lower:
        evidence = (
            f"NDBI (buildings) rose {ndbi_delta:+.2f} and nightlight increased "
            f"{nightlight_delta:+.0f}. Signatures confirm urban expansion."
        )
    elif "mining" in threat_lower:
        evidence = (
            f"Bare soil index (BSI) rose {bsi_delta:+.2f}, exposing subsoil. "
            f"No significant building construction detected. Pattern matches mining."
        )
    else:  # deforestation / localized / unclassified
        evidence = (
            "No construction or exposed-soil signatures. Forest clearing detected. "
            "Vegetation recovery unlikely without intervention."
        )

    # Urgency sentence
    urgency_map = {
        "critical": "URGENT: Deploy field team immediately for ground verification.",
        "severe":   "URGENT: Deploy field team immediately for ground verification.",
        "high":     "Send field team within 48 hours for full assessment.",
        "medium":   "Monitor closely. Verify during next scheduled field visit.",
        "low":      "Document for next quarter's survey.",
    }
    urgency = urgency_map.get(severity.lower(), "Document for next quarter's survey.")

    is_isolated_str = (
        "isolated from" if zone.get("is_isolated", False) else "embedded within"
    )

    lines = [
        f"{threat} DETECTED — Zone {zone_id}",
        f"Location: {lat:.3f}°N, {lon:.3f}°E at {elevation:.0f}m elevation.",
        "",
        f"NDVI dropped {abs(ndvi_delta):.2f} (current: {ndvi_current:.2f}). "
        f"DSR = {dsr:.1f}x seasonal variation — this is NOT a seasonal change.",
        f"Confidence: {confidence:.0f}%.",
        "",
        evidence,
        "",
        f"Temporal: {consecutive_declines} consecutive declining months.",
        f"Spatial: Anomaly score {local_anomaly_score:.1f}. "
        f"Zone is {is_isolated_str} healthy neighbours.",
        "",
        f"Recommended Action: {urgency}",
        "",
        "— Aravalli Intelligence v1.0",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(zone: dict[str, Any]) -> str:
    """Interpolate zone data into the user prompt template.

    Args:
        zone: Dictionary of zone data values.

    Returns:
        Fully formatted user prompt string.
    """
    safe = {
        "threat_type":         str(zone.get("threat_type", "unclassified")),
        "confidence":          float(zone.get("confidence", 0)),
        "zone_id":             str(zone.get("zone_id", "")),
        "lat":                 float(zone.get("lat", 0)),
        "lon":                 float(zone.get("lon", 0)),
        "elevation":           float(zone.get("elevation", 0)),
        "ndvi_delta":          float(zone.get("ndvi_delta", 0)),
        "ndvi_current":        float(zone.get("ndvi_current", 0)),
        "ndbi_delta":          float(zone.get("ndbi_delta", 0)),
        "bsi_delta":           float(zone.get("bsi_delta", 0)),
        "nightlight_delta":    float(zone.get("nightlight_delta", 0)),
        "consecutive_declines":int(zone.get("consecutive_declines", 0)),
        "slope_short":         float(zone.get("slope_short", 0)),
        "recovery_signal":     float(zone.get("recovery_signal", 0)),
        "volatility_ratio":    float(zone.get("volatility_ratio", 0)),
        "local_anomaly_score": float(zone.get("local_anomaly_score", 0)),
        "is_isolated":         bool(zone.get("is_isolated", False)),
        "regional_health":     float(zone.get("regional_health", 0)),
        "ensemble_votes":      int(zone.get("ensemble_votes", 0)),
        "dsr":                 float(zone.get("dsr", 0)),
        "dsr_classification":  str(zone.get("dsr_classification", "unknown")),
        "drift_score":         float(zone.get("drift_score", 0)),
        "threat_score":        float(zone.get("threat_score", 0)),
        "severity":            str(zone.get("severity", "low")),
        "driver_1":            str(zone.get("driver_1", "NDVI change")),
        "driver_1_zscore":     float(zone.get("driver_1_zscore", 0)),
        "driver_2":            str(zone.get("driver_2", "Consecutive declines")),
        "driver_2_zscore":     float(zone.get("driver_2_zscore", 0)),
        "driver_3":            str(zone.get("driver_3", "Anomaly score")),
        "driver_3_zscore":     float(zone.get("driver_3_zscore", 0)),
    }
    return USER_PROMPT_TEMPLATE.format(**safe)


# ─────────────────────────────────────────────────────────────────────────────
# Generic LLM caller via OpenAI-compatible REST
# ─────────────────────────────────────────────────────────────────────────────
def _call_llm(
    prompt: str,
    system_prompt: str,
    api_endpoint: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    extra_params: dict[str, Any] | None = None,
) -> str | None:
    """Call any OpenAI-compatible API endpoint.

    Args:
        prompt:        The user-facing prompt text.
        system_prompt: The system instruction prompt.
        api_endpoint:  Full URL to the /chat/completions endpoint.
        api_key:       Bearer token / API key.
        model:         Model identifier string.
        temperature:   Sampling temperature (0.0–1.0).
        max_tokens:    Maximum output tokens.
        timeout:       Request timeout in seconds.
        extra_params:  Optional dict of extra payload parameters (e.g. reasoning_effort).

    Returns:
        Report text string, or None on any failure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if extra_params:
        payload.update(extra_params)
        
    try:
        resp = requests.post(
            api_endpoint, headers=headers, json=payload, timeout=timeout
        )
        if resp.status_code != 200:
            logger.warning("LLM API returned HTTP %s: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return text if text else None
    except requests.Timeout:
        logger.warning("LLM call timed out after %ss", timeout)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM call error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public callers — PRIMARY and FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def call_gpt_oss(prompt: str, system_prompt: str, config: dict[str, Any]) -> str | None:
    """Call GPT OSS 120B (primary reasoning model).

    Args:
        prompt:        User prompt for a single zone.
        system_prompt: Ecological analyst system instruction.
        config:        Full application config dict.

    Returns:
        Report text or None on failure.
    """
    llm_cfg = config.get("llm", {}).get("primary", {})
    
    # Extract extra_params if provided in config
    extra_params = {}
    if "reasoning_effort" in llm_cfg:
        extra_params["reasoning_effort"] = llm_cfg["reasoning_effort"]
        
    result = _call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        api_endpoint=llm_cfg.get("api_endpoint", ""),
        api_key=llm_cfg.get("api_key", ""),
        model=llm_cfg.get("model", "openai/gpt-oss-120b"),
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 300),
        timeout=llm_cfg.get("timeout", 30),
        extra_params=extra_params if extra_params else None,
    )
    delay = config.get("llm", {}).get("primary", {}).get("delay_between_calls", 2)
    time.sleep(delay)
    return result


def call_llama(prompt: str, system_prompt: str, config: dict[str, Any]) -> str | None:
    """Call Llama 3.3 70B (text-to-text fallback model).

    Args:
        prompt:        User prompt for a single zone.
        system_prompt: Ecological analyst system instruction.
        config:        Full application config dict.

    Returns:
        Report text or None on failure.
    """
    llm_cfg = config.get("llm", {}).get("fallback", {})
    result = _call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        api_endpoint=llm_cfg.get("api_endpoint", ""),
        api_key=llm_cfg.get("api_key", ""),
        model=llm_cfg.get("model", "llama-3.3-70b"),
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 300),
        timeout=llm_cfg.get("timeout", 20),
    )
    delay = config.get("llm", {}).get("fallback", {}).get("delay_between_calls", 1)
    time.sleep(delay)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_cache(cache_path: Path) -> dict[str, Any]:
    """Load cached_reports.json, returning an empty dict on any error.

    Args:
        cache_path: Path to the cache file.

    Returns:
        Dictionary mapping zone_id → report record.
    """
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cache load failed (%s) — starting fresh.", exc)
        return {}


def save_cache(cache_path: Path, cache_dict: dict[str, Any]) -> None:
    """Save cached_reports.json with pretty-print formatting.

    Args:
        cache_path:  Destination path.
        cache_dict:  Full cache dictionary to persist.
    """
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(cache_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.warning("Cache save failed: %s", exc)


def _make_cache_record(text: str, source: str) -> dict[str, Any]:
    return {
        "text":      text,
        "source":    source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main per-zone orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def get_or_create_report(
    zone_id: str,
    zone_data: dict[str, Any],
    config: dict[str, Any],
    cache: dict[str, Any],
) -> tuple[str, str]:
    """Return (report_text, source_tag) for a single zone.

    Implements the full 3-tier chain:
        GPT OSS 120B  →  Llama 3.3 70B  →  Template

    Also reads from / writes to the supplied cache dict in-memory.
    The caller is responsible for persisting the cache to disk.

    Args:
        zone_id:   Zone identifier string (e.g. "zone_0042").
        zone_data: Dictionary of all zone feature values.
        config:    Full application config dict.
        cache:     In-memory cache dict (mutated in-place).

    Returns:
        Tuple of (report_text, source_tag).
    """
    # ── 1. Check cache ──────────────────────────────────────────────────────
    if zone_id in cache:
        logger.debug("Cache hit: %s", zone_id)
        record = cache[zone_id]
        return record["text"], "cached"

    # ── 2. Build prompt ─────────────────────────────────────────────────────
    prompt = build_prompt(zone_data)

    # ── 3. Try GPT OSS 120B (Primary — Reasoning) ───────────────────────────
    text = call_gpt_oss(prompt, SYSTEM_PROMPT, config)
    if text:
        source = "live_gpt_oss"
        logger.info("GPT OSS 120B report generated for %s", zone_id)
        cache[zone_id] = _make_cache_record(text, source)
        return text, source

    logger.warning("GPT OSS 120B failed for %s — trying Llama 3.3 70B.", zone_id)

    # ── 4. Try Llama 3.3 70B (Fallback — Text-to-Text) ─────────────────────
    text = call_llama(prompt, SYSTEM_PROMPT, config)
    if text:
        source = "live_llama"
        logger.info("Llama 3.3 70B report generated for %s", zone_id)
        cache[zone_id] = _make_cache_record(text, source)
        return text, source

    logger.warning("Llama 3.3 70B failed for %s — using template.", zone_id)

    # ── 5. Template fallback (Guaranteed) ───────────────────────────────────
    text = template_report(zone_data)
    source = "template"
    cache[zone_id] = _make_cache_record(text, source)
    return text, source


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_explain(config: dict[str, Any], force_regenerate: bool = False) -> dict[str, Any]:
    """Run the explanation pipeline for all detected zones.

    Reads output/detections.csv, generates AI reports for every detected
    zone, caches results to output/cached_reports.json.

    Args:
        config: Full application config dict (from config_loader.py).
        force_regenerate: If True, ignores existing cache and regenerates all.

    Returns:
        Dictionary of zone_id → report record (mirrors cache structure).
    """
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    detections_path = output_dir / "detections.csv"
    cache_path      = output_dir / "cached_reports.json"

    if not detections_path.exists():
        logger.error("detections.csv not found. Run pipeline/detect.py first.")
        return {}

    df = pd.read_csv(detections_path)
    detected = df[df["is_anomaly"] == 1].copy()
    logger.info("Loaded %d detected zones for reporting.", len(detected))

    cache = load_cache(cache_path)
    if force_regenerate:
        logger.info("Force regeneration requested. Ignoring existing cache.")
        cache = {}
        
    results: dict[str, Any] = {}
    counts = {"live_gpt_oss": 0, "live_llama": 0, "cached": 0, "template": 0}

    t0 = time.time()
    for _, row in detected.iterrows():
        zone_id   = str(row.get("zone_id", ""))
        zone_data = row.to_dict()

        text, source = get_or_create_report(zone_id, zone_data, config, cache)
        results[zone_id] = _make_cache_record(text, source)
        counts[source] = counts.get(source, 0) + 1

        # Persist cache after every zone so partial runs are never lost
        save_cache(cache_path, cache)

    elapsed = time.time() - t0
    logger.info(
        "Explain complete in %.1fs — GPT OSS: %d, Llama: %d, cached: %d, template: %d",
        elapsed, counts["live_gpt_oss"], counts["live_llama"],
        counts["cached"], counts["template"],
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_loader import load_config  # type: ignore[import]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    force = "--force" in sys.argv
    cfg = load_config()
    reports = run_explain(cfg, force_regenerate=force)
    print(f"\n{'='*60}")
    print(f" ARAVALLI INTELLIGENCE — EXPLAIN COMPLETE")
    print(f" Reports generated: {len(reports)}")
    print(f"{'='*60}\n")
