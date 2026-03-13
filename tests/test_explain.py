"""
tests/test_explain.py

10 pytest tests for the AI Explanation Layer (pipeline/explain.py).
All LLM API calls are mocked — tests never require live API keys.

PRD Reference: Section 12 — AI Explanation Layer
Author: Team BIOBYTES
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.explain import (
    build_prompt,
    call_gpt_oss,
    call_llama,
    get_or_create_report,
    load_cache,
    save_cache,
    template_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_zone() -> dict:
    """Minimal realistic zone record matching detections.csv columns."""
    return {
        "zone_id":              "zone_0042",
        "lat":                  25.412,
        "lon":                  73.891,
        "elevation":            820,
        "threat_type":          "deforestation",
        "confidence":           72.5,
        "ndvi_delta":           -0.21,
        "ndvi_current":         0.29,
        "ndbi_delta":           0.02,
        "bsi_delta":            0.04,
        "nightlight_delta":     1.0,
        "consecutive_declines": 8,
        "slope_short":          -0.012,
        "recovery_signal":      -0.05,
        "volatility_ratio":     1.3,
        "local_anomaly_score":  2.8,
        "is_isolated":          True,
        "regional_health":      0.61,
        "ensemble_votes":       3,
        "dsr":                  3.4,
        "dsr_classification":   "confirmed_degradation",
        "drift_score":          6.8,
        "threat_score":         71.0,
        "severity":             "high",
        "driver_1":             "Change exceeds seasonal expectation",
        "driver_1_zscore":      3.4,
        "driver_2":             "Consecutive declining months",
        "driver_2_zscore":      2.9,
        "driver_3":             "Bare soil exposure",
        "driver_3_zscore":      2.1,
    }


@pytest.fixture()
def base_config() -> dict:
    """Minimal config for the explain module."""
    return {
        "llm": {
            "primary": {
                "model":                "gpt-oss-120b-reasoning",
                "api_endpoint":         "https://api.example.com/chat",
                "api_key":              "fake-primary-key",
                "temperature":          0.3,
                "max_tokens":           300,
                "timeout":              30,
                "delay_between_calls":  0,   # 0 so tests run fast
            },
            "fallback": {
                "model":                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "api_endpoint":         "https://api.example.com/chat",
                "api_key":              "fake-fallback-key",
                "temperature":          0.3,
                "max_tokens":           300,
                "timeout":              20,
                "delay_between_calls":  0,
            },
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Cache loading
# ─────────────────────────────────────────────────────────────────────────────
def test_cache_loading(tmp_path: Path) -> None:
    """load_cache() reads an existing JSON file correctly."""
    cache_data = {
        "zone_0001": {
            "text": "Test report for zone 1.",
            "source": "live_gpt_oss",
            "timestamp": "2025-02-25T14:30:00Z",
        }
    }
    cache_file = tmp_path / "cached_reports.json"
    cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

    loaded = load_cache(cache_file)
    assert loaded["zone_0001"]["text"] == "Test report for zone 1."
    assert loaded["zone_0001"]["source"] == "live_gpt_oss"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Cache saving and round-trip
# ─────────────────────────────────────────────────────────────────────────────
def test_cache_saving(tmp_path: Path) -> None:
    """save_cache() persists data and load_cache() retrieves it unchanged."""
    cache: dict = {}
    cache_file = tmp_path / "cached_reports.json"

    cache["zone_0042"] = {
        "text": "Deforestation alert.",
        "source": "template",
        "timestamp": "2025-02-25T00:00:00Z",
    }
    save_cache(cache_file, cache)

    reloaded = load_cache(cache_file)
    assert reloaded["zone_0042"]["source"] == "template"
    assert reloaded["zone_0042"]["text"] == "Deforestation alert."


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Template generation quality
# ─────────────────────────────────────────────────────────────────────────────
def test_template_generation(sample_zone: dict) -> None:
    """template_report() always produces a report ≤200 words with key fields."""
    report = template_report(sample_zone)

    word_count = len(report.split())
    assert word_count <= 200, f"Template report is {word_count} words (limit: 200)"

    content = report.lower()
    assert "deforestation" in content, "Threat type must appear in report"
    assert "dsr" in content or "seasonal" in content, "Seasonal proof must be mentioned"
    assert "zone_0042" in content, "Zone ID must be present"
    assert "action" in content or "recommended" in content or "deploy" in content \
        or "send" in content or "monitor" in content or "document" in content, \
        "An action recommendation must be present"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Prompt building
# ─────────────────────────────────────────────────────────────────────────────
def test_prompt_building(sample_zone: dict) -> None:
    """build_prompt() includes all required data fields in the output string."""
    prompt = build_prompt(sample_zone)

    required_fields = [
        "zone_0042",       # zone_id
        "deforestation",   # threat_type
        "-0.210",          # ndvi_delta (formatted as +/-)
        "confirmed_degradation",  # dsr_classification
        "3.40",            # dsr value
        "Change exceeds seasonal expectation",  # driver_1
    ]
    for field in required_fields:
        assert field in prompt, f"Expected '{field}' in prompt output"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — GPT OSS 120B integration (mocked)
# ─────────────────────────────────────────────────────────────────────────────
def test_gpt_oss_integration(sample_zone: dict, base_config: dict) -> None:
    """call_gpt_oss() correctly parses the API response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "GPT OSS report: Deforestation confirmed."}}]
    }

    prompt = build_prompt(sample_zone)
    with patch("pipeline.explain.requests.post", return_value=mock_response):
        result = call_gpt_oss(prompt, "system", base_config)

    assert result == "GPT OSS report: Deforestation confirmed."


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Llama 3.3 70B integration (mocked)
# ─────────────────────────────────────────────────────────────────────────────
def test_llama_integration(sample_zone: dict, base_config: dict) -> None:
    """call_llama() correctly parses the API response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Llama report: Forest clearing detected."}}]
    }

    prompt = build_prompt(sample_zone)
    with patch("pipeline.explain.requests.post", return_value=mock_response):
        result = call_llama(prompt, "system", base_config)

    assert result == "Llama report: Forest clearing detected."


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Full fallback chain
# ─────────────────────────────────────────────────────────────────────────────
def test_fallback_chain(sample_zone: dict, base_config: dict) -> None:
    """When GPT OSS fails → Llama succeeds → source is 'live_llama'."""
    # GPT OSS returns None (failure), Llama returns a report
    with patch("pipeline.explain.call_gpt_oss", return_value=None):
        with patch("pipeline.explain.call_llama", return_value="Llama fallback report."):
            cache: dict = {}
            _, source = get_or_create_report(
                "zone_0042", sample_zone, base_config, cache
            )

    assert source == "live_llama"
    assert cache["zone_0042"]["source"] == "live_llama"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Timeout handling / full template fallback
# ─────────────────────────────────────────────────────────────────────────────
def test_timeout_handling(sample_zone: dict, base_config: dict) -> None:
    """When BOTH LLMs fail → template is used → source is 'template'."""
    with patch("pipeline.explain.call_gpt_oss", return_value=None):
        with patch("pipeline.explain.call_llama", return_value=None):
            cache: dict = {}
            text, source = get_or_create_report(
                "zone_0042", sample_zone, base_config, cache
            )

    assert source == "template"
    assert len(text) > 0
    assert "zone_0042" in text.lower() or "DEFORESTATION" in text


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Error resilience (various API errors never crash)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("http_status", [400, 401, 429, 500, 503])
def test_error_resilience(
    sample_zone: dict, base_config: dict, http_status: int
) -> None:
    """Any HTTP error code from the API never crashes — template is returned."""
    mock_response = MagicMock()
    mock_response.status_code = http_status
    mock_response.text = f"HTTP {http_status} error"

    with patch("pipeline.explain.requests.post", return_value=mock_response):
        cache: dict = {}
        text, source = get_or_create_report(
            "zone_0042", sample_zone, base_config, cache
        )

    assert text, "Every zone must produce a non-empty report"
    assert source in {"live_gpt_oss", "live_llama", "cached", "template"}


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — Word count enforcement
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("threat_type,severity", [
    ("deforestation",        "high"),
    ("mining",               "severe"),
    ("encroachment",         "medium"),
    ("localized_disturbance","low"),
])
def test_word_count(
    sample_zone: dict, threat_type: str, severity: str
) -> None:
    """Template reports must stay under 200 words for all threat types."""
    zone = dict(sample_zone, threat_type=threat_type, severity=severity)
    report = template_report(zone)
    word_count = len(report.split())
    assert word_count <= 200, (
        f"Report for {threat_type}/{severity} is {word_count} words (≤200 required)"
    )
