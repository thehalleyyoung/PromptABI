"""Tests for training-contract drift alarms (step 274)."""

from __future__ import annotations

from promptabi.training_drift_alarm import (
    AlarmLevel,
    ContractMetrics,
    DriftTolerance,
    detect_drift,
    render_drift_alarm_text,
)

BASE = ContractMetrics(
    run_id="base",
    valid_mask_ratio=0.99,
    supervised_token_ratio=0.30,
    forgery_rejection_rate=0.001,
    template_digest="d1",
)


def test_no_drift_is_ok() -> None:
    current = ContractMetrics("r", 0.99, 0.30, 0.001, "d1")
    assert detect_drift(BASE, current).level is AlarmLevel.OK


def test_warn_drift() -> None:
    current = ContractMetrics("r", 0.96, 0.30, 0.001, "d1")
    result = detect_drift(BASE, current)
    assert result.level is AlarmLevel.WARN


def test_critical_drift() -> None:
    current = ContractMetrics("r", 0.90, 0.30, 0.001, "d1")
    result = detect_drift(BASE, current)
    assert result.level is AlarmLevel.CRITICAL


def test_template_change_is_critical() -> None:
    current = ContractMetrics("r", 0.99, 0.30, 0.001, "d2")
    result = detect_drift(BASE, current)
    assert result.level is AlarmLevel.CRITICAL
    assert any(a.metric == "template_digest" for a in result.alarms)


def test_custom_tolerance() -> None:
    current = ContractMetrics("r", 0.97, 0.30, 0.001, "d1")
    tol = DriftTolerance(warn_delta=0.05, critical_delta=0.1)
    assert detect_drift(BASE, current, tol).level is AlarmLevel.OK


def test_render_text_smoke() -> None:
    current = ContractMetrics("r", 0.90, 0.30, 0.001, "d1")
    assert "drift" in render_drift_alarm_text(detect_drift(BASE, current))
