"""Training-contract drift alarms (step 274).

A long-running training program produces a stream of *contract metrics* per run:
the fraction of examples with a valid loss mask, the supervised-token ratio, the
rate of role-forgery rejections, the template-digest in use.  Quiet drift in any
of these is an early warning that the data or interface changed.  This module
compares a new run's metrics against a baseline and raises tiered alarms when a
metric drifts beyond a configured tolerance, or when a categorical fingerprint
(template digest) changes at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

DRIFT_ALARM_VERSION = "promptabi.training-drift-alarm.v1"


class AlarmLevel(StrEnum):
    OK = "ok"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ContractMetrics:
    run_id: str
    valid_mask_ratio: float
    supervised_token_ratio: float
    forgery_rejection_rate: float
    template_digest: str


@dataclass(frozen=True, slots=True)
class DriftTolerance:
    warn_delta: float = 0.02
    critical_delta: float = 0.05


@dataclass(frozen=True, slots=True)
class MetricAlarm:
    metric: str
    level: AlarmLevel
    delta: float
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "level": self.level.value,
            "delta": round(self.delta, 4),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class DriftAlarmResult:
    version: str
    level: AlarmLevel
    alarms: tuple[MetricAlarm, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "level": self.level.value,
            "alarms": [a.to_dict() for a in self.alarms],
        }


_NUMERIC = (
    ("valid_mask_ratio", "valid_mask_ratio"),
    ("supervised_token_ratio", "supervised_token_ratio"),
    ("forgery_rejection_rate", "forgery_rejection_rate"),
)


def _level_for(delta: float, tol: DriftTolerance) -> AlarmLevel:
    if delta >= tol.critical_delta:
        return AlarmLevel.CRITICAL
    if delta >= tol.warn_delta:
        return AlarmLevel.WARN
    return AlarmLevel.OK


def detect_drift(
    baseline: ContractMetrics,
    current: ContractMetrics,
    tol: DriftTolerance = DriftTolerance(),
) -> DriftAlarmResult:
    alarms: list[MetricAlarm] = []

    for metric, attr in _NUMERIC:
        delta = abs(getattr(current, attr) - getattr(baseline, attr))
        level = _level_for(delta, tol)
        if level is not AlarmLevel.OK:
            alarms.append(
                MetricAlarm(
                    metric=metric,
                    level=level,
                    delta=delta,
                    detail=f"{getattr(baseline, attr)} -> {getattr(current, attr)}",
                )
            )

    if baseline.template_digest != current.template_digest:
        alarms.append(
            MetricAlarm(
                metric="template_digest",
                level=AlarmLevel.CRITICAL,
                delta=1.0,
                detail=f"{baseline.template_digest} -> {current.template_digest}",
            )
        )

    overall = AlarmLevel.OK
    if any(a.level is AlarmLevel.CRITICAL for a in alarms):
        overall = AlarmLevel.CRITICAL
    elif any(a.level is AlarmLevel.WARN for a in alarms):
        overall = AlarmLevel.WARN

    return DriftAlarmResult(
        version=DRIFT_ALARM_VERSION,
        level=overall,
        alarms=tuple(alarms),
    )


def render_drift_alarm_text(result: DriftAlarmResult) -> str:
    lines = [
        f"PromptABI training-contract drift alarm ({result.version})",
        f"level: {result.level.value.upper()}",
    ]
    for a in result.alarms:
        lines.append(f"  [{a.level.value}] {a.metric}: Δ={a.delta:.3f} ({a.detail})")
    return "\n".join(lines) + "\n"
