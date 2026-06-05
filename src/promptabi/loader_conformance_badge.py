"""Data-loader conformance badges (step 271).

A reusable badge summarizes whether a data loader conforms to PromptABI's
training-data battery: loss-mask correctness (step 263), target-span survival
(step 270), and contract-preserving transforms (step 262).  The badge is a small,
shareable artifact (think shields.io) with a colour, a pass ratio, and the list
of failing checks, so a loader's README can advertise a verifiable status.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

LOADER_BADGE_VERSION = "promptabi.loader-badge.v1"


class BadgeColor(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class LoaderBadge:
    version: str
    loader: str
    color: BadgeColor
    passed: int
    total: int
    failing: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"promptabi conformance {self.passed}/{self.total}"

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "loader": self.loader,
            "color": self.color.value,
            "passed": self.passed,
            "total": self.total,
            "failing": list(self.failing),
            "label": self.label,
        }

    def to_shields_endpoint(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "label": "promptabi",
            "message": f"{self.passed}/{self.total}",
            "color": self.color.value,
        }


def build_loader_badge(
    loader: str,
    outcomes: tuple[CheckOutcome, ...],
) -> LoaderBadge:
    total = len(outcomes)
    passed = sum(1 for o in outcomes if o.passed)
    failing = tuple(o.name for o in outcomes if not o.passed)

    if total == 0:
        color = BadgeColor.RED
    elif passed == total:
        color = BadgeColor.GREEN
    elif passed == 0:
        color = BadgeColor.RED
    else:
        color = BadgeColor.YELLOW

    return LoaderBadge(
        version=LOADER_BADGE_VERSION,
        loader=loader,
        color=color,
        passed=passed,
        total=total,
        failing=failing,
    )


def render_badge_text(badge: LoaderBadge) -> str:
    lines = [
        f"PromptABI loader conformance badge ({badge.version})",
        f"loader: {badge.loader}",
        f"status: {badge.color.value.upper()} {badge.passed}/{badge.total}",
    ]
    for name in badge.failing:
        lines.append(f"  ! failing: {name}")
    return "\n".join(lines) + "\n"
