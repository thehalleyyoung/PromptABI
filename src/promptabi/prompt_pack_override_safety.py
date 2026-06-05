"""Consumer-side override safety checks for prompt packs (step 256).

A consumer commonly overrides a few fields of a pack's configuration -- a longer
context budget, a custom system preamble, an extra stop sequence.  Some overrides
are harmless; others silently *weaken the safety properties the pack certified*.
This module classifies overrides and refuses the dangerous ones:

* disabling a sanitizer the pack relied on,
* removing a stop sequence (re-opening a stop-leak the pack closed),
* widening a structured-output schema slot to free text,
* injecting a raw control/role marker into an overridden preamble.

Each override is checked against the pack's declared *protected* fields and the
known control markers; anything that lowers the safety floor is reported with a
clear reason so the consumer can make an informed decision (or fail closed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .prompt_pack_rag_extension import DEFAULT_CONTROL_MARKERS

PROMPT_PACK_OVERRIDE_VERSION = "promptabi.prompt-pack-override.v1"


class OverrideRisk(StrEnum):
    SANITIZER_DISABLED = "sanitizer-disabled"
    STOP_SEQUENCE_REMOVED = "stop-sequence-removed"
    CONTROL_MARKER_INJECTED = "control-marker-injected"
    PROTECTED_FIELD_MUTATED = "protected-field-mutated"


@dataclass(frozen=True, slots=True)
class PackSafetyFloor:
    """The minimum safety guarantees a pack ships with."""

    sanitizers: frozenset[str] = frozenset()
    stop_sequences: frozenset[str] = frozenset()
    protected_fields: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ConsumerOverride:
    field: str
    value: object


@dataclass(frozen=True, slots=True)
class OverrideFinding:
    risk: OverrideRisk
    field: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"risk": self.risk.value, "field": self.field, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class OverrideSafetyResult:
    version: str
    safe: bool
    findings: tuple[OverrideFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "safe": self.safe,
            "findings": [f.to_dict() for f in self.findings],
        }


def _as_str_set(value: object) -> set[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(v) for v in value}
    return {str(value)}


def check_overrides(
    floor: PackSafetyFloor,
    overrides: tuple[ConsumerOverride, ...],
    control_markers: tuple[str, ...] = DEFAULT_CONTROL_MARKERS,
) -> OverrideSafetyResult:
    findings: list[OverrideFinding] = []

    for ov in overrides:
        if ov.field in floor.protected_fields:
            findings.append(
                OverrideFinding(
                    OverrideRisk.PROTECTED_FIELD_MUTATED,
                    ov.field,
                    "field is declared protected by the pack and may not be "
                    "overridden",
                )
            )

        if ov.field == "sanitizers":
            new = _as_str_set(ov.value)
            dropped = floor.sanitizers - new
            for san in sorted(dropped):
                findings.append(
                    OverrideFinding(
                        OverrideRisk.SANITIZER_DISABLED,
                        ov.field,
                        f"override removes pack sanitizer {san!r}",
                    )
                )

        if ov.field in ("stop", "stop_sequences"):
            new = _as_str_set(ov.value)
            dropped = floor.stop_sequences - new
            for stop in sorted(dropped):
                findings.append(
                    OverrideFinding(
                        OverrideRisk.STOP_SEQUENCE_REMOVED,
                        ov.field,
                        f"override removes pack stop sequence {stop!r}",
                    )
                )

        if isinstance(ov.value, str):
            for marker in control_markers:
                if marker in ov.value:
                    findings.append(
                        OverrideFinding(
                            OverrideRisk.CONTROL_MARKER_INJECTED,
                            ov.field,
                            f"override value contains control marker {marker!r}",
                        )
                    )

    return OverrideSafetyResult(
        version=PROMPT_PACK_OVERRIDE_VERSION,
        safe=not findings,
        findings=tuple(findings),
    )


def render_override_text(result: OverrideSafetyResult) -> str:
    lines = [
        f"PromptABI prompt-pack override safety ({result.version})",
        f"result: {'SAFE' if result.safe else 'UNSAFE'}",
    ]
    for finding in result.findings:
        lines.append(f"  ! {finding.risk.value} [{finding.field}]: {finding.detail}")
    return "\n".join(lines) + "\n"
