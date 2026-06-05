"""Rank diagnostic fix suggestions for safe triage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .diagnostics import Diagnostic


class FixSafety(StrEnum):
    """How likely a suggested change is to make the interface safer."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FixCompatibility(StrEnum):
    """How likely a suggested change is to preserve downstream compatibility."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FixBlastRadius(StrEnum):
    """How much of the application a suggested change is expected to touch."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class RankedFixSuggestion:
    """A deterministic ranked fix candidate with explicit risk dimensions."""

    text: str
    rank: int
    score: int
    safety: FixSafety
    compatibility: FixCompatibility
    blast_radius: FixBlastRadius
    changes_user_visible_prompt_behavior: bool
    rationale: tuple[str, ...]
    diagnostic_count: int
    rules: tuple[str, ...]
    artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "text": self.text,
            "score": self.score,
            "safety": self.safety.value,
            "compatibility": self.compatibility.value,
            "blast_radius": self.blast_radius.value,
            "changes_user_visible_prompt_behavior": self.changes_user_visible_prompt_behavior,
            "rationale": list(self.rationale),
            "diagnostic_count": self.diagnostic_count,
            "rules": list(self.rules),
            "artifacts": list(self.artifacts),
        }


def rank_fix_suggestions(diagnostics: Sequence[Diagnostic] | Iterable[Diagnostic]) -> tuple[RankedFixSuggestion, ...]:
    """Rank unique diagnostic suggestions by safety, compatibility, scope, and UX impact."""

    grouped: dict[str, list[Diagnostic]] = defaultdict(list)
    for diagnostic in diagnostics:
        for suggestion in diagnostic.suggestions:
            grouped[suggestion].append(diagnostic)

    ranked_without_position = [
        _score_suggestion(text, tuple(sorted(items, key=lambda diagnostic: diagnostic.sort_key)))
        for text, items in grouped.items()
    ]
    ordered = sorted(
        ranked_without_position,
        key=lambda item: (
            -item.score,
            item.changes_user_visible_prompt_behavior,
            _blast_radius_rank(item.blast_radius),
            -item.diagnostic_count,
            item.text.lower(),
        ),
    )
    return tuple(
        RankedFixSuggestion(
            text=item.text,
            rank=index,
            score=item.score,
            safety=item.safety,
            compatibility=item.compatibility,
            blast_radius=item.blast_radius,
            changes_user_visible_prompt_behavior=item.changes_user_visible_prompt_behavior,
            rationale=item.rationale,
            diagnostic_count=item.diagnostic_count,
            rules=item.rules,
            artifacts=item.artifacts,
        )
        for index, item in enumerate(ordered, start=1)
    )


def ordered_suggestion_texts(diagnostics: Sequence[Diagnostic] | Iterable[Diagnostic]) -> tuple[str, ...]:
    """Return suggestion text ordered by the fix-ranking policy."""

    return tuple(suggestion.text for suggestion in rank_fix_suggestions(diagnostics))


def _score_suggestion(text: str, diagnostics: tuple[Diagnostic, ...]) -> RankedFixSuggestion:
    properties = _merged_fix_properties(diagnostics)
    safety = _coerce_safety(properties.get("fix_safety")) or _infer_safety(text, diagnostics)
    compatibility = _coerce_compatibility(properties.get("fix_compatibility")) or _infer_compatibility(text, diagnostics)
    blast_radius = _coerce_blast_radius(properties.get("fix_blast_radius")) or _infer_blast_radius(text, diagnostics)
    user_visible = _coerce_bool(
        properties.get("fix_changes_user_visible_prompt_behavior")
        if "fix_changes_user_visible_prompt_behavior" in properties
        else properties.get("fix_user_visible_prompt_behavior")
    )
    if user_visible is None:
        user_visible = _infer_user_visible_prompt_change(text, diagnostics)

    score = (
        _SAFETY_POINTS[safety]
        + _COMPATIBILITY_POINTS[compatibility]
        + _BLAST_RADIUS_POINTS[blast_radius]
        + (0 if user_visible else 10)
        + min(len(diagnostics), 5)
    )
    rules = tuple(sorted({diagnostic.rule_id for diagnostic in diagnostics}))
    artifacts = tuple(sorted(_artifact_label(diagnostic) for diagnostic in diagnostics if diagnostic.artifact is not None))
    rationale = _rationale(safety, compatibility, blast_radius, user_visible, diagnostics)
    return RankedFixSuggestion(
        text=text,
        rank=0,
        score=score,
        safety=safety,
        compatibility=compatibility,
        blast_radius=blast_radius,
        changes_user_visible_prompt_behavior=user_visible,
        rationale=rationale,
        diagnostic_count=len({diagnostic.fingerprint for diagnostic in diagnostics}),
        rules=rules,
        artifacts=artifacts,
    )


def _merged_fix_properties(diagnostics: tuple[Diagnostic, ...]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for diagnostic in diagnostics:
        for key, value in diagnostic.properties:
            if key.startswith("fix_") and value is not None and str(value):
                merged.setdefault(key, str(value))
    return merged


def _infer_safety(text: str, diagnostics: tuple[Diagnostic, ...]) -> FixSafety:
    lowered = _haystack(text, diagnostics)
    if any(term in lowered for term in ("suppress", "ignore", "disable", "remove the check", "accepted risk")):
        return FixSafety.LOW
    if any(term in lowered for term in ("escape", "encode", "redact", "sanitize", "pin", "lockfile", "hash", "verify")):
        return FixSafety.HIGH
    return FixSafety.MEDIUM


def _infer_compatibility(text: str, diagnostics: tuple[Diagnostic, ...]) -> FixCompatibility:
    lowered = _haystack(text, diagnostics)
    if any(term in lowered for term in ("rewrite", "replace", "change provider", "change the provider", "lower context", "truncate")):
        return FixCompatibility.LOW
    if any(term in text.lower() for term in ("document", "annotation", "metadata", "lockfile", "pin")):
        return FixCompatibility.HIGH
    if any(term in lowered for term in ("template", "schema", "stop", "tool", "parser", "role delimiter")):
        return FixCompatibility.MEDIUM
    if any(term in lowered for term in ("pin", "lockfile", "metadata", "document", "annotation", "explicit")):
        return FixCompatibility.HIGH
    return FixCompatibility.MEDIUM


def _infer_blast_radius(text: str, diagnostics: tuple[Diagnostic, ...]) -> FixBlastRadius:
    lowered = _haystack(text, diagnostics)
    if any(term in text.lower() for term in ("document", "annotation", "metadata", "lockfile", "pin")):
        return FixBlastRadius.LOW
    if any(term in lowered for term in ("provider", "template", "truncation", "context budget", "framework")):
        return FixBlastRadius.HIGH
    if any(term in lowered for term in ("schema", "tool", "stop", "parser", "grammar")):
        return FixBlastRadius.MEDIUM
    if any(term in lowered for term in ("pin", "lockfile", "metadata", "document", "annotation", "explicit")):
        return FixBlastRadius.LOW
    return FixBlastRadius.MEDIUM


def _infer_user_visible_prompt_change(text: str, diagnostics: tuple[Diagnostic, ...]) -> bool:
    lowered = _haystack(text, diagnostics)
    if any(term in lowered for term in ("lockfile", "metadata", "document", "annotation", "sha256", "hash", "policy")):
        return False
    return any(
        term in lowered
        for term in (
            "prompt",
            "template",
            "escape",
            "encode",
            "schema",
            "stop",
            "tool",
            "parser",
            "truncation",
            "context",
            "provider",
            "role",
        )
    )


def _rationale(
    safety: FixSafety,
    compatibility: FixCompatibility,
    blast_radius: FixBlastRadius,
    user_visible: bool,
    diagnostics: tuple[Diagnostic, ...],
) -> tuple[str, ...]:
    reasons = [
        f"safety={safety.value}",
        f"compatibility={compatibility.value}",
        f"blast_radius={blast_radius.value}",
        "changes user-visible prompt behavior" if user_visible else "does not change user-visible prompt behavior",
    ]
    if len(diagnostics) > 1:
        reasons.append(f"addresses {len({diagnostic.fingerprint for diagnostic in diagnostics})} diagnostics")
    return tuple(reasons)


def _haystack(text: str, diagnostics: tuple[Diagnostic, ...]) -> str:
    parts = [text]
    for diagnostic in diagnostics:
        parts.append(diagnostic.rule_id)
        parts.append(diagnostic.message)
        if diagnostic.artifact is not None:
            parts.append(diagnostic.artifact.kind)
            parts.append(diagnostic.artifact.name)
    return " ".join(parts).lower()


def _coerce_safety(value: str | None) -> FixSafety | None:
    return _coerce_enum(FixSafety, value)


def _coerce_compatibility(value: str | None) -> FixCompatibility | None:
    return _coerce_enum(FixCompatibility, value)


def _coerce_blast_radius(value: str | None) -> FixBlastRadius | None:
    return _coerce_enum(FixBlastRadius, value)


def _coerce_enum(enum_type: type[FixSafety] | type[FixCompatibility] | type[FixBlastRadius], value: str | None):
    if value is None:
        return None
    normalized = value.strip().lower().replace("_", "-")
    try:
        return enum_type(normalized)
    except ValueError:
        return None


def _coerce_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _artifact_label(diagnostic: Diagnostic) -> str:
    assert diagnostic.artifact is not None
    location = diagnostic.artifact.location_uri
    if location is not None:
        return f"{diagnostic.artifact.kind}:{diagnostic.artifact.name}@{location}"
    return f"{diagnostic.artifact.kind}:{diagnostic.artifact.name}"


def _blast_radius_rank(blast_radius: FixBlastRadius) -> int:
    return {
        FixBlastRadius.LOW: 0,
        FixBlastRadius.MEDIUM: 1,
        FixBlastRadius.HIGH: 2,
    }[blast_radius]


_SAFETY_POINTS = {
    FixSafety.HIGH: 40,
    FixSafety.MEDIUM: 24,
    FixSafety.LOW: -20,
}
_COMPATIBILITY_POINTS = {
    FixCompatibility.HIGH: 30,
    FixCompatibility.MEDIUM: 18,
    FixCompatibility.LOW: 0,
}
_BLAST_RADIUS_POINTS = {
    FixBlastRadius.LOW: 20,
    FixBlastRadius.MEDIUM: 10,
    FixBlastRadius.HIGH: 0,
}
