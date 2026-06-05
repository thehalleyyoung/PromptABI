"""Add prompt-pack deprecation and LTS metadata (step 248).

Long-lived prompt packs need a lifecycle contract so consumers can plan upgrades:
when a version was released, whether it is **long-term-support (LTS)**, when it
was *deprecated*, what supersedes it, and when support *ends*.  This module
models that metadata and -- crucially -- validates it so a pack cannot publish an
incoherent lifecycle (deprecating before release, ending support before
deprecation, marking a version both LTS and end-of-life, or pointing ``superseded_by``
at itself).

:func:`evaluate_lifecycle` answers the consumer's real questions against a
reference date: *is this version still supported? is it deprecated? how many days
of support remain?* and yields an actionable status plus the recommended
successor, so a CI gate can warn before a pack reaches end of life.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

PROMPT_PACK_LIFECYCLE_VERSION = "promptabi.prompt-pack-lifecycle.v1"


class LifecycleStatus(StrEnum):
    SUPPORTED = "supported"
    LTS = "lts"
    DEPRECATED = "deprecated"
    END_OF_LIFE = "end-of-life"
    PRERELEASE = "prerelease"


class LifecycleValidationKind(StrEnum):
    DEPRECATED_BEFORE_RELEASE = "deprecated-before-release"
    EOL_BEFORE_DEPRECATION = "eol-before-deprecation"
    EOL_BEFORE_RELEASE = "eol-before-release"
    LTS_WITH_EOL_IN_PAST = "lts-with-eol-in-past"
    SELF_SUPERSEDE = "self-supersede"
    MISSING_SUCCESSOR = "missing-successor"


@dataclass(frozen=True, slots=True)
class LifecycleValidation:
    kind: LifecycleValidationKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class PackLifecycle:
    pack_name: str
    version: str
    released_on: date
    lts: bool = False
    deprecated_on: date | None = None
    end_of_life_on: date | None = None
    superseded_by: str | None = None

    def validate(self) -> tuple[LifecycleValidation, ...]:
        problems: list[LifecycleValidation] = []
        if self.deprecated_on is not None and self.deprecated_on < self.released_on:
            problems.append(
                LifecycleValidation(
                    LifecycleValidationKind.DEPRECATED_BEFORE_RELEASE,
                    f"{self.deprecated_on} < release {self.released_on}",
                )
            )
        if self.end_of_life_on is not None:
            if self.end_of_life_on < self.released_on:
                problems.append(
                    LifecycleValidation(
                        LifecycleValidationKind.EOL_BEFORE_RELEASE,
                        f"{self.end_of_life_on} < release {self.released_on}",
                    )
                )
            if (
                self.deprecated_on is not None
                and self.end_of_life_on < self.deprecated_on
            ):
                problems.append(
                    LifecycleValidation(
                        LifecycleValidationKind.EOL_BEFORE_DEPRECATION,
                        f"eol {self.end_of_life_on} < deprecated {self.deprecated_on}",
                    )
                )
        if self.superseded_by == self.version:
            problems.append(
                LifecycleValidation(
                    LifecycleValidationKind.SELF_SUPERSEDE,
                    f"{self.version} cannot supersede itself",
                )
            )
        if self.deprecated_on is not None and self.superseded_by is None:
            problems.append(
                LifecycleValidation(
                    LifecycleValidationKind.MISSING_SUCCESSOR,
                    "deprecated version declares no superseded_by successor",
                )
            )
        return tuple(problems)

    def to_dict(self) -> dict[str, object]:
        return {
            "pack_name": self.pack_name,
            "version": self.version,
            "released_on": self.released_on.isoformat(),
            "lts": self.lts,
            "deprecated_on": self.deprecated_on.isoformat() if self.deprecated_on else None,
            "end_of_life_on": self.end_of_life_on.isoformat() if self.end_of_life_on else None,
            "superseded_by": self.superseded_by,
        }


@dataclass(frozen=True, slots=True)
class LifecycleEvaluation:
    version: str
    pack_name: str
    pack_version: str
    status: LifecycleStatus
    supported: bool
    days_until_end_of_life: int | None
    recommended_successor: str | None
    validation: tuple[LifecycleValidation, ...] = field(default=())

    @property
    def coherent(self) -> bool:
        return not self.validation

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "pack_name": self.pack_name,
            "pack_version": self.pack_version,
            "status": self.status.value,
            "supported": self.supported,
            "days_until_end_of_life": self.days_until_end_of_life,
            "recommended_successor": self.recommended_successor,
            "coherent": self.coherent,
            "validation": [v.to_dict() for v in self.validation],
        }


def evaluate_lifecycle(lifecycle: PackLifecycle, *, as_of: date) -> LifecycleEvaluation:
    """Resolve a pack version's lifecycle status against the date ``as_of``."""

    validation = lifecycle.validate()

    if as_of < lifecycle.released_on:
        status = LifecycleStatus.PRERELEASE
        supported = False
    elif lifecycle.end_of_life_on is not None and as_of >= lifecycle.end_of_life_on:
        status = LifecycleStatus.END_OF_LIFE
        supported = False
    elif lifecycle.deprecated_on is not None and as_of >= lifecycle.deprecated_on:
        status = LifecycleStatus.DEPRECATED
        supported = True
    elif lifecycle.lts:
        status = LifecycleStatus.LTS
        supported = True
    else:
        status = LifecycleStatus.SUPPORTED
        supported = True

    days_remaining: int | None = None
    if lifecycle.end_of_life_on is not None:
        days_remaining = (lifecycle.end_of_life_on - as_of).days

    successor = lifecycle.superseded_by if status in (
        LifecycleStatus.DEPRECATED,
        LifecycleStatus.END_OF_LIFE,
    ) else None

    return LifecycleEvaluation(
        version=PROMPT_PACK_LIFECYCLE_VERSION,
        pack_name=lifecycle.pack_name,
        pack_version=lifecycle.version,
        status=status,
        supported=supported,
        days_until_end_of_life=days_remaining,
        recommended_successor=successor,
        validation=validation,
    )


def render_lifecycle_json(evaluation: LifecycleEvaluation) -> str:
    return json.dumps(evaluation.to_dict(), indent=2, sort_keys=True) + "\n"


def render_lifecycle_text(evaluation: LifecycleEvaluation) -> str:
    lines = [
        f"PromptABI prompt-pack lifecycle ({evaluation.version})",
        f"{evaluation.pack_name}@{evaluation.pack_version}",
        f"status: {evaluation.status.value}",
        f"supported: {'YES' if evaluation.supported else 'NO'}",
    ]
    if evaluation.days_until_end_of_life is not None:
        lines.append(f"days until end-of-life: {evaluation.days_until_end_of_life}")
    if evaluation.recommended_successor is not None:
        lines.append(f"upgrade to: {evaluation.recommended_successor}")
    for problem in evaluation.validation:
        lines.append(f"  ! incoherent: {problem.kind.value}: {problem.detail}")
    return "\n".join(lines) + "\n"
