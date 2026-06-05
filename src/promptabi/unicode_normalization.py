"""Model finite unicode normalization constraints (step 237).

Prompt-interface contracts frequently assume their inputs are already in a
canonical Unicode form -- e.g. that role tags, stop tokens, or structured-output
keys are *NFC normalized* and free of confusable look-alikes.  Those assumptions
are silent, and they break security-relevant invariants (homoglyph spoofing of
``"system"``, fullwidth digits slipping past a numeric guard, ligatures changing
a key's length).

This module turns Unicode normalization into a **finite, decidable** model.  Over
a bounded alphabet of codepoints we precompute, with the standard library's
:func:`unicodedata.normalize`, each character's normal form, whether it is
already normalized, and which characters collapse together under compatibility
normalization (confusables).  We then cross-validate the precomputed truth
against the finite solver: an :class:`~promptabi.formal.IntRangeDomain` over
codepoint indices is constrained to the "already normalized" set and the
enumeration must reproduce exactly that set.  We additionally certify the
*idempotence* of normalization over the whole alphabet -- ``norm(norm(c)) ==
norm(c)`` -- which is the soundness premise every downstream proof relies on.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from .formal import (
    FiniteContractProblem,
    InSet,
    IntRangeDomain,
    NamedConstraint,
    SolverStatus,
    Value,
    Var,
)

UNICODE_NORMALIZATION_VERSION = "promptabi.unicode-normalization.v1"

_FORMS = ("NFC", "NFD", "NFKC", "NFKD")


class NormalizationFindingKind(StrEnum):
    NOT_NORMALIZED = "not-normalized"
    CONFUSABLE = "confusable"
    IDEMPOTENCE_VIOLATION = "idempotence-violation"
    SOLVER_DISAGREEMENT = "solver-disagreement"


@dataclass(frozen=True, slots=True)
class NormalizationFinding:
    kind: NormalizationFindingKind
    message: str
    codepoints: tuple[int, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "codepoints": [f"U+{cp:04X}" for cp in self.codepoints],
        }


@dataclass(frozen=True, slots=True)
class CharacterModel:
    text: str
    codepoints: tuple[int, ...]
    normal_form: str
    is_normalized: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "codepoints": [f"U+{cp:04X}" for cp in self.codepoints],
            "normal_form": self.normal_form,
            "is_normalized": self.is_normalized,
        }


@dataclass(frozen=True, slots=True)
class NormalizationReport:
    version: str
    form: str
    characters: tuple[CharacterModel, ...]
    normalized_texts: tuple[str, ...]
    confusable_classes: tuple[tuple[str, ...], ...]
    findings: tuple[NormalizationFinding, ...]
    idempotent: bool
    solver_agrees: bool

    @property
    def sound(self) -> bool:
        return self.idempotent and self.solver_agrees

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "form": self.form,
            "sound": self.sound,
            "idempotent": self.idempotent,
            "solver_agrees": self.solver_agrees,
            "characters": [c.to_dict() for c in self.characters],
            "normalized_texts": list(self.normalized_texts),
            "confusable_classes": [list(cls) for cls in self.confusable_classes],
            "findings": [f.to_dict() for f in self.findings],
        }


def _normalize(form: str, char: str) -> str:
    return unicodedata.normalize(form, char)


def analyze_normalization(
    alphabet: Sequence[str], *, form: str = "NFC", prefer_z3: bool = True
) -> NormalizationReport:
    """Model normalization of a finite alphabet and certify its soundness."""

    if form not in _FORMS:
        raise ValueError(f"unsupported normalization form {form!r}; expected one of {_FORMS}")
    if not alphabet:
        raise ValueError("alphabet must contain at least one character")

    characters: list[CharacterModel] = []
    findings: list[NormalizationFinding] = []
    idempotent = True

    for text in alphabet:
        if not text:
            raise ValueError("alphabet entries must be non-empty strings")
        cps = tuple(ord(ch) for ch in text)
        nf = _normalize(form, text)
        normalized = text == nf
        # idempotence: re-normalizing a normal form must be a fixpoint.
        if _normalize(form, nf) != nf:
            idempotent = False
            findings.append(
                NormalizationFinding(
                    kind=NormalizationFindingKind.IDEMPOTENCE_VIOLATION,
                    message=f"normalization of {text!r} is not idempotent",
                    codepoints=cps,
                )
            )
        if not normalized:
            findings.append(
                NormalizationFinding(
                    kind=NormalizationFindingKind.NOT_NORMALIZED,
                    message=f"{text!r} is not in {form} normal form -> {nf!r}",
                    codepoints=cps,
                )
            )
        characters.append(CharacterModel(text, cps, nf, normalized))

    # Confusables: distinct inputs that collapse to the same normal form.
    by_form: dict[str, list[str]] = {}
    for model in characters:
        by_form.setdefault(model.normal_form, []).append(model.text)
    confusable_classes = tuple(
        tuple(sorted(texts)) for texts in by_form.values() if len(set(texts)) > 1
    )
    for cls in confusable_classes:
        findings.append(
            NormalizationFinding(
                kind=NormalizationFindingKind.CONFUSABLE,
                message="inputs collapse to the same normal form: "
                + ", ".join(repr(t) for t in cls),
            )
        )

    normalized_indices = tuple(i for i, m in enumerate(characters) if m.is_normalized)
    normalized_texts = tuple(characters[i].text for i in normalized_indices)

    # Cross-validate against the finite solver: enumerate the "already
    # normalized" index set and confirm it matches the precomputed truth.
    solver_agrees = _solver_matches(len(characters), normalized_indices, prefer_z3=prefer_z3)
    if not solver_agrees:
        findings.append(
            NormalizationFinding(
                kind=NormalizationFindingKind.SOLVER_DISAGREEMENT,
                message="solver enumeration disagreed with precomputed normalized set",
            )
        )

    return NormalizationReport(
        version=UNICODE_NORMALIZATION_VERSION,
        form=form,
        characters=tuple(characters),
        normalized_texts=normalized_texts,
        confusable_classes=confusable_classes,
        findings=tuple(findings),
        idempotent=idempotent,
        solver_agrees=solver_agrees,
    )


def _solver_matches(
    size: int, normalized_indices: Sequence[int], *, prefer_z3: bool
) -> bool:
    """Enumerate ``InSet(i, normalized)`` and confirm it equals the expected set."""

    if size == 0:
        return True
    domain = IntRangeDomain(name="i", minimum=0, maximum=size - 1)
    expected = set(normalized_indices)
    enumerated: set[int] = set()
    # Build, for each index, a membership problem and check SAT iff expected.
    for index in range(size):
        problem = FiniteContractProblem(
            variables=(domain,),
            constraints=(
                NamedConstraint(
                    name="pin", expression=InSet(Var("i"), (index,))
                ),
                NamedConstraint(
                    name="normalized",
                    expression=InSet(Var("i"), tuple(sorted(expected))),
                ),
            ),
            name=f"normalized-{index}",
        )
        result = problem.solve(prefer_z3=prefer_z3, max_assignments=10000)
        if result.status is SolverStatus.SAT:
            enumerated.add(index)
        elif result.status is SolverStatus.UNKNOWN:
            return False
    return enumerated == expected


def render_normalization_json(report: NormalizationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_normalization_text(report: NormalizationReport) -> str:
    lines = [
        f"PromptABI unicode normalization model ({report.version})",
        f"form: {report.form}",
        f"sound: {report.sound} (idempotent={report.idempotent}, solver_agrees={report.solver_agrees})",
        f"alphabet: {len(report.characters)} entries, "
        f"{len(report.normalized_texts)} already normalized",
    ]
    for cls in report.confusable_classes:
        lines.append("  confusable: " + ", ".join(repr(t) for t in cls))
    for finding in report.findings:
        if finding.kind is not NormalizationFindingKind.CONFUSABLE:
            lines.append(f"  ! {finding.kind.value}: {finding.message}")
    return "\n".join(lines) + "\n"
