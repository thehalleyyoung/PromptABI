"""Model provider-native grammar backends (step 282).

Constrained decoding is implemented by different *grammar backends* -- xgrammar,
llguidance, Outlines, a provider-native JSON-mode -- each supporting a different
fragment of grammar features (regex lookahead, recursion, unicode classes,
context-free vs regular).  Handing a grammar to a backend that does not support a
feature it uses yields silent fallback or an error.  This module models each
backend's supported feature set and proves whether a grammar's required features
are covered, naming the unsupported feature and suggesting a backend that does
cover it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

GRAMMAR_BACKEND_VERSION = "promptabi.grammar-backend.v1"


class GrammarFeature(StrEnum):
    REGEX = "regex"
    LOOKAHEAD = "lookahead"
    RECURSION = "recursion"
    UNICODE_CLASS = "unicode-class"
    CONTEXT_FREE = "context-free"
    JSON_SCHEMA = "json-schema"


@dataclass(frozen=True, slots=True)
class GrammarBackend:
    name: str
    supported: frozenset[GrammarFeature]


@dataclass(frozen=True, slots=True)
class GrammarSpec:
    name: str
    required: frozenset[GrammarFeature]


class BackendFindingKind(StrEnum):
    UNSUPPORTED_FEATURE = "unsupported-feature"


@dataclass(frozen=True, slots=True)
class BackendFinding:
    kind: BackendFindingKind
    feature: GrammarFeature
    alternative: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "feature": self.feature.value,
            "alternative": self.alternative,
        }


@dataclass(frozen=True, slots=True)
class BackendCompatResult:
    version: str
    backend: str
    grammar: str
    compatible: bool
    findings: tuple[BackendFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "backend": self.backend,
            "grammar": self.grammar,
            "compatible": self.compatible,
            "findings": [f.to_dict() for f in self.findings],
        }


def check_backend(
    grammar: GrammarSpec,
    backend: GrammarBackend,
    alternatives: tuple[GrammarBackend, ...] = (),
) -> BackendCompatResult:
    findings: list[BackendFinding] = []
    for feature in sorted(grammar.required - backend.supported, key=lambda f: f.value):
        alt = next(
            (b.name for b in alternatives if feature in b.supported), None
        )
        findings.append(
            BackendFinding(BackendFindingKind.UNSUPPORTED_FEATURE, feature, alt)
        )
    return BackendCompatResult(
        version=GRAMMAR_BACKEND_VERSION,
        backend=backend.name,
        grammar=grammar.name,
        compatible=not findings,
        findings=tuple(findings),
    )


def render_backend_text(result: BackendCompatResult) -> str:
    lines = [
        f"PromptABI grammar backend compatibility ({result.version})",
        f"{result.grammar} on {result.backend}: "
        f"{'COMPATIBLE' if result.compatible else 'INCOMPATIBLE'}",
    ]
    for f in result.findings:
        alt = f" (try: {f.alternative})" if f.alternative else ""
        lines.append(f"  ! unsupported feature {f.feature.value}{alt}")
    return "\n".join(lines) + "\n"
