"""Shared prompt-interface standards proposal (step 299).

The capstone of the provider-conformance layer: assemble the obligations,
capability vocabulary, error envelope, and stop semantics validated by the
preceding steps into a single, versioned *standards proposal* document object --
a normative checklist with stable clause ids -- and score a provider's declared
support against it to produce an adoption report.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

STANDARDS_PROPOSAL_VERSION = "promptabi.standards-proposal.v1"


class ClauseLevel(StrEnum):
    MUST = "must"
    SHOULD = "should"
    MAY = "may"


@dataclass(frozen=True, slots=True)
class StandardsClause:
    clause_id: str
    level: ClauseLevel
    title: str


def default_clauses() -> tuple[StandardsClause, ...]:
    return (
        StandardsClause("PA-1", ClauseLevel.MUST, "Role boundaries are non-forgeable"),
        StandardsClause("PA-2", ClauseLevel.MUST, "Declared stop policy terminates output"),
        StandardsClause("PA-3", ClauseLevel.MUST, "Structured output validates against schema"),
        StandardsClause("PA-4", ClauseLevel.MUST, "Tool calls are well-formed and accounted for"),
        StandardsClause("PA-5", ClauseLevel.MUST, "Error envelope is conformant"),
        StandardsClause("PA-6", ClauseLevel.SHOULD, "Refusals use a dedicated channel"),
        StandardsClause("PA-7", ClauseLevel.SHOULD, "Context-window semantics are documented per revision"),
        StandardsClause("PA-8", ClauseLevel.SHOULD, "Capability negotiation is supported"),
        StandardsClause("PA-9", ClauseLevel.MAY, "Migration patches are published for breaking changes"),
        StandardsClause("PA-10", ClauseLevel.MAY, "Annual provider-semantics report is published"),
    )


@dataclass(frozen=True, slots=True)
class StandardsProposal:
    version: str
    title: str
    clauses: tuple[StandardsClause, ...]

    def clause_ids(self) -> frozenset[str]:
        return frozenset(c.clause_id for c in self.clauses)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "title": self.title,
            "clauses": [
                {"clause_id": c.clause_id, "level": c.level.value, "title": c.title}
                for c in self.clauses
            ],
        }


def default_proposal() -> StandardsProposal:
    return StandardsProposal(
        version=STANDARDS_PROPOSAL_VERSION,
        title="PromptABI Shared Prompt-Interface Standard",
        clauses=default_clauses(),
    )


@dataclass(frozen=True, slots=True)
class AdoptionReport:
    version: str
    provider: str
    must_total: int
    must_satisfied: int
    should_total: int
    should_satisfied: int
    unmet_must: tuple[str, ...]

    @property
    def compliant(self) -> bool:
        return self.must_satisfied == self.must_total

    @property
    def adoption_score(self) -> float:
        total = self.must_total + self.should_total
        got = self.must_satisfied + self.should_satisfied
        return got / total if total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "provider": self.provider,
            "compliant": self.compliant,
            "adoption_score": self.adoption_score,
            "must_satisfied": f"{self.must_satisfied}/{self.must_total}",
            "should_satisfied": f"{self.should_satisfied}/{self.should_total}",
            "unmet_must": list(self.unmet_must),
        }


def score_adoption(
    proposal: StandardsProposal,
    provider: str,
    satisfied_clause_ids: frozenset[str],
) -> AdoptionReport:
    must = [c for c in proposal.clauses if c.level == ClauseLevel.MUST]
    should = [c for c in proposal.clauses if c.level == ClauseLevel.SHOULD]

    must_ok = [c for c in must if c.clause_id in satisfied_clause_ids]
    should_ok = [c for c in should if c.clause_id in satisfied_clause_ids]
    unmet_must = tuple(
        c.clause_id for c in must if c.clause_id not in satisfied_clause_ids
    )

    return AdoptionReport(
        version=STANDARDS_PROPOSAL_VERSION,
        provider=provider,
        must_total=len(must),
        must_satisfied=len(must_ok),
        should_total=len(should),
        should_satisfied=len(should_ok),
        unmet_must=unmet_must,
    )


def render_adoption_report_text(report: AdoptionReport) -> str:
    lines = [
        f"PromptABI standards adoption report ({report.version})",
        f"provider: {report.provider}",
        f"compliant: {'YES' if report.compliant else 'NO'} "
        f"(score={report.adoption_score:.2f})",
        f"MUST: {report.must_satisfied}/{report.must_total}",
        f"SHOULD: {report.should_satisfied}/{report.should_total}",
    ]
    if report.unmet_must:
        lines.append(f"  unmet MUST clauses: {list(report.unmet_must)}")
    return "\n".join(lines) + "\n"
