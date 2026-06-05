"""Annual provider-semantics reports (step 298).

To make prompt-interface behavior accountable over time, this module compiles an
*annual report* from a year's worth of provider revisions and conformance runs:
how many revisions shipped, which obligations regressed and recovered, the net
change in pass-rate, and the longest-standing unresolved failure.  It turns the
raw drift series into a stable, citable summary artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ANNUAL_REPORT_VERSION = "promptabi.annual-report.v1"


@dataclass(frozen=True, slots=True)
class RevisionRecord:
    revision: str
    order: int  # chronological index within the year
    pass_rate: float
    failing_obligations: frozenset[str] = field(default=frozenset())


@dataclass(frozen=True, slots=True)
class AnnualReport:
    version: str
    year: int
    provider: str
    revisions_shipped: int
    net_pass_rate_change: float
    regressed_obligations: tuple[str, ...]
    recovered_obligations: tuple[str, ...]
    longest_standing_failure: str | None
    longest_standing_span: int

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "year": self.year,
            "provider": self.provider,
            "revisions_shipped": self.revisions_shipped,
            "net_pass_rate_change": self.net_pass_rate_change,
            "regressed_obligations": list(self.regressed_obligations),
            "recovered_obligations": list(self.recovered_obligations),
            "longest_standing_failure": self.longest_standing_failure,
            "longest_standing_span": self.longest_standing_span,
        }


def compile_annual_report(
    *, year: int, provider: str, records: tuple[RevisionRecord, ...]
) -> AnnualReport:
    ordered = sorted(records, key=lambda r: r.order)
    if not ordered:
        return AnnualReport(
            ANNUAL_REPORT_VERSION, year, provider, 0, 0.0, (), (), None, 0
        )

    first, last = ordered[0], ordered[-1]
    net_change = last.pass_rate - first.pass_rate

    regressed: set[str] = set()
    recovered: set[str] = set()
    prev: frozenset[str] = ordered[0].failing_obligations
    for rec in ordered[1:]:
        regressed |= rec.failing_obligations - prev
        recovered |= prev - rec.failing_obligations
        prev = rec.failing_obligations

    # Longest contiguous failing span per obligation (in revision count).
    span_by_ob: dict[str, int] = {}
    current_run: dict[str, int] = {}
    best_run: dict[str, int] = {}
    for rec in ordered:
        for ob in rec.failing_obligations:
            current_run[ob] = current_run.get(ob, 0) + 1
            best_run[ob] = max(best_run.get(ob, 0), current_run[ob])
        for ob in list(current_run):
            if ob not in rec.failing_obligations:
                current_run[ob] = 0
    span_by_ob = best_run

    longest_failure: str | None = None
    longest_span = 0
    for ob, span in sorted(span_by_ob.items()):
        if span > longest_span:
            longest_span = span
            longest_failure = ob

    return AnnualReport(
        version=ANNUAL_REPORT_VERSION,
        year=year,
        provider=provider,
        revisions_shipped=len(ordered),
        net_pass_rate_change=net_change,
        regressed_obligations=tuple(sorted(regressed)),
        recovered_obligations=tuple(sorted(recovered)),
        longest_standing_failure=longest_failure,
        longest_standing_span=longest_span,
    )


def render_annual_report_text(report: AnnualReport) -> str:
    lines = [
        f"PromptABI annual provider-semantics report ({report.version})",
        f"{report.provider} -- {report.year}",
        f"revisions shipped: {report.revisions_shipped}",
        f"net pass-rate change: {report.net_pass_rate_change:+.3f}",
        f"regressed: {list(report.regressed_obligations)}",
        f"recovered: {list(report.recovered_obligations)}",
        f"longest-standing failure: {report.longest_standing_failure!r} "
        f"({report.longest_standing_span} revisions)",
    ]
    return "\n".join(lines) + "\n"
