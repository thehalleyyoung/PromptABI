"""Conference-demo scenarios that replay real PromptABI deployment gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .diagnostics import Diagnostic, DiagnosticSeverity
from .session import VerificationResult, VerificationSession


class ConferenceDemoError(ValueError):
    """Raised when a conference demo scenario cannot be proven against real configs."""


@dataclass(frozen=True, slots=True)
class ConferenceDemoSpec:
    """One narrative demo backed by a buggy/fixed PromptABI config pair."""

    id: str
    title: str
    moment: str
    risk: str
    buggy_config: Path
    fixed_config: Path
    expected_error_rules: frozenset[str]


@dataclass(frozen=True, slots=True)
class ConferenceDemoCase:
    """Observed verifier outcome for one conference demo spec."""

    id: str
    title: str
    moment: str
    risk: str
    buggy_config: str
    fixed_config: str
    expected_error_rules: tuple[str, ...]
    observed_error_rules: tuple[str, ...]
    fixed_error_rules: tuple[str, ...]
    caught: bool
    fixed_clean: bool
    headline: str
    witness_steps: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.caught and self.fixed_clean

    def to_dict(self) -> dict[str, object]:
        return {
            "buggy_config": self.buggy_config,
            "caught": self.caught,
            "expected_error_rules": list(self.expected_error_rules),
            "fixed_clean": self.fixed_clean,
            "fixed_config": self.fixed_config,
            "fixed_error_rules": list(self.fixed_error_rules),
            "headline": self.headline,
            "id": self.id,
            "moment": self.moment,
            "observed_error_rules": list(self.observed_error_rules),
            "ok": self.ok,
            "risk": self.risk,
            "title": self.title,
            "witness_steps": list(self.witness_steps),
        }


@dataclass(frozen=True, slots=True)
class ConferenceDemoReport:
    """A deterministic report for the four paper/conference demo moments."""

    cases: tuple[ConferenceDemoCase, ...]
    root: str

    @property
    def ok(self) -> bool:
        return all(case.ok for case in self.cases)

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "ok": self.ok,
            "root": self.root,
            "summary": {
                "caught": sum(1 for case in self.cases if case.caught),
                "fixed_clean": sum(1 for case in self.cases if case.fixed_clean),
                "scenarios": len(self.cases),
            },
        }


def run_conference_demos(root: str | Path | None = None) -> ConferenceDemoReport:
    """Replay the curated demo configs and require the intended bugs to be caught."""

    repo_root = Path(root).resolve() if root is not None else _default_repo_root()
    specs = _demo_specs(repo_root)
    cases = tuple(_run_demo_case(spec, repo_root=repo_root) for spec in specs)
    return ConferenceDemoReport(cases=cases, root=str(repo_root))


def render_conference_demo_json(report: ConferenceDemoReport) -> str:
    """Render conference demos as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_conference_demo_text(report: ConferenceDemoReport) -> str:
    """Render conference demos as a stage-ready narrative."""

    lines = [
        "PromptABI conference demos",
        f"root: {report.root}",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        "",
    ]
    for case in report.cases:
        lines.append(f"{case.id}: {case.title}")
        lines.append(f"  moment: {case.moment}")
        lines.append(f"  risk: {case.risk}")
        lines.append(f"  buggy -> {'caught' if case.caught else 'missed'}: {', '.join(case.observed_error_rules) or '(none)'}")
        lines.append(f"  fixed -> {'clean' if case.fixed_clean else 'still failing'}: {case.fixed_config}")
        lines.append(f"  headline: {case.headline}")
        if case.witness_steps:
            lines.append("  witness:")
            for index, step in enumerate(case.witness_steps, start=1):
                lines.append(f"    {index}. {step}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _run_demo_case(spec: ConferenceDemoSpec, *, repo_root: Path) -> ConferenceDemoCase:
    buggy_result = _run_config(spec.buggy_config, scenario_id=spec.id, label="buggy")
    fixed_result = _run_config(spec.fixed_config, scenario_id=spec.id, label="fixed")
    buggy_errors = _error_diagnostics(buggy_result)
    fixed_errors = _error_diagnostics(fixed_result)
    observed_rules = tuple(sorted({diagnostic.rule_id for diagnostic in buggy_errors}))
    expected_rules = tuple(sorted(spec.expected_error_rules))
    caught = spec.expected_error_rules.issubset(observed_rules)
    fixed_error_rules = tuple(sorted({diagnostic.rule_id for diagnostic in fixed_errors}))
    fixed_clean = not fixed_errors
    headline_diagnostic = _headline_diagnostic(buggy_errors, spec.expected_error_rules)
    return ConferenceDemoCase(
        id=spec.id,
        title=spec.title,
        moment=spec.moment,
        risk=spec.risk,
        buggy_config=_relative_to(spec.buggy_config, repo_root),
        fixed_config=_relative_to(spec.fixed_config, repo_root),
        expected_error_rules=expected_rules,
        observed_error_rules=observed_rules,
        fixed_error_rules=fixed_error_rules,
        caught=caught,
        fixed_clean=fixed_clean,
        headline=_headline(headline_diagnostic, expected_rules=expected_rules, observed_rules=observed_rules),
        witness_steps=_witness_steps(headline_diagnostic),
    )


def _run_config(config_path: Path, *, scenario_id: str, label: str) -> VerificationResult:
    if not config_path.is_file():
        raise ConferenceDemoError(f"demo '{scenario_id}' {label} config is missing: {config_path}")
    try:
        return VerificationSession.from_config_file(config_path).run()
    except ValueError as exc:
        raise ConferenceDemoError(f"demo '{scenario_id}' {label} config failed to run: {exc}") from exc


def _error_diagnostics(result: VerificationResult) -> tuple[Diagnostic, ...]:
    return tuple(diagnostic for diagnostic in result.diagnostics if diagnostic.severity is DiagnosticSeverity.ERROR)


def _headline_diagnostic(errors: tuple[Diagnostic, ...], expected_rules: frozenset[str]) -> Diagnostic | None:
    for diagnostic in errors:
        if diagnostic.rule_id in expected_rules:
            return diagnostic
    return errors[0] if errors else None


def _headline(
    diagnostic: Diagnostic | None,
    *,
    expected_rules: tuple[str, ...],
    observed_rules: tuple[str, ...],
) -> str:
    if diagnostic is not None:
        return f"{diagnostic.rule_id}: {diagnostic.message}"
    expected = ", ".join(expected_rules)
    observed = ", ".join(observed_rules) or "none"
    return f"expected {expected}; observed {observed}"


def _witness_steps(diagnostic: Diagnostic | None) -> tuple[str, ...]:
    if diagnostic is None or diagnostic.witness is None:
        return ()
    rendered = []
    for step in diagnostic.witness.steps[:5]:
        line = step.action
        if step.input is not None:
            line += f" | input: {step.input}"
        if step.output is not None:
            line += f" | output: {step.output}"
        rendered.append(line)
    return tuple(rendered)


def _demo_specs(repo_root: Path) -> tuple[ConferenceDemoSpec, ...]:
    examples = repo_root / "examples"
    return (
        ConferenceDemoSpec(
            id="before-deployment",
            title="Catch a tool-call serialization bug before deploy",
            moment="CI gate before an agent service ships",
            risk="provider and application disagree on streamed tool-call IDs and arguments",
            buggy_config=examples / "end-to-end" / "tool-calling" / "buggy.promptabi.json",
            fixed_config=examples / "end-to-end" / "tool-calling" / "fixed.promptabi.json",
            expected_error_rules=frozenset({"tool-serialization"}),
        ),
        ConferenceDemoSpec(
            id="before-fine-tuning",
            title="Catch training/interface drift before fine-tuning",
            moment="dataset manifest gate before a fine-tune job starts",
            risk="assistant targets, packing boundaries, and training contracts no longer match serving-time roles",
            buggy_config=examples / "end-to-end" / "training-quickstart" / "buggy.promptabi.json",
            fixed_config=examples / "end-to-end" / "training-quickstart" / "fixed.promptabi.json",
            expected_error_rules=frozenset({"static-contract-violation", "training-packing-boundary"}),
        ),
        ConferenceDemoSpec(
            id="before-eval-publication",
            title="Catch benchmark contract leakage before eval publication",
            moment="paper/eval harness gate before leaderboard numbers are published",
            risk="answer keys, grading rubrics, parser mismatches, tool gaps, and truncation contaminate results",
            buggy_config=examples / "evaluation-harness" / "unsafe.promptabi.json",
            fixed_config=examples / "evaluation-harness" / "safe.promptabi.json",
            expected_error_rules=frozenset(
                {
                    "evaluation-harness-answer-key-leakage",
                    "evaluation-harness-grading-rubric-leakage",
                    "evaluation-harness-stop-policy-mismatch",
                }
            ),
        ),
        ConferenceDemoSpec(
            id="before-provider-migration",
            title="Catch provider contract drift before migration",
            moment="release gate before moving from one OpenAI-compatible target to another provider",
            risk="tool schema, stop behavior, streaming chunks, and model-family assumptions change under the app",
            buggy_config=examples / "end-to-end" / "provider-migration" / "buggy.promptabi.json",
            fixed_config=examples / "end-to-end" / "provider-migration" / "fixed.promptabi.json",
            expected_error_rules=frozenset({"provider-migration"}),
        ),
    )


def _relative_to(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
