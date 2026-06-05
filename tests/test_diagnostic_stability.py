import json
from pathlib import Path

from promptabi import (
    DiagnosticStabilityFindingKind,
    prove_diagnostic_stability_under_formatting,
)
from promptabi.cli import main


EXAMPLE_CONFIG = Path("examples/diagnostic-stability/promptabi.json")


def test_diagnostics_are_stable_under_formatting_only_changes() -> None:
    report = prove_diagnostic_stability_under_formatting(EXAMPLE_CONFIG)

    assert report.ok
    assert report.baseline_count > 0
    assert len(report.variants) == 4
    # Every formatting variant preserved diagnostics while moving byte offsets,
    # so the perturbation is real, not vacuous.
    for variant in report.variants:
        assert variant.stable
        assert variant.variant_count == report.baseline_count
        assert variant.spans_shifted > 0


def test_format_sensitive_checker_is_detected(monkeypatch) -> None:
    # Inject a synthetic checker whose finding span encodes the byte length of the
    # source file; reformatting changes file size, fabricating/dropping a finding.
    from promptabi.diagnostic_stability import (
        DiagnosticStabilityFindingKind as Kind,
        prove_diagnostic_stability_under_formatting as prove,
    )
    from promptabi.diagnostics import Diagnostic, DiagnosticSeverity
    from promptabi.session import VerificationSession, VerificationResult

    original_run = VerificationSession.run

    def flaky_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        # Emit an extra diagnostic whose message depends on raw byte length of the
        # first file-backed artifact, making it formatting-sensitive.
        total_bytes = 0
        for artifact in self.config.artifact_bundle:
            path = getattr(artifact.location, "path", None)
            if path:
                try:
                    total_bytes += len(Path(path).read_text(encoding="utf-8"))
                except OSError:
                    pass
        extra = Diagnostic(
            rule_id="formatting-sensitive-demo",
            severity=DiagnosticSeverity.WARNING,
            message=f"observed {total_bytes} raw bytes",
        )
        return VerificationResult(config=result.config, diagnostics=result.diagnostics + (extra,))

    monkeypatch.setattr(VerificationSession, "run", flaky_run)

    report = prove(EXAMPLE_CONFIG)

    assert not report.ok
    kinds = {finding.kind for finding in report.findings}
    assert Kind.DROPPED_DIAGNOSTIC in kinds or Kind.FABRICATED_DIAGNOSTIC in kinds
    assert DiagnosticStabilityFindingKind is Kind


def test_diagnostic_stability_cli(capsys) -> None:
    exit_code = main(
        ["diagnostic-stability", "--config", str(EXAMPLE_CONFIG), "--format", "json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["version"] == "promptabi.diagnostic-stability.v1"
    assert len(payload["variants"]) == 4
