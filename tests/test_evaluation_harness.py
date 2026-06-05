import json
from pathlib import Path

from promptabi.artifacts import ArtifactKind, EvaluationHarnessArtifact, artifact_from_config
from promptabi.cli import main
from promptabi.evaluation_harness import analyze_evaluation_harness_contracts
from promptabi.session import CHECK_MODE_CATALOG, VerificationSession


ROOT = Path(__file__).resolve().parents[1]
SAFE_CONFIG = ROOT / "examples" / "evaluation-harness" / "safe.promptabi.json"
UNSAFE_CONFIG = ROOT / "examples" / "evaluation-harness" / "unsafe.promptabi.json"


def test_evaluation_harness_artifact_parses_manifest_fields() -> None:
    artifact = artifact_from_config(
        "eval",
        {
            "kind": "evaluation-harness",
            "path": "examples/evaluation-harness/harness.safe.json",
            "benchmark_name": "promptabi-contract-eval",
            "provider": "openai-compatible",
            "tokenizer": "byte-bpe",
            "prompt_template": "template",
            "answer_parser": "json-schema",
            "stop_sequences": ["</answer>", "</answer>"],
            "few_shot_examples": [{"id": "one", "role": "user", "content": "Q", "token_count": 1}],
        },
        base_dir=ROOT,
    )

    assert isinstance(artifact, EvaluationHarnessArtifact)
    assert artifact.kind is ArtifactKind.EVALUATION_HARNESS
    assert artifact.stop_sequences == ("</answer>",)
    assert artifact.few_shot_examples[0].token_count == 1


def test_safe_evaluation_harness_runs_through_verification_session() -> None:
    result = VerificationSession.from_config_file(SAFE_CONFIG).run()
    rule_ids = [diagnostic.rule_id for diagnostic in result.diagnostics]

    assert result.ok is True
    assert "evaluation-harness-verified" in rule_ids
    assert all(rule_id in CHECK_MODE_CATALOG for rule_id in rule_ids)
    assert not any(rule_id.endswith("mismatch") for rule_id in rule_ids)


def test_unsafe_evaluation_harness_reports_contract_breaks_with_spans() -> None:
    result = VerificationSession.from_config_file(UNSAFE_CONFIG).run()
    by_rule = {diagnostic.rule_id: diagnostic for diagnostic in result.diagnostics}

    expected = {
        "evaluation-harness-provider-mismatch",
        "evaluation-harness-model-mismatch",
        "evaluation-harness-tokenizer-mismatch",
        "evaluation-harness-prompt-template-mismatch",
        "evaluation-harness-stop-policy-mismatch",
        "evaluation-harness-answer-parser-mismatch",
        "evaluation-harness-prompt-variable-missing",
        "evaluation-harness-few-shot-role-mismatch",
        "evaluation-harness-few-shot-budget-overflow",
    }
    assert result.ok is False
    assert expected.issubset(by_rule)
    assert by_rule["evaluation-harness-stop-policy-mismatch"].span is not None
    assert by_rule["evaluation-harness-few-shot-role-mismatch"].witness is not None
    assert "evaluation-harness-verified" not in by_rule


def test_evaluation_harness_cli_json_output(capsys) -> None:
    exit_code = main(["verify", "--config", str(UNSAFE_CONFIG), "--format", "json", "--fail-on", "never"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert any(
        diagnostic["rule_id"] == "evaluation-harness-stop-policy-mismatch"
        for diagnostic in payload["diagnostics"]
    )


def test_evaluation_harness_analyzer_emits_missing_contract_abstention() -> None:
    session = VerificationSession.from_config_file(SAFE_CONFIG)
    loaded = session.load_artifacts()
    harness = next(artifact.artifact for artifact in loaded if isinstance(artifact.artifact, EvaluationHarnessArtifact))

    report = analyze_evaluation_harness_contracts(harness, (harness,))

    assert any(finding.rule_id == "evaluation-harness-contract-missing" for finding in report.findings)
