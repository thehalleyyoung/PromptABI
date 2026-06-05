import json
from pathlib import Path

from promptabi.cli import main
from promptabi.deployment_gates import (
    DEPLOYMENT_GATE_SURFACES,
    DeploymentGateError,
    build_deployment_gate_report,
    render_deployment_gate_json,
    render_deployment_gate_text,
    write_deployment_gate_examples,
)


def test_deployment_gate_report_requires_current_signed_bundle_evidence() -> None:
    report = build_deployment_gate_report(
        "examples/minimal/promptabi.json",
        bundle_key="deployment-secret",
        bundle_key_id="deploy-test",
    )
    payload = json.loads(render_deployment_gate_json(report))

    assert report.ok is True
    assert report.bundle_hash
    assert report.reproducibility_hash
    assert payload["ok"] is True
    assert payload["bundle"]["signing_key_id"] == "deploy-test"
    assert payload["safe_deployment_cores"][0]["core"] == ["required-tokens-exceed-input-budget"]
    assert payload["safe_deployment_cores"][0]["minimal"] is True
    assert tuple(example.surface for example in report.examples) == DEPLOYMENT_GATE_SURFACES
    assert all(report.bundle_hash in example.content for example in report.examples)
    assert all(report.reproducibility_hash in example.content for example in report.examples)
    assert any("ValidatingAdmissionPolicy" in example.content for example in report.examples)
    assert any("terraform_data" in example.content for example in report.examples)
    assert any("environment:" in example.content for example in report.examples)
    assert any("release_system" in example.content for example in report.examples)


def test_deployment_gate_report_blocks_missing_bundle_key() -> None:
    try:
        build_deployment_gate_report("examples/minimal/promptabi.json")
    except DeploymentGateError as exc:
        assert "bundle signing key" in str(exc)
    else:
        raise AssertionError("expected deployment gates to require signed bundle evidence")


def test_deployment_gate_text_and_writer_create_real_examples(tmp_path: Path) -> None:
    report = build_deployment_gate_report("examples/minimal/promptabi.json", bundle_key="deployment-secret")
    text = render_deployment_gate_text(report)
    written = write_deployment_gate_examples(
        tmp_path / "deployment-gates",
        "examples/minimal/promptabi.json",
        bundle_key="deployment-secret",
    )

    assert "PromptABI deployment gates" in text
    assert "status: PASS" in text
    assert "unsat-core budget-unsat-survival" in text
    assert sorted(path.name for path in written.written_files) == [
        "deployment-gates.json",
        "github-environments.yml",
        "internal-release-system.json",
        "kubernetes.yaml",
        "terraform.hcl",
    ]
    manifest = json.loads((written.output_dir / "deployment-gates.json").read_text(encoding="utf-8"))
    assert manifest["manifest_sha256"] == written.report.manifest_sha256
    assert "bundle_hash" in (written.output_dir / "kubernetes.yaml").read_text(encoding="utf-8")


def test_deployment_gate_cli_writes_examples(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "gates"

    exit_code = main(
        [
            "deployment-gates",
            "--config",
            "examples/minimal/promptabi.json",
            "--bundle-key",
            "deployment-secret",
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "PromptABI deployment gates" in captured.out
    assert (output_dir / "deployment-gates.json").is_file()

    exit_code = main(
        [
            "deployment-gates",
            "--config",
            "examples/minimal/promptabi.json",
            "--bundle-key",
            "deployment-secret",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["ok"] is True
