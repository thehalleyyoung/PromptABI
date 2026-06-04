import json
from pathlib import Path

from promptabi.cli import main
from promptabi.init import available_stacks


def test_verify_text_output_passes_for_example_config(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI verification: minimal-chat-template" in captured.out
    assert "status: PASS" in captured.out
    assert captured.err == ""


def test_verify_discovers_config_from_nested_directory(tmp_path, monkeypatch, capsys) -> None:
    config_dir = tmp_path / "project"
    nested = config_dir / "app" / "prompts"
    nested.mkdir(parents=True)
    config = config_dir / "promptabi.json"
    schema = config_dir / "schema.json"
    cache_dir = tmp_path / "cache"
    schema.write_text("{}", encoding="utf-8")
    config.write_text(
        '{"name": "discovered", "artifacts": {"schema": "schema.json"}}',
        encoding="utf-8",
    )
    monkeypatch.chdir(nested)

    exit_code = main(["verify", "--cache-dir", str(cache_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI verification: discovered" in captured.out
    assert cache_dir.is_dir()
    assert captured.err == ""


def test_verify_json_output_is_stable(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["diagnostics"][0]["rule_id"] == "repository-skeleton"
    assert payload["diagnostics"][0]["check_modes"] == ["heuristic"]
    assert "fingerprint" in payload["diagnostics"][0]
    assert payload["diagnostics"][0]["witness"]["steps"][0]["action"] == "load JSON config"
    assert list(payload) == ["config", "diagnostics", "ok"]


def test_verify_artifact_override_replaces_configured_location(tmp_path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    existing = tmp_path / "schema.json"
    missing = tmp_path / "missing.schema.json"
    existing.write_text("{}", encoding="utf-8")
    config.write_text(
        f'{{"name": "override", "artifacts": {{"schema": "{missing.name}"}}}}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "verify",
            "--config",
            str(config),
            "--artifact",
            f"schema={existing}",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["config"]["artifacts"] == {"schema": str(existing)}


def test_verify_missing_artifact_fails_with_error(tmp_path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text(
        '{"name": "bad", "artifacts": {"schema": "missing.schema.json"}}',
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["diagnostics"][0]["rule_id"] == "artifact-missing"
    assert payload["diagnostics"][0]["witness"]["steps"][1]["output"] == "missing"


def test_verify_exit_code_policy_can_fail_on_any_diagnostic(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json", "--fail-on", "any"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "INFO repository-skeleton [heuristic]" in captured.out
    assert captured.err == ""


def test_verify_quiet_suppresses_info_diagnostics(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json", "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status: PASS" in captured.out
    assert "INFO repository-skeleton" not in captured.out


def test_verify_verbose_prints_workflow_metadata(tmp_path, capsys) -> None:
    cache_dir = tmp_path / "promptabi-cache"

    exit_code = main(
        [
            "verify",
            "--config",
            "examples/minimal/promptabi.json",
            "--cache-dir",
            str(cache_dir),
            "--verbose",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"cache: {cache_dir}" in captured.out
    assert "artifacts: 3" in captured.out


def test_verify_sarif_output_is_code_scanning_compatible(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json", "--format", "sarif"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    result = payload["runs"][0]["results"][0]
    assert exit_code == 0
    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["tool"]["driver"]["name"] == "PromptABI"
    assert result["ruleId"] == "repository-skeleton"
    assert result["level"] == "note"
    assert result["properties"]["checkModes"] == ["heuristic"]
    assert payload["runs"][0]["tool"]["driver"]["rules"][0]["properties"]["checkModes"] == ["heuristic"]
    assert "promptabiFingerprint" in result["partialFingerprints"]


def test_verify_sarif_can_emit_github_code_scanning_metadata(tmp_path, capsys) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "promptabi.json"
    config.write_text(
        '{"name": "github-sarif", "artifacts": {"schema": "missing.schema.json"}}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "verify",
            "--config",
            str(config),
            "--format",
            "sarif",
            "--sarif-category",
            "pull-request",
            "--sarif-checkout-uri-base",
            str(repo),
            "--sarif-include-invocation",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    run = payload["runs"][0]
    result = run["results"][0]
    assert exit_code == 1
    assert run["automationDetails"]["id"] == "pull-request/"
    assert "originalUriBaseIds" in run
    assert run["invocations"][0]["commandLine"].startswith("promptabi verify --config")
    location = result["locations"][0]["physicalLocation"]["artifactLocation"]
    assert location == {"uri": "promptabi.json", "uriBaseId": "PROJECTROOT"}
    assert result["level"] == "error"
    assert "promptabiLocationFingerprint" in result["partialFingerprints"]
    assert captured.err == ""


def test_verify_github_annotations_output_uses_workflow_commands(tmp_path, capsys) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "promptabi.json"
    config.write_text(
        '{"name": "github-annotations", "artifacts": {"schema": "missing.schema.json"}}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "verify",
            "--config",
            str(config),
            "--format",
            "github-annotations",
            "--sarif-checkout-uri-base",
            str(repo),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.startswith("::error title=artifact-missing,file=promptabi.json,line=1,col=")
    assert "artifact 'schema' does not exist" in captured.out
    assert captured.err == ""


def test_verify_role_boundary_nonforgeability_reports_real_fixture(tmp_path, capsys) -> None:
    fixture = Path("fixtures/seed_corpus/llama/tokenizer_config.json").resolve()
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "llama-role-boundary",
                "checks": ["role-boundary-nonforgeability"],
                "artifacts": {
                    "llama": {
                        "kind": "chat-template",
                        "path": str(fixture),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    role_diagnostics = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "role-boundary-nonforgeability"
    ]
    assert exit_code == 1
    assert role_diagnostics
    assert any("role-header 'assistant'" in diagnostic["message"] for diagnostic in role_diagnostics)
    assert any("<|start_header_id|>" in diagnostic["message"] for diagnostic in role_diagnostics)
    assert role_diagnostics[0]["check_modes"] == ["bounded", "sound"]
    witness_steps = role_diagnostics[0]["witness"]["steps"]
    assert any(step["action"] == "tokenize forged excerpt" for step in witness_steps)
    assert any(step["action"] == "locate forged boundary" for step in witness_steps)
    assert any("byte-level" in step.get("output", "") for step in witness_steps)


def test_verify_role_boundary_example_reports_structural_not_semantic_boundary(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/role-boundary/unsafe.promptabi.json", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    diagnostics = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "role-boundary-nonforgeability"
    ]

    assert exit_code == 1
    assert diagnostics
    assert any("role-header 'assistant'" in diagnostic["message"] for diagnostic in diagnostics)
    assert any("assistant-prefix '<|im_start|>'" in diagnostic["message"] for diagnostic in diagnostics)
    assert all("model will" not in diagnostic["message"] for diagnostic in diagnostics)
    witness_steps = diagnostics[0]["witness"]["steps"]
    assert any(step["action"] == "render forged boundary excerpt" for step in witness_steps)
    assert any(step["action"] == "tokenize forged excerpt" for step in witness_steps)
    assert any(step["action"] == "locate forged boundary" for step in witness_steps)


def test_verify_role_boundary_example_accepts_sanitized_template(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/role-boundary/safe.promptabi.json", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "role-boundary-nonforgeability"
    ] == []


def test_explain_expands_role_boundary_diagnostic_from_real_fixture(capsys) -> None:
    exit_code = main(
        [
            "explain",
            "--config",
            "examples/role-boundary/unsafe.promptabi.json",
            "--index",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI explanation: role-boundary-nonforgeability" in captured.out
    assert "Formal property" in captured.out
    assert "Attacker-controlled content must not render as structural role delimiters" in captured.out
    assert "Artifact snippet" in captured.out
    assert "<|im_start|>" in captured.out
    assert "Witness" in captured.out
    assert "Likely production symptom" in captured.out
    assert "role delimiters" in captured.out
    assert "Avoid raw dynamic role headers" in captured.out
    assert captured.err == ""


def test_explain_json_output_is_structured_for_single_diagnostic(capsys) -> None:
    exit_code = main(
        [
            "explain",
            "--config",
            "examples/minimal/promptabi.json",
            "--rule-id",
            "repository-skeleton",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["diagnostic"]["rule_id"] == "repository-skeleton"
    assert payload["property_checked"].startswith("The configured PromptABI project should load")
    assert payload["proof_modes"] == [
        "The check is useful evidence but is not a proof over a fully modeled fragment."
    ]
    assert payload["fix_suggestions"] == []
    assert "source_snippet" not in payload
    assert captured.err == ""


def test_explain_rule_id_requires_disambiguation_for_repeated_findings(capsys) -> None:
    exit_code = main(
        [
            "explain",
            "--config",
            "examples/role-boundary/unsafe.promptabi.json",
            "--rule-id",
            "role-boundary-nonforgeability",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "matched" in captured.err
    assert "rerun with --fingerprint or --index" in captured.err


def test_init_scaffolds_each_supported_stack_with_verifiable_config(tmp_path, capsys) -> None:
    for stack in available_stacks():
        output_dir = tmp_path / stack
        init_exit = main(["init", "--stack", stack, "--output-dir", str(output_dir), "--name", f"{stack}-demo"])
        init_output = capsys.readouterr()

        assert init_exit == 0
        assert f"wrote PromptABI {stack} scaffold" in init_output.out
        config_path = output_dir / "promptabi.json"
        assert config_path.is_file()

        verify_exit = main(["verify", "--config", str(config_path), "--format", "json", "--fail-on", "never"])
        verify_output = capsys.readouterr()
        payload = json.loads(verify_output.out)

        assert verify_exit == 0
        assert payload["config"]["name"] == f"{stack}-demo"
        assert payload["config"]["artifact_bundle"]["artifacts"]
        assert {
            diagnostic["rule_id"]
            for diagnostic in payload["diagnostics"]
            if diagnostic["rule_id"] in {"artifact-missing", "artifact-load-failed", "check-unknown", "check-failed"}
        } == set()


def test_init_refuses_to_overwrite_without_force(tmp_path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text("{}", encoding="utf-8")

    exit_code = main(["init", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "refusing to overwrite" in captured.err
