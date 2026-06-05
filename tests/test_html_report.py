import json
from pathlib import Path

from promptabi import render_result, run_verification
from promptabi.cli import main


def test_verify_html_report_is_self_contained_and_escapes_content(tmp_path: Path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "html <contract>",
                "artifacts": {"schema<script>": "missing.schema.json"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "html"])

    html = capsys.readouterr().out
    assert exit_code == 1
    assert html.startswith("<!doctype html>")
    assert "PromptABI static report: html &lt;contract&gt;" in html
    assert "schema&lt;script&gt;" in html
    assert "<script" not in html
    assert "<link" not in html
    assert "http://" not in html
    assert "https://" not in html


def test_token_budget_html_report_renders_real_chart(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/token-budget/promptabi.json", "--format", "html"])

    html = capsys.readouterr().out
    assert exit_code == 1
    assert "Token-budget charts" in html
    assert "context=96, reserved=36, input=60" in html
    assert "system-policy" in html
    assert "retrieval-context" in html
    assert 'class="bar dropped"' in html


def test_static_contract_html_report_renders_smt_witnesses(tmp_path: Path, capsys) -> None:
    segments = tmp_path / "segments.json"
    budget = tmp_path / "budget.json"
    stops = tmp_path / "stops.json"
    specials = tmp_path / "specials.json"
    segments.write_text(
        json.dumps({"segments": [{"name": "system", "role": "system", "required": True, "token_count": 8}]}),
        encoding="utf-8",
    )
    for path in (budget, stops, specials):
        path.write_text("{}", encoding="utf-8")
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "static-contract-html",
                "checks": ["static-contracts"],
                "artifacts": {
                    "segments": {
                        "kind": "prompt-segment",
                        "path": segments.name,
                        "segments": [{"name": "system", "role": "system", "required": True, "token_count": 8}],
                    },
                    "budget": {
                        "kind": "framework-truncation-config",
                        "path": budget.name,
                        "framework": "vllm",
                        "strategy": "left",
                        "max_context_tokens": 32,
                    },
                    "stops": {
                        "kind": "stop-policy",
                        "path": stops.name,
                        "stop_sequences": ["</tool_call>"],
                    },
                    "specials": {
                        "kind": "special-token-map",
                        "path": specials.name,
                        "tokens": {"eos": "</s>"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "html"])

    html = capsys.readouterr().out
    assert exit_code == 0
    assert "SMT and finite-contract witnesses" in html
    assert "static-contract-proved" in html
    assert "solve finite contract" in html
    assert "classify SMT diagnostic" in html
    assert "unsat core" in html


def test_role_boundary_html_report_has_interactive_witness_explorer(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/role-boundary/unsafe.promptabi.json", "--format", "html"])

    html = capsys.readouterr().out
    assert exit_code == 1
    assert "Interactive witness explorer" in html
    assert "Role-region overlays" in html
    assert "Token-boundary view" in html
    assert 'class="token-boundary"' in html
    assert "role-boundary-nonforgeability" in html
    assert "<script" not in html


def test_static_contract_html_report_renders_solver_assignment_table(tmp_path: Path, capsys) -> None:
    segments = tmp_path / "segments.json"
    budget = tmp_path / "budget.json"
    segments.write_text(
        json.dumps({"segments": [{"name": "system", "role": "system", "required": True, "token_count": 24}]}),
        encoding="utf-8",
    )
    budget.write_text("{}", encoding="utf-8")
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "static-contract-assignment-html",
                "checks": ["static-contracts"],
                "artifacts": {
                    "segments": {
                        "kind": "prompt-segment",
                        "path": segments.name,
                        "segments": [{"name": "system", "role": "system", "required": True, "token_count": 24}],
                    },
                    "budget": {
                        "kind": "framework-truncation-config",
                        "path": budget.name,
                        "framework": "vllm",
                        "strategy": "left",
                        "max_context_tokens": 16,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "html"])

    html = capsys.readouterr().out
    assert exit_code == 1
    assert "Solver-assignment tables" in html
    assert "required_prompt_tokens" in html
    assert "input_budget_tokens" in html


def test_diff_html_report_renders_artifact_diff_section(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "baseline.promptabi.json"
    current = tmp_path / "current.promptabi.json"
    baseline.write_text(
        json.dumps({"name": "baseline", "checks": ["repository-skeleton"], "artifacts": {}}),
        encoding="utf-8",
    )
    current.write_text(
        json.dumps({"name": "current", "checks": ["repository-skeleton", "token-budget-model"], "artifacts": {}}),
        encoding="utf-8",
    )

    exit_code = main(["diff", str(baseline), str(current), "--format", "html"])

    html = capsys.readouterr().out
    assert exit_code == 0
    assert "Artifact diffs" in html
    assert "diff-check-added" in html
    assert "token-budget-model" in html


def test_public_api_can_render_html() -> None:
    result = run_verification("examples/minimal/promptabi.json")

    html = render_result(result, output_format="html")

    assert html.startswith("<!doctype html>")
    assert "minimal-chat-template" in html
    assert "Diagnostics" in html
