import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.mutation_fuzzing import (
    ALL_FUZZ_SURFACES,
    FuzzSurface,
    MutationFuzzingError,
    render_mutation_fuzz_json,
    render_mutation_fuzz_text,
    run_mutation_fuzzing,
)


def test_mutation_fuzzer_covers_all_required_artifact_surfaces() -> None:
    report = run_mutation_fuzzing()
    payload = report.to_dict()

    assert tuple(payload["surfaces"]) == tuple(surface.value for surface in ALL_FUZZ_SURFACES)
    assert payload["case_count"] == 24
    assert payload["mutation_count"] == 16
    assert payload["introduced_violation_count"] >= 12

    discovered = set(payload["discovered_rule_ids"])
    assert {
        "chat-template-unsupported-construct",
        "tokenizer-control-token-reachable",
        "stop-overreachability",
        "schema-json-schema-recursion-limit",
        "grammar-regex-lookaround",
        "tool-schema-precondition-satisfiability",
        "prompt-segment-survival-violation",
        "smt-counterexample",
    }.issubset(discovered)


def test_mutation_fuzzer_reports_only_rules_introduced_by_mutants() -> None:
    report = run_mutation_fuzzing((FuzzSurface.STOP_POLICIES, "smt-encodings"))
    by_id = {result.case.case_id: result for result in report.mutation_results}

    assert "stop-overreachability" in by_id["stop-policy-json-overreach"].introduced_rule_ids
    assert "stop-token-unreachable" in by_id["stop-policy-prefix-collision"].introduced_rule_ids
    assert by_id["smt-satisfiable-violation"].introduced_rule_ids == ("smt-counterexample",)
    assert by_id["smt-unsat-core-conflict"].introduced_rule_ids == ()


def test_mutation_fuzz_renderers_and_cli_shape(tmp_path: Path, capsys) -> None:
    report = run_mutation_fuzzing(("tool-definitions", "truncation-configs"))
    json_payload = json.loads(render_mutation_fuzz_json(report))
    text_payload = render_mutation_fuzz_text(report)

    assert json_payload["introduced_violation_count"] > 0
    assert "PromptABI mutation fuzzing" in text_payload
    assert "tool-schema-precondition-satisfiability" in text_payload

    exit_code = main(["fuzz", "mutations", "--surface", "schemas", "--format", "json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out)["surfaces"] == ["schemas"]

    output = tmp_path / "fuzz.json"
    exit_code = main(["fuzz", "mutations", "--surface", "tokenizers", "--output", str(output)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wrote mutation-fuzzing report" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["discovered_rule_ids"]


def test_public_api_mutation_fuzzing_returns_report_and_rendered_forms() -> None:
    report = promptabi.fuzz_mutations(("chat-templates",))
    rendered = promptabi.fuzz_mutations(("chat-templates",), output_format="json")

    assert isinstance(report, promptabi.MutationFuzzReport)
    assert json.loads(rendered)["surfaces"] == ["chat-templates"]


def test_mutation_fuzzer_rejects_unknown_surfaces() -> None:
    try:
        run_mutation_fuzzing(("not-a-surface",))
    except MutationFuzzingError as exc:
        assert "unknown fuzz surface" in str(exc)
    else:
        raise AssertionError("expected unknown fuzz surface to be rejected")
