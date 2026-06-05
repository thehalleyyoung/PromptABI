import json

from promptabi.cli import main
from promptabi.conference_demos import render_conference_demo_text, run_conference_demos


EXPECTED_SCENARIOS = {
    "before-deployment": {"tool-serialization"},
    "before-fine-tuning": {"static-contract-violation", "training-packing-boundary"},
    "before-eval-publication": {
        "evaluation-harness-answer-key-leakage",
        "evaluation-harness-grading-rubric-leakage",
        "evaluation-harness-stop-policy-mismatch",
    },
    "before-provider-migration": {"provider-migration"},
}


def test_conference_demos_replay_real_buggy_and_fixed_configs() -> None:
    report = run_conference_demos()

    assert report.ok is True
    assert {case.id for case in report.cases} == set(EXPECTED_SCENARIOS)
    for case in report.cases:
        assert case.caught is True
        assert case.fixed_clean is True
        assert EXPECTED_SCENARIOS[case.id] <= set(case.observed_error_rules)
        assert case.fixed_error_rules == ()
        assert case.headline.startswith(tuple(EXPECTED_SCENARIOS[case.id]))
        assert case.witness_steps


def test_conference_demo_text_is_stage_ready() -> None:
    output = render_conference_demo_text(run_conference_demos())

    assert "PromptABI conference demos" in output
    assert "status: PASS" in output
    assert "buggy -> caught" in output
    assert "fixed -> clean" in output
    assert "before-eval-publication: Catch benchmark contract leakage before eval publication" in output
    assert "witness:" in output


def test_conference_demo_cli_json_uses_demo_success_exit_code(capsys) -> None:
    exit_code = main(["conference-demo", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["summary"] == {"caught": 4, "fixed_clean": 4, "scenarios": 4}
    by_id = {case["id"]: case for case in payload["cases"]}
    assert set(by_id) == set(EXPECTED_SCENARIOS)
    assert EXPECTED_SCENARIOS["before-provider-migration"] <= set(
        by_id["before-provider-migration"]["observed_error_rules"]
    )
