import json
from pathlib import Path

import pytest

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ByteLevelTokenizer,
    MinimizationError,
    MinimizationKind,
    StopPolicyArtifact,
    contains_oracle,
    minimize_failure_repro,
    render_minimization,
)
from promptabi.cli import main
from promptabi.minimization import load_minimization_case
from promptabi.stop_analysis import analyze_stop_policy_tokenizer


@pytest.mark.parametrize(
    ("kind", "value", "needle"),
    [
        (
            MinimizationKind.TEMPLATE,
            "system prelude\n{% for message in messages %}\n{{ message.content }} BUG_DELIMITER\n{% endfor %}\ntrailer",
            "BUG_DELIMITER",
        ),
        (
            MinimizationKind.SCHEMA,
            {
                "title": "large",
                "type": "object",
                "properties": {
                    "safe": {"type": "string"},
                    "bad": {"const": "BUG_SCHEMA_VALUE", "description": "keep this one"},
                },
                "required": ["bad"],
            },
            "BUG_SCHEMA_VALUE",
        ),
        (
            MinimizationKind.STOP_STRINGS,
            ["### harmless", "END", "prefix BUG_STOP suffix", "Observation:"],
            "BUG_STOP",
        ),
        (
            MinimizationKind.MESSAGE_SET,
            [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "hello BUG_MESSAGE"},
                {"role": "assistant", "content": "verbose answer"},
            ],
            "BUG_MESSAGE",
        ),
        (
            MinimizationKind.SOLVER_CONSTRAINTS,
            [
                {"name": "budget", "expr": {"le": ["tokens", 8192]}},
                {"name": "failing", "expr": {"contains": ["rendered", "BUG_CONSTRAINT"]}},
                {"name": "role", "expr": {"eq": ["role", "assistant"]}},
            ],
            "BUG_CONSTRAINT",
        ),
        (
            MinimizationKind.PROVIDER_FIXTURE,
            {
                "provider": "openai-compatible",
                "request": {"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
                "response": {"choices": [{"message": {"tool_calls": [{"function": {"name": "BUG_TOOL"}}]}}]},
                "stream": [{"delta": "unused"}],
            },
            "BUG_TOOL",
        ),
    ],
)
def test_minimizers_shrink_each_promptabi_repro_surface(kind, value, needle) -> None:
    result = minimize_failure_repro(value, contains_oracle(needle), kind=kind)

    rendered_json = render_minimization(result, output_format="json")
    payload = json.loads(rendered_json)
    assert result.changed
    assert result.stats.minimized_size < result.stats.original_size
    assert needle in json.dumps(result.minimized)
    assert payload["kind"] == kind.value
    assert result.witness().steps[-1].action == "validate minimized repro"

    second = minimize_failure_repro(result.minimized, contains_oracle(needle), kind=kind)
    assert second.minimized == result.minimized


def test_minimizer_rejects_non_reproducing_original() -> None:
    with pytest.raises(MinimizationError, match="original repro does not satisfy"):
        minimize_failure_repro(["safe", "still-safe"], contains_oracle("missing"), kind="stop-strings")


def test_stop_string_minimization_preserves_real_promptabi_collision() -> None:
    tokenizer = ByteLevelTokenizer()

    def still_reports_end_collision(value) -> bool:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return False
        policy = StopPolicyArtifact(
            kind=ArtifactKind.STOP_POLICY,
            name="stops",
            location=ArtifactLocation(uri="memory://stops"),
            stop_sequences=tuple(value),
        )
        report = analyze_stop_policy_tokenizer(policy, tokenizer)
        return any(
            collision.level == "string"
            and collision.relation == "prefix"
            and collision.shorter == "END"
            and collision.longer == "ENDIF"
            for collision in report.collisions
        )

    result = minimize_failure_repro(
        ["Observation:", "END", "noise", "ENDIF", "</tool_call>"],
        still_reports_end_collision,
        kind=MinimizationKind.STOP_STRINGS,
    )

    assert result.minimized == ["END", "ENDIF"]
    assert still_reports_end_collision(result.minimized)
    assert result.stats.predicate_calls >= 2


def test_minimize_cli_outputs_stable_json(tmp_path: Path, capsys) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "kind": "message-set",
                "input": [
                    {"role": "system", "content": "unused"},
                    {"role": "user", "content": "please keep BUG_CLI here"},
                    {"role": "assistant", "content": "unused"},
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "minimize",
            str(case_path),
            "--keep-substring",
            "BUG_CLI",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["kind"] == "message-set"
    assert payload["changed"] is True
    assert "BUG_CLI" in json.dumps(payload["minimized"])


def test_load_minimization_case_validates_shape(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text('{"kind": "schema", "input": {"type": "object"}}', encoding="utf-8")

    kind, value = load_minimization_case(case_path)

    assert kind is MinimizationKind.SCHEMA
    assert value == {"type": "object"}
