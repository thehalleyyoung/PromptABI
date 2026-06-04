import json
from pathlib import Path

from promptabi import ArtifactKind, ArtifactLocation, StopPolicyArtifact
from promptabi.cli import main
from promptabi.stop_differential import (
    StopTraceCase,
    StopTraceExpectation,
    analyze_stop_differential,
    simulate_stop_trace,
)
from promptabi.artifacts import ProviderConfigArtifact


def test_stop_simulator_models_provider_family_inclusion_rules() -> None:
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="stops",
        location=ArtifactLocation(uri="memory://stops"),
        stop_sequences=("END",),
    )

    openai = simulate_stop_trace(
        policy,
        StopTraceCase(
            name="openai",
            family="openai-compatible",
            chunks=("alpha ", "END trailing"),
            token_ids=(),
            expectation=StopTraceExpectation(stopped=True, output="alpha "),
        ),
    )
    huggingface = simulate_stop_trace(
        policy,
        StopTraceCase(
            name="hf",
            family="huggingface",
            chunks=("alpha ", "END trailing"),
            token_ids=(),
            expectation=StopTraceExpectation(stopped=True, output="alpha END"),
        ),
    )

    assert openai.stopped is True
    assert openai.output == "alpha "
    assert openai.include_stop_in_output is False
    assert huggingface.stopped is True
    assert huggingface.output == "alpha END"
    assert huggingface.include_stop_in_output is True


def test_stop_differential_replays_recorded_provider_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "openai-stop-traces.json"
    fixture.write_text(
        json.dumps(
            {
                "provider": "openai-compatible",
                "request_shape": {"messages": "array", "stop": "array"},
                "stop_traces": [
                    {
                        "name": "substring-stop-across-streaming-deltas",
                        "family": "openai-compatible",
                        "chunks": ["answer: ", "safe", "END ignored"],
                        "expected": {
                            "stopped": True,
                            "output": "answer: safe",
                            "matched_stop": "END",
                            "finish_reason": "stop",
                            "include_stop_in_output": False,
                        },
                    },
                    {
                        "name": "hf-stop-string-retained-in-generated-text",
                        "family": "huggingface",
                        "chunks": ["answer: ", "safe", "END ignored"],
                        "expected": {
                            "stopped": True,
                            "output": "answer: safeEND",
                            "matched_stop": "END",
                            "finish_reason": "stop",
                            "include_stop_in_output": True,
                        },
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="stops",
        location=ArtifactLocation(uri="memory://stops"),
        stop_sequences=("END",),
    )
    provider = ProviderConfigArtifact(
        kind=ArtifactKind.PROVIDER_CONFIG,
        name="provider-fixture",
        location=ArtifactLocation(path=str(fixture)),
        provider="openai-compatible",
    )

    report = analyze_stop_differential(policy, (provider,))

    assert [case.name for case in report.cases] == [
        "substring-stop-across-streaming-deltas",
        "hf-stop-string-retained-in-generated-text",
    ]
    assert len(report.matches) == 2
    assert report.mismatches == ()
    assert report.abstentions == ()


def test_stop_simulator_uses_only_explicit_stop_token_ids() -> None:
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="token-stops",
        location=ArtifactLocation(uri="memory://stops"),
        stop_token_ids=(42,),
    )

    explicit = simulate_stop_trace(
        policy,
        StopTraceCase(
            name="explicit-token",
            family="vllm",
            chunks=("payload",),
            token_ids=(10, 42, 99),
            expectation=StopTraceExpectation(stopped=True, output="payload"),
        ),
    )
    not_guessed = simulate_stop_trace(
        policy,
        StopTraceCase(
            name="eos-not-guessed",
            family="vllm",
            chunks=("payload",),
            token_ids=(2,),
            expectation=StopTraceExpectation(stopped=False, output="payload"),
        ),
    )

    assert explicit.stopped is True
    assert explicit.matched_stop == "token:42"
    assert explicit.token_index == 1
    assert not_guessed.stopped is False


def test_stop_differential_cli_reports_fixture_mismatch(tmp_path: Path, capsys) -> None:
    stop_path = tmp_path / "stops.json"
    fixture_path = tmp_path / "provider.json"
    config_path = tmp_path / "promptabi.json"
    stop_path.write_text('{"provider": "openai", "stop": ["END"]}', encoding="utf-8")
    fixture_path.write_text(
        json.dumps(
            {
                "provider": "openai-compatible",
                "request_shape": {"messages": "array", "stop": "array"},
                "stop_trace": {
                    "name": "bad-retains-stop",
                    "family": "openai-compatible",
                    "chunks": ["payload", "END", "tail"],
                    "expected": {
                        "stopped": True,
                        "output": "payloadEND",
                        "matched_stop": "END",
                        "finish_reason": "stop",
                        "include_stop_in_output": True,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "name": "stop-differential-fixture",
                "checks": ["stop-differential"],
                "artifacts": {
                    "stops": {"kind": "stop-policy", "path": str(stop_path)},
                    "provider": {
                        "kind": "provider-config",
                        "path": str(fixture_path),
                        "provider": "openai-compatible",
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    mismatch = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "stop-differential-mismatch"
    ][0]
    assert exit_code == 1
    assert payload["ok"] is False
    assert "bad-retains-stop" in mismatch["message"]
    assert "output" in mismatch["message"]
    assert mismatch["check_modes"] == ["heuristic"]
    assert any(step["action"] == "replay text chunks" for step in mismatch["witness"]["steps"])
    assert captured.err == ""
