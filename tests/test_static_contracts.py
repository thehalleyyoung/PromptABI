import json
from pathlib import Path

from promptabi.artifacts import (
    ArtifactKind,
    ArtifactLocation,
    ChatTemplateArtifact,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    SpecialToken,
    SpecialTokenMapArtifact,
    StopPolicyArtifact,
    TrainingManifestArtifact,
    TruncationStrategy,
)
from promptabi.cli import main
from promptabi.config import VerificationConfig
from promptabi.loaders import LoadedArtifact
from promptabi.static_contracts import analyze_static_contracts


def _loaded(artifact):
    return LoadedArtifact(artifact=artifact, source_type="memory", pinned=True, resolved=True)


def test_static_contracts_prove_budget_and_stop_exclusion_with_enumeration() -> None:
    location = ArtifactLocation(uri="memory://static-contract")
    config = VerificationConfig(name="static")
    loaded = (
        _loaded(
            PromptSegmentArtifact(
                kind=ArtifactKind.PROMPT_SEGMENT,
                name="segments",
                location=location,
                segments=(
                    PromptSegment("system", role="system", required=True, token_count=10),
                    PromptSegment("retrieval", role="user", token_count=20),
                ),
            )
        ),
        _loaded(
            FrameworkTruncationConfigArtifact(
                kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
                name="budget",
                location=location,
                framework="vllm",
                strategy=TruncationStrategy.LEFT,
                max_context_tokens=64,
                reserve_output_tokens=8,
            )
        ),
        _loaded(
            StopPolicyArtifact(
                kind=ArtifactKind.STOP_POLICY,
                name="stops",
                location=location,
                stop_sequences=("</tool_call>",),
            )
        ),
        _loaded(
            SpecialTokenMapArtifact(
                kind=ArtifactKind.SPECIAL_TOKEN_MAP,
                name="specials",
                location=location,
                tokens=(SpecialToken("eos", "</s>", 2),),
            )
        ),
    )

    report = analyze_static_contracts(config, loaded, prefer_z3=False)

    assert not report.violations
    names = {finding.name for finding in report.findings}
    assert "prompt-segment-survival-violation" in names
    assert "stop-control-token-collision" in names
    assert all(finding.result is None or finding.result.backend.value == "finite-enumeration" for finding in report.findings)


def test_static_contracts_extract_counterexamples_for_real_conflicts() -> None:
    location = ArtifactLocation(uri="memory://static-contract")
    config = VerificationConfig(name="static")
    loaded = (
        _loaded(
            PromptSegmentArtifact(
                kind=ArtifactKind.PROMPT_SEGMENT,
                name="segments",
                location=location,
                segments=(PromptSegment("must", role="system", required=True, token_count=80),),
            )
        ),
        _loaded(
            FrameworkTruncationConfigArtifact(
                kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
                name="budget",
                location=location,
                framework="langchain",
                strategy=TruncationStrategy.OLDEST_MESSAGE,
                max_context_tokens=64,
                reserve_output_tokens=8,
            )
        ),
        _loaded(
            StopPolicyArtifact(
                kind=ArtifactKind.STOP_POLICY,
                name="stops",
                location=location,
                stop_sequences=("</s>",),
            )
        ),
        _loaded(
            SpecialTokenMapArtifact(
                kind=ArtifactKind.SPECIAL_TOKEN_MAP,
                name="specials",
                location=location,
                tokens=(SpecialToken("eos", "</s>", 2),),
            )
        ),
        _loaded(
            TrainingManifestArtifact(
                kind=ArtifactKind.TRAINING_MANIFEST,
                name="train",
                location=location,
                target_roles=("assistant", "critic"),
            )
        ),
        _loaded(
            ChatTemplateArtifact(
                kind=ArtifactKind.CHAT_TEMPLATE,
                name="template",
                location=location,
                roles=("system", "user", "assistant"),
            )
        ),
    )

    report = analyze_static_contracts(config, loaded, prefer_z3=False)

    violations = {finding.name: finding for finding in report.violations}
    assert violations["prompt-segment-survival-violation"].result.assignment == {
        "input_budget_tokens": 56,
        "required_prompt_tokens": 80,
    }
    assert violations["stop-control-token-collision"].result.assignment["stop_sequence"] == "</s>"
    assert violations["training-target-role-alignment"].result.assignment == {"target_role": "critic"}


def test_verify_static_contracts_cli_reports_z3_backed_contract(tmp_path: Path, capsys) -> None:
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
                "name": "static-contract-cli",
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

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    diagnostics = [diagnostic for diagnostic in payload["diagnostics"] if diagnostic["rule_id"] == "static-contract-proved"]
    assert diagnostics
    assert diagnostics[0]["check_modes"] == ["bounded", "sound", "z3-backed-smt"]
    assert any(step["action"] == "solve finite contract" for step in diagnostics[0]["witness"]["steps"])
