import json
from hashlib import sha256
from pathlib import Path

from promptabi.artifacts import ArtifactKind, EvaluationHarnessArtifact, EvaluationTurnContract, artifact_from_config
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
            "answer_key_variables": ["expected"],
            "grading_rubric_fields": ["private_rubric"],
            "chain_of_thought_variables": ["grader_notes"],
            "required_tools": ["calculator"],
            "available_tools": ["calculator", "search"],
            "max_history_messages": 2,
            "max_history_tokens": 32,
            "preserve_system_prompt": True,
            "preserve_tool_messages": True,
            "retained_turn_ids": ["system", "question"],
            "conversation_turns": [
                {
                    "id": "system",
                    "role": "system",
                    "content": "Follow rubric.",
                    "token_count": 3,
                    "system_prompt_required": True,
                },
                {
                    "id": "question",
                    "role": "user",
                    "content": "Q",
                    "token_count": 1,
                    "tools_required": ["calculator"],
                    "tools_available": ["calculator"],
                },
            ],
            "benchmark_tokenizer": {
                "harness_family": "helm",
                "name": "byte-bpe",
                "chat_template_sha256": "abc123",
                "special_tokens": [{"name": "eos_token", "value": "</s>", "token_id": 2}],
                "added_tokens": [{"content": "<tool>", "id": 32000, "special": False}],
            },
            "few_shot_examples": [{"id": "one", "role": "user", "content": "Q", "token_count": 1}],
        },
        base_dir=ROOT,
    )

    assert isinstance(artifact, EvaluationHarnessArtifact)
    assert artifact.kind is ArtifactKind.EVALUATION_HARNESS
    assert artifact.stop_sequences == ("</answer>",)
    assert artifact.answer_key_variables == ("expected",)
    assert artifact.grading_rubric_variables == ("private_rubric",)
    assert artifact.chain_of_thought_variables == ("grader_notes",)
    assert artifact.required_tools == ("calculator",)
    assert artifact.available_tools == ("calculator", "search")
    assert artifact.max_history_messages == 2
    assert artifact.max_history_tokens == 32
    assert artifact.preserve_system_prompt is True
    assert artifact.preserve_tool_messages is True
    assert artifact.retained_turn_ids == ("question", "system")
    assert isinstance(artifact.conversation_turns[0], EvaluationTurnContract)
    assert [turn.turn_id for turn in artifact.conversation_turns] == ["system", "question"]
    assert artifact.conversation_turns[1].tools_required == ("calculator",)
    assert artifact.benchmark_tokenizer is not None
    assert artifact.benchmark_tokenizer.harness_family == "helm"
    assert artifact.benchmark_tokenizer.pinned_fields == ("special_tokens", "added_tokens", "chat_template_sha256")
    assert artifact.few_shot_examples[0].token_count == 1


def test_safe_evaluation_harness_runs_through_verification_session() -> None:
    result = VerificationSession.from_config_file(SAFE_CONFIG).run()
    rule_ids = [diagnostic.rule_id for diagnostic in result.diagnostics]

    assert result.ok is True
    assert "evaluation-harness-verified" in rule_ids
    assert "evaluation-harness-tokenizer-unpinned" not in rule_ids
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
        "evaluation-harness-grading-parser-mismatch",
        "evaluation-harness-answer-key-leakage",
        "evaluation-harness-chain-of-thought-leakage",
        "evaluation-harness-prompt-variable-missing",
        "evaluation-harness-few-shot-role-mismatch",
        "evaluation-harness-few-shot-budget-overflow",
        "evaluation-harness-grading-rubric-leakage",
        "evaluation-harness-history-truncation-mismatch",
        "evaluation-harness-role-alternation-mismatch",
        "evaluation-harness-system-prompt-truncated",
        "evaluation-harness-tool-unavailable",
    }
    assert result.ok is False
    assert expected.issubset(by_rule)
    grading_parser_diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.rule_id == "evaluation-harness-grading-parser-mismatch"
    ]
    assert by_rule["evaluation-harness-stop-policy-mismatch"].span is not None
    assert grading_parser_diagnostics
    assert all(diagnostic.span is not None for diagnostic in grading_parser_diagnostics)
    assert any(
        step.action == "select bounded answer sample" and step.output == "{}"
        for diagnostic in grading_parser_diagnostics
        for step in diagnostic.witness.steps
    )
    assert any(("subject", "answer_parser/application_parser") in diagnostic.properties for diagnostic in grading_parser_diagnostics)
    assert all(("actual", "grading-parser-broader") in diagnostic.properties for diagnostic in grading_parser_diagnostics)
    assert ("actual", "answer_key") in by_rule["evaluation-harness-answer-key-leakage"].properties
    assert by_rule["evaluation-harness-chain-of-thought-leakage"].span is not None
    assert by_rule["evaluation-harness-few-shot-role-mismatch"].witness is not None
    assert by_rule["evaluation-harness-role-alternation-mismatch"].span is not None
    assert by_rule["evaluation-harness-system-prompt-truncated"].span is not None
    assert ("actual", "retriever") in by_rule["evaluation-harness-tool-unavailable"].properties
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
    assert any(
        diagnostic["rule_id"] == "evaluation-harness-grading-rubric-leakage"
        for diagnostic in payload["diagnostics"]
    )
    assert any(
        diagnostic["rule_id"] == "evaluation-harness-tool-unavailable"
        for diagnostic in payload["diagnostics"]
    )


def test_safe_evaluation_harness_covers_multi_turn_contracts() -> None:
    result = VerificationSession.from_config_file(SAFE_CONFIG).run()
    verified = next(diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "evaluation-harness-verified")

    assert result.ok is True
    assert all("multi-turn" not in diagnostic.rule_id for diagnostic in result.diagnostics if diagnostic.severity.value == "error")
    assert any(step.output == "no contract-breaking mismatch found" for step in verified.witness.steps)


def test_evaluation_harness_analyzer_emits_missing_contract_abstention() -> None:
    session = VerificationSession.from_config_file(SAFE_CONFIG)
    loaded = session.load_artifacts()
    harness = next(artifact.artifact for artifact in loaded if isinstance(artifact.artifact, EvaluationHarnessArtifact))

    report = analyze_evaluation_harness_contracts(harness, (harness,))

    assert any(finding.rule_id == "evaluation-harness-contract-missing" for finding in report.findings)


def test_evaluation_harness_detects_benchmark_tokenizer_drift_with_real_files(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_eval_tokenizer(
        baseline,
        chat_template="{{ messages[0].content }}",
        eos_id=2,
        normalizer={"type": "Lowercase"},
        stop_strings=["</answer>"],
        add_bos=False,
    )
    _write_eval_tokenizer(
        current,
        chat_template="<|start_header_id|>{{ messages[0].role }}<|end_header_id|>",
        eos_id=128009,
        normalizer={"type": "NFC"},
        stop_strings=["<|eot_id|>"],
        add_bos=True,
    )
    harness = tmp_path / "harness.json"
    harness.write_text(
        json.dumps(
            {
                "benchmark_name": "helm-mmlu-local",
                "provider": "openai-compatible",
                "model": "llama-eval",
                "tokenizer": "eval-tokenizer",
                "prompt_template": "template",
                "answer_parser": "json-schema",
                "answer_schema": "schema",
                "stop_sequences": ["<|eot_id|>"],
                "prompt_variables": ["question"],
                "required_prompt_variables": ["question"],
                "benchmark_tokenizer": {
                    "harness_family": "helm",
                    "name": "eval-tokenizer",
                    "chat_template_sha256": sha256("{{ messages[0].content }}".encode()).hexdigest(),
                    "special_tokens": [{"name": "eos_token", "value": "</s>", "token_id": 2}],
                    "normalizer_signature": '{"type":"Lowercase"}',
                    "add_bos_token": False,
                    "stop_sequences": ["</answer>"],
                    "stop_token_ids": [2],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "eval-tokenizer-drift",
                "checks": ["evaluation-harness-contracts"],
                "artifacts": {
                    "eval": {"kind": "evaluation-harness", "path": "harness.json"},
                    "provider": {
                        "kind": "provider-config",
                        "path": "provider.json",
                        "provider": "openai-compatible",
                        "metadata": {"model": "llama-eval"},
                    },
                    "tokenizer": {"kind": "tokenizer", "path": "current", "family": "eval-tokenizer"},
                    "template": {"kind": "chat-template", "path": "current/tokenizer_config.json"},
                    "stops": {"kind": "stop-policy", "path": "stop-policy.json"},
                    "schema": {"kind": "schema", "path": "answer.schema.json", "dialect": "json-schema"},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (tmp_path / "provider.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stop-policy.json").write_text(json.dumps({"stop": ["<|eot_id|>"]}), encoding="utf-8")
    (tmp_path / "answer.schema.json").write_text('{"type":"object"}', encoding="utf-8")

    result = VerificationSession.from_config_file(config).run()
    drift = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "evaluation-harness-tokenizer-drift"]

    assert result.ok is False
    assert {dict(diagnostic.properties)["subject"] for diagnostic in drift} >= {
        "benchmark_tokenizer.chat_template_sha256",
        "benchmark_tokenizer.special_tokens",
        "benchmark_tokenizer.normalizer_signature",
        "benchmark_tokenizer.add_bos_token",
        "benchmark_tokenizer.stop_sequences",
        "benchmark_tokenizer.stop_token_ids",
    }
    assert all(diagnostic.span is not None for diagnostic in drift)
    assert any(step.action == "compare benchmark tokenizer field" for step in drift[0].witness.steps)


def test_provider_hosted_evaluation_tokenizer_abstains_without_local_snapshot(tmp_path: Path) -> None:
    harness = tmp_path / "harness.json"
    harness.write_text(
        json.dumps(
            {
                "benchmark_name": "provider-hosted-eval",
                "provider": "openai-compatible",
                "model": "remote-eval",
                "tokenizer": "remote-tokenizer",
                "prompt_template": "template",
                "answer_parser": "json-schema",
                "answer_schema": "schema",
                "stop_sequences": ["</answer>"],
                "prompt_variables": ["question"],
                "required_prompt_variables": ["question"],
                "benchmark_tokenizer": {
                    "harness_family": "provider-hosted",
                    "name": "remote-tokenizer",
                    "revision": "provider-release-2026-06",
                    "eos_token": "</s>",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "provider-hosted-eval",
                "checks": ["evaluation-harness-contracts"],
                "artifacts": {
                    "eval": {"kind": "evaluation-harness", "path": "harness.json"},
                    "tokenizer": {"kind": "tokenizer", "path": "tokenizer.json", "family": "remote-tokenizer"},
                    "template": {"kind": "chat-template", "path": "tokenizer.json"},
                    "stops": {"kind": "stop-policy", "path": "stop-policy.json"},
                    "schema": {"kind": "schema", "path": "answer.schema.json", "dialect": "json-schema"},
                    "provider": {
                        "kind": "provider-config",
                        "path": "provider.json",
                        "provider": "openai-compatible",
                        "metadata": {"model": "remote-eval"},
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (tmp_path / "provider.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stop-policy.json").write_text(json.dumps({"stop": ["</answer>"]}), encoding="utf-8")
    (tmp_path / "answer.schema.json").write_text('{"type":"object"}', encoding="utf-8")

    diagnostics = VerificationSession.from_config_file(config).run().diagnostics

    assert any(diagnostic.rule_id == "evaluation-harness-tokenizer-unpinned" for diagnostic in diagnostics)
    assert not any(diagnostic.rule_id == "evaluation-harness-tokenizer-drift" for diagnostic in diagnostics)


def test_evaluation_harness_reports_cross_provider_eval_matrix(tmp_path: Path) -> None:
    config = _write_cross_provider_eval_config(tmp_path)

    result = VerificationSession.from_config_file(config).run()
    reports = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "evaluation-harness-cross-provider-report"]

    assert result.ok is True
    assert reports
    assert "7 adapter families" in reports[0].message
    assert ("actual", "anthropic, bedrock, gemini, llama.cpp-server, ollama, openai, vllm-openai-server") in reports[0].properties
    assert any(step.action == "collect provider adapter families" for step in reports[0].witness.steps)
    assert not any(diagnostic.rule_id == "evaluation-harness-cross-provider-incomplete" for diagnostic in result.diagnostics)


def test_evaluation_harness_detects_cross_provider_eval_mismatch(tmp_path: Path) -> None:
    config = _write_cross_provider_eval_config(
        tmp_path,
        provider_overrides={
            "ollama": {
                "response_fields": ["message"],
                "stop_sequences": ["<|done|>"],
                "structured_output_modes": [],
                "supports_parallel_tools": False,
            }
        },
    )

    result = VerificationSession.from_config_file(config).run()
    mismatches = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "evaluation-harness-cross-provider-mismatch"]

    assert result.ok is False
    assert {dict(diagnostic.properties)["subject"] for diagnostic in mismatches} >= {
        "providers.ollama.stop_policy",
        "providers.ollama.structured_outputs",
        "providers.ollama.request_response",
        "providers.ollama.parallel_tools",
    }
    assert any(
        step.action == "select candidate provider adapter" and step.output == "ollama"
        for diagnostic in mismatches
        for step in diagnostic.witness.steps
    )
    assert all(diagnostic.span is not None for diagnostic in mismatches)


def _write_eval_tokenizer(
    root: Path,
    *,
    chat_template: str,
    eos_id: int,
    normalizer: dict[str, str],
    stop_strings: list[str],
    add_bos: bool,
) -> None:
    root.mkdir()
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "eos_token_id": eos_id,
                "add_bos_token": add_bos,
                "chat_template": chat_template,
                "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "normalizer": normalizer,
                "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"stop_strings": stop_strings, "eos_token_id": eos_id}, sort_keys=True),
        encoding="utf-8",
    )


def _write_cross_provider_eval_config(
    tmp_path: Path,
    *,
    provider_overrides: dict[str, dict[str, object]] | None = None,
) -> Path:
    provider_overrides = provider_overrides or {}
    harness = tmp_path / "harness.json"
    harness.write_text(
        json.dumps(
            {
                "benchmark_name": "cross-provider-contract-eval",
                "provider": "openai-compatible",
                "model": "eval-model",
                "tokenizer": "byte-bpe",
                "prompt_template": "template",
                "answer_parser": "json-schema",
                "answer_schema": "schema",
                "stop_sequences": ["</answer>"],
                "prompt_variables": ["question"],
                "required_prompt_variables": ["question"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "template.json").write_text('{"chat_template":"{{ messages[0].content }}"}', encoding="utf-8")
    (tmp_path / "stop-policy.json").write_text(json.dumps({"stop": ["</answer>"]}), encoding="utf-8")
    (tmp_path / "answer.schema.json").write_text('{"type":"object"}', encoding="utf-8")

    base_metadata: dict[str, object] = {
        "model": "eval-model",
        "request_fields": ["messages", "tools", "response_format"],
        "response_fields": ["choices", "message", "tool_calls"],
        "tool_argument_encoding": "json-string",
        "supports_parallel_tools": True,
        "stop_sequences": ["</answer>"],
        "structured_output_modes": ["json_schema"],
    }
    provider_specs = {
        "openai": ("openai-compatible", "openai"),
        "anthropic": ("anthropic", "anthropic"),
        "gemini": ("gemini", "gemini"),
        "bedrock": ("bedrock", "bedrock"),
        "vllm": ("vllm-openai-server", "vllm-openai-server"),
        "llamacpp": ("llama.cpp-server", "llama.cpp-server"),
        "ollama": ("ollama", "ollama"),
    }
    artifacts: dict[str, object] = {
        "eval": {"kind": "evaluation-harness", "path": "harness.json"},
        "tokenizer": {"kind": "tokenizer", "path": "tokenizer.json", "family": "byte-bpe"},
        "template": {"kind": "chat-template", "path": "template.json"},
        "stops": {"kind": "stop-policy", "path": "stop-policy.json"},
        "schema": {"kind": "schema", "path": "answer.schema.json", "dialect": "json-schema"},
    }
    for artifact_name, (provider_name, family) in provider_specs.items():
        provider_path = tmp_path / f"{artifact_name}.json"
        provider_path.write_text(
            json.dumps(
                {
                    "provider": provider_name,
                    "api_family": family,
                    "request_shape": {"messages": "array", "tools": "array", "response_format": "object"},
                    "response_shape": {"choices": "array", "message": "object", "tool_calls": "array"},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        metadata = {**base_metadata, "provider_family": family, **provider_overrides.get(artifact_name, {})}
        artifacts[artifact_name] = {
            "kind": "provider-config",
            "path": f"{artifact_name}.json",
            "provider": provider_name,
            "metadata": metadata,
        }

    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "cross-provider-eval",
                "checks": ["evaluation-harness-contracts"],
                "artifacts": artifacts,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config
