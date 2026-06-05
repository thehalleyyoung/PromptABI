import json
from pathlib import Path
from typing import Any

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ChatTemplateSymbolicBounds,
    SchemaArtifact,
    StopPolicyArtifact,
    ToolDefinitionArtifact,
    analyze_role_boundary_nonforgeability,
    analyze_stop_overreachability,
    parse_hf_chat_template_config,
)
from promptabi.production_code_bugs import ProductionCodeBugCorpusError, load_production_code_bug_corpus


CORPUS_PATH = Path("fixtures/real_world_bugs/corpus.json")
PRODUCTION_CODE_PATH = Path("fixtures/real_world_bugs/production_code.json")


def _load_corpus() -> dict[str, Any]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def test_real_world_bug_corpus_records_public_provenance() -> None:
    corpus = _load_corpus()

    assert corpus["version"] == 1
    all_cases = (*corpus["role_boundary_cases"], *corpus["stop_overreachability_cases"])
    assert len(all_cases) >= 4
    for case in all_cases:
        assert case["public_reference"].startswith("https://github.com/")
        assert case["bug_class"]
        assert case["expected_witnesses"] if "expected_witnesses" in case else case["expected_findings"]


def test_production_code_bug_corpus_traverses_exact_pinned_source() -> None:
    corpus = load_production_code_bug_corpus(PRODUCTION_CODE_PATH)
    replays = {replay.case_id: replay for replay in corpus.replay(real_world_corpus_path=CORPUS_PATH)}

    assert set(replays) == {
        "gemma4_quote_sentinel_truncation",
        "gemma4_streaming_html_tag_duplication",
        "llama_cpp_qwen_array_object_tool_leak",
        "phi_system_turn_forgery",
        "qwen3_xml_tool_parameter_stop",
        "vllm_qwen_multi_function_block_boundary",
    }
    assert all(replay.passed for replay in replays.values())
    assert replays["gemma4_quote_sentinel_truncation"].rule_ids == ("parser-quote-truncation",)
    assert replays["gemma4_streaming_html_tag_duplication"].rule_ids == ("streaming-buffer-reparse",)
    assert replays["llama_cpp_qwen_array_object_tool_leak"].rule_ids == ("tagged-json-parameter-parser-boundary",)
    assert replays["phi_system_turn_forgery"].extracted_values["expected_witnesses_matched"] == 3
    assert replays["qwen3_xml_tool_parameter_stop"].extracted_values["stop_sequences"] == [
        "</tool_call>",
        "</function>",
        "</parameter>",
    ]
    assert replays["vllm_qwen_multi_function_block_boundary"].rule_ids == ("streaming-function-state-leak",)
    for case in corpus.cases:
        assert case.reference.public_url.startswith("https://github.com/")
        assert case.source_sha256
        assert case.source_excerpt
        raw_case = next(item for item in json.loads(PRODUCTION_CODE_PATH.read_text(encoding="utf-8"))["cases"] if item["id"] == case.case_id)
        assert raw_case["recorded_bug"]["url"].startswith("https://github.com/")
        assert raw_case["recorded_bug"]["record_type"] in {"issue", "pull_request"}


def test_production_code_bug_corpus_rejects_tampered_source_excerpt(tmp_path: Path) -> None:
    payload = json.loads(PRODUCTION_CODE_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["source_excerpt"] += "\n# tampered after hashing"
    fixture = tmp_path / "production_code.json"
    fixture.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    try:
        load_production_code_bug_corpus(fixture)
    except ProductionCodeBugCorpusError as exc:
        assert "source_excerpt sha256 mismatch" in str(exc)
    else:
        raise AssertionError("expected production-code corpus hash validation failure")


def test_production_code_bug_replay_rejects_non_source_derived_stop(tmp_path: Path) -> None:
    payload = json.loads(PRODUCTION_CODE_PATH.read_text(encoding="utf-8"))
    stop_case = next(case for case in payload["cases"] if case["id"] == "qwen3_xml_tool_parameter_stop")
    stop_case["extraction"]["expected_stop_sequences"][0] = "</not_in_source>"
    fixture = tmp_path / "production_code.json"
    fixture.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    corpus = load_production_code_bug_corpus(fixture)

    try:
        corpus.replay_case("qwen3_xml_tool_parameter_stop", real_world_corpus_path=CORPUS_PATH)
    except ProductionCodeBugCorpusError as exc:
        assert "is not in source excerpt" in str(exc)
    else:
        raise AssertionError("expected source-derived stop validation failure")


def test_mined_role_boundary_bugs_are_detected() -> None:
    corpus = _load_corpus()
    bounds = ChatTemplateSymbolicBounds(
        max_messages=1,
        max_tools=0,
        max_loop_iterations=1,
        max_paths=32,
    )

    for case in corpus["role_boundary_cases"]:
        parsed = parse_hf_chat_template_config(case["template_config"])
        report = analyze_role_boundary_nonforgeability(parsed, bounds=bounds)
        actual = {(finding.input_expression, finding.marker, finding.marker_kind) for finding in report.findings}
        expected = {tuple(witness) for witness in case["expected_witnesses"]}

        assert report.model.supported, case["id"]
        assert expected <= actual, case["id"]
        for input_expression, marker, marker_kind in expected:
            finding = next(
                finding
                for finding in report.findings
                if (
                    finding.input_expression == input_expression
                    and finding.marker == marker
                    and finding.marker_kind == marker_kind
                )
            )
            assert finding.malicious_input == marker, case["id"]
            assert marker in finding.rendered_excerpt, case["id"]
            assert marker in finding.tokenized_representation, case["id"]
            assert finding.marker_start_offset < finding.marker_end_offset, case["id"]


def test_mined_stop_overreachability_bugs_are_detected(tmp_path: Path) -> None:
    corpus = _load_corpus()

    for case in corpus["stop_overreachability_cases"]:
        policy = StopPolicyArtifact(
            kind=ArtifactKind.STOP_POLICY,
            name=f"{case['id']}-stops",
            location=ArtifactLocation(uri=f"memory://real-world-bugs/{case['id']}/stops"),
            stop_sequences=tuple(case["stop_sequences"]),
        )
        artifacts = _structured_artifacts_for_case(case, tmp_path)

        report = analyze_stop_overreachability(policy, artifacts)

        assert report.findings, case["id"]
        for expected in case["expected_findings"]:
            finding = next(
                (
                    finding
                    for finding in report.findings
                    if finding.stop_sequence == expected["stop_sequence"]
                    and finding.category == expected["category"]
                    and finding.region.kind == expected["region_kind"]
                    and expected["parser_state_contains"] in finding.resulting_state
                ),
                None,
            )
            assert finding is not None, case["id"]
            assert finding.truncated_prefix == finding.valid_output[: finding.firing_offset], case["id"]
            assert finding.stop_sequence in finding.valid_output_prefix, case["id"]
            assert finding.resulting_structure.startswith(
                ("malformed", "prematurely accepted")
            ), case["id"]


def _structured_artifacts_for_case(case: dict[str, Any], tmp_path: Path):
    artifact_kind = case["artifact_kind"]
    if artifact_kind == "builtin-structural":
        return ()

    path = tmp_path / f"{case['id']}.json"
    path.write_text(json.dumps(case["artifact_json"], sort_keys=True), encoding="utf-8")
    if artifact_kind == "schema":
        return (
            SchemaArtifact(
                kind=ArtifactKind.SCHEMA,
                name=case["id"],
                location=ArtifactLocation(path=str(path)),
            ),
        )
    if artifact_kind == "tool-definition":
        return (
            ToolDefinitionArtifact(
                kind=ArtifactKind.TOOL_DEFINITION,
                name=case["id"],
                location=ArtifactLocation(path=str(path)),
            ),
        )
    raise AssertionError(f"unsupported real-world bug artifact kind: {artifact_kind}")
