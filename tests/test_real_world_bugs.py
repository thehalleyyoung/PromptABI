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


CORPUS_PATH = Path("fixtures/real_world_bugs/corpus.json")


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
