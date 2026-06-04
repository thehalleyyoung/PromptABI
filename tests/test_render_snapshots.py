import json

from promptabi.artifacts import ArtifactBundle, ArtifactKind, ArtifactLocation, ArtifactProvenance, SchemaArtifact
from promptabi.config import VerificationConfig
from promptabi.diagnostics import (
    ArtifactRef,
    CheckMode,
    Diagnostic,
    DiagnosticSeverity,
    SourceSpan,
    WitnessStep,
    WitnessTrace,
)
from promptabi.render import SarifRenderOptions, render_github_annotations, render_json, render_sarif, render_text
from promptabi.session import VerificationResult


def _snapshot_result() -> VerificationResult:
    schema = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="answer-schema",
        location=ArtifactLocation(path="schemas/answer.schema.json"),
        provenance=ArtifactProvenance(sha256="0123456789abcdef"),
        dialect="json-schema-2020-12",
        source_span=SourceSpan(
            path="promptabi.json",
            start_line=5,
            start_column=15,
            end_line=9,
            end_column=6,
        ),
    )
    artifact_ref = schema.to_ref()
    return VerificationResult(
        config=VerificationConfig(
            name="snapshot-contract",
            artifacts={"answer-schema": "schemas/answer.schema.json"},
            artifact_bundle=ArtifactBundle((schema,)),
            checks=("artifact-missing", "repository-skeleton"),
            max_context_tokens=2048,
        ),
        diagnostics=(
            Diagnostic(
                rule_id="artifact-missing",
                severity=DiagnosticSeverity.ERROR,
                message="artifact 'answer-schema' does not exist",
                artifact=artifact_ref,
                span=SourceSpan(
                    path="promptabi.json",
                    start_line=5,
                    start_column=15,
                    end_line=9,
                    end_column=6,
                ),
                witness=WitnessTrace(
                    summary="The configured local artifact path was resolved but was absent on disk.",
                    steps=(
                        WitnessStep(
                            action="resolve artifact path",
                            output="schemas/answer.schema.json",
                        ),
                        WitnessStep(action="check local filesystem", output="missing"),
                    ),
                    artifacts=(artifact_ref,),
                ),
                suggestions=("Check the path relative to the PromptABI config file.",),
                check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
            ),
            Diagnostic(
                rule_id="artifact-unpinned",
                severity=DiagnosticSeverity.WARNING,
                message="artifact 'answer-schema' is not pinned by sha256",
                artifact=artifact_ref,
                span=SourceSpan(path="schemas/answer.schema.json"),
                witness=WitnessTrace(
                    summary="The artifact loaded, but its provenance is not fully reproducible.",
                    steps=(WitnessStep(action="inspect provenance", output="sha256 missing"),),
                    artifacts=(artifact_ref,),
                ),
                suggestions=("Add a sha256 pin for reproducible verification.",),
                check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
            ),
            Diagnostic(
                rule_id="repository-skeleton",
                severity=DiagnosticSeverity.INFO,
                message="PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired.",
                witness=WitnessTrace(
                    summary="The verification session constructed a typed config and produced deterministic output.",
                    steps=(
                        WitnessStep(
                            action="load JSON config",
                            input="snapshot-contract",
                            output="1 artifacts",
                        ),
                        WitnessStep(action="render stable diagnostics"),
                    ),
                ),
                check_modes=(CheckMode.HEURISTIC,),
            ),
        ),
    )


def test_text_diagnostic_output_matches_snapshot() -> None:
    assert render_text(_snapshot_result()) == (
        "PromptABI verification: snapshot-contract\n"
        "checks: artifact-missing, repository-skeleton\n"
        "status: FAIL\n"
        "ERROR artifact-missing [complete, sound]: artifact 'answer-schema' does not exist\n"
        "  fingerprint: 5bb6ab676b2e7756\n"
        "  artifact: schema:answer-schema (schemas/answer.schema.json)\n"
        "  span: promptabi.json:5:15-9:6\n"
        "  witness: The configured local artifact path was resolved but was absent on disk.\n"
        "    1. resolve artifact path | output: schemas/answer.schema.json\n"
        "    2. check local filesystem | output: missing\n"
        "  suggestion: Check the path relative to the PromptABI config file.\n"
        "WARNING artifact-unpinned [complete, sound]: artifact 'answer-schema' is not pinned by sha256\n"
        "  fingerprint: 55d4ad24d45eb4b1\n"
        "  artifact: schema:answer-schema (schemas/answer.schema.json)\n"
        "  span: schemas/answer.schema.json:1:1\n"
        "  witness: The artifact loaded, but its provenance is not fully reproducible.\n"
        "    1. inspect provenance | output: sha256 missing\n"
        "  suggestion: Add a sha256 pin for reproducible verification.\n"
        "INFO repository-skeleton [heuristic]: PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired.\n"
        "  fingerprint: 966044f6134aa008\n"
        "  witness: The verification session constructed a typed config and produced deterministic output.\n"
        "    1. load JSON config | input: snapshot-contract | output: 1 artifacts\n"
        "    2. render stable diagnostics\n"
    )


def test_json_diagnostic_output_matches_structured_snapshot() -> None:
    assert json.loads(render_json(_snapshot_result())) == {
        "config": {
            "artifact_bundle": {
                "artifacts": [
                    {
                        "dialect": "json-schema-2020-12",
                        "kind": "schema",
                        "location": {"path": "schemas/answer.schema.json"},
                        "name": "answer-schema",
                        "provenance": {"sha256": "0123456789abcdef"},
                        "source_span": {
                            "end_column": 6,
                            "end_line": 9,
                            "path": "promptabi.json",
                            "start_column": 15,
                            "start_line": 5,
                        },
                    }
                ]
            },
            "artifacts": {"answer-schema": "schemas/answer.schema.json"},
            "checks": ["artifact-missing", "repository-skeleton"],
            "max_context_tokens": 2048,
            "name": "snapshot-contract",
        },
        "diagnostics": [
            {
                "artifact": {
                    "kind": "schema",
                    "name": "answer-schema",
                    "path": "schemas/answer.schema.json",
                    "sha256": "0123456789abcdef",
                },
                "check_modes": ["complete", "sound"],
                "fingerprint": "5bb6ab676b2e7756",
                "message": "artifact 'answer-schema' does not exist",
                "rule_id": "artifact-missing",
                "severity": "error",
                "span": {
                    "end_column": 6,
                    "end_line": 9,
                    "path": "promptabi.json",
                    "start_column": 15,
                    "start_line": 5,
                },
                "suggestions": ["Check the path relative to the PromptABI config file."],
                "witness": {
                    "artifacts": [
                        {
                            "kind": "schema",
                            "name": "answer-schema",
                            "path": "schemas/answer.schema.json",
                            "sha256": "0123456789abcdef",
                        }
                    ],
                    "steps": [
                        {
                            "action": "resolve artifact path",
                            "output": "schemas/answer.schema.json",
                        },
                        {"action": "check local filesystem", "output": "missing"},
                    ],
                    "summary": "The configured local artifact path was resolved but was absent on disk.",
                },
            },
            {
                "artifact": {
                    "kind": "schema",
                    "name": "answer-schema",
                    "path": "schemas/answer.schema.json",
                    "sha256": "0123456789abcdef",
                },
                "check_modes": ["complete", "sound"],
                "fingerprint": "55d4ad24d45eb4b1",
                "message": "artifact 'answer-schema' is not pinned by sha256",
                "rule_id": "artifact-unpinned",
                "severity": "warning",
                "span": {
                    "path": "schemas/answer.schema.json",
                    "start_column": 1,
                    "start_line": 1,
                },
                "suggestions": ["Add a sha256 pin for reproducible verification."],
                "witness": {
                    "artifacts": [
                        {
                            "kind": "schema",
                            "name": "answer-schema",
                            "path": "schemas/answer.schema.json",
                            "sha256": "0123456789abcdef",
                        }
                    ],
                    "steps": [{"action": "inspect provenance", "output": "sha256 missing"}],
                    "summary": "The artifact loaded, but its provenance is not fully reproducible.",
                },
            },
            {
                "check_modes": ["heuristic"],
                "fingerprint": "966044f6134aa008",
                "message": "PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired.",
                "rule_id": "repository-skeleton",
                "severity": "info",
                "suggestions": [],
                "witness": {
                    "artifacts": [],
                    "steps": [
                        {
                            "action": "load JSON config",
                            "input": "snapshot-contract",
                            "output": "1 artifacts",
                        },
                        {"action": "render stable diagnostics"},
                    ],
                    "summary": "The verification session constructed a typed config and produced deterministic output.",
                },
            },
        ],
        "ok": False,
    }


def test_sarif_diagnostic_output_matches_structured_snapshot() -> None:
    assert json.loads(render_sarif(_snapshot_result())) == {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "results": [
                    {
                        "level": "error",
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "promptabi.json"},
                                    "region": {
                                        "endColumn": 6,
                                        "endLine": 9,
                                        "startColumn": 15,
                                        "startLine": 5,
                                    },
                                }
                            }
                        ],
                        "message": {"text": "artifact 'answer-schema' does not exist"},
                        "partialFingerprints": {"promptabiFingerprint": "5bb6ab676b2e7756"},
                        "properties": {
                            "checkModes": ["complete", "sound"],
                            "severity": "error",
                            "suggestions": ["Check the path relative to the PromptABI config file."],
                            "witness": {
                                "artifacts": [
                                    {
                                        "kind": "schema",
                                        "name": "answer-schema",
                                        "path": "schemas/answer.schema.json",
                                        "sha256": "0123456789abcdef",
                                    }
                                ],
                                "steps": [
                                    {
                                        "action": "resolve artifact path",
                                        "output": "schemas/answer.schema.json",
                                    },
                                    {"action": "check local filesystem", "output": "missing"},
                                ],
                                "summary": "The configured local artifact path was resolved but was absent on disk.",
                            },
                        },
                        "ruleId": "artifact-missing",
                    },
                    {
                        "level": "warning",
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "schemas/answer.schema.json"},
                                    "region": {"startColumn": 1, "startLine": 1},
                                }
                            }
                        ],
                        "message": {"text": "artifact 'answer-schema' is not pinned by sha256"},
                        "partialFingerprints": {"promptabiFingerprint": "55d4ad24d45eb4b1"},
                        "properties": {
                            "checkModes": ["complete", "sound"],
                            "severity": "warning",
                            "suggestions": ["Add a sha256 pin for reproducible verification."],
                            "witness": {
                                "artifacts": [
                                    {
                                        "kind": "schema",
                                        "name": "answer-schema",
                                        "path": "schemas/answer.schema.json",
                                        "sha256": "0123456789abcdef",
                                    }
                                ],
                                "steps": [
                                    {"action": "inspect provenance", "output": "sha256 missing"}
                                ],
                                "summary": "The artifact loaded, but its provenance is not fully reproducible.",
                            },
                        },
                        "ruleId": "artifact-unpinned",
                    },
                    {
                        "level": "note",
                        "message": {
                            "text": "PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired."
                        },
                        "partialFingerprints": {"promptabiFingerprint": "966044f6134aa008"},
                        "properties": {
                            "checkModes": ["heuristic"],
                            "severity": "info",
                            "suggestions": [],
                            "witness": {
                                "artifacts": [],
                                "steps": [
                                    {
                                        "action": "load JSON config",
                                        "input": "snapshot-contract",
                                        "output": "1 artifacts",
                                    },
                                    {"action": "render stable diagnostics"},
                                ],
                                "summary": "The verification session constructed a typed config and produced deterministic output.",
                            },
                        },
                        "ruleId": "repository-skeleton",
                    },
                ],
                "tool": {
                    "driver": {
                        "informationUri": "https://github.com/thehalleyyoung/PromptABI",
                        "name": "PromptABI",
                        "rules": [
                            {
                                "defaultConfiguration": {"level": "error"},
                                "id": "artifact-missing",
                                "name": "artifact-missing",
                                "properties": {
                                    "checkModes": ["complete", "sound"],
                                    "precision": "high",
                                },
                                "shortDescription": {
                                    "text": "artifact 'answer-schema' does not exist"
                                },
                            },
                            {
                                "defaultConfiguration": {"level": "warning"},
                                "id": "artifact-unpinned",
                                "name": "artifact-unpinned",
                                "properties": {
                                    "checkModes": ["complete", "sound"],
                                    "precision": "high",
                                },
                                "shortDescription": {
                                    "text": "artifact 'answer-schema' is not pinned by sha256"
                                },
                            },
                            {
                                "defaultConfiguration": {"level": "note"},
                                "id": "repository-skeleton",
                                "name": "repository-skeleton",
                                "properties": {
                                    "checkModes": ["heuristic"],
                                    "precision": "high",
                                },
                                "shortDescription": {
                                    "text": "PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired."
                                },
                            },
                        ],
                    }
                },
            }
        ],
        "version": "2.1.0",
    }


def test_sarif_github_code_scanning_options_rewrite_locations_and_suppressions(tmp_path) -> None:
    repo = tmp_path / "repo"
    artifact_path = repo / "schemas" / "answer.schema.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")
    artifact_ref = ArtifactRef(
        kind="schema",
        name="answer-schema",
        path=str(artifact_path),
        sha256="0123456789abcdef",
    )
    result = VerificationResult(
        config=VerificationConfig(name="github-code-scanning", checks=("artifact-unpinned",)),
        diagnostics=(
            Diagnostic(
                rule_id="artifact-unpinned",
                severity=DiagnosticSeverity.WARNING,
                message="artifact 'answer-schema' is not pinned by sha256",
                artifact=artifact_ref,
                span=SourceSpan(
                    path=str(artifact_path),
                    start_line=2,
                    start_column=3,
                    end_line=2,
                    end_column=9,
                ),
                suggestions=("Add a sha256 pin for reproducible verification.",),
                check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
                properties=(
                    ("owner", "platform"),
                    ("sarif_suppression_justification", "accepted test fixture risk"),
                ),
            ),
        ),
    )

    payload = json.loads(
        render_sarif(
            result,
            options=SarifRenderOptions(
                category="pull-request",
                checkout_uri_base=repo,
                include_invocation=True,
                command_line="promptabi verify --format sarif",
                working_directory=repo,
            ),
        )
    )

    run = payload["runs"][0]
    assert run["automationDetails"]["id"] == "pull-request/"
    assert run["originalUriBaseIds"]["PROJECTROOT"]["uri"].startswith("file://")
    assert run["invocations"] == [
        {
            "commandLine": "promptabi verify --format sarif",
            "executionSuccessful": True,
            "workingDirectory": {"uri": repo.resolve().as_uri() + "/"},
        }
    ]
    result_payload = run["results"][0]
    artifact_location = result_payload["locations"][0]["physicalLocation"]["artifactLocation"]
    assert artifact_location == {"uri": "schemas/answer.schema.json", "uriBaseId": "PROJECTROOT"}
    assert result_payload["suppressions"] == [
        {"kind": "external", "justification": "accepted test fixture risk"}
    ]
    assert result_payload["properties"]["owner"] == "platform"
    assert "sarif_suppression_justification" not in result_payload["properties"]
    assert "promptabiLocationFingerprint" in result_payload["partialFingerprints"]
    assert "primaryLocationLineHash" not in result_payload["partialFingerprints"]


def test_github_annotations_escape_messages_and_rewrite_locations(tmp_path) -> None:
    repo = tmp_path / "repo"
    artifact_path = repo / "schemas" / "answer.schema.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")
    result = VerificationResult(
        config=VerificationConfig(name="github-annotations", checks=("schema-warning",)),
        diagnostics=(
            Diagnostic(
                rule_id="schema-warning",
                severity=DiagnosticSeverity.ERROR,
                message="bad schema\nneeds escaping, urgently",
                span=SourceSpan(
                    path=str(artifact_path),
                    start_line=7,
                    start_column=5,
                    end_line=8,
                    end_column=2,
                ),
            ),
        ),
    )

    assert render_github_annotations(result, checkout_uri_base=repo) == (
        "::error title=schema-warning,file=schemas/answer.schema.json,line=7,col=5,"
        "endLine=8,endColumn=2::bad schema%0Aneeds escaping, urgently\n"
    )
