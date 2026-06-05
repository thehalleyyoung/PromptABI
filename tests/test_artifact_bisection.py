import json
from pathlib import Path

import promptabi
from promptabi.artifact_bisection import (
    ArtifactDriftSurface,
    ArtifactRevision,
    artifact_revision_from_cli,
    bisect_artifact_drift,
    render_artifact_bisection_json,
    render_artifact_bisection_text,
)
from promptabi.cli import main


def test_tokenizer_drift_bisection_finds_exact_first_bad_revision(tmp_path: Path) -> None:
    baseline = tmp_path / "tok-r0"
    rev1 = tmp_path / "tok-r1"
    rev2 = tmp_path / "tok-r2"
    rev3 = tmp_path / "tok-r3"
    _write_tokenizer_revision(baseline, chat_template="{{ messages[0].content }}", eos_id=2, stop_strings=["</s>"])
    _write_tokenizer_revision(rev1, chat_template="{{ messages[0].content }}", eos_id=2, stop_strings=["</s>"])
    _write_tokenizer_revision(rev2, chat_template="{{ messages[0].content }}", eos_id=128009, stop_strings=["<|eot_id|>"])
    _write_tokenizer_revision(rev3, chat_template="{{ messages[0].content }}", eos_id=128009, stop_strings=["<|eot_id|>"])

    report = bisect_artifact_drift(
        ArtifactDriftSurface.TOKENIZER,
        baseline,
        (
            ArtifactRevision("tokenizer-1.0.1", str(rev1)),
            ArtifactRevision("tokenizer-1.1.0", str(rev2)),
            ArtifactRevision("tokenizer-1.1.1", str(rev3)),
        ),
        bad_fields=("eos_token_id",),
    )

    assert report.ok is False
    assert report.first_bad_index == 1
    assert report.first_bad_revision is not None
    assert report.first_bad_revision.label == "tokenizer-1.1.0"
    assert report.previous_good_revision is not None
    assert report.previous_good_revision.label == "tokenizer-1.0.1"
    assert [finding.field for finding in report.findings] == ["eos_token_id"]
    assert [probe.revision.label for probe in report.probes] == ["tokenizer-1.0.1", "tokenizer-1.1.0"]
    assert "first bad revision: #1 tokenizer-1.1.0" in render_artifact_bisection_text(report)


def test_template_bisection_only_treats_template_changes_as_bad(tmp_path: Path) -> None:
    baseline = tmp_path / "tok-r0"
    token_only = tmp_path / "tok-r1"
    template_change = tmp_path / "tok-r2"
    _write_tokenizer_revision(baseline, chat_template="{{ messages[0].content }}", eos_id=2, stop_strings=["</s>"])
    _write_tokenizer_revision(token_only, chat_template="{{ messages[0].content }}", eos_id=128009, stop_strings=["</s>"])
    _write_tokenizer_revision(template_change, chat_template="<s>{{ messages[0].content }}", eos_id=128009, stop_strings=["</s>"])

    report = bisect_artifact_drift(
        "template",
        baseline,
        (
            ArtifactRevision("token-id-only", str(token_only)),
            ArtifactRevision("template-wrapper", str(template_change)),
        ),
    )

    assert report.first_bad_revision is not None
    assert report.first_bad_revision.label == "template-wrapper"
    assert {finding.field for finding in report.findings} == {"chat_template_sha256", "chat_template_length"}


def test_schema_drift_bisection_cli_reports_json_pointer_regression(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "schema-r0.json"
    rev1 = tmp_path / "schema-r1.json"
    rev2 = tmp_path / "schema-r2.json"
    _write_schema(baseline, required=["answer"])
    _write_schema(rev1, required=["answer"])
    _write_schema(rev2, required=["answer", "citation"])

    exit_code = main(
        [
            "release",
            "drift-bisect",
            "--surface",
            "schema",
            "--baseline",
            str(baseline),
            "--revision",
            f"schema-1={rev1}",
            "--revision",
            f"schema-2={rev2}",
            "--bad-field",
            "/required/1",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert captured.err == ""
    assert payload["first_bad_revision"]["label"] == "schema-2"
    assert payload["findings"][0]["field"] == "/required/1"
    assert payload["findings"][0]["current"] == "citation"


def test_artifact_bisection_renderers_and_public_api_are_stable(tmp_path: Path) -> None:
    baseline = tmp_path / "provider-r0.json"
    rev1 = tmp_path / "provider-r1.json"
    baseline.write_text(json.dumps({"provider": "openai", "max_input_tokens": 8192}), encoding="utf-8")
    rev1.write_text(json.dumps({"provider": "openai", "max_input_tokens": 4096}), encoding="utf-8")

    revision = artifact_revision_from_cli(f"provider-2026-06={rev1}")
    report = bisect_artifact_drift("provider", baseline, (revision,), bad_fields=("json-field-changed",))
    payload = json.loads(render_artifact_bisection_json(report))
    api_payload = json.loads(
        promptabi.artifact_drift_bisection(
            "provider",
            baseline,
            (revision,),
            bad_fields=("json-field-changed",),
            output_format="json",
        )
    )

    assert payload == api_payload
    assert payload["surface"] == "provider"
    assert payload["first_bad_revision"]["label"] == "provider-2026-06"
    assert "max_input_tokens" in render_artifact_bisection_text(report)


def _write_schema(path: Path, *, required: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "object",
                "required": required,
                "properties": {
                    "answer": {"type": "string"},
                    "citation": {"type": "string"},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_tokenizer_revision(
    root: Path,
    *,
    chat_template: str,
    eos_id: int,
    stop_strings: list[str],
) -> None:
    root.mkdir()
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "eos_token_id": eos_id,
                "chat_template": chat_template,
                "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps({"normalizer": {"type": "NFC"}, "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}]}, sort_keys=True),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"stop_strings": stop_strings, "eos_token_id": eos_id}, sort_keys=True),
        encoding="utf-8",
    )
