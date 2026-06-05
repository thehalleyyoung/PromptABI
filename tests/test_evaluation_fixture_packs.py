import json
from pathlib import Path

import pytest

import promptabi
from promptabi.cli import main
from promptabi.evaluation_fixture_packs import (
    EVALUATION_FIXTURE_PACK_MANIFEST_VERSION,
    REQUIRED_EVALUATION_BUG_CLASSES,
    EvaluationFixturePackError,
    build_evaluation_fixture_pack_manifest,
    load_evaluation_fixture_pack,
    replay_evaluation_fixture_pack,
    write_evaluation_fixture_pack_manifest,
)


def test_evaluation_fixture_pack_replays_required_bug_classes() -> None:
    pack = load_evaluation_fixture_pack()
    results = replay_evaluation_fixture_pack()
    by_id = {result.case_id: result for result in results}

    assert set(pack.bug_classes) == set(REQUIRED_EVALUATION_BUG_CLASSES)
    assert all(result.passed for result in results)
    assert by_id["eval-stop-string-drift"].observed_rule_ids.count("evaluation-harness-stop-policy-mismatch") == 1
    assert "evaluation-harness-grading-parser-mismatch" in by_id["eval-parser-schema-drift"].observed_rule_ids
    assert "rag-payload-truncation" in by_id["eval-truncation-rag-loss"].observed_rule_ids
    assert "role-boundary-nonforgeability" in by_id["eval-role-boundary-forgery"].observed_rule_ids
    assert "evaluation-harness-tokenizer-mismatch" in by_id["eval-tokenizer-mismatch"].observed_rule_ids


def test_evaluation_fixture_pack_manifest_records_hashes_and_cli_writes(tmp_path: Path, capsys) -> None:
    manifest = build_evaluation_fixture_pack_manifest()
    output = tmp_path / "evaluation-fixtures.manifest.json"
    written = write_evaluation_fixture_pack_manifest(output)

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert written == manifest
    assert payload == manifest
    assert manifest["manifest_version"] == EVALUATION_FIXTURE_PACK_MANIFEST_VERSION
    assert manifest["case_count"] == 5
    assert manifest["all_cases_passed"] is True
    assert len(manifest["manifest_sha256"]) == 64
    assert all(len(entry["case_sha256"]) == 64 for entry in manifest["entries"])

    exit_code = main(["corpus", "evaluation-fixture-pack"])
    captured = capsys.readouterr()
    cli_payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert cli_payload["manifest_sha256"] == manifest["manifest_sha256"]

    cli_output = tmp_path / "manifest.json"
    exit_code = main(["corpus", "evaluation-fixture-pack", "--output", str(cli_output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert cli_output.is_file()
    assert "wrote evaluation fixture pack manifest" in captured.out


def test_public_api_exposes_evaluation_fixture_pack_manifest() -> None:
    manifest = promptabi.build_evaluation_fixture_pack_manifest()

    assert manifest["all_cases_passed"] is True
    assert set(manifest["bug_classes"]) == set(promptabi.REQUIRED_EVALUATION_BUG_CLASSES)


def test_evaluation_fixture_pack_validation_rejects_missing_bug_class(tmp_path: Path) -> None:
    pack_dir = tmp_path / "fixtures" / "evaluation_fixture_packs"
    pack_dir.mkdir(parents=True)
    config = tmp_path / "examples" / "evaluation-harness" / "unsafe.promptabi.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    pack = pack_dir / "pack.json"
    pack.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "methodology": "test",
                "cases": [
                    {
                        "id": "only-parser",
                        "bug_class": "parser",
                        "display_name": "Only parser",
                        "config": "examples/evaluation-harness/unsafe.promptabi.json",
                        "evidence": "test",
                        "labels": ["parser"],
                        "expected_rule_ids": ["evaluation-harness-answer-parser-mismatch"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationFixturePackError, match="missing required bug classes"):
        load_evaluation_fixture_pack(pack)
