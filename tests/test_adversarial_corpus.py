import json
from pathlib import Path

import promptabi
from promptabi import (
    REQUIRED_ADVERSARIAL_SURFACES,
    AdversarialCorpusReport,
    AdversarialSurface,
    build_adversarial_corpus_manifest,
    generate_adversarial_cases,
    generate_adversarial_corpus,
    replay_adversarial_corpus,
    write_adversarial_corpus_manifest,
)
from promptabi.cli import main


def test_adversarial_corpus_generates_every_required_attack_surface() -> None:
    cases = generate_adversarial_corpus()

    assert {case.surface for case in cases} == set(REQUIRED_ADVERSARIAL_SURFACES)
    assert len(cases) == 7
    assert {case.payload_sha256 for case in cases}
    assert all(case.expected_rule_ids for case in cases)
    assert AdversarialSurface.ROLE_DELIMITERS in {case.surface for case in cases}


def test_adversarial_corpus_replays_against_real_analyzers() -> None:
    cases = generate_adversarial_corpus()
    replays = replay_adversarial_corpus(cases)
    observed_by_surface = {replay.case.surface: set(replay.observed_rule_ids) for replay in replays}

    assert all(replay.passed for replay in replays)
    assert "role-boundary-nonforgeability" in observed_by_surface[AdversarialSurface.ROLE_DELIMITERS]
    assert "tokenizer-control-token-reachable" in observed_by_surface[AdversarialSurface.SPECIAL_TOKENS]
    assert "tokenizer-normalization-drift" in observed_by_surface[AdversarialSurface.UNICODE_NORMALIZATION]
    assert "stop-overreachability" in observed_by_surface[AdversarialSurface.JSON_ESCAPING]
    assert "stop-overreachability" in observed_by_surface[AdversarialSurface.MARKDOWN_FENCES]
    assert "tool-schema-open-string-arguments" in observed_by_surface[AdversarialSurface.XML_TOOL_TAGS]
    assert "provider-migration" in observed_by_surface[AdversarialSurface.PROVIDER_ENVELOPES]


def test_adversarial_corpus_manifest_records_hashes_and_replay_results(tmp_path: Path) -> None:
    manifest = build_adversarial_corpus_manifest()
    output = tmp_path / "adversarial-corpus.manifest.json"
    written = write_adversarial_corpus_manifest(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert written == manifest
    assert payload == manifest
    assert manifest["manifest_version"] == 1
    assert manifest["case_count"] == 7
    assert manifest["all_cases_passed"] is True
    assert len(manifest["manifest_sha256"]) == 64
    assert {entry["surface"] for entry in manifest["cases"]} == {surface.value for surface in REQUIRED_ADVERSARIAL_SURFACES}
    assert all(entry["payload_sha256"] for entry in manifest["cases"])


def test_adversarial_corpus_cli_and_public_api(tmp_path: Path, capsys) -> None:
    report = generate_adversarial_cases()
    rendered = promptabi.generate_adversarial_cases(output_format="json")

    assert isinstance(report, AdversarialCorpusReport)
    assert json.loads(rendered)["all_cases_passed"] is True

    exit_code = main(["corpus", "adversarial", "--format", "text"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "PromptABI adversarial corpus" in captured.out

    output = tmp_path / "manifest.json"
    exit_code = main(["corpus", "adversarial", "--output", str(output)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wrote adversarial corpus manifest" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["all_cases_passed"] is True
