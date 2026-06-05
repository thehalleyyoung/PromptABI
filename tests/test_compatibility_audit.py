import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.compatibility_audit import (
    CompatibilityAuditStatus,
    normalize_candidate_versions,
    render_compatibility_audit_json,
    render_compatibility_audit_text,
    run_compatibility_audit,
)


PINNED_VERSIONS = {
    "tokenizer": "seed-v1",
    "template": "seed-v1",
    "provider": "provider-fixtures-v1",
    "grammar": "grammar-differential-v1",
    "framework": "structured-schemas-v1",
}


def test_compatibility_audit_replays_real_pinned_fixture_surfaces() -> None:
    report = run_compatibility_audit(PINNED_VERSIONS)
    payload = report.to_dict()
    targets = {target["surface"]: target for target in payload["targets"]}

    assert report.ok is True
    assert payload["coverage_count"] >= 30
    assert set(targets) == {"tokenizer", "template", "provider", "grammar", "framework"}
    assert targets["tokenizer"]["observed_versions"] == ["seed-v1"]
    assert targets["template"]["coverage_count"] >= 10
    assert targets["provider"]["evidence"]["provider_families"]
    assert targets["grammar"]["evidence"]["mismatches"] == 0
    assert targets["framework"]["evidence"]["entry_types"] == ["grammar", "schema", "tool-definition"]


def test_compatibility_audit_abstains_for_unpinned_candidate_version() -> None:
    report = run_compatibility_audit({**PINNED_VERSIONS, "tokenizer": "new-tokenizer-v999"})
    tokenizer = next(target for target in report.targets if target.surface == "tokenizer")

    assert report.ok is False
    assert tokenizer.status is CompatibilityAuditStatus.ABSTAINED
    assert tokenizer.observed_versions == ("seed-v1",)
    assert "no pinned seed-corpus fixture" in tokenizer.failures[0]


def test_compatibility_audit_renderers_and_public_api_are_stable() -> None:
    report = run_compatibility_audit(PINNED_VERSIONS)
    text = render_compatibility_audit_text(report)
    payload = json.loads(render_compatibility_audit_json(report))
    api_payload = json.loads(promptabi.compatibility_audit(PINNED_VERSIONS, output_format="json"))

    assert "PromptABI compatibility audit" in text
    assert "- provider: PASS" in text
    assert payload["ok"] is True
    assert api_payload == payload
    assert render_compatibility_audit_json(report) == render_compatibility_audit_json(report)


def test_compatibility_audit_cli_writes_json_and_unknown_versions_fail(tmp_path: Path, capsys) -> None:
    output = tmp_path / "compatibility-audit.json"
    args = ["release", "compatibility-audit", "--format", "json", "--output", str(output)]
    for surface, version in PINNED_VERSIONS.items():
        args.extend(["--candidate-version", f"{surface}={version}"])

    exit_code = main(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "wrote compatibility audit report" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True

    exit_code = main(["release", "compatibility-audit", "--candidate-version", "all=unknown-v1"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "ABSTAINED" in captured.out
    assert captured.err == ""


def test_candidate_versions_accept_all_default_and_require_missing_surfaces() -> None:
    assert normalize_candidate_versions({"all": "fixture-v1"}) == {
        "tokenizer": "fixture-v1",
        "template": "fixture-v1",
        "provider": "fixture-v1",
        "grammar": "fixture-v1",
        "framework": "fixture-v1",
    }
