import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.release import (
    LTSReleaseStatus,
    build_lts_release_plan,
    lts_item_from_string,
    render_lts_release_plan_json,
    render_lts_release_plan_text,
)


LTS_ITEMS = (
    lts_item_from_string("checker_fix:role-boundary-nonforgeability"),
    lts_item_from_string("security_patch:security-model"),
    lts_item_from_string("corpus_update:seed-v1"),
    lts_item_from_string("compatibility_metadata:provider-fixtures-v1"),
)


def test_lts_release_plan_routes_all_required_maintenance_lanes() -> None:
    plan = build_lts_release_plan(
        LTS_ITEMS,
        series="1.0",
        base_version="1.0.0",
        target_version="1.0.1",
    )
    payload = plan.to_dict()
    lanes = {decision["item"]["category"]: decision["lane"] for decision in payload["decisions"]}

    assert plan.ok is True
    assert lanes == {
        "checker_fix": "checker-backport",
        "security_patch": "security-hotfix",
        "corpus_update": "corpus-refresh",
        "compatibility_metadata": "compatibility-metadata",
    }
    assert {check["name"] for check in payload["checks"]} == {
        "version-window",
        "maintenance-category-coverage",
        "compatibility-metadata",
        "corpus-update-evidence",
        "release-metadata",
    }
    assert len(payload["manifest_sha256"]) == 64
    assert all(decision.status is LTSReleaseStatus.PASS for decision in plan.decisions)


def test_lts_release_plan_fails_missing_category_without_raising() -> None:
    plan = build_lts_release_plan(
        LTS_ITEMS[:-1],
        series="1.0",
        base_version="1.0.0",
        target_version="1.0.1",
    )
    checks = {check.name: check for check in plan.checks}

    assert plan.ok is False
    assert checks["maintenance-category-coverage"].status is LTSReleaseStatus.FAIL
    assert checks["maintenance-category-coverage"].evidence[-1] == ("missing", ["compatibility_metadata"])


def test_lts_release_plan_json_is_deterministic_for_reordered_inputs() -> None:
    forward = build_lts_release_plan(
        LTS_ITEMS,
        series="1.0",
        base_version="1.0.0",
        target_version="1.0.1",
    )
    reversed_plan = build_lts_release_plan(
        tuple(reversed(LTS_ITEMS)),
        series="1.0",
        base_version="1.0.0",
        target_version="1.0.1",
    )

    assert render_lts_release_plan_json(forward) == render_lts_release_plan_json(reversed_plan)
    assert forward.manifest_sha256 == reversed_plan.manifest_sha256


def test_lts_release_plan_renderers_public_api_and_cli(tmp_path: Path, capsys) -> None:
    output = tmp_path / "lts-plan.json"
    args = [
        "release",
        "lts-plan",
        "--series",
        "1.0",
        "--base-version",
        "1.0.0",
        "--target-version",
        "1.0.1",
    ]
    for item in LTS_ITEMS:
        args.extend(["--item", f"{item.category}:{item.item_id}"])
    args.extend(["--format", "json", "--output", str(output)])

    exit_code = main(args)
    captured = capsys.readouterr()
    api_payload = json.loads(promptabi.lts_release_plan(LTS_ITEMS, series="1.0", base_version="1.0.0", target_version="1.0.1", output_format="json"))
    text = render_lts_release_plan_text(
        build_lts_release_plan(LTS_ITEMS, series="1.0", base_version="1.0.0", target_version="1.0.1")
    )

    assert exit_code == 0
    assert captured.err == ""
    assert "wrote LTS release plan" in captured.out
    assert json.loads(output.read_text(encoding="utf-8")) == api_payload
    assert "PromptABI LTS release plan" in text
    assert "checker_fix:role-boundary-nonforgeability" in text


def test_lts_release_plan_cli_malformed_item_returns_parse_error(capsys) -> None:
    exit_code = main(
        [
            "release",
            "lts-plan",
            "--series",
            "1.0",
            "--base-version",
            "1.0.0",
            "--target-version",
            "1.0.1",
            "--item",
            "unknown:thing",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "unknown LTS item category" in captured.err


def test_release_workflow_runs_lts_plan_gate() -> None:
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "promptabi release lts-plan" in workflow
    assert "--item checker_fix:role-boundary-nonforgeability" in workflow
    assert "--item security_patch:security-model" in workflow
