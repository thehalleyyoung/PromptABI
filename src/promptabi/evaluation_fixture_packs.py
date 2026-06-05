"""Replayable evaluation fixture packs with known benchmark-interface bugs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .session import VerificationSession


DEFAULT_EVALUATION_FIXTURE_PACK_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "evaluation_fixture_packs" / "pack.json"
EVALUATION_FIXTURE_PACK_MANIFEST_VERSION = 1
REQUIRED_EVALUATION_BUG_CLASSES = frozenset(
    {
        "stop-string",
        "parser",
        "truncation",
        "role-boundary",
        "tokenizer-mismatch",
    }
)


class EvaluationFixturePackError(ValueError):
    """Raised when an evaluation fixture pack is incomplete or fails replay."""


@dataclass(frozen=True, slots=True)
class EvaluationFixtureCase:
    """One labeled eval-harness or benchmark-interface regression fixture."""

    case_id: str
    bug_class: str
    display_name: str
    config_path: Path
    expected_rule_ids: tuple[str, ...]
    expected_absent_rule_ids: tuple[str, ...]
    labels: tuple[str, ...]
    evidence: str

    def to_manifest_entry(self, result: "EvaluationFixtureResult") -> dict[str, object]:
        return {
            "id": self.case_id,
            "bug_class": self.bug_class,
            "display_name": self.display_name,
            "config": result.config,
            "labels": list(self.labels),
            "expected_rule_ids": list(self.expected_rule_ids),
            "expected_absent_rule_ids": list(self.expected_absent_rule_ids),
            "observed_rule_ids": list(result.observed_rule_ids),
            "passed": result.passed,
            "evidence": self.evidence,
            "evidence_summary": result.evidence_summary,
            "case_sha256": _stable_json_hash(
                {
                    "bug_class": self.bug_class,
                    "config": result.config,
                    "expected_absent_rule_ids": self.expected_absent_rule_ids,
                    "expected_rule_ids": self.expected_rule_ids,
                    "labels": self.labels,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class EvaluationFixtureResult:
    """Replay result for one evaluation fixture case."""

    case_id: str
    bug_class: str
    config: str
    observed_rule_ids: tuple[str, ...]
    passed: bool
    evidence_summary: str
    diagnostic_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "bug_class": self.bug_class,
            "config": self.config,
            "observed_rule_ids": list(self.observed_rule_ids),
            "passed": self.passed,
            "evidence_summary": self.evidence_summary,
            "diagnostic_count": self.diagnostic_count,
        }


@dataclass(frozen=True, slots=True)
class EvaluationFixturePack:
    """A deterministic collection of benchmark-interface bug fixtures."""

    path: Path
    methodology: str
    cases: tuple[EvaluationFixtureCase, ...]

    @property
    def bug_classes(self) -> tuple[str, ...]:
        return tuple(sorted({case.bug_class for case in self.cases}))

    def replay(self) -> tuple[EvaluationFixtureResult, ...]:
        repo_root = self.path.parent.parent.parent
        return tuple(_replay_case(case, repo_root) for case in self.cases)

    def manifest(self) -> dict[str, object]:
        results = self.replay()
        by_id = {result.case_id: result for result in results}
        entries = [case.to_manifest_entry(by_id[case.case_id]) for case in self.cases]
        manifest: dict[str, object] = {
            "manifest_version": EVALUATION_FIXTURE_PACK_MANIFEST_VERSION,
            "methodology": self.methodology,
            "path": str(self.path),
            "case_count": len(entries),
            "bug_classes": list(self.bug_classes),
            "required_bug_classes": sorted(REQUIRED_EVALUATION_BUG_CLASSES),
            "all_cases_passed": all(result.passed for result in results),
            "entries": entries,
        }
        manifest["manifest_sha256"] = _stable_json_hash(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        return manifest


def load_evaluation_fixture_pack(path: str | Path | None = None) -> EvaluationFixturePack:
    """Load and validate the maintained evaluation fixture pack."""

    pack_path = Path(path) if path is not None else DEFAULT_EVALUATION_FIXTURE_PACK_PATH
    payload = _read_json_object(pack_path)
    if payload.get("manifest_version") != EVALUATION_FIXTURE_PACK_MANIFEST_VERSION:
        raise EvaluationFixturePackError(f"{pack_path} has unsupported evaluation fixture manifest_version")
    methodology = payload.get("methodology")
    if not isinstance(methodology, str) or not methodology:
        raise EvaluationFixturePackError(f"{pack_path} field 'methodology' must be a non-empty string")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationFixturePackError(f"{pack_path} field 'cases' must be a non-empty list")
    repo_root = pack_path.parent.parent.parent
    cases = tuple(sorted((_case_from_mapping(pack_path, repo_root, item) for item in raw_cases), key=lambda item: item.case_id))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationFixturePackError(f"{pack_path} contains duplicate case ids")
    missing = REQUIRED_EVALUATION_BUG_CLASSES.difference(case.bug_class for case in cases)
    if missing:
        raise EvaluationFixturePackError(
            "evaluation fixture pack is missing required bug classes: " + ", ".join(sorted(missing))
        )
    return EvaluationFixturePack(path=pack_path, methodology=methodology, cases=cases)


def replay_evaluation_fixture_pack(path: str | Path | None = None) -> tuple[EvaluationFixtureResult, ...]:
    """Replay every evaluation fixture through real PromptABI verification configs."""

    results = load_evaluation_fixture_pack(path).replay()
    failures = tuple(result for result in results if not result.passed)
    if failures:
        failed = ", ".join(f"{result.case_id}: {result.evidence_summary}" for result in failures)
        raise EvaluationFixturePackError(f"evaluation fixture pack replay failed: {failed}")
    return results


def build_evaluation_fixture_pack_manifest(path: str | Path | None = None) -> dict[str, object]:
    """Validate, replay, and return a deterministic evaluation fixture manifest."""

    manifest = load_evaluation_fixture_pack(path).manifest()
    if not manifest["all_cases_passed"]:
        failed = ", ".join(entry["id"] for entry in manifest["entries"] if not entry["passed"])  # type: ignore[index]
        raise EvaluationFixturePackError(f"evaluation fixture pack replay failed: {failed}")
    return manifest


def write_evaluation_fixture_pack_manifest(output: str | Path, *, path: str | Path | None = None) -> dict[str, object]:
    """Write the deterministic evaluation fixture pack manifest."""

    manifest = build_evaluation_fixture_pack_manifest(path)
    output_path = Path(output)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _case_from_mapping(pack_path: Path, repo_root: Path, raw: object) -> EvaluationFixtureCase:
    if not isinstance(raw, dict):
        raise EvaluationFixturePackError(f"{pack_path} cases must be JSON objects")
    case_id = _required_string(pack_path, raw, "id")
    bug_class = _required_string(pack_path, raw, "bug_class")
    if bug_class not in REQUIRED_EVALUATION_BUG_CLASSES:
        raise EvaluationFixturePackError(f"{pack_path} case {case_id!r} has unsupported bug_class {bug_class!r}")
    display_name = _required_string(pack_path, raw, "display_name")
    config = _required_string(pack_path, raw, "config")
    config_path = (repo_root / config).resolve()
    if not config_path.is_file():
        raise EvaluationFixturePackError(f"{pack_path} case {case_id!r} config does not exist: {config}")
    return EvaluationFixtureCase(
        case_id=case_id,
        bug_class=bug_class,
        display_name=display_name,
        config_path=config_path,
        expected_rule_ids=_string_tuple(raw.get("expected_rule_ids"), pack_path, case_id, "expected_rule_ids", allow_empty=False),
        expected_absent_rule_ids=_string_tuple(raw.get("expected_absent_rule_ids", ()), pack_path, case_id, "expected_absent_rule_ids"),
        labels=_string_tuple(raw.get("labels"), pack_path, case_id, "labels", allow_empty=False),
        evidence=_required_string(pack_path, raw, "evidence"),
    )


def _replay_case(case: EvaluationFixtureCase, repo_root: Path) -> EvaluationFixtureResult:
    result = VerificationSession.from_config_file(case.config_path).run()
    observed = tuple(sorted({diagnostic.rule_id for diagnostic in result.diagnostics}))
    expected = set(case.expected_rule_ids)
    absent = set(case.expected_absent_rule_ids)
    missing = sorted(expected.difference(observed))
    forbidden = sorted(absent.intersection(observed))
    passed = not missing and not forbidden
    relative_config = str(case.config_path.relative_to(repo_root))
    evidence = f"{len(result.diagnostics)} diagnostic(s) from {relative_config}"
    if missing:
        evidence += "; missing expected rule(s): " + ", ".join(missing)
    if forbidden:
        evidence += "; observed forbidden rule(s): " + ", ".join(forbidden)
    return EvaluationFixtureResult(
        case_id=case.case_id,
        bug_class=case.bug_class,
        config=relative_config,
        observed_rule_ids=observed,
        passed=passed,
        evidence_summary=evidence,
        diagnostic_count=len(result.diagnostics),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationFixturePackError(f"evaluation fixture pack file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationFixturePackError(
            f"evaluation fixture pack file is not valid JSON: {path}:{exc.lineno}:{exc.colno}"
        ) from exc
    if not isinstance(raw, dict):
        raise EvaluationFixturePackError(f"evaluation fixture pack file must contain a JSON object: {path}")
    return raw


def _required_string(path: Path, raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationFixturePackError(f"{path} case field '{key}' must be a non-empty string")
    return value


def _string_tuple(
    value: object,
    path: Path,
    case_id: str,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item for item in value):
        raise EvaluationFixturePackError(f"{path} case {case_id!r} field '{field_name}' must be a list of non-empty strings")
    normalized = tuple(sorted(dict.fromkeys(value)))
    if not normalized and not allow_empty:
        raise EvaluationFixturePackError(f"{path} case {case_id!r} field '{field_name}' must not be empty")
    return normalized


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
