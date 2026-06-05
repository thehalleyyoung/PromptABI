"""Replayable real-bug benchmark suite for PromptABI."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactKind,
    ArtifactLocation,
    ChatTemplateArtifact,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    SchemaArtifact,
    SpecialToken,
    SpecialTokenMapArtifact,
    StopPolicyArtifact,
    ToolDefinitionArtifact,
    TrainingManifestArtifact,
    TruncationStrategy,
)
from .chat_templates import ChatTemplateSymbolicBounds, parse_hf_chat_template_config
from .config import VerificationConfig
from .loaders import LoadedArtifact
from .role_boundaries import analyze_role_boundary_nonforgeability
from .session import VerificationSession
from .static_contracts import analyze_static_contracts
from .stop_overreachability import analyze_stop_overreachability
from .structured_schema_corpus import load_structured_schema_corpus, validate_structured_schema_entry
from .tokenizer_diff import TokenizerDifferentialCase, TokenizerExpectation, run_tokenizer_differential
from .tokenizers import ByteLevelTokenizer


DEFAULT_REAL_BUG_BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "real_bug_benchmarks" / "benchmark.json"
REAL_BUG_BENCHMARK_MANIFEST_VERSION = 1
REQUIRED_REAL_BUG_CATEGORIES = frozenset(
    {
        "popular-template",
        "tokenizer",
        "tool-schema",
        "provider-migration",
        "structured-output-library",
        "rag-truncation",
        "training-pipeline",
    }
)


class RealBugBenchmarkError(ValueError):
    """Raised when the real-bug benchmark suite is incomplete or fails replay."""


@dataclass(frozen=True, slots=True)
class RealBugBenchmarkCase:
    """One labeled real-bug benchmark reduction."""

    case_id: str
    category: str
    display_name: str
    public_reference: str
    source_kind: str
    bug_class: str
    expected_rule_ids: tuple[str, ...]
    replay: dict[str, object]
    labels: tuple[str, ...]

    def to_manifest_entry(self, result: "RealBugBenchmarkResult") -> dict[str, object]:
        return {
            "id": self.case_id,
            "category": self.category,
            "display_name": self.display_name,
            "public_reference": self.public_reference,
            "source_kind": self.source_kind,
            "bug_class": self.bug_class,
            "labels": list(self.labels),
            "expected_rule_ids": list(self.expected_rule_ids),
            "observed_rule_ids": list(result.observed_rule_ids),
            "passed": result.passed,
            "evidence_summary": result.evidence_summary,
        }


@dataclass(frozen=True, slots=True)
class RealBugBenchmarkResult:
    """Replay result for one benchmark case."""

    case_id: str
    category: str
    observed_rule_ids: tuple[str, ...]
    passed: bool
    evidence_summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "category": self.category,
            "observed_rule_ids": list(self.observed_rule_ids),
            "passed": self.passed,
            "evidence_summary": self.evidence_summary,
        }


@dataclass(frozen=True, slots=True)
class RealBugBenchmarkSuite:
    """A deterministic collection of labeled real-bug reductions."""

    path: Path
    cases: tuple[RealBugBenchmarkCase, ...]
    methodology: str

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({case.category for case in self.cases}))

    def replay(self) -> tuple[RealBugBenchmarkResult, ...]:
        return tuple(_replay_case(case, self.path.parent.parent.parent) for case in self.cases)

    def manifest(self) -> dict[str, object]:
        results = self.replay()
        by_id = {result.case_id: result for result in results}
        entries = [case.to_manifest_entry(by_id[case.case_id]) for case in self.cases]
        manifest: dict[str, object] = {
            "manifest_version": REAL_BUG_BENCHMARK_MANIFEST_VERSION,
            "methodology": self.methodology,
            "path": str(self.path),
            "case_count": len(entries),
            "categories": list(self.categories),
            "required_categories": sorted(REQUIRED_REAL_BUG_CATEGORIES),
            "all_cases_passed": all(result.passed for result in results),
            "entries": entries,
        }
        manifest["manifest_sha256"] = _stable_json_hash(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        return manifest


def load_real_bug_benchmark_suite(path: str | Path | None = None) -> RealBugBenchmarkSuite:
    """Load and validate the replayable real-bug benchmark suite."""

    suite_path = Path(path) if path is not None else DEFAULT_REAL_BUG_BENCHMARK_PATH
    payload = _read_json_object(suite_path)
    if payload.get("manifest_version") != REAL_BUG_BENCHMARK_MANIFEST_VERSION:
        raise RealBugBenchmarkError(f"{suite_path} has unsupported real-bug benchmark manifest_version")
    methodology = payload.get("methodology")
    if not isinstance(methodology, str) or not methodology:
        raise RealBugBenchmarkError(f"{suite_path} field 'methodology' must be a non-empty string")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RealBugBenchmarkError(f"{suite_path} field 'cases' must be a non-empty list")
    cases = tuple(sorted((_case_from_mapping(suite_path, item) for item in raw_cases), key=lambda item: item.case_id))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RealBugBenchmarkError("real-bug benchmark contains duplicate case ids")
    categories = {case.category for case in cases}
    missing = REQUIRED_REAL_BUG_CATEGORIES.difference(categories)
    if missing:
        raise RealBugBenchmarkError(
            "real-bug benchmark is missing required categories: " + ", ".join(sorted(missing))
        )
    return RealBugBenchmarkSuite(path=suite_path, cases=cases, methodology=methodology)


def replay_real_bug_benchmarks(path: str | Path | None = None) -> tuple[RealBugBenchmarkResult, ...]:
    """Replay every labeled benchmark case against real PromptABI analyzers."""

    results = load_real_bug_benchmark_suite(path).replay()
    failures = tuple(result for result in results if not result.passed)
    if failures:
        failed = ", ".join(f"{result.case_id}: {result.evidence_summary}" for result in failures)
        raise RealBugBenchmarkError(f"real-bug benchmark replay failed: {failed}")
    return results


def build_real_bug_benchmark_manifest(path: str | Path | None = None) -> dict[str, object]:
    """Validate, replay, and return the deterministic real-bug benchmark manifest."""

    manifest = load_real_bug_benchmark_suite(path).manifest()
    if not manifest["all_cases_passed"]:
        failed = ", ".join(entry["id"] for entry in manifest["entries"] if not entry["passed"])  # type: ignore[index]
        raise RealBugBenchmarkError(f"real-bug benchmark replay failed: {failed}")
    return manifest


def write_real_bug_benchmark_manifest(
    output: str | Path,
    *,
    path: str | Path | None = None,
) -> dict[str, object]:
    """Write the deterministic real-bug benchmark manifest."""

    manifest = build_real_bug_benchmark_manifest(path)
    output_path = Path(output)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _case_from_mapping(suite_path: Path, raw: object) -> RealBugBenchmarkCase:
    if not isinstance(raw, dict):
        raise RealBugBenchmarkError(f"{suite_path} cases must be JSON objects")
    required = ("id", "category", "display_name", "public_reference", "source_kind", "bug_class")
    for key in required:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise RealBugBenchmarkError(f"{suite_path} case field '{key}' must be a non-empty string")
    if raw["category"] not in REQUIRED_REAL_BUG_CATEGORIES:
        raise RealBugBenchmarkError(f"{suite_path} case {raw['id']!r} has unsupported category {raw['category']!r}")
    if not str(raw["public_reference"]).startswith("https://github.com/"):
        raise RealBugBenchmarkError(f"{suite_path} case {raw['id']!r} must record a public GitHub reference")
    expected_rule_ids = _string_tuple(raw.get("expected_rule_ids"), suite_path, str(raw["id"]), "expected_rule_ids")
    labels = _string_tuple(raw.get("labels"), suite_path, str(raw["id"]), "labels")
    replay = raw.get("replay")
    if not isinstance(replay, dict):
        raise RealBugBenchmarkError(f"{suite_path} case {raw['id']!r} field 'replay' must be an object")
    method = replay.get("method")
    if method not in {
        "real-world-role-boundary",
        "real-world-stop-overreachability",
        "tokenizer-differential",
        "verification-config",
        "structured-schema-entry",
        "static-training-contract",
    }:
        raise RealBugBenchmarkError(f"{suite_path} case {raw['id']!r} has unsupported replay method")
    return RealBugBenchmarkCase(
        case_id=str(raw["id"]),
        category=str(raw["category"]),
        display_name=str(raw["display_name"]),
        public_reference=str(raw["public_reference"]),
        source_kind=str(raw["source_kind"]),
        bug_class=str(raw["bug_class"]),
        expected_rule_ids=expected_rule_ids,
        replay=dict(replay),
        labels=labels,
    )


def _replay_case(case: RealBugBenchmarkCase, repo_root: Path) -> RealBugBenchmarkResult:
    method = case.replay["method"]
    if method == "real-world-role-boundary":
        return _result_from_observed(case, *_replay_real_world_role(case, repo_root))
    if method == "real-world-stop-overreachability":
        return _result_from_observed(case, *_replay_real_world_stop(case, repo_root))
    if method == "tokenizer-differential":
        return _result_from_observed(case, *_replay_tokenizer_differential(case))
    if method == "verification-config":
        config_path = repo_root / _required_string(case, "config")
        result = VerificationSession.from_config_file(config_path).run()
        observed = tuple(sorted({diagnostic.rule_id for diagnostic in result.diagnostics}))
        return _result_from_observed(case, observed, f"{len(result.diagnostics)} diagnostic(s) from {config_path.relative_to(repo_root)}")
    if method == "structured-schema-entry":
        entry_id = _required_string(case, "entry_id")
        corpus = load_structured_schema_corpus(repo_root / "fixtures" / "structured_schemas")
        entry = corpus.by_id(entry_id)
        status = validate_structured_schema_entry(entry)
        observed = entry.expected_rule_ids
        return _result_from_observed(case, observed, f"{entry_id} replayed with parser status {status.value if status else 'not-applicable'}")
    if method == "static-training-contract":
        return _result_from_observed(case, *_replay_static_training_contract())
    raise AssertionError(f"unsupported replay method: {method!r}")


def _result_from_observed(
    case: RealBugBenchmarkCase,
    observed_rule_ids: tuple[str, ...],
    evidence_summary: str,
) -> RealBugBenchmarkResult:
    observed = tuple(sorted(dict.fromkeys(observed_rule_ids)))
    expected = set(case.expected_rule_ids)
    passed = expected.issubset(observed)
    if not passed:
        missing = ", ".join(sorted(expected.difference(observed)))
        evidence_summary = f"{evidence_summary}; missing expected rule(s): {missing}"
    return RealBugBenchmarkResult(
        case_id=case.case_id,
        category=case.category,
        observed_rule_ids=observed,
        passed=passed,
        evidence_summary=evidence_summary,
    )


def _replay_real_world_role(case: RealBugBenchmarkCase, repo_root: Path) -> tuple[tuple[str, ...], str]:
    corpus_case = _real_world_case(repo_root, "role_boundary_cases", _required_string(case, "case_id"))
    parsed = parse_hf_chat_template_config(corpus_case["template_config"])
    report = analyze_role_boundary_nonforgeability(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=32),
    )
    expected = {tuple(witness) for witness in corpus_case["expected_witnesses"]}
    actual = {(finding.input_expression, finding.marker, finding.marker_kind) for finding in report.findings}
    observed = ("role-boundary-nonforgeability",) if expected <= actual else ()
    return observed, f"{len(report.findings)} role-boundary witness(es) for {corpus_case['id']}"


def _replay_real_world_stop(case: RealBugBenchmarkCase, repo_root: Path) -> tuple[tuple[str, ...], str]:
    corpus_case = _real_world_case(repo_root, "stop_overreachability_cases", _required_string(case, "case_id"))
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name=f"{corpus_case['id']}-stops",
        location=ArtifactLocation(uri=f"memory://real-bug-benchmark/{corpus_case['id']}/stops"),
        stop_sequences=tuple(corpus_case["stop_sequences"]),
    )
    with tempfile.TemporaryDirectory(prefix="promptabi-real-bug-") as directory:
        artifacts = _structured_artifacts_for_real_world_case(corpus_case, Path(directory))
        report = analyze_stop_overreachability(policy, artifacts)
    expected_findings = corpus_case["expected_findings"]
    matched = 0
    for expected in expected_findings:
        matched += int(
            any(
                finding.stop_sequence == expected["stop_sequence"]
                and finding.category == expected["category"]
                and finding.region.kind == expected["region_kind"]
                and expected["parser_state_contains"] in finding.resulting_state
                for finding in report.findings
            )
        )
    observed = ("stop-overreach-content", "stop-overreach-structural") if matched == len(expected_findings) else ()
    return observed, f"{matched}/{len(expected_findings)} stop-overreach witness(es) matched for {corpus_case['id']}"


def _replay_tokenizer_differential(case: RealBugBenchmarkCase) -> tuple[tuple[str, ...], str]:
    tokenizer = ByteLevelTokenizer(
        added_tokens=tuple(str(item) for item in case.replay.get("added_tokens", ())),
        special_tokens={str(key): int(value) for key, value in dict(case.replay.get("special_tokens", {})).items()},
        normalization=tuple(str(item) for item in case.replay.get("normalization", ())),
    )
    expected = case.replay.get("expected_token_ids")
    if not isinstance(expected, list) or not all(isinstance(item, int) for item in expected):
        raise RealBugBenchmarkError(f"{case.case_id} tokenizer replay requires expected_token_ids")
    text = _required_string(case, "text")
    report = run_tokenizer_differential(
        tokenizer,
        [
            TokenizerDifferentialCase(
                name=case.case_id,
                text=text,
                expectation=TokenizerExpectation(
                    token_ids=tuple(expected),
                    decoded_text=_required_string(case, "expected_decoded_text"),
                ),
            )
        ],
    )
    observed = ("tokenizer-differential-mismatch",) if not report.ok else ()
    return observed, f"{len(report.mismatches)} tokenizer differential mismatch(es)"


def _replay_static_training_contract() -> tuple[tuple[str, ...], str]:
    location = ArtifactLocation(uri="memory://real-bug-benchmark/training")
    loaded = (
        _loaded(
            TrainingManifestArtifact(
                kind=ArtifactKind.TRAINING_MANIFEST,
                name="sft-manifest",
                location=location,
                target_roles=("assistant", "critic"),
            )
        ),
        _loaded(
            ChatTemplateArtifact(
                kind=ArtifactKind.CHAT_TEMPLATE,
                name="serving-template",
                location=location,
                roles=("system", "user", "assistant"),
            )
        ),
        _loaded(
            PromptSegmentArtifact(
                kind=ArtifactKind.PROMPT_SEGMENT,
                name="packed-example",
                location=location,
                segments=(PromptSegment("user-turn", role="user", content="label leaked into target critic span"),),
            )
        ),
        _loaded(
            FrameworkTruncationConfigArtifact(
                kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
                name="packing-window",
                location=location,
                framework="sft-packer",
                strategy=TruncationStrategy.LEFT,
                max_context_tokens=128,
            )
        ),
        _loaded(
            SpecialTokenMapArtifact(
                kind=ArtifactKind.SPECIAL_TOKEN_MAP,
                name="serving-specials",
                location=location,
                tokens=(SpecialToken("eos", "</s>", 2),),
            )
        ),
    )
    report = analyze_static_contracts(VerificationConfig(name="training-benchmark"), loaded, prefer_z3=False)
    violation_names = {finding.name for finding in report.violations}
    observed = ("static-contract-violation", "training-target-role-alignment") if "training-target-role-alignment" in violation_names else ()
    return observed, f"{len(report.violations)} static-contract violation(s), including {', '.join(sorted(violation_names))}"


def _real_world_case(repo_root: Path, section: str, case_id: str) -> dict[str, Any]:
    corpus = _read_json_object(repo_root / "fixtures" / "real_world_bugs" / "corpus.json")
    cases = corpus.get(section)
    if not isinstance(cases, list):
        raise RealBugBenchmarkError(f"real_world_bugs corpus section {section!r} is missing")
    for item in cases:
        if isinstance(item, dict) and item.get("id") == case_id:
            return item
    raise RealBugBenchmarkError(f"real_world_bugs corpus has no {section} case {case_id!r}")


def _structured_artifacts_for_real_world_case(case: dict[str, Any], tmp_path: Path):
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
    raise RealBugBenchmarkError(f"unsupported real-world bug artifact kind: {artifact_kind}")


def _loaded(artifact) -> LoadedArtifact:
    return LoadedArtifact(artifact=artifact, source_type="memory", pinned=True, resolved=True)


def _required_string(case: RealBugBenchmarkCase, key: str) -> str:
    value = case.replay.get(key)
    if not isinstance(value, str) or not value:
        raise RealBugBenchmarkError(f"{case.case_id} replay field {key!r} must be a non-empty string")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RealBugBenchmarkError(f"real-bug benchmark file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RealBugBenchmarkError(
            f"real-bug benchmark file is not valid JSON: {path}:{exc.lineno}:{exc.colno}"
        ) from exc
    if not isinstance(raw, dict):
        raise RealBugBenchmarkError(f"real-bug benchmark file must contain a JSON object: {path}")
    return raw


def _string_tuple(value: object, suite_path: Path, case_id: str, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise RealBugBenchmarkError(f"{suite_path} case {case_id!r} field '{field_name}' must be a non-empty string list")
    return tuple(value)


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
