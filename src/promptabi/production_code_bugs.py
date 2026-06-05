"""Replay public bug cases by traversing exact production-code excerpts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, ArtifactLocation, SchemaArtifact, StopPolicyArtifact, ToolDefinitionArtifact
from .chat_templates import ChatTemplateSymbolicBounds, parse_hf_chat_template_config
from .role_boundaries import analyze_role_boundary_nonforgeability
from .stop_overreachability import analyze_stop_overreachability


DEFAULT_PRODUCTION_CODE_BUG_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "real_world_bugs" / "production_code.json"
)
DEFAULT_REAL_WORLD_BUG_CORPUS_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "real_world_bugs" / "corpus.json"
PRODUCTION_CODE_BUG_CORPUS_VERSION = 1
_STRING_LITERAL_RE = re.compile(
    r"""(?P<prefix>[rRuUbBfF]{0,2})(?P<quote>'''|\"\"\"|'|")(?P<body>.*?)(?<!\\)(?P=quote)""",
    re.DOTALL,
)


class ProductionCodeBugCorpusError(ValueError):
    """Raised when production-code bug replay cannot be validated."""


@dataclass(frozen=True, slots=True)
class ProductionCodeReference:
    """Pinned public source-code location for one bug replay."""

    repository: str
    ref: str
    path: str
    public_url: str
    license: str
    source_file_sha: str

    @classmethod
    def from_mapping(cls, raw: object, *, case_id: str) -> "ProductionCodeReference":
        if not isinstance(raw, dict):
            raise ProductionCodeBugCorpusError(f"{case_id} production_code.reference must be an object")
        required = ("repository", "ref", "path", "public_url", "license", "source_file_sha")
        values: dict[str, str] = {}
        for key in required:
            value = raw.get(key)
            if not isinstance(value, str) or not value:
                raise ProductionCodeBugCorpusError(f"{case_id} production_code.reference.{key} must be non-empty")
            values[key] = value
        if not values["public_url"].startswith("https://github.com/"):
            raise ProductionCodeBugCorpusError(f"{case_id} production_code.reference.public_url must be a GitHub URL")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ProductionCodeBugCase:
    """One public bug case backed by an exact upstream source excerpt."""

    case_id: str
    analysis: str
    reference: ProductionCodeReference
    source_excerpt: str
    source_sha256: str
    extraction: dict[str, object]

    @classmethod
    def from_mapping(cls, raw: object) -> "ProductionCodeBugCase":
        if not isinstance(raw, dict):
            raise ProductionCodeBugCorpusError("production-code cases must be objects")
        case_id = _required_string(raw, "id", "production-code case")
        analysis = _required_string(raw, "analysis", case_id)
        source_excerpt = _required_string(raw, "source_excerpt", case_id)
        source_sha256 = _required_string(raw, "source_sha256", case_id)
        extraction = raw.get("extraction")
        if not isinstance(extraction, dict):
            raise ProductionCodeBugCorpusError(f"{case_id} extraction must be an object")
        actual_sha = _sha256(source_excerpt)
        if actual_sha != source_sha256:
            raise ProductionCodeBugCorpusError(
                f"{case_id} source_excerpt sha256 mismatch: expected {source_sha256}, got {actual_sha}"
            )
        if analysis not in {"role-boundary-template", "stop-overreachability-source"}:
            raise ProductionCodeBugCorpusError(f"{case_id} has unsupported production-code analysis {analysis!r}")
        return cls(
            case_id=case_id,
            analysis=analysis,
            reference=ProductionCodeReference.from_mapping(raw.get("reference"), case_id=case_id),
            source_excerpt=source_excerpt,
            source_sha256=source_sha256,
            extraction=dict(extraction),
        )


@dataclass(frozen=True, slots=True)
class ProductionCodeBugReplay:
    """Replay result from exact production code into PromptABI analyzers."""

    case_id: str
    rule_ids: tuple[str, ...]
    evidence_summary: str
    extracted_values: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.rule_ids)


@dataclass(frozen=True, slots=True)
class ProductionCodeBugCorpus:
    """Production-code replay corpus for public bug cases."""

    path: Path
    methodology: str
    cases: tuple[ProductionCodeBugCase, ...]

    def by_id(self, case_id: str) -> ProductionCodeBugCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise ProductionCodeBugCorpusError(f"production-code corpus has no case {case_id!r}")

    def replay_case(self, case_id: str, *, real_world_corpus_path: str | Path | None = None) -> ProductionCodeBugReplay:
        return replay_production_code_bug_case(
            self.by_id(case_id),
            real_world_corpus_path=real_world_corpus_path,
        )

    def replay(self, *, real_world_corpus_path: str | Path | None = None) -> tuple[ProductionCodeBugReplay, ...]:
        return tuple(
            replay_production_code_bug_case(case, real_world_corpus_path=real_world_corpus_path) for case in self.cases
        )


def load_production_code_bug_corpus(path: str | Path | None = None) -> ProductionCodeBugCorpus:
    """Load the production-code bug corpus and validate source excerpt hashes."""

    corpus_path = Path(path) if path is not None else DEFAULT_PRODUCTION_CODE_BUG_CORPUS_PATH
    payload = _read_json_object(corpus_path, "production-code bug corpus")
    if payload.get("version") != PRODUCTION_CODE_BUG_CORPUS_VERSION:
        raise ProductionCodeBugCorpusError(f"{corpus_path} has unsupported production-code corpus version")
    methodology = payload.get("methodology")
    if not isinstance(methodology, str) or not methodology:
        raise ProductionCodeBugCorpusError(f"{corpus_path} methodology must be a non-empty string")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ProductionCodeBugCorpusError(f"{corpus_path} cases must be a non-empty list")
    cases = tuple(sorted((ProductionCodeBugCase.from_mapping(item) for item in raw_cases), key=lambda item: item.case_id))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ProductionCodeBugCorpusError("production-code corpus contains duplicate case ids")
    return ProductionCodeBugCorpus(path=corpus_path, methodology=methodology, cases=cases)


def replay_production_code_bug_case(
    case: ProductionCodeBugCase,
    *,
    real_world_corpus_path: str | Path | None = None,
) -> ProductionCodeBugReplay:
    """Traverse one exact production-code excerpt and replay it with local analyzers."""

    if case.analysis == "role-boundary-template":
        return _replay_role_boundary_source(case)
    if case.analysis == "stop-overreachability-source":
        return _replay_stop_overreachability_source(case, real_world_corpus_path=real_world_corpus_path)
    raise AssertionError(f"unsupported production-code analysis: {case.analysis!r}")


def replay_production_code_bug(case_id: str) -> ProductionCodeBugReplay:
    """Replay one public bug case from exact production-code excerpts."""

    return load_production_code_bug_corpus().replay_case(case_id)


def _replay_role_boundary_source(case: ProductionCodeBugCase) -> ProductionCodeBugReplay:
    source_kind = _required_extraction_string(case, "template_source")
    if source_kind == "jinja-file":
        template = case.source_excerpt
    elif source_kind == "python-return-concat":
        template = _extract_python_return_template(case.source_excerpt, case_id=case.case_id)
    elif source_kind == "python-assignment-concat":
        template = _extract_python_assignment_template(
            case.source_excerpt,
            case_id=case.case_id,
            target=_required_extraction_string(case, "template_variable"),
        )
    else:
        raise ProductionCodeBugCorpusError(f"{case.case_id} unsupported template_source {source_kind!r}")
    config = {
        "chat_template": template,
        "eos_token": _required_extraction_string(case, "eos_token"),
        "additional_special_tokens": _required_extraction_string_list(case, "additional_special_tokens"),
    }
    parsed = parse_hf_chat_template_config(config)
    report = analyze_role_boundary_nonforgeability(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=32),
    )
    expected = {tuple(item) for item in _required_extraction_list(case, "expected_witnesses")}
    actual = {(finding.input_expression, finding.marker, finding.marker_kind) for finding in report.findings}
    rule_ids = ("role-boundary-nonforgeability",) if report.model.supported and expected <= actual else ()
    return ProductionCodeBugReplay(
        case_id=case.case_id,
        rule_ids=rule_ids,
        evidence_summary=(
            f"{len(report.findings)} role-boundary witness(es) after traversing exact production source and extracting "
            f"{len(template)} template byte(s) from {case.reference.repository}:{case.reference.path}"
        ),
        extracted_values={
            "template_sha256": _sha256(template),
            "source_sha256": case.source_sha256,
            "expected_witnesses_matched": len(expected.intersection(actual)),
        },
    )


def _replay_stop_overreachability_source(
    case: ProductionCodeBugCase,
    *,
    real_world_corpus_path: str | Path | None = None,
) -> ProductionCodeBugReplay:
    expected_stops = tuple(_required_extraction_string_list(case, "expected_stop_sequences"))
    extracted_literals = _extract_source_string_literals(case.source_excerpt)
    for stop in expected_stops:
        if not _stop_is_source_derived(stop, extracted_literals, case.source_excerpt):
            raise ProductionCodeBugCorpusError(f"{case.case_id} expected stop sequence {stop!r} is not in source excerpt")
    corpus_case = _real_world_case(
        _required_extraction_string(case, "real_world_section"),
        _required_extraction_string(case, "real_world_case_id"),
        path=real_world_corpus_path,
    )
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name=f"{case.case_id}-production-code-stops",
        location=ArtifactLocation(uri=f"memory://production-code-bugs/{case.case_id}/stops"),
        stop_sequences=expected_stops,
    )
    with tempfile.TemporaryDirectory(prefix="promptabi-production-code-bug-") as directory:
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
    rule_ids = ("stop-overreach-content", "stop-overreach-structural") if matched == len(expected_findings) else ()
    return ProductionCodeBugReplay(
        case_id=case.case_id,
        rule_ids=rule_ids,
        evidence_summary=(
            f"{matched}/{len(expected_findings)} stop-overreach witness(es) after traversing exact production source for "
            f"{len(expected_stops)} source-derived stop sequence(s)"
        ),
        extracted_values={
            "stop_sequences": list(expected_stops),
            "source_literal_count": len(extracted_literals),
            "source_sha256": case.source_sha256,
        },
    )


def _extract_python_return_template(source: str, *, case_id: str) -> str:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise ProductionCodeBugCorpusError(f"{case_id} source excerpt is not parseable Python: {exc}") from exc
    returns = [node for node in ast.walk(module) if isinstance(node, ast.Return)]
    if len(returns) != 1:
        raise ProductionCodeBugCorpusError(f"{case_id} source excerpt must contain exactly one Python return")
    try:
        value = ast.literal_eval(returns[0].value)
    except (ValueError, TypeError) as exc:
        raise ProductionCodeBugCorpusError(f"{case_id} Python return is not a literal template") from exc
    if not isinstance(value, str) or not value:
        raise ProductionCodeBugCorpusError(f"{case_id} Python return did not produce a non-empty template")
    return value


def _extract_python_assignment_template(source: str, *, case_id: str, target: str) -> str:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise ProductionCodeBugCorpusError(f"{case_id} source excerpt is not parseable Python: {exc}") from exc
    matches: list[ast.expr] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(item, ast.Name) and item.id == target for item in node.targets):
            matches.append(node.value)
    if len(matches) != 1:
        raise ProductionCodeBugCorpusError(f"{case_id} source excerpt must assign {target!r} exactly once")
    try:
        value = ast.literal_eval(matches[0])
    except (ValueError, TypeError) as exc:
        raise ProductionCodeBugCorpusError(f"{case_id} assignment {target!r} is not a literal template") from exc
    if not isinstance(value, str) or not value:
        raise ProductionCodeBugCorpusError(f"{case_id} assignment {target!r} did not produce a non-empty template")
    return value


def _extract_source_string_literals(source: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _STRING_LITERAL_RE.finditer(source):
        token = match.group(0)
        prefix = match.group("prefix").lower()
        if "f" in prefix:
            continue
        try:
            value = ast.literal_eval(token)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, str):
            values.append(value)
    return tuple(dict.fromkeys(values))


def _stop_is_source_derived(stop: str, literals: tuple[str, ...], source: str) -> bool:
    return stop in source or any(stop == literal or stop in literal for literal in literals)


def _real_world_case(section: str, case_id: str, *, path: str | Path | None = None) -> dict[str, Any]:
    corpus_path = Path(path) if path is not None else DEFAULT_REAL_WORLD_BUG_CORPUS_PATH
    corpus = _read_json_object(corpus_path, "real-world bug corpus")
    cases = corpus.get(section)
    if not isinstance(cases, list):
        raise ProductionCodeBugCorpusError(f"{corpus_path} section {section!r} is missing")
    for item in cases:
        if isinstance(item, dict) and item.get("id") == case_id:
            return item
    raise ProductionCodeBugCorpusError(f"{corpus_path} has no case {case_id!r} in {section}")


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
                name=str(case["id"]),
                location=ArtifactLocation(path=str(path)),
            ),
        )
    if artifact_kind == "tool-definition":
        return (
            ToolDefinitionArtifact(
                kind=ArtifactKind.TOOL_DEFINITION,
                name=str(case["id"]),
                location=ArtifactLocation(path=str(path)),
            ),
        )
    raise ProductionCodeBugCorpusError(f"unsupported real-world artifact kind: {artifact_kind}")


def _required_extraction_string(case: ProductionCodeBugCase, key: str) -> str:
    return _required_string(case.extraction, key, case.case_id)


def _required_extraction_string_list(case: ProductionCodeBugCase, key: str) -> tuple[str, ...]:
    values = _required_extraction_list(case, key)
    if not all(isinstance(item, str) and item for item in values):
        raise ProductionCodeBugCorpusError(f"{case.case_id} extraction.{key} must be a non-empty string list")
    return tuple(values)


def _required_extraction_list(case: ProductionCodeBugCase, key: str) -> list[object]:
    value = case.extraction.get(key)
    if not isinstance(value, list) or not value:
        raise ProductionCodeBugCorpusError(f"{case.case_id} extraction.{key} must be a non-empty list")
    return value


def _required_string(raw: dict[str, object], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ProductionCodeBugCorpusError(f"{context} field {key!r} must be a non-empty string")
    return value


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProductionCodeBugCorpusError(f"{description} file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProductionCodeBugCorpusError(f"{description} file is not valid JSON: {path}:{exc.lineno}:{exc.colno}") from exc
    if not isinstance(raw, dict):
        raise ProductionCodeBugCorpusError(f"{description} must contain a JSON object: {path}")
    return raw


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
