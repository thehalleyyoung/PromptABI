"""Offline replay checks for recorded provider fixture packs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, ProviderConfigArtifact
from .diagnostics import SourceSpan
from .loaders import LoadedArtifact
from .provider_migration import SUPPORTED_PROVIDER_FAMILIES, canonical_provider_family
from .source import build_json_source_map


class ProviderFixtureReplayFindingKind(StrEnum):
    """Concrete replay failures in a recorded provider fixture pack."""

    UNSUPPORTED_PROVIDER = "unsupported-provider"
    MISSING_SURFACE = "missing-surface"
    INVALID_REQUEST_SHAPE = "invalid-request-shape"
    INVALID_RESPONSE_SHAPE = "invalid-response-shape"
    INVALID_STOP_SHAPE = "invalid-stop-shape"
    INVALID_STREAMING_SHAPE = "invalid-streaming-shape"
    INVALID_ERROR_SHAPE = "invalid-error-shape"
    INVALID_LIMIT_SHAPE = "invalid-limit-shape"
    EDGE_CASE_UNRESOLVED = "edge-case-unresolved"
    SECRET_LIKE_REPLAY_DATA = "secret-like-replay-data"


@dataclass(frozen=True, slots=True)
class ProviderFixtureReplayFinding:
    """One deterministic provider-fixture replay failure."""

    kind: ProviderFixtureReplayFindingKind
    message: str
    severity: str
    artifact_name: str
    provider: str
    span: SourceSpan | None = None
    evidence: tuple[tuple[str, str], ...] = ()
    suggestion: str = "Fix the recorded provider fixture pack before using it as an offline oracle."


@dataclass(frozen=True, slots=True)
class ProviderFixtureReplayCase:
    """One provider fixture replayed entirely from local recorded JSON."""

    artifact_name: str
    provider: str
    provider_family: str
    replay_hash: str
    surfaces: tuple[str, ...]
    edge_cases: tuple[str, ...]
    request_fields: tuple[str, ...]
    response_fields: tuple[str, ...]
    stop_sequences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProviderFixtureReplayReport:
    """Bounded replay result for loaded provider fixture packs."""

    cases: tuple[ProviderFixtureReplayCase, ...]
    findings: tuple[ProviderFixtureReplayFinding, ...]
    fixtures_checked: int
    provider_families: tuple[str, ...]
    replay_hash: str


_REQUIRED_SURFACES = ("request", "response", "stops", "streaming", "errors", "limits", "edge_cases")
_SECRET_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "secret",
        "password",
        "x-api-key",
    }
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
_SURFACE_ALIASES = {
    "stops.finish_reason": "stops.finish_reason_path",
    "streaming.delta": "streaming.delta_path",
}


def analyze_provider_fixture_replay(
    loaded_artifacts: tuple[LoadedArtifact, ...],
) -> ProviderFixtureReplayReport:
    """Replay recorded provider fixture packs without network calls.

    The replay is intentionally bounded to the secret-free contract data in each
    local JSON pack: request/response shapes, tool-call encoding, stop behavior,
    streaming chunking, error envelopes, limits, and labeled edge cases.
    """

    providers = tuple(
        sorted(
            (loaded for loaded in loaded_artifacts if _is_provider_snapshot(loaded)),
            key=lambda loaded: loaded.artifact.name,
        )
    )
    cases: list[ProviderFixtureReplayCase] = []
    findings: list[ProviderFixtureReplayFinding] = []

    for loaded in providers:
        case, case_findings = _replay_provider_fixture(loaded)
        if case is not None:
            cases.append(case)
        findings.extend(case_findings)

    replay_hash = _stable_json_hash(
        {
            "cases": [
                {
                    "artifact_name": case.artifact_name,
                    "provider_family": case.provider_family,
                    "replay_hash": case.replay_hash,
                }
                for case in cases
            ],
            "findings": [
                {
                    "artifact_name": finding.artifact_name,
                    "kind": finding.kind.value,
                    "message": finding.message,
                    "severity": finding.severity,
                }
                for finding in findings
            ],
        }
    )
    return ProviderFixtureReplayReport(
        cases=tuple(cases),
        findings=tuple(sorted(findings, key=lambda item: (item.severity, item.kind.value, item.message))),
        fixtures_checked=len(providers),
        provider_families=tuple(sorted({case.provider_family for case in cases})),
        replay_hash=replay_hash,
    )


def _replay_provider_fixture(
    loaded: LoadedArtifact,
) -> tuple[ProviderFixtureReplayCase | None, tuple[ProviderFixtureReplayFinding, ...]]:
    artifact = loaded.artifact
    assert isinstance(artifact, ProviderConfigArtifact)
    path = Path(artifact.location.path) if artifact.location.path is not None else None
    if path is None:
        return None, ()
    raw, spans = _read_fixture(path)
    provider = _string(raw.get("provider")) or artifact.provider
    provider_family = (
        canonical_provider_family(_string(raw.get("provider_family")) or artifact.api_family or provider)
        or _string(raw.get("provider_family"))
        or provider
    )
    findings: list[ProviderFixtureReplayFinding] = []

    if provider_family not in SUPPORTED_PROVIDER_FAMILIES:
        findings.append(
            _finding(
                ProviderFixtureReplayFindingKind.UNSUPPORTED_PROVIDER,
                "error",
                artifact.name,
                provider,
                f"provider fixture '{artifact.name}' uses unsupported provider family '{provider_family}'",
                spans.get("provider_family"),
                (("provider_family", provider_family), ("supported families", ", ".join(SUPPORTED_PROVIDER_FAMILIES))),
                "Use a supported provider family or add a replay adapter for this provider.",
            )
        )

    for surface in _REQUIRED_SURFACES:
        if surface not in raw:
            findings.append(
                _finding(
                    ProviderFixtureReplayFindingKind.MISSING_SURFACE,
                    "error",
                    artifact.name,
                    provider,
                    f"provider fixture '{artifact.name}' lacks required replay surface '{surface}'",
                    spans.get(surface),
                    (("surface", surface), ("required surfaces", ", ".join(_REQUIRED_SURFACES))),
                    "Record every required surface so offline replay covers the provider contract.",
                )
            )

    findings.extend(_secret_findings(artifact.name, provider, raw, spans))
    findings.extend(_validate_request(artifact.name, provider, raw, spans))
    findings.extend(_validate_response(artifact.name, provider, raw, spans))
    findings.extend(_validate_stops(artifact.name, provider, raw, spans))
    findings.extend(_validate_streaming(artifact.name, provider, raw, spans))
    findings.extend(_validate_errors(artifact.name, provider, raw, spans))
    findings.extend(_validate_limits(artifact.name, provider, raw, spans))
    findings.extend(_validate_edge_cases(artifact.name, provider, raw, spans))

    request = _mapping(raw.get("request"))
    response = _mapping(raw.get("response"))
    stops = _mapping(raw.get("stops"))
    edge_cases = _edge_cases(raw.get("edge_cases"))
    canonical_pack = {
        "provider": provider,
        "provider_family": provider_family,
        "request": request,
        "response": response,
        "stops": stops,
        "streaming": _mapping(raw.get("streaming")),
        "errors": _mapping(raw.get("errors")),
        "limits": _mapping(raw.get("limits")),
        "edge_cases": edge_cases,
    }
    case = ProviderFixtureReplayCase(
        artifact_name=artifact.name,
        provider=provider,
        provider_family=provider_family,
        replay_hash=_stable_json_hash(canonical_pack),
        surfaces=tuple(surface for surface in _REQUIRED_SURFACES if surface in raw),
        edge_cases=tuple(str(item["id"]) for item in edge_cases if isinstance(item.get("id"), str)),
        request_fields=_string_tuple(request.get("fields")),
        response_fields=_string_tuple(response.get("fields")),
        stop_sequences=_string_tuple(stops.get("sequences")),
    )
    return case, tuple(findings)


def _validate_request(
    artifact_name: str,
    provider: str,
    raw: dict[str, Any],
    spans: dict[str, SourceSpan],
) -> tuple[ProviderFixtureReplayFinding, ...]:
    request = _mapping(raw.get("request"))
    findings = []
    if not _string(request.get("method")):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_REQUEST_SHAPE, artifact_name, provider, "request.method", spans))
    if not _string(request.get("endpoint")):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_REQUEST_SHAPE, artifact_name, provider, "request.endpoint", spans))
    if not _string_tuple(request.get("fields")):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_REQUEST_SHAPE, artifact_name, provider, "request.fields", spans))
    return tuple(findings)


def _validate_response(
    artifact_name: str,
    provider: str,
    raw: dict[str, Any],
    spans: dict[str, SourceSpan],
) -> tuple[ProviderFixtureReplayFinding, ...]:
    response = _mapping(raw.get("response"))
    tool_calls = _mapping(response.get("tool_calls"))
    findings = []
    if not _string_tuple(response.get("fields")):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_RESPONSE_SHAPE, artifact_name, provider, "response.fields", spans))
    if not _string_tuple(response.get("finish_reasons")):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_RESPONSE_SHAPE, artifact_name, provider, "response.finish_reasons", spans))
    for field in ("name_path", "arguments_path", "argument_encoding"):
        if not _string(tool_calls.get(field)):
            findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_RESPONSE_SHAPE, artifact_name, provider, f"response.tool_calls.{field}", spans))
    parallel = tool_calls.get("supports_parallel_tool_calls")
    if not isinstance(parallel, bool):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_RESPONSE_SHAPE, artifact_name, provider, "response.tool_calls.supports_parallel_tool_calls", spans))
    return tuple(findings)


def _validate_stops(
    artifact_name: str,
    provider: str,
    raw: dict[str, Any],
    spans: dict[str, SourceSpan],
) -> tuple[ProviderFixtureReplayFinding, ...]:
    stops = _mapping(raw.get("stops"))
    findings = []
    if not _string_tuple(stops.get("sequences")):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_STOP_SHAPE, artifact_name, provider, "stops.sequences", spans))
    if not _string(stops.get("finish_reason_path")):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_STOP_SHAPE, artifact_name, provider, "stops.finish_reason_path", spans))
    if not isinstance(stops.get("truncates_before_parser"), bool):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_STOP_SHAPE, artifact_name, provider, "stops.truncates_before_parser", spans))
    return tuple(findings)


def _validate_streaming(
    artifact_name: str,
    provider: str,
    raw: dict[str, Any],
    spans: dict[str, SourceSpan],
) -> tuple[ProviderFixtureReplayFinding, ...]:
    streaming = _mapping(raw.get("streaming"))
    findings = []
    if not _string(streaming.get("delta_path")):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_STREAMING_SHAPE, artifact_name, provider, "streaming.delta_path", spans))
    if not isinstance(streaming.get("emits_argument_fragments"), bool):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_STREAMING_SHAPE, artifact_name, provider, "streaming.emits_argument_fragments", spans))
    if streaming.get("emits_argument_fragments") is True and not _string(streaming.get("assembly_key")):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_STREAMING_SHAPE, artifact_name, provider, "streaming.assembly_key", spans))
    return tuple(findings)


def _validate_errors(
    artifact_name: str,
    provider: str,
    raw: dict[str, Any],
    spans: dict[str, SourceSpan],
) -> tuple[ProviderFixtureReplayFinding, ...]:
    errors = _mapping(raw.get("errors"))
    findings = []
    for field in ("code_path", "message_path", "rate_limit_path"):
        if not _string(errors.get(field)):
            findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_ERROR_SHAPE, artifact_name, provider, f"errors.{field}", spans))
    sample = errors.get("sample")
    if sample is not None and not isinstance(sample, dict):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_ERROR_SHAPE, artifact_name, provider, "errors.sample", spans))
    return tuple(findings)


def _validate_limits(
    artifact_name: str,
    provider: str,
    raw: dict[str, Any],
    spans: dict[str, SourceSpan],
) -> tuple[ProviderFixtureReplayFinding, ...]:
    limits = _mapping(raw.get("limits"))
    findings = []
    for field in ("max_input_tokens", "max_output_tokens"):
        value = limits.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_LIMIT_SHAPE, artifact_name, provider, f"limits.{field}", spans))
    parallel_limit = limits.get("parallel_tool_call_limit")
    if parallel_limit is not None and (not isinstance(parallel_limit, int) or isinstance(parallel_limit, bool) or parallel_limit <= 0):
        findings.append(_shape_finding(ProviderFixtureReplayFindingKind.INVALID_LIMIT_SHAPE, artifact_name, provider, "limits.parallel_tool_call_limit", spans))
    return tuple(findings)


def _validate_edge_cases(
    artifact_name: str,
    provider: str,
    raw: dict[str, Any],
    spans: dict[str, SourceSpan],
) -> tuple[ProviderFixtureReplayFinding, ...]:
    findings = []
    for index, edge_case in enumerate(_edge_cases(raw.get("edge_cases"))):
        edge_id = _string(edge_case.get("id")) or f"edge_cases[{index}]"
        surface = _string(edge_case.get("surface"))
        expected = _string(edge_case.get("expected_behavior"))
        if surface is None or expected is None:
            findings.append(
                _finding(
                    ProviderFixtureReplayFindingKind.EDGE_CASE_UNRESOLVED,
                    "error",
                    artifact_name,
                    provider,
                    f"provider fixture '{artifact_name}' has an incomplete edge-case record",
                    spans.get(f"edge_cases.{index}"),
                    (("edge case", edge_id), ("surface", surface or "<missing>")),
                    "Record edge-case id, surface, and expected_behavior fields.",
                )
            )
            continue
        if not _surface_exists(raw, surface):
            findings.append(
                _finding(
                    ProviderFixtureReplayFindingKind.EDGE_CASE_UNRESOLVED,
                    "error",
                    artifact_name,
                    provider,
                    f"provider fixture '{artifact_name}' edge case '{edge_id}' references missing surface '{surface}'",
                    spans.get(f"edge_cases.{index}.surface") or spans.get("edge_cases"),
                    (("edge case", edge_id), ("surface", surface), ("expected behavior", expected)),
                    "Point the edge case at a recorded request, response, stop, streaming, error, or limits surface.",
                )
            )
    return tuple(findings)


def _secret_findings(
    artifact_name: str,
    provider: str,
    value: object,
    spans: dict[str, SourceSpan],
    path: str = "$",
) -> tuple[ProviderFixtureReplayFinding, ...]:
    findings: list[ProviderFixtureReplayFinding] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_")
            child_path = f"{path}.{key_text}"
            if normalized in _SECRET_KEY_NAMES:
                findings.append(
                    _finding(
                        ProviderFixtureReplayFindingKind.SECRET_LIKE_REPLAY_DATA,
                        "error",
                        artifact_name,
                        provider,
                        f"provider fixture '{artifact_name}' contains secret-like replay field at {child_path}",
                        spans.get(_span_key_from_path(child_path)),
                        (("field", child_path),),
                        "Redact credentials and record only structural API-contract fields.",
                    )
                )
            findings.extend(_secret_findings(artifact_name, provider, child, spans, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_secret_findings(artifact_name, provider, child, spans, f"{path}.{index}"))
    elif isinstance(value, str):
        for pattern in _SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                findings.append(
                    _finding(
                        ProviderFixtureReplayFindingKind.SECRET_LIKE_REPLAY_DATA,
                        "error",
                        artifact_name,
                        provider,
                        f"provider fixture '{artifact_name}' contains secret-like replay value at {path}",
                        spans.get(_span_key_from_path(path)),
                        (("field", path),),
                        "Replace secret-like values with redacted placeholders before committing fixture packs.",
                    )
                )
                break
    return tuple(findings)


def _shape_finding(
    kind: ProviderFixtureReplayFindingKind,
    artifact_name: str,
    provider: str,
    field: str,
    spans: dict[str, SourceSpan],
) -> ProviderFixtureReplayFinding:
    return _finding(
        kind,
        "error",
        artifact_name,
        provider,
        f"provider fixture '{artifact_name}' has invalid replay field '{field}'",
        spans.get(field) or spans.get(field.rsplit(".", 1)[0]),
        (("field", field),),
        "Record this replay field with the expected scalar/list/object type.",
    )


def _finding(
    kind: ProviderFixtureReplayFindingKind,
    severity: str,
    artifact_name: str,
    provider: str,
    message: str,
    span: SourceSpan | None,
    evidence: tuple[tuple[str, str], ...],
    suggestion: str,
) -> ProviderFixtureReplayFinding:
    return ProviderFixtureReplayFinding(
        kind=kind,
        severity=severity,
        artifact_name=artifact_name,
        provider=provider,
        message=message,
        span=span,
        evidence=evidence,
        suggestion=suggestion,
    )


def _read_fixture(path: Path) -> tuple[dict[str, Any], dict[str, SourceSpan]]:
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        return {}, {}
    source_map = build_json_source_map(text, path)
    spans: dict[str, SourceSpan] = {}
    for json_path, span in source_map.spans.items():
        if json_path and json_path[-1] == "@key":
            key = ".".join(json_path[:-1])
            spans.setdefault(key, span)
            continue
        spans.setdefault(".".join(json_path), span)
    return raw, spans


def _surface_exists(raw: dict[str, Any], surface: str) -> bool:
    surface = _SURFACE_ALIASES.get(surface, surface)
    current: object = raw
    for part in surface.replace("[]", "").split("."):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _is_provider_snapshot(loaded: LoadedArtifact) -> bool:
    return loaded.artifact.kind is ArtifactKind.PROVIDER_CONFIG and loaded.source_type == "provider-config-snapshot"


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _edge_cases(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str) and value:
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return tuple(value)
    if isinstance(value, tuple) and all(isinstance(item, str) and item for item in value):
        return value
    return ()


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _span_key_from_path(path: str) -> str:
    return path.removeprefix("$.").replace("[", ".").replace("]", "")


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
