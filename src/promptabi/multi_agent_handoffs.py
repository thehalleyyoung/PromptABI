"""Contract witnesses for multi-agent handoff boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .diagnostics import ArtifactRef, WitnessStep, WitnessTrace


MULTI_AGENT_HANDOFF_VERSION = "1.0"


class MultiAgentHandoffError(ValueError):
    """Raised when a multi-agent handoff manifest is invalid."""


class HandoffViolationKind(StrEnum):
    """Kinds of finite handoff contract violations PromptABI can witness."""

    UNKNOWN_AGENT = "unknown-agent"
    ROLE_REJECTED = "role-rejected"
    MISSING_REQUIRED_FIELD = "missing-required-field"
    TYPE_MISMATCH = "type-mismatch"
    FORBIDDEN_MARKER = "forbidden-marker"


@dataclass(frozen=True, slots=True)
class HandoffAgentContract:
    """One agent's declared input/output boundary for handoffs."""

    name: str
    accepts_roles: tuple[str, ...]
    emits_roles: tuple[str, ...]
    required_fields: tuple[str, ...] = ()
    input_schema: dict[str, str] | None = None
    forbidden_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("agent name", self.name)
        object.__setattr__(self, "accepts_roles", _unique_non_empty(self.accepts_roles, "agent accepts_roles"))
        object.__setattr__(self, "emits_roles", _unique_non_empty(self.emits_roles, "agent emits_roles"))
        if not self.accepts_roles:
            raise MultiAgentHandoffError("agent accepts_roles must be non-empty")
        if not self.emits_roles:
            raise MultiAgentHandoffError("agent emits_roles must be non-empty")
        object.__setattr__(self, "required_fields", _unique_non_empty(self.required_fields, "agent required_fields"))
        object.__setattr__(
            self,
            "forbidden_markers",
            _unique_non_empty(self.forbidden_markers, "agent forbidden_markers"),
        )
        if self.input_schema is not None:
            normalized = {str(key): str(value) for key, value in self.input_schema.items()}
            if any(not key or not value for key, value in normalized.items()):
                raise MultiAgentHandoffError("agent input_schema keys and values must be non-empty strings")
            unknown = set(normalized.values()) - {"string", "integer", "number", "boolean", "object", "array", "null"}
            if unknown:
                raise MultiAgentHandoffError(f"unsupported input_schema type(s): {', '.join(sorted(unknown))}")
            object.__setattr__(self, "input_schema", normalized)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "accepts_roles": list(self.accepts_roles),
            "emits_roles": list(self.emits_roles),
            "name": self.name,
        }
        if self.required_fields:
            data["required_fields"] = list(self.required_fields)
        if self.input_schema:
            data["input_schema"] = dict(self.input_schema)
        if self.forbidden_markers:
            data["forbidden_markers"] = list(self.forbidden_markers)
        return data


@dataclass(frozen=True, slots=True)
class HandoffPayload:
    """Concrete payload crossing from one agent to another."""

    role: str
    content: str
    fields: dict[str, object]

    def __post_init__(self) -> None:
        _require_non_empty("handoff payload role", self.role)
        object.__setattr__(self, "content", str(self.content))
        object.__setattr__(self, "fields", dict(self.fields))

    def to_dict(self) -> dict[str, object]:
        return {
            "content": self.content,
            "fields": dict(self.fields),
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class MultiAgentHandoff:
    """A concrete handoff edge in a multi-agent system."""

    name: str
    source: str
    target: str
    payload: HandoffPayload
    provenance_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("handoff name", self.name)
        _require_non_empty("handoff source", self.source)
        _require_non_empty("handoff target", self.target)
        object.__setattr__(
            self,
            "provenance_fields",
            _unique_non_empty(self.provenance_fields, "handoff provenance_fields"),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "from": self.source,
            "name": self.name,
            "payload": self.payload.to_dict(),
            "to": self.target,
        }
        if self.provenance_fields:
            data["provenance_fields"] = list(self.provenance_fields)
        return data


@dataclass(frozen=True, slots=True)
class HandoffContractViolation:
    """One violated handoff contract with a replayable witness."""

    handoff: str
    kind: HandoffViolationKind
    message: str
    source: str
    target: str
    witness: WitnessTrace
    suggestion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "handoff": self.handoff,
            "kind": self.kind.value,
            "message": self.message,
            "source": self.source,
            "suggestion": self.suggestion,
            "target": self.target,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MultiAgentHandoffReport:
    """Verification result for a multi-agent handoff manifest."""

    name: str
    agents: tuple[HandoffAgentContract, ...]
    handoffs: tuple[MultiAgentHandoff, ...]
    violations: tuple[HandoffContractViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "agents": [agent.to_dict() for agent in self.agents],
            "handoffs": [handoff.to_dict() for handoff in self.handoffs],
            "name": self.name,
            "ok": self.ok,
            "version": MULTI_AGENT_HANDOFF_VERSION,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def load_multi_agent_handoff_manifest(path: str | Path) -> MultiAgentHandoffReport:
    """Load and analyze a concrete multi-agent handoff manifest."""

    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MultiAgentHandoffError(f"multi-agent handoff manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise MultiAgentHandoffError(
            f"multi-agent handoff manifest is not valid JSON at {manifest_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise MultiAgentHandoffError("multi-agent handoff manifest root must be a JSON object")
    return analyze_multi_agent_handoffs(raw, manifest_path=manifest_path)


def analyze_multi_agent_handoffs(
    data: dict[str, Any],
    *,
    manifest_path: str | Path = "memory://multi-agent-handoffs",
) -> MultiAgentHandoffReport:
    """Analyze multi-agent handoff contracts and emit concrete witnesses."""

    name = _optional_str(data.get("name")) or "multi-agent-handoffs"
    agents = _parse_agents(data.get("agents"))
    handoffs = _parse_handoffs(data.get("handoffs"))
    agent_by_name = {agent.name: agent for agent in agents}
    if len(agent_by_name) != len(agents):
        raise MultiAgentHandoffError("agent names must be unique")
    if len({handoff.name for handoff in handoffs}) != len(handoffs):
        raise MultiAgentHandoffError("handoff names must be unique")

    manifest_ref = ArtifactRef(kind="multi-agent-handoff", name=name, path=str(manifest_path))
    violations: list[HandoffContractViolation] = []
    for handoff in handoffs:
        source = agent_by_name.get(handoff.source)
        target = agent_by_name.get(handoff.target)
        if source is None or target is None:
            missing = handoff.source if source is None else handoff.target
            violations.append(
                _violation(
                    handoff,
                    HandoffViolationKind.UNKNOWN_AGENT,
                    f"handoff references unknown agent '{missing}'",
                    manifest_ref=manifest_ref,
                    step_output=missing,
                    suggestion="Declare every handoff endpoint in the manifest agents list.",
                )
            )
            continue

        if handoff.payload.role not in source.emits_roles:
            violations.append(
                _violation(
                    handoff,
                    HandoffViolationKind.ROLE_REJECTED,
                    f"source agent '{source.name}' does not declare emitted role '{handoff.payload.role}'",
                    manifest_ref=manifest_ref,
                    step_output=f"{source.name} emits {', '.join(source.emits_roles)}",
                    suggestion="Normalize the handoff role to one emitted by the source agent.",
                )
            )
        if handoff.payload.role not in target.accepts_roles:
            violations.append(
                _violation(
                    handoff,
                    HandoffViolationKind.ROLE_REJECTED,
                    f"target agent '{target.name}' rejects role '{handoff.payload.role}'",
                    manifest_ref=manifest_ref,
                    step_output=f"{target.name} accepts {', '.join(target.accepts_roles)}",
                    suggestion="Insert a handoff adapter that rewrites the payload role into the target's accepted role set.",
                )
            )

        required = (*target.required_fields, *handoff.provenance_fields)
        for field in dict.fromkeys(required):
            value = handoff.payload.fields.get(field)
            if value in (None, ""):
                violations.append(
                    _violation(
                        handoff,
                        HandoffViolationKind.MISSING_REQUIRED_FIELD,
                        f"handoff payload omits required field '{field}' for target '{target.name}'",
                        manifest_ref=manifest_ref,
                        step_output=field,
                        suggestion="Carry the field through the handoff payload or remove it from the target contract.",
                    )
                )

        for field, expected_type in (target.input_schema or {}).items():
            if field not in handoff.payload.fields:
                continue
            actual = _json_type(handoff.payload.fields[field])
            if actual != expected_type and not (expected_type == "number" and actual == "integer"):
                violations.append(
                    _violation(
                        handoff,
                        HandoffViolationKind.TYPE_MISMATCH,
                        f"field '{field}' has type '{actual}' but target '{target.name}' expects '{expected_type}'",
                        manifest_ref=manifest_ref,
                        step_output=f"{field}: {actual} != {expected_type}",
                        suggestion="Serialize or validate handoff fields before the target agent consumes them.",
                    )
                )

        searchable_values = _payload_strings(handoff.payload)
        markers = tuple(dict.fromkeys((*source.forbidden_markers, *target.forbidden_markers)))
        for marker in markers:
            for location, value in searchable_values:
                if marker in value:
                    violations.append(
                        _violation(
                            handoff,
                            HandoffViolationKind.FORBIDDEN_MARKER,
                            f"handoff field '{location}' contains forbidden control marker {marker!r}",
                            manifest_ref=manifest_ref,
                            step_output=f"{location} contains {marker}",
                            suggestion="Escape, JSON-encode, or structurally wrap handoff text before forwarding it.",
                            rendered=value,
                        )
                    )
                    break

    return MultiAgentHandoffReport(
        name=name,
        agents=agents,
        handoffs=handoffs,
        violations=tuple(violations),
    )


def render_multi_agent_handoff_json(report: MultiAgentHandoffReport) -> str:
    """Render a handoff report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_multi_agent_handoff_text(report: MultiAgentHandoffReport) -> str:
    """Render a handoff report for CLI users."""

    status = "PASS" if report.ok else "FAIL"
    lines = [
        "PromptABI multi-agent handoff contracts",
        f"name: {report.name}",
        f"status: {status}",
        f"agents: {len(report.agents)}",
        f"handoffs: {len(report.handoffs)}",
    ]
    if not report.violations:
        lines.append("violations: none")
        return "\n".join(lines) + "\n"
    lines.append(f"violations: {len(report.violations)}")
    for violation in report.violations:
        lines.append(f"ERROR {violation.kind.value} [{violation.handoff}]: {violation.message}")
        lines.append(f"  source -> target: {violation.source} -> {violation.target}")
        lines.append(f"  witness: {violation.witness.summary}")
        for index, step in enumerate(violation.witness.steps, start=1):
            detail = f"    {index}. {step.action}"
            if step.input is not None:
                detail += f" | input: {step.input}"
            if step.output is not None:
                detail += f" | output: {step.output}"
            lines.append(detail)
        lines.append(f"  suggestion: {violation.suggestion}")
    return "\n".join(lines) + "\n"


def _parse_agents(raw: object) -> tuple[HandoffAgentContract, ...]:
    if not isinstance(raw, list) or not raw:
        raise MultiAgentHandoffError("multi-agent handoff manifest must include a non-empty agents list")
    agents: list[HandoffAgentContract] = []
    for item in raw:
        if not isinstance(item, dict):
            raise MultiAgentHandoffError("each agent entry must be an object")
        agents.append(
            HandoffAgentContract(
                name=_required_str(item, "name"),
                accepts_roles=_string_tuple(item.get("accepts_roles", ()), "accepts_roles"),
                emits_roles=_string_tuple(item.get("emits_roles", ()), "emits_roles"),
                required_fields=_string_tuple(item.get("required_fields", ()), "required_fields"),
                input_schema=_optional_schema(item.get("input_schema")),
                forbidden_markers=_string_tuple(item.get("forbidden_markers", ()), "forbidden_markers"),
            )
        )
    return tuple(agents)


def _parse_handoffs(raw: object) -> tuple[MultiAgentHandoff, ...]:
    if not isinstance(raw, list) or not raw:
        raise MultiAgentHandoffError("multi-agent handoff manifest must include a non-empty handoffs list")
    handoffs: list[MultiAgentHandoff] = []
    for item in raw:
        if not isinstance(item, dict):
            raise MultiAgentHandoffError("each handoff entry must be an object")
        payload_raw = item.get("payload")
        if not isinstance(payload_raw, dict):
            raise MultiAgentHandoffError("each handoff payload must be an object")
        fields = payload_raw.get("fields", {})
        if not isinstance(fields, dict):
            raise MultiAgentHandoffError("handoff payload fields must be an object")
        handoffs.append(
            MultiAgentHandoff(
                name=_required_str(item, "name"),
                source=_required_str(item, "from"),
                target=_required_str(item, "to"),
                payload=HandoffPayload(
                    role=_required_str(payload_raw, "role"),
                    content=str(payload_raw.get("content", "")),
                    fields=fields,
                ),
                provenance_fields=_string_tuple(item.get("provenance_fields", ()), "provenance_fields"),
            )
        )
    return tuple(handoffs)


def _violation(
    handoff: MultiAgentHandoff,
    kind: HandoffViolationKind,
    message: str,
    *,
    manifest_ref: ArtifactRef,
    step_output: str,
    suggestion: str,
    rendered: str | None = None,
) -> HandoffContractViolation:
    rendered_strings = (rendered,) if rendered else ()
    return HandoffContractViolation(
        handoff=handoff.name,
        kind=kind,
        message=message,
        source=handoff.source,
        target=handoff.target,
        witness=WitnessTrace(
            summary=f"{handoff.name} violates {kind.value}",
            steps=(
                WitnessStep(
                    action="select concrete handoff edge",
                    input=f"{handoff.source}->{handoff.target}",
                    output=handoff.name,
                ),
                WitnessStep(
                    action="replay handoff payload contract",
                    input=f"role={handoff.payload.role}",
                    output=step_output,
                ),
                WitnessStep(
                    action="emit minimal handoff fix",
                    input=kind.value,
                    output=suggestion,
                ),
            ),
            artifacts=(manifest_ref,),
            rendered_strings=rendered_strings,
            minimal_fixes=(suggestion,),
        ),
        suggestion=suggestion,
    )


def _payload_strings(payload: HandoffPayload) -> tuple[tuple[str, str], ...]:
    values = [("content", payload.content)]
    for key, value in sorted(payload.fields.items()):
        if isinstance(value, str):
            values.append((f"fields.{key}", value))
    return tuple(values)


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _optional_schema(raw: object) -> dict[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise MultiAgentHandoffError("agent input_schema must be an object")
    return {str(key): str(value) for key, value in raw.items()}


def _string_tuple(raw: object, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple):
        raise MultiAgentHandoffError(f"{field_name} must be a list of strings")
    if any(not isinstance(item, str) for item in raw):
        raise MultiAgentHandoffError(f"{field_name} must be a list of strings")
    return tuple(raw)


def _unique_non_empty(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value) for value in values))
    if any(not value for value in normalized):
        raise MultiAgentHandoffError(f"{field_name} values must be non-empty")
    return normalized


def _required_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise MultiAgentHandoffError(f"required field '{key}' must be a non-empty string")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise MultiAgentHandoffError("optional string fields must be non-empty strings when present")
    return value


def _require_non_empty(label: str, value: str) -> None:
    if not value:
        raise MultiAgentHandoffError(f"{label} must be non-empty")
