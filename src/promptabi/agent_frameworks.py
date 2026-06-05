"""Adapters for agent frameworks that assemble prompts from prompt packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, PromptPackArtifact, PromptPackTemplate, PromptSegment, artifact_from_config
from .loaders import ArtifactLoader


class AgentFrameworkIntegrationError(ValueError):
    """Raised when a dynamic agent prompt-pack integration spec is invalid."""


@dataclass(frozen=True, slots=True)
class DynamicContextSource:
    """One runtime source that a framework may splice into a prompt-pack template."""

    name: str
    segment: str
    role: str
    source_type: str = "runtime"
    required: bool = False

    def __post_init__(self) -> None:
        _require_non_empty("dynamic context source name", self.name)
        _require_non_empty("dynamic context source segment", self.segment)
        _require_non_empty("dynamic context source role", self.role)
        _require_non_empty("dynamic context source type", self.source_type)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required": self.required,
            "role": self.role,
            "segment": self.segment,
            "source_type": self.source_type,
        }


@dataclass(frozen=True, slots=True)
class AgentPromptPackAssembly:
    """A deterministic PromptABI view of a framework's dynamic prompt assembly."""

    name: str
    framework: str
    prompt_pack_path: str
    prompt_pack_version: str | None
    template_name: str
    selected_template: PromptPackTemplate
    segments: tuple[PromptSegment, ...]
    tool_names: tuple[str, ...]
    stop_sequences: tuple[str, ...]
    provider: str
    model_family: str
    dynamic_context_sources: tuple[DynamicContextSource, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("agent assembly name", self.name)
        _require_non_empty("agent assembly framework", self.framework)
        _require_non_empty("agent assembly prompt_pack_path", self.prompt_pack_path)
        _require_non_empty("agent assembly template_name", self.template_name)
        _require_non_empty("agent assembly provider", self.provider)
        _require_non_empty("agent assembly model_family", self.model_family)
        object.__setattr__(self, "tool_names", _unique_non_empty(self.tool_names, "agent assembly tool_names"))
        object.__setattr__(self, "stop_sequences", _unique_non_empty(self.stop_sequences, "agent assembly stop_sequences"))
        if not self.segments:
            raise AgentFrameworkIntegrationError("agent assembly must declare at least one prompt segment")
        if len({segment.name for segment in self.segments}) != len(self.segments):
            raise AgentFrameworkIntegrationError("agent assembly prompt segment names must be unique")

    @property
    def prompt_pack_artifact_name(self) -> str:
        return "support-pack"

    def to_promptabi_config(self) -> dict[str, object]:
        """Render the dynamic assembly as a normal PromptABI verification config."""

        prompt_pack: dict[str, object] = {
            "kind": ArtifactKind.PROMPT_PACK.value,
            "path": self.prompt_pack_path,
        }
        if self.prompt_pack_version is not None:
            prompt_pack["version"] = self.prompt_pack_version

        return {
            "name": f"{self.name}-prompt-pack-contract",
            "checks": ["prompt-pack-contracts"],
            "artifacts": {
                self.prompt_pack_artifact_name: prompt_pack,
                "messages": {
                    "kind": ArtifactKind.PROMPT_SEGMENT.value,
                    "uri": f"memory://agent-frameworks/{self.name}/messages",
                    "segments": [segment.to_dict() for segment in self.segments],
                },
                "tools": {
                    "kind": ArtifactKind.TOOL_DEFINITION.value,
                    "uri": f"memory://agent-frameworks/{self.name}/tools",
                    "provider": self.provider,
                    "tool_names": list(self.tool_names),
                },
                "stops": {
                    "kind": ArtifactKind.STOP_POLICY.value,
                    "uri": f"memory://agent-frameworks/{self.name}/stops",
                    "stop_sequences": list(self.stop_sequences),
                },
                "provider": {
                    "kind": ArtifactKind.PROVIDER_CONFIG.value,
                    "uri": f"memory://agent-frameworks/{self.name}/provider",
                    "provider": self.model_family,
                },
            },
        }

    def render_prompt_preview(self, values: dict[str, str]) -> str:
        """Render a compact deterministic preview for examples and tests."""

        missing = [variable for variable in self.selected_template.variables if variable not in values and variable != "messages"]
        if missing:
            raise AgentFrameworkIntegrationError(
                f"agent assembly preview is missing template variable(s): {', '.join(missing)}"
            )
        lines = [f"# {self.framework} assembly using {self.template_name}"]
        for segment in self.segments:
            value = values.get(segment.name, segment.content or f"<{segment.name}>")
            role = segment.role or "unknown"
            lines.append(f"{role}: {value}")
        return "\n".join(lines) + "\n"


def load_agent_prompt_pack_assembly(path: str | Path) -> AgentPromptPackAssembly:
    """Load and validate a dynamic agent prompt-pack integration spec."""

    spec_path = Path(path)
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AgentFrameworkIntegrationError(f"agent integration spec not found: {spec_path}") from exc
    except json.JSONDecodeError as exc:
        raise AgentFrameworkIntegrationError(
            f"agent integration spec is not valid JSON at {spec_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise AgentFrameworkIntegrationError("agent integration spec root must be a JSON object")
    return agent_prompt_pack_assembly_from_mapping(raw, base_dir=spec_path.parent)


def agent_prompt_pack_assembly_from_mapping(
    data: dict[str, Any],
    *,
    base_dir: str | Path,
) -> AgentPromptPackAssembly:
    """Build an assembly from JSON-like data and the prompt-pack file it references."""

    base_path = Path(base_dir)
    name = _required_str(data, "name")
    framework = _required_str(data, "framework")
    prompt_pack_spec = data.get("prompt_pack")
    if not isinstance(prompt_pack_spec, dict):
        raise AgentFrameworkIntegrationError("agent integration field 'prompt_pack' must be an object")
    prompt_pack_path = _required_str(prompt_pack_spec, "path")
    resolved_prompt_pack_path = str((base_path / prompt_pack_path).resolve()) if not Path(prompt_pack_path).is_absolute() else prompt_pack_path
    template_name = _required_str(prompt_pack_spec, "template")
    prompt_pack_version = _optional_str(prompt_pack_spec, "version")
    loaded_pack = _load_prompt_pack(resolved_prompt_pack_path, version=prompt_pack_version, base_dir=base_path)
    selected_template = _select_template(loaded_pack, template_name)

    segments = _segments(data.get("segments"))
    dynamic_sources = _dynamic_sources(data.get("dynamic_context_sources", []))
    _validate_dynamic_sources(dynamic_sources, segments)
    _validate_template_contract(selected_template, segments)

    provider = _required_str(data, "provider")
    model_family = _optional_str(data, "model_family") or provider
    return AgentPromptPackAssembly(
        name=name,
        framework=framework,
        prompt_pack_path=resolved_prompt_pack_path,
        prompt_pack_version=prompt_pack_version,
        template_name=template_name,
        selected_template=selected_template,
        segments=segments,
        tool_names=_string_tuple(data.get("tools", []), "tools"),
        stop_sequences=_string_tuple(data.get("stops", []), "stops"),
        provider=provider,
        model_family=model_family,
        dynamic_context_sources=dynamic_sources,
    )


def write_agent_promptabi_config(assembly: AgentPromptPackAssembly, path: str | Path) -> None:
    """Write the generated PromptABI config for a dynamic agent assembly."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(assembly.to_promptabi_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_agent_prompt_pack_plan(assembly: AgentPromptPackAssembly) -> str:
    """Render a concise human-readable description of the dynamic integration."""

    dynamic = ", ".join(source.name for source in assembly.dynamic_context_sources) or "none"
    required = ", ".join(segment.name for segment in assembly.segments if segment.required) or "none"
    return (
        "PromptABI agent prompt-pack assembly\n"
        f"name: {assembly.name}\n"
        f"framework: {assembly.framework}\n"
        f"template: {assembly.template_name}\n"
        f"provider/model-family: {assembly.provider}/{assembly.model_family}\n"
        f"tools: {', '.join(assembly.tool_names) or 'none'}\n"
        f"stops: {', '.join(assembly.stop_sequences) or 'none'}\n"
        f"required regions: {required}\n"
        f"dynamic sources: {dynamic}\n"
    )


def _load_prompt_pack(path: str, *, version: str | None, base_dir: Path) -> PromptPackArtifact:
    spec: dict[str, object] = {"kind": ArtifactKind.PROMPT_PACK.value, "path": path}
    if version is not None:
        spec["version"] = version
    artifact = artifact_from_config("support-pack", spec, base_dir=base_dir)
    loaded = ArtifactLoader().load(artifact)
    if not isinstance(loaded.artifact, PromptPackArtifact):
        raise AgentFrameworkIntegrationError("referenced prompt pack did not load as a prompt-pack artifact")
    return loaded.artifact


def _select_template(prompt_pack: PromptPackArtifact, template_name: str) -> PromptPackTemplate:
    for template in prompt_pack.exported_templates:
        if template.name == template_name:
            return template
    available = ", ".join(template.name for template in prompt_pack.exported_templates) or "none"
    raise AgentFrameworkIntegrationError(
        f"prompt pack '{prompt_pack.pack_name}' does not export template '{template_name}' (available: {available})"
    )


def _validate_template_contract(template: PromptPackTemplate, segments: tuple[PromptSegment, ...]) -> None:
    segment_names = {segment.name for segment in segments}
    missing_regions = [region for region in template.required_regions if region not in segment_names]
    if missing_regions:
        raise AgentFrameworkIntegrationError(
            f"agent assembly omits prompt-pack required region(s): {', '.join(missing_regions)}"
        )
    segment_roles = {segment.role for segment in segments if segment.role is not None}
    missing_roles = [role for role in template.roles if role not in segment_roles]
    if missing_roles:
        raise AgentFrameworkIntegrationError(
            f"agent assembly omits role(s) required by template '{template.name}': {', '.join(missing_roles)}"
        )


def _validate_dynamic_sources(sources: tuple[DynamicContextSource, ...], segments: tuple[PromptSegment, ...]) -> None:
    segment_names = {segment.name for segment in segments}
    missing = [source.segment for source in sources if source.segment not in segment_names]
    if missing:
        raise AgentFrameworkIntegrationError(
            f"dynamic context source(s) target unknown prompt segment(s): {', '.join(sorted(set(missing)))}"
        )


def _segments(value: object) -> tuple[PromptSegment, ...]:
    if not isinstance(value, list):
        raise AgentFrameworkIntegrationError("agent integration field 'segments' must be a list")
    segments: list[PromptSegment] = []
    for item in value:
        if not isinstance(item, dict):
            raise AgentFrameworkIntegrationError("agent integration segment entries must be objects")
        segments.append(
            PromptSegment(
                name=_required_str(item, "name"),
                role=_optional_str(item, "role"),
                required=_bool(item, "required", default=False),
                content=_optional_str(item, "content"),
                token_count=_optional_int(item, "token_count"),
                max_tokens=_optional_int(item, "max_tokens"),
            )
        )
    return tuple(segments)


def _dynamic_sources(value: object) -> tuple[DynamicContextSource, ...]:
    if not isinstance(value, list):
        raise AgentFrameworkIntegrationError("agent integration field 'dynamic_context_sources' must be a list")
    sources: list[DynamicContextSource] = []
    for item in value:
        if not isinstance(item, dict):
            raise AgentFrameworkIntegrationError("dynamic context source entries must be objects")
        sources.append(
            DynamicContextSource(
                name=_required_str(item, "name"),
                segment=_required_str(item, "segment"),
                role=_required_str(item, "role"),
                source_type=_optional_str(item, "source_type") or "runtime",
                required=_bool(item, "required", default=False),
            )
        )
    if len({source.name for source in sources}) != len(sources):
        raise AgentFrameworkIntegrationError("dynamic context source names must be unique")
    return tuple(sorted(sources, key=lambda source: source.name))


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise AgentFrameworkIntegrationError(f"agent integration field '{key}' must be a non-empty string")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AgentFrameworkIntegrationError(f"agent integration field '{key}' must be a non-empty string when present")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AgentFrameworkIntegrationError(f"agent integration field '{key}' must be a non-negative integer when present")
    return value


def _bool(data: dict[str, Any], key: str, *, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise AgentFrameworkIntegrationError(f"agent integration field '{key}' must be a boolean")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise AgentFrameworkIntegrationError(f"agent integration field '{field_name}' must be a list of non-empty strings")
    return _unique_non_empty(tuple(value), field_name)


def _unique_non_empty(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise AgentFrameworkIntegrationError(f"{field_name} values must be non-empty strings")
        result.append(value)
    return tuple(sorted(dict.fromkeys(result)))


def _require_non_empty(field_name: str, value: str) -> None:
    if not value:
        raise AgentFrameworkIntegrationError(f"{field_name} must be non-empty")
