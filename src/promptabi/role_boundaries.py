"""Role-region modeling over bounded chat-template renderings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chat_templates import (
    ChatTemplateParseResult,
    ChatTemplateSymbolicBounds,
    ChatTemplateSymbolicPath,
    symbolically_execute_chat_template,
)


DEFAULT_STRUCTURAL_ROLES = (
    "system",
    "user",
    "assistant",
    "tool",
    "developer",
    "function",
)

_MESSAGE_PLACEHOLDER_RE = re.compile(r"\{messages\[(?P<index>\d+)\]\.(?P<field>[A-Za-z_][A-Za-z0-9_]*)\}")
_POSITIVE_ROLE_RE = re.compile(
    r"""messages\[(?P<index>\d+)\](?:\[['"]role['"]\]|\.role)\s*==\s*(['"])(?P<role>[^'"]+)\2"""
)
_NEGATIVE_ROLE_RE = re.compile(
    r"""(?:not\()?\s*messages\[(?P<index>\d+)\](?:\[['"]role['"]\]|\.role)\s*(?:!=|==)\s*(['"])(?P<role>[^'"]+)\2"""
)


@dataclass(frozen=True, slots=True)
class RoleBoundaryRegion:
    """One structural role region in a symbolic rendered prompt pattern.

    Offsets are character offsets into the symbolic rendered pattern, where
    variable content is represented by placeholders such as
    ``{messages[0].content}``.
    """

    path_index: int
    region_index: int
    role: str
    role_source: str
    start_offset: int
    end_offset: int
    segment_indexes: tuple[int, ...]
    message_index: int | None = None
    content_expressions: tuple[str, ...] = ()
    control_text: str = ""
    excluded_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.path_index < 0:
            raise ValueError("path_index must be non-negative")
        if self.region_index < 0:
            raise ValueError("region_index must be non-negative")
        if not self.role:
            raise ValueError("role must be non-empty")
        if not self.role_source:
            raise ValueError("role_source must be non-empty")
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("region offsets must be monotonic")
        if self.message_index is not None and self.message_index < 0:
            raise ValueError("message_index must be non-negative")
        object.__setattr__(self, "segment_indexes", tuple(self.segment_indexes))
        object.__setattr__(self, "content_expressions", tuple(dict.fromkeys(self.content_expressions)))
        object.__setattr__(self, "excluded_roles", tuple(sorted(dict.fromkeys(self.excluded_roles))))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "path_index": self.path_index,
            "region_index": self.region_index,
            "role": self.role,
            "role_source": self.role_source,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "segment_indexes": list(self.segment_indexes),
        }
        if self.message_index is not None:
            data["message_index"] = self.message_index
        if self.content_expressions:
            data["content_expressions"] = list(self.content_expressions)
        if self.control_text:
            data["control_text"] = self.control_text
        if self.excluded_roles:
            data["excluded_roles"] = list(self.excluded_roles)
        return data


@dataclass(frozen=True, slots=True)
class RoleBoundaryPath:
    """Role-region decomposition for one bounded symbolic template path."""

    path_index: int
    conditions: tuple[str, ...]
    loop_iterations: tuple[tuple[str, int], ...]
    rendered_pattern: str
    regions: tuple[RoleBoundaryRegion, ...] = ()
    unassigned_segment_indexes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.path_index < 0:
            raise ValueError("path_index must be non-negative")
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "loop_iterations", tuple(self.loop_iterations))
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "unassigned_segment_indexes", tuple(self.unassigned_segment_indexes))

    def to_dict(self) -> dict[str, object]:
        return {
            "path_index": self.path_index,
            "conditions": list(self.conditions),
            "loop_iterations": [
                {"iterable": iterable, "count": count}
                for iterable, count in self.loop_iterations
            ],
            "rendered_pattern": self.rendered_pattern,
            "regions": [region.to_dict() for region in self.regions],
            "unassigned_segment_indexes": list(self.unassigned_segment_indexes),
        }


@dataclass(frozen=True, slots=True)
class RoleBoundaryModel:
    """Bounded structural role model derived from chat-template rendering."""

    supported: bool
    roles: tuple[str, ...]
    paths: tuple[RoleBoundaryPath, ...] = ()
    abstentions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(sorted(dict.fromkeys(self.roles))))
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "abstentions", tuple(dict.fromkeys(self.abstentions)))

    def to_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "roles": list(self.roles),
            "path_count": len(self.paths),
            "paths": [path.to_dict() for path in self.paths],
            "abstentions": list(self.abstentions),
        }


@dataclass(frozen=True, slots=True)
class RoleBoundaryForgeryFinding:
    """A bounded structural role-boundary forgery witness."""

    path_index: int
    region_index: int
    input_expression: str
    input_role: str
    marker: str
    marker_kind: str
    rendered_excerpt: str
    boundary_description: str

    def __post_init__(self) -> None:
        if self.path_index < 0:
            raise ValueError("path_index must be non-negative")
        if self.region_index < 0:
            raise ValueError("region_index must be non-negative")
        for field_name in (
            "input_expression",
            "input_role",
            "marker",
            "marker_kind",
            "rendered_excerpt",
            "boundary_description",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "path_index": self.path_index,
            "region_index": self.region_index,
            "input_expression": self.input_expression,
            "input_role": self.input_role,
            "marker": self.marker,
            "marker_kind": self.marker_kind,
            "rendered_excerpt": self.rendered_excerpt,
            "boundary_description": self.boundary_description,
        }


@dataclass(frozen=True, slots=True)
class RoleBoundaryNonforgeabilityReport:
    """Result of the first bounded role-boundary non-forgeability check."""

    model: RoleBoundaryModel
    findings: tuple[RoleBoundaryForgeryFinding, ...] = ()
    marker_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        if self.marker_count < 0:
            raise ValueError("marker_count must be non-negative")

    @property
    def ok(self) -> bool:
        return not self.findings and self.model.supported

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "marker_count": self.marker_count,
            "model": self.model.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def build_role_boundary_model(
    parsed: ChatTemplateParseResult,
    *,
    bounds: ChatTemplateSymbolicBounds | None = None,
) -> RoleBoundaryModel:
    """Build a bounded structural role-region model for a chat template."""

    symbolic = symbolically_execute_chat_template(parsed, bounds=bounds)
    role_paths = tuple(
        _build_path(path, path_index, parsed.role_assumptions)
        for path_index, path in enumerate(symbolic.paths)
    )
    concrete_roles = {
        region.role
        for path in role_paths
        for region in path.regions
        if region.role != "unknown" and not region.role.startswith("{")
    }
    concrete_roles.update(parsed.role_assumptions)
    abstentions = tuple(f"{item.kind}: {item.expression}" for item in symbolic.abstentions)
    return RoleBoundaryModel(
        supported=symbolic.supported,
        roles=tuple(concrete_roles),
        paths=role_paths,
        abstentions=abstentions,
    )


def analyze_role_boundary_nonforgeability(
    parsed: ChatTemplateParseResult,
    *,
    bounds: ChatTemplateSymbolicBounds | None = None,
) -> RoleBoundaryNonforgeabilityReport:
    """Check whether user-controlled fields can render structural controls.

    The first check is exact for the bounded symbolic template paths it models:
    raw message content and raw dynamic role fields are treated as arbitrary
    attacker-controlled strings, while sanitizer recognition is intentionally
    deferred to the later sanitizer-awareness milestone.
    """

    model = build_role_boundary_model(parsed, bounds=bounds)
    markers = _structural_marker_catalog(parsed, model)
    findings: list[RoleBoundaryForgeryFinding] = []
    seen: set[tuple[int, int, str, str, str]] = set()

    for path in model.paths:
        for region in path.regions:
            for finding in _role_header_findings(path, region):
                key = (
                    finding.path_index,
                    finding.region_index,
                    finding.input_expression,
                    finding.marker,
                    finding.marker_kind,
                )
                if key not in seen:
                    seen.add(key)
                    findings.append(finding)
            if not _is_user_controlled_region(region):
                continue
            for expression in region.content_expressions:
                for marker, marker_kind in markers:
                    if marker in expression:
                        continue
                    finding = RoleBoundaryForgeryFinding(
                        path_index=path.path_index,
                        region_index=region.region_index,
                        input_expression=expression,
                        input_role=region.role,
                        marker=marker,
                        marker_kind=marker_kind,
                        rendered_excerpt=_rendered_excerpt(path.rendered_pattern, expression, marker),
                        boundary_description=(
                            f"{expression} can render {marker_kind} marker {marker!r} "
                            f"inside a {region.role} region"
                        ),
                    )
                    key = (
                        finding.path_index,
                        finding.region_index,
                        finding.input_expression,
                        finding.marker,
                        finding.marker_kind,
                    )
                    if key not in seen:
                        seen.add(key)
                        findings.append(finding)

    findings.sort(
        key=lambda finding: (
            finding.path_index,
            finding.region_index,
            finding.input_expression,
            finding.marker_kind,
            finding.marker,
        )
    )
    return RoleBoundaryNonforgeabilityReport(
        model=model,
        findings=tuple(findings),
        marker_count=len(markers),
    )


def _build_path(
    path: ChatTemplateSymbolicPath,
    path_index: int,
    declared_roles: tuple[str, ...],
) -> RoleBoundaryPath:
    offsets = _segment_offsets(path)
    positive_roles = _positive_roles(path.conditions)
    excluded_roles = _excluded_roles(path.conditions)
    message_segments = _message_segments(path)
    generation_start = _generation_prompt_start(path)
    ranges = _message_region_ranges(message_segments, len(path.segments), generation_start)
    regions: list[RoleBoundaryRegion] = []
    assigned_segments: set[int] = set()

    for message_index, segment_indexes in sorted(message_segments.items()):
        start_segment, end_segment = ranges[message_index]
        segment_range = tuple(range(start_segment, end_segment + 1))
        assigned_segments.update(segment_range)
        role, role_source = _region_role(
            path,
            segment_range,
            message_index,
            positive_roles,
            declared_roles,
        )
        content_expressions = _content_expressions(path, message_index)
        start_offset = offsets[start_segment][0]
        end_offset = offsets[end_segment][1]
        regions.append(
            RoleBoundaryRegion(
                path_index=path_index,
                region_index=len(regions),
                role=role,
                role_source=role_source,
                message_index=message_index,
                start_offset=start_offset,
                end_offset=end_offset,
                segment_indexes=segment_range,
                content_expressions=content_expressions,
                control_text=_control_text(path, segment_range, message_index),
                excluded_roles=excluded_roles.get(message_index, ()),
            )
        )

    generation_region = _generation_prompt_region(
        path,
        path_index,
        len(regions),
        assigned_segments,
        offsets,
    )
    if generation_region is not None:
        regions.append(generation_region)
        assigned_segments.update(generation_region.segment_indexes)

    unassigned = tuple(index for index in range(len(path.segments)) if index not in assigned_segments)
    return RoleBoundaryPath(
        path_index=path_index,
        conditions=path.conditions,
        loop_iterations=path.loop_iterations,
        rendered_pattern=path.rendered_pattern,
        regions=tuple(regions),
        unassigned_segment_indexes=unassigned,
    )


def _segment_offsets(path: ChatTemplateSymbolicPath) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for segment in path.segments:
        end = cursor + len(segment.value)
        offsets.append((cursor, end))
        cursor = end
    return tuple(offsets)


def _message_segments(path: ChatTemplateSymbolicPath) -> dict[int, tuple[int, ...]]:
    by_message: dict[int, list[int]] = {}
    for index, segment in enumerate(path.segments):
        for match in _MESSAGE_PLACEHOLDER_RE.finditer(segment.value):
            by_message.setdefault(int(match.group("index")), []).append(index)
    return {message_index: tuple(indexes) for message_index, indexes in by_message.items()}


def _message_region_ranges(
    message_segments: dict[int, tuple[int, ...]],
    segment_count: int,
    generation_start: int | None,
) -> dict[int, tuple[int, int]]:
    ordered = sorted(
        (min(indexes), max(indexes), message_index)
        for message_index, indexes in message_segments.items()
    )
    ranges: dict[int, tuple[int, int]] = {}
    cursor = 0
    for position, (first, last, message_index) in enumerate(ordered):
        if position + 1 < len(ordered):
            next_first = ordered[position + 1][0]
        elif generation_start is not None:
            next_first = min(generation_start, segment_count)
        else:
            next_first = segment_count
        start = min(cursor, first)
        end = max(last, next_first - 1)
        ranges[message_index] = (start, end)
        cursor = end + 1
    return ranges


def _positive_roles(conditions: tuple[str, ...]) -> dict[int, str]:
    roles: dict[int, str] = {}
    for condition in conditions:
        if condition.strip().startswith(("not(", "else after")):
            continue
        for match in _POSITIVE_ROLE_RE.finditer(condition):
            roles[int(match.group("index"))] = match.group("role")
    return roles


def _excluded_roles(conditions: tuple[str, ...]) -> dict[int, tuple[str, ...]]:
    excluded: dict[int, set[str]] = {}
    for condition in conditions:
        for match in _NEGATIVE_ROLE_RE.finditer(condition):
            if "!=" in match.group(0) or condition.strip().startswith(("not(", "else after")):
                excluded.setdefault(int(match.group("index")), set()).add(match.group("role"))
    return {index: tuple(sorted(roles)) for index, roles in excluded.items()}


def _region_role(
    path: ChatTemplateSymbolicPath,
    segment_range: tuple[int, ...],
    message_index: int,
    positive_roles: dict[int, str],
    declared_roles: tuple[str, ...],
) -> tuple[str, str]:
    if message_index in positive_roles:
        return positive_roles[message_index], "condition"
    for segment_index in segment_range:
        segment = path.segments[segment_index]
        if f"{{messages[{message_index}].role}}" in segment.value:
            return f"{{messages[{message_index}].role}}", "variable"
    literal_role = _literal_role(_control_text(path, segment_range, message_index), declared_roles)
    if literal_role is not None:
        return literal_role, "literal"
    return "unknown", "residual"


def _literal_role(control_text: str, declared_roles: tuple[str, ...]) -> str | None:
    for role in tuple(sorted(set(DEFAULT_STRUCTURAL_ROLES).union(declared_roles), key=len, reverse=True)):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(role)}(?![A-Za-z0-9_])", control_text):
            return role
    return None


def _content_expressions(path: ChatTemplateSymbolicPath, message_index: int) -> tuple[str, ...]:
    expressions: list[str] = []
    for segment in path.segments:
        for match in _MESSAGE_PLACEHOLDER_RE.finditer(segment.value):
            if int(match.group("index")) != message_index or match.group("field") == "role":
                continue
            expressions.append(match.group(0))
    return tuple(dict.fromkeys(expressions))


def _control_text(
    path: ChatTemplateSymbolicPath,
    segment_range: tuple[int, ...],
    message_index: int,
) -> str:
    pieces: list[str] = []
    for segment_index in segment_range:
        value = path.segments[segment_index].value
        for match in _MESSAGE_PLACEHOLDER_RE.finditer(value):
            if int(match.group("index")) == message_index and match.group("field") != "role":
                value = value.replace(match.group(0), "")
        pieces.append(value)
    return "".join(pieces)


def _generation_prompt_region(
    path: ChatTemplateSymbolicPath,
    path_index: int,
    region_index: int,
    assigned_segments: set[int],
    offsets: tuple[tuple[int, int], ...],
) -> RoleBoundaryRegion | None:
    if "add_generation_prompt" not in path.conditions:
        return None
    candidates = _generation_prompt_segment_indexes(path, assigned_segments)
    if not candidates:
        return None
    start_segment = min(candidates)
    end_segment = max(candidates)
    segment_indexes = tuple(range(start_segment, end_segment + 1))
    return RoleBoundaryRegion(
        path_index=path_index,
        region_index=region_index,
        role="assistant",
        role_source="generation-prompt",
        start_offset=offsets[start_segment][0],
        end_offset=offsets[end_segment][1],
        segment_indexes=segment_indexes,
        control_text="".join(path.segments[index].value for index in segment_indexes),
    )


def _generation_prompt_start(path: ChatTemplateSymbolicPath) -> int | None:
    indexes = _generation_prompt_segment_indexes(path, set())
    return min(indexes) if indexes else None


def _generation_prompt_segment_indexes(
    path: ChatTemplateSymbolicPath,
    assigned_segments: set[int],
) -> tuple[int, ...]:
    if "add_generation_prompt" not in path.conditions:
        return ()
    return tuple(
        index
        for index, segment in enumerate(path.segments)
        if index not in assigned_segments and "assistant" in segment.value
    )


def _role_header_findings(
    path: RoleBoundaryPath,
    region: RoleBoundaryRegion,
) -> tuple[RoleBoundaryForgeryFinding, ...]:
    if region.role_source != "variable" or region.message_index is None:
        return ()
    expression = f"{{messages[{region.message_index}].role}}"
    forgeable_roles = tuple(
        role
        for role in DEFAULT_STRUCTURAL_ROLES
        if role not in region.excluded_roles and role != "user"
    )
    findings: list[RoleBoundaryForgeryFinding] = []
    for role in forgeable_roles:
        findings.append(
            RoleBoundaryForgeryFinding(
                path_index=path.path_index,
                region_index=region.region_index,
                input_expression=expression,
                input_role=region.role,
                marker=role,
                marker_kind="role-header",
                rendered_excerpt=_rendered_excerpt(path.rendered_pattern, expression, role),
                boundary_description=(
                    f"{expression} is rendered directly into a role header and can become {role!r}"
                ),
            )
        )
    return tuple(findings)


def _is_user_controlled_region(region: RoleBoundaryRegion) -> bool:
    if region.role in {"user", "tool", "function"}:
        return True
    if region.role_source in {"variable", "residual"}:
        return True
    return region.role not in {"assistant", "system", "developer"}


def _structural_marker_catalog(
    parsed: ChatTemplateParseResult,
    model: RoleBoundaryModel,
) -> tuple[tuple[str, str], ...]:
    markers: dict[str, str] = {}
    for token in parsed.special_tokens:
        _add_marker(markers, token.text, "special-token")
    for path in model.paths:
        for region in path.regions:
            for literal in _literal_control_runs(region.control_text):
                for marker in _extract_marker_candidates(literal):
                    _add_marker(markers, marker, _marker_kind(marker, region))
    return tuple(sorted(markers.items(), key=lambda item: (item[1], item[0])))


def _literal_control_runs(control_text: str) -> tuple[str, ...]:
    runs: list[str] = []
    cursor = 0
    for match in _MESSAGE_PLACEHOLDER_RE.finditer(control_text):
        if match.start() > cursor:
            runs.append(control_text[cursor : match.start()])
        cursor = match.end()
    if cursor < len(control_text):
        runs.append(control_text[cursor:])
    return tuple(run for run in runs if run)


_ANGLE_SENTINEL_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9_:-]*(?:\s[^>\n]{0,120})?>|<\|[^|\n]{1,120}\|>")
_BRACKET_SENTINEL_RE = re.compile(r"\[/?[A-Za-z][A-Za-z0-9_ -]{1,80}\]")
_FENCE_SENTINEL_RE = re.compile(r"```[A-Za-z0-9_-]*")


def _extract_marker_candidates(literal: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for regex in (_ANGLE_SENTINEL_RE, _BRACKET_SENTINEL_RE, _FENCE_SENTINEL_RE):
        candidates.extend(match.group(0) for match in regex.finditer(literal))
    stripped = literal.strip()
    if _significant_marker(stripped) and any(role in stripped for role in DEFAULT_STRUCTURAL_ROLES):
        candidates.append(stripped)
    return tuple(dict.fromkeys(candidate for candidate in candidates if _significant_marker(candidate)))


def _significant_marker(marker: str) -> bool:
    if len(marker.strip()) < 3:
        return False
    if marker.strip() in DEFAULT_STRUCTURAL_ROLES:
        return False
    return bool(re.search(r"[<>\[\]`|/]", marker))


def _marker_kind(marker: str, region: RoleBoundaryRegion) -> str:
    lowered = marker.lower()
    if "tool" in lowered or "function" in lowered:
        return "tool-call-sentinel"
    if region.role == "assistant" or "assistant" in lowered:
        return "assistant-prefix"
    if any(role in lowered for role in DEFAULT_STRUCTURAL_ROLES):
        return "role-header"
    return "control-delimiter"


def _add_marker(markers: dict[str, str], marker: str, kind: str) -> None:
    if not _significant_marker(marker):
        return
    previous = markers.get(marker)
    if previous is None or _marker_kind_rank(kind) < _marker_kind_rank(previous):
        markers[marker] = kind


def _marker_kind_rank(kind: str) -> int:
    return {
        "role-header": 0,
        "assistant-prefix": 1,
        "tool-call-sentinel": 2,
        "special-token": 3,
        "control-delimiter": 4,
    }.get(kind, 5)


def _rendered_excerpt(rendered_pattern: str, expression: str, replacement: str) -> str:
    rendered = rendered_pattern.replace(expression, replacement, 1)
    index = rendered.find(replacement)
    if index < 0:
        return rendered[:160]
    start = max(0, index - 60)
    end = min(len(rendered), index + len(replacement) + 60)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(rendered) else ""
    return prefix + rendered[start:end] + suffix
