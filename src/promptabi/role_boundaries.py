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
