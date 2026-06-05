"""Provider migration dry-run patches (step 287).

Migrating a request from one provider to another is a mechanical-but-error-prone
remapping of parameter names and value ranges (``max_tokens`` ->
``max_output_tokens``, ``stop`` -> ``stop_sequences``, unsupported params
dropped, clamped ranges).  This module computes a *dry-run patch*: the exact set
of rename/clamp/drop operations needed to turn a source request into a valid
target request, plus warnings for any semantically lossy change -- so a migration
can be reviewed before it is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

PROVIDER_MIGRATION_DRYRUN_VERSION = "promptabi.provider-migration-dryrun.v1"


class PatchOpKind(StrEnum):
    RENAME = "rename"
    CLAMP = "clamp"
    DROP = "drop"
    KEEP = "keep"


@dataclass(frozen=True, slots=True)
class ParamSpec:
    name: str
    max_value: float | None = None


@dataclass(frozen=True, slots=True)
class TargetSchema:
    """Target provider's accepted params and source->target renames."""

    accepted: dict[str, ParamSpec]
    renames: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PatchOp:
    kind: PatchOpKind
    source_param: str
    target_param: str | None
    detail: str
    lossy: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "source_param": self.source_param,
            "target_param": self.target_param,
            "detail": self.detail,
            "lossy": self.lossy,
        }


@dataclass(frozen=True, slots=True)
class MigrationPatch:
    version: str
    ops: tuple[PatchOp, ...]
    target_request: dict[str, object]

    @property
    def lossy(self) -> bool:
        return any(op.lossy for op in self.ops)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "lossy": self.lossy,
            "ops": [op.to_dict() for op in self.ops],
            "target_request": self.target_request,
        }


def dry_run_migration(
    source_request: dict[str, object],
    target: TargetSchema,
) -> MigrationPatch:
    ops: list[PatchOp] = []
    target_request: dict[str, object] = {}

    for name, value in source_request.items():
        target_name = target.renames.get(name, name)
        spec = target.accepted.get(target_name)

        if spec is None:
            ops.append(
                PatchOp(
                    PatchOpKind.DROP,
                    name,
                    None,
                    f"{name!r} has no equivalent on target",
                    lossy=True,
                )
            )
            continue

        new_value = value
        clamped = False
        if (
            spec.max_value is not None
            and isinstance(value, (int, float))
            and value > spec.max_value
        ):
            new_value = spec.max_value
            clamped = True

        if target_name != name:
            ops.append(
                PatchOp(
                    PatchOpKind.RENAME,
                    name,
                    target_name,
                    f"{name!r} -> {target_name!r}",
                )
            )
        if clamped:
            ops.append(
                PatchOp(
                    PatchOpKind.CLAMP,
                    name,
                    target_name,
                    f"{value} -> {new_value} (max {spec.max_value})",
                    lossy=True,
                )
            )
        if target_name == name and not clamped:
            ops.append(PatchOp(PatchOpKind.KEEP, name, target_name, "unchanged"))

        target_request[target_name] = new_value

    return MigrationPatch(
        version=PROVIDER_MIGRATION_DRYRUN_VERSION,
        ops=tuple(ops),
        target_request=target_request,
    )


def render_migration_patch_text(patch: MigrationPatch) -> str:
    lines = [
        f"PromptABI provider migration dry-run ({patch.version})",
        f"lossy: {'yes' if patch.lossy else 'no'}",
    ]
    for op in patch.ops:
        flag = " (lossy)" if op.lossy else ""
        lines.append(f"  {op.kind.value}: {op.detail}{flag}")
    return "\n".join(lines) + "\n"
