"""Model prompt-pack imports as a static module system with a verified linker.

PromptABI prompt packs (:class:`promptabi.artifacts.PromptPackArtifact`) export
named symbols: chat templates, tool schemas, and stop policies.  Real prompt
libraries reuse one another -- an application pack depends on a shared "base"
pack -- which is exactly a *module system*.  This module treats every loaded
prompt pack as a module that **exports** qualified symbols and may **import**
symbols from other packs, then statically links the program:

* every import must resolve to a symbol actually exported by the named pack
  (no dangling imports);
* an imported symbol must satisfy the importer's declared minimum pack version
  (no silent ABI regressions);
* the import graph must be acyclic so a deterministic link order exists;
* a qualified symbol exported by two different packs is a namespace collision;
* two imports in one pack may not bind the same local name to different
  sources.

Imports are declared in the pack JSON under a top-level ``imports`` list and are
read offline; no code is executed.  When the program links cleanly the resolver
returns the unique topological link order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import PromptPackArtifact
from .diagnostics import ArtifactRef, WitnessStep, WitnessTrace
from .loaders import LoadedArtifact


PROMPT_PACK_MODULE_VERSION = "promptabi.prompt-pack-modules.v1"

_SYMBOL_KINDS = ("template", "tool", "stop")


class PromptPackModuleFindingKind(StrEnum):
    """Concrete static-linking defects in a prompt-pack module program."""

    UNRESOLVED_PACK = "unresolved-pack"
    UNRESOLVED_SYMBOL = "unresolved-symbol"
    MALFORMED_IMPORT = "malformed-import"
    VERSION_INCOMPATIBLE = "version-incompatible"
    IMPORT_CYCLE = "import-cycle"
    DUPLICATE_EXPORT = "duplicate-export"
    ALIAS_COLLISION = "alias-collision"


@dataclass(frozen=True, slots=True)
class PromptPackImport:
    """One declared import edge from an importer pack to an exported symbol."""

    importer: str
    pack: str
    symbol: str
    alias: str
    min_version: str | None

    @property
    def local_name(self) -> str:
        return self.alias or self.symbol

    def to_dict(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "importer": self.importer,
            "local_name": self.local_name,
            "min_version": self.min_version,
            "pack": self.pack,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class PromptPackModule:
    """A prompt pack viewed as a module with exports and imports."""

    pack_name: str
    version: str | None
    exports: tuple[str, ...]
    imports: tuple[PromptPackImport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "exports": list(self.exports),
            "imports": [edge.to_dict() for edge in self.imports],
            "pack_name": self.pack_name,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class PromptPackModuleFinding:
    """One static-linking defect with a replayable witness."""

    kind: PromptPackModuleFindingKind
    importer: str
    message: str
    detail: tuple[tuple[str, str], ...]
    witness: WitnessTrace

    def to_dict(self) -> dict[str, object]:
        return {
            "detail": [list(pair) for pair in self.detail],
            "importer": self.importer,
            "kind": self.kind.value,
            "message": self.message,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PromptPackModuleGraph:
    """Result of statically linking a prompt-pack module program."""

    version: str
    modules: tuple[PromptPackModule, ...]
    findings: tuple[PromptPackModuleFinding, ...]
    link_order: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "link_order": list(self.link_order),
            "modules": [module.to_dict() for module in self.modules],
            "ok": self.ok,
            "version": self.version,
        }


def resolve_prompt_pack_modules(
    loaded_artifacts: tuple[LoadedArtifact, ...],
) -> PromptPackModuleGraph:
    """Statically link every loaded prompt pack as a module program."""

    modules = tuple(
        sorted(
            (_module(loaded) for loaded in loaded_artifacts if isinstance(loaded.artifact, PromptPackArtifact)),
            key=lambda module: module.pack_name,
        )
    )
    modules_by_name = {module.pack_name: module for module in modules}
    exports_by_name = {module.pack_name: set(module.exports) for module in modules}

    findings: list[PromptPackModuleFinding] = []
    findings.extend(_duplicate_export_findings(modules))

    edges: dict[str, list[str]] = {module.pack_name: [] for module in modules}
    for module in modules:
        seen_local: dict[str, PromptPackImport] = {}
        for edge in module.imports:
            collision = seen_local.get(edge.local_name)
            if collision is not None and (collision.pack, collision.symbol) != (edge.pack, edge.symbol):
                findings.append(_alias_collision_finding(edge, collision))
            seen_local.setdefault(edge.local_name, edge)

            target = modules_by_name.get(edge.pack)
            if target is None:
                findings.append(_unresolved_pack_finding(edge, tuple(sorted(modules_by_name))))
                continue
            if edge.symbol not in exports_by_name[edge.pack]:
                findings.append(_unresolved_symbol_finding(edge, tuple(sorted(exports_by_name[edge.pack]))))
                continue
            if not _version_satisfied(edge.min_version, target.version):
                findings.append(_version_finding(edge, target.version))
                continue
            edges[module.pack_name].append(edge.pack)

    cycle = _detect_cycle(edges)
    if cycle is not None:
        findings.append(_cycle_finding(cycle))
        link_order: tuple[str, ...] = ()
    else:
        link_order = _topological_order(modules, edges)

    return PromptPackModuleGraph(
        version=PROMPT_PACK_MODULE_VERSION,
        modules=modules,
        findings=tuple(findings),
        link_order=link_order if not findings else (),
    )


def render_prompt_pack_modules_json(graph: PromptPackModuleGraph) -> str:
    """Render the module graph as stable JSON."""

    return json.dumps(graph.to_dict(), indent=2, sort_keys=True) + "\n"


def render_prompt_pack_modules_text(graph: PromptPackModuleGraph) -> str:
    """Render the module graph for CI logs and reviewers."""

    lines = [
        f"PromptABI prompt-pack module system ({graph.version})",
        f"status: {'LINKED' if graph.ok else 'UNRESOLVED'}",
        f"modules: {len(graph.modules)}",
    ]
    if graph.ok:
        lines.append("link order: " + (" -> ".join(graph.link_order) or "<none>"))
        return "\n".join(lines) + "\n"
    lines.append(f"findings: {len(graph.findings)}")
    for finding in graph.findings:
        lines.append(f"{finding.kind.value} [{finding.importer}]: {finding.message}")
        for key, value in finding.detail:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def _module(loaded: LoadedArtifact) -> PromptPackModule:
    artifact = loaded.artifact
    assert isinstance(artifact, PromptPackArtifact)
    exports = (
        tuple(f"template:{template.name}" for template in artifact.exported_templates)
        + tuple(f"tool:{tool.name}" for tool in artifact.tool_schemas)
        + tuple(f"stop:{policy.name}" for policy in artifact.stop_policies)
    )
    imports = _read_imports(artifact)
    return PromptPackModule(
        pack_name=artifact.pack_name,
        version=artifact.pack_version,
        exports=tuple(sorted(exports)),
        imports=imports,
    )


def _read_imports(artifact: PromptPackArtifact) -> tuple[PromptPackImport, ...]:
    path = artifact.location.path
    if not path:
        return ()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(raw, dict):
        return ()
    entries = raw.get("imports")
    if not isinstance(entries, list):
        return ()
    imports: list[PromptPackImport] = []
    for entry in entries:
        edge = _import_from_entry(artifact.pack_name, entry)
        if edge is not None:
            imports.append(edge)
    return tuple(imports)


def _import_from_entry(importer: str, entry: Any) -> PromptPackImport | None:
    if not isinstance(entry, dict):
        return None
    pack = entry.get("pack")
    symbol = entry.get("symbol")
    if not isinstance(pack, str) or not isinstance(symbol, str):
        return None
    alias = entry.get("as")
    min_version = entry.get("min_version")
    return PromptPackImport(
        importer=importer,
        pack=pack,
        symbol=symbol,
        alias=alias if isinstance(alias, str) else "",
        min_version=min_version if isinstance(min_version, str) else None,
    )


def _duplicate_export_findings(
    modules: tuple[PromptPackModule, ...],
) -> list[PromptPackModuleFinding]:
    owners: dict[str, list[str]] = {}
    for module in modules:
        for symbol in module.exports:
            owners.setdefault(symbol, []).append(module.pack_name)
    findings: list[PromptPackModuleFinding] = []
    for symbol, packs in sorted(owners.items()):
        if len(packs) > 1:
            message = f"symbol '{symbol}' is exported by multiple packs: {', '.join(sorted(packs))}"
            findings.append(
                PromptPackModuleFinding(
                    kind=PromptPackModuleFindingKind.DUPLICATE_EXPORT,
                    importer=sorted(packs)[0],
                    message=message,
                    detail=(("symbol", symbol), ("packs", ", ".join(sorted(packs)))),
                    witness=_witness(
                        sorted(packs)[0],
                        message,
                        ("read each pack's export table", symbol, ", ".join(sorted(packs))),
                        "Rename the colliding symbol or namespace it under a single owning pack.",
                    ),
                )
            )
    return findings


def _unresolved_pack_finding(edge: PromptPackImport, available: tuple[str, ...]) -> PromptPackModuleFinding:
    message = f"pack '{edge.importer}' imports from unknown pack '{edge.pack}'"
    return PromptPackModuleFinding(
        kind=PromptPackModuleFindingKind.UNRESOLVED_PACK,
        importer=edge.importer,
        message=message,
        detail=(("missing pack", edge.pack), ("available packs", ", ".join(available) or "<none>")),
        witness=_witness(
            edge.importer,
            message,
            ("resolve import target pack", edge.pack, "not loaded"),
            f"Load the '{edge.pack}' pack or remove the import.",
        ),
    )


def _unresolved_symbol_finding(edge: PromptPackImport, exports: tuple[str, ...]) -> PromptPackModuleFinding:
    message = f"pack '{edge.importer}' imports '{edge.symbol}' not exported by '{edge.pack}'"
    return PromptPackModuleFinding(
        kind=PromptPackModuleFindingKind.UNRESOLVED_SYMBOL,
        importer=edge.importer,
        message=message,
        detail=(("missing symbol", edge.symbol), ("pack exports", ", ".join(exports) or "<none>")),
        witness=_witness(
            edge.importer,
            message,
            ("look up symbol in exporter table", edge.symbol, ", ".join(exports) or "<none>"),
            f"Export '{edge.symbol}' from '{edge.pack}' or fix the import name.",
        ),
    )


def _version_finding(edge: PromptPackImport, exporter_version: str | None) -> PromptPackModuleFinding:
    message = (
        f"pack '{edge.importer}' requires '{edge.symbol}' at >= {edge.min_version} "
        f"but '{edge.pack}' is version {exporter_version or '<unversioned>'}"
    )
    return PromptPackModuleFinding(
        kind=PromptPackModuleFindingKind.VERSION_INCOMPATIBLE,
        importer=edge.importer,
        message=message,
        detail=(
            ("required min_version", edge.min_version or "<none>"),
            ("exporter version", exporter_version or "<none>"),
        ),
        witness=_witness(
            edge.importer,
            message,
            ("compare required and exported versions", edge.min_version or "<none>", exporter_version or "<none>"),
            f"Upgrade '{edge.pack}' to >= {edge.min_version} or relax the import constraint.",
        ),
    )


def _alias_collision_finding(edge: PromptPackImport, existing: PromptPackImport) -> PromptPackModuleFinding:
    message = (
        f"pack '{edge.importer}' binds local name '{edge.local_name}' to both "
        f"'{existing.pack}:{existing.symbol}' and '{edge.pack}:{edge.symbol}'"
    )
    return PromptPackModuleFinding(
        kind=PromptPackModuleFindingKind.ALIAS_COLLISION,
        importer=edge.importer,
        message=message,
        detail=(
            ("local name", edge.local_name),
            ("first source", f"{existing.pack}:{existing.symbol}"),
            ("second source", f"{edge.pack}:{edge.symbol}"),
        ),
        witness=_witness(
            edge.importer,
            message,
            ("bind import into local namespace", edge.local_name, "already bound"),
            "Give one of the conflicting imports a distinct 'as' alias.",
        ),
    )


def _cycle_finding(cycle: tuple[str, ...]) -> PromptPackModuleFinding:
    message = "prompt-pack import graph has a cycle: " + " -> ".join(cycle)
    return PromptPackModuleFinding(
        kind=PromptPackModuleFindingKind.IMPORT_CYCLE,
        importer=cycle[0],
        message=message,
        detail=(("cycle", " -> ".join(cycle)),),
        witness=_witness(
            cycle[0],
            message,
            ("walk the import graph", cycle[0], " -> ".join(cycle)),
            "Break the dependency cycle so a deterministic link order exists.",
        ),
    )


def _witness(
    importer: str,
    summary: str,
    step: tuple[str, str, str],
    fix: str,
) -> WitnessTrace:
    return WitnessTrace(
        summary=summary,
        steps=(WitnessStep(action=step[0], input=step[1], output=step[2]),),
        artifacts=(ArtifactRef(kind="prompt-pack", name=importer, path=None),),
        minimal_fixes=(fix,),
    )


def _version_satisfied(required: str | None, available: str | None) -> bool:
    if required is None:
        return True
    if available is None:
        return False
    return _version_tuple(available) >= _version_tuple(required)


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in value.strip().split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _detect_cycle(edges: dict[str, list[str]]) -> tuple[str, ...] | None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in edges}
    stack: list[str] = []

    def visit(node: str) -> tuple[str, ...] | None:
        color[node] = GRAY
        stack.append(node)
        for neighbor in edges.get(node, ()):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                index = stack.index(neighbor)
                return tuple(stack[index:]) + (neighbor,)
            if color[neighbor] == WHITE:
                found = visit(neighbor)
                if found is not None:
                    return found
        stack.pop()
        color[node] = BLACK
        return None

    for node in sorted(edges):
        if color[node] == WHITE:
            found = visit(node)
            if found is not None:
                return found
    return None


def _topological_order(
    modules: tuple[PromptPackModule, ...],
    edges: dict[str, list[str]],
) -> tuple[str, ...]:
    order: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        for dependency in sorted(set(edges.get(node, ()))):
            visit(dependency)
        order.append(node)

    for module in modules:
        visit(module.pack_name)
    return tuple(order)
