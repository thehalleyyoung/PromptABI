"""Reusable certified demo packs and their certifier (step 257).

This module ships a small set of *certified* demo prompt packs (under
``fixtures/demo_packs``) and the certifier that proves they are safe to reuse.
A demo pack is exactly the artifact a downstream consumer would copy, so the
certifier runs the full reusable-pack battery on it:

* every declared RAG extension point is present, sanitized, and outside any
  control region (step 251),
* every shipped structured-output schema is inside PromptABI's supported
  fragment and non-vacuous (step 252),
* the pack's stop sequences are non-empty and its exported roles are declared.

The result is a deterministic :class:`DemoPackCertificate` that other steps
(third-party certification, the interop leaderboard) build on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .prompt_pack_rag_extension import (
    ExtensionPoint,
    PackTemplate,
    verify_extension_points,
)
from .prompt_pack_schema_certification import (
    ShippedSchema,
    certify_pack_schemas,
)

DEMO_PACK_VERSION = "promptabi.demo-pack.v1"

_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "demo_packs"


@dataclass(frozen=True, slots=True)
class DemoPack:
    name: str
    version: str
    description: str
    license: str
    exported_roles: tuple[str, ...]
    template: str
    control_markers: tuple[str, ...]
    extension_points: tuple[ExtensionPoint, ...]
    stop_sequences: tuple[str, ...]
    sanitizers: tuple[str, ...]
    schemas: tuple[ShippedSchema, ...]
    models: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "DemoPack":
        points = tuple(
            ExtensionPoint(
                name=str(p["name"]),
                placeholder=str(p["placeholder"]),
                sanitizer=(str(p["sanitizer"]) if p.get("sanitizer") else None),
            )
            for p in raw.get("extension_points", [])  # type: ignore[union-attr]
        )
        schemas = tuple(
            ShippedSchema(name=name, schema=schema)
            for name, schema in dict(raw.get("schemas", {})).items()  # type: ignore[arg-type]
        )
        return cls(
            name=str(raw["name"]),
            version=str(raw["version"]),
            description=str(raw.get("description", "")),
            license=str(raw.get("license", "")),
            exported_roles=tuple(raw.get("exported_roles", [])),  # type: ignore[arg-type]
            template=str(raw["template"]),
            control_markers=tuple(raw.get("control_markers", [])),  # type: ignore[arg-type]
            extension_points=points,
            stop_sequences=tuple(raw.get("stop_sequences", [])),  # type: ignore[arg-type]
            sanitizers=tuple(raw.get("sanitizers", [])),  # type: ignore[arg-type]
            schemas=schemas,
            models=tuple(raw.get("models", [])),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DemoPackCertificate:
    version: str
    pack: str
    pack_version: str
    certified: bool
    reasons: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "pack": self.pack,
            "pack_version": self.pack_version,
            "certified": self.certified,
            "reasons": list(self.reasons),
        }


def load_demo_pack(path: str | Path) -> DemoPack:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return DemoPack.from_mapping(raw)


def load_demo_packs(root: str | Path | None = None) -> tuple[DemoPack, ...]:
    base = Path(root) if root is not None else _FIXTURE_ROOT
    return tuple(
        load_demo_pack(p) for p in sorted(base.glob("*.json"))
    )


def certify_demo_pack(pack: DemoPack) -> DemoPackCertificate:
    reasons: list[str] = []

    template = PackTemplate(source=pack.template, control_markers=pack.control_markers)
    rag = verify_extension_points(template, pack.extension_points)
    if not rag.safe:
        reasons.extend(f"rag:{f.kind.value}:{f.slot}" for f in rag.findings)

    schema_cert = certify_pack_schemas(pack.schemas)
    if not schema_cert.certified:
        for cert in schema_cert.certificates:
            reasons.extend(f"schema:{f.kind.value}:{f.schema}" for f in cert.findings)

    if not pack.stop_sequences:
        reasons.append("stop:no-stop-sequence")
    if not pack.exported_roles:
        reasons.append("roles:no-exported-roles")
    if not pack.models:
        reasons.append("models:no-target-models")

    return DemoPackCertificate(
        version=DEMO_PACK_VERSION,
        pack=pack.name,
        pack_version=pack.version,
        certified=not reasons,
        reasons=tuple(reasons),
    )


def certify_all_demo_packs(
    root: str | Path | None = None,
) -> tuple[DemoPackCertificate, ...]:
    return tuple(certify_demo_pack(p) for p in load_demo_packs(root))


def render_demo_pack_text(cert: DemoPackCertificate) -> str:
    lines = [
        f"PromptABI demo pack {cert.pack}@{cert.pack_version} ({cert.version})",
        f"result: {'CERTIFIED' if cert.certified else 'REJECTED'}",
    ]
    for reason in cert.reasons:
        lines.append(f"  ! {reason}")
    return "\n".join(lines) + "\n"
