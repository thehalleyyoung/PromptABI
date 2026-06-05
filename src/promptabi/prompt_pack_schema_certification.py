"""Certify structured-output schemas shipped inside prompt packs (step 252).

Packs that promise structured output ship JSON Schemas the consumer is expected
to hand to a constrained decoder.  A pack should not ship a schema PromptABI
cannot reason about, or one that pins the decoder into an *unsatisfiable* shape.
This module certifies every shipped schema by:

* normalizing it through :func:`promptabi.json_schema.normalize_json_schema_mapping`
  and rejecting anything outside the supported fragment (so the pack cannot make
  a "structured output" claim PromptABI cannot back up), and
* requiring at least one ``required`` field for object roots, so the pack does
  not ship a vacuous ``{}``-accepts-everything schema while claiming structure.

The output is a per-schema certificate plus an aggregate pack-level verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .json_schema import normalize_json_schema_mapping

PROMPT_PACK_SCHEMA_CERT_VERSION = "promptabi.prompt-pack-schema-cert.v1"


class SchemaCertFindingKind(StrEnum):
    OUTSIDE_FRAGMENT = "outside-supported-fragment"
    VACUOUS_OBJECT = "vacuous-object-schema"
    EMPTY_SCHEMA = "empty-schema"


@dataclass(frozen=True, slots=True)
class ShippedSchema:
    name: str
    schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class SchemaCertFinding:
    kind: SchemaCertFindingKind
    schema: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "schema": self.schema, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class SchemaCertificate:
    schema: str
    certified: bool
    feature_count: int
    findings: tuple[SchemaCertFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "certified": self.certified,
            "feature_count": self.feature_count,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass(frozen=True, slots=True)
class PackSchemaCertification:
    version: str
    certified: bool
    certificates: tuple[SchemaCertificate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "certified": self.certified,
            "certificates": [c.to_dict() for c in self.certificates],
        }


def certify_schema(shipped: ShippedSchema) -> SchemaCertificate:
    findings: list[SchemaCertFinding] = []

    if not shipped.schema:
        findings.append(
            SchemaCertFinding(
                SchemaCertFindingKind.EMPTY_SCHEMA,
                shipped.name,
                "schema is empty and constrains nothing",
            )
        )
        return SchemaCertificate(shipped.name, False, 0, tuple(findings))

    result = normalize_json_schema_mapping(shipped.schema)
    if not result.supported_fragment:
        abstentions = [i.message for i in result.issues if i.severity == "abstention"]
        findings.append(
            SchemaCertFinding(
                SchemaCertFindingKind.OUTSIDE_FRAGMENT,
                shipped.name,
                "uses constructs outside the supported fragment: "
                + "; ".join(abstentions[:3]),
            )
        )

    root = result.root
    if root.kind.value == "object" and not root.required:
        findings.append(
            SchemaCertFinding(
                SchemaCertFindingKind.VACUOUS_OBJECT,
                shipped.name,
                "object root declares no required fields; structured-output claim "
                "is vacuous",
            )
        )

    return SchemaCertificate(
        schema=shipped.name,
        certified=not findings,
        feature_count=len(result.features),
        findings=tuple(findings),
    )


def certify_pack_schemas(
    schemas: tuple[ShippedSchema, ...],
) -> PackSchemaCertification:
    certs = tuple(certify_schema(s) for s in schemas)
    return PackSchemaCertification(
        version=PROMPT_PACK_SCHEMA_CERT_VERSION,
        certified=bool(certs) and all(c.certified for c in certs),
        certificates=certs,
    )


def render_schema_certification_text(result: PackSchemaCertification) -> str:
    lines = [
        f"PromptABI prompt-pack schema certification ({result.version})",
        f"result: {'CERTIFIED' if result.certified else 'REJECTED'}",
    ]
    for cert in result.certificates:
        status = "ok" if cert.certified else "FAIL"
        lines.append(f"  [{status}] {cert.schema} ({cert.feature_count} features)")
        for finding in cert.findings:
            lines.append(f"      ! {finding.kind.value}: {finding.detail}")
    return "\n".join(lines) + "\n"
