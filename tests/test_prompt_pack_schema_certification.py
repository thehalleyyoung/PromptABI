"""Tests for prompt-pack structured-output schema certification (step 252)."""

from __future__ import annotations

from promptabi.prompt_pack_schema_certification import (
    SchemaCertFindingKind,
    ShippedSchema,
    certify_pack_schemas,
    certify_schema,
    render_schema_certification_text,
)

GOOD = ShippedSchema(
    name="triage",
    schema={
        "type": "object",
        "properties": {
            "priority": {"type": "string", "enum": ["low", "high"]},
            "category": {"type": "string"},
        },
        "required": ["priority", "category"],
    },
)


def test_good_schema_certified() -> None:
    cert = certify_schema(GOOD)
    assert cert.certified, cert.findings
    assert cert.feature_count > 0


def test_empty_schema_rejected() -> None:
    cert = certify_schema(ShippedSchema("empty", {}))
    assert not cert.certified
    assert any(f.kind is SchemaCertFindingKind.EMPTY_SCHEMA for f in cert.findings)


def test_vacuous_object_rejected() -> None:
    cert = certify_schema(
        ShippedSchema("vac", {"type": "object", "properties": {"a": {"type": "string"}}})
    )
    assert not cert.certified
    assert any(f.kind is SchemaCertFindingKind.VACUOUS_OBJECT for f in cert.findings)


def test_pack_certification_aggregates() -> None:
    result = certify_pack_schemas((GOOD,))
    assert result.certified
    bad = certify_pack_schemas((GOOD, ShippedSchema("empty", {})))
    assert not bad.certified


def test_render_text_smoke() -> None:
    result = certify_pack_schemas((GOOD,))
    assert "certification" in render_schema_certification_text(result)
