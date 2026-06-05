"""Tests for reusable certified demo packs (step 257)."""

from __future__ import annotations

from promptabi.demo_packs import (
    certify_all_demo_packs,
    certify_demo_pack,
    load_demo_pack,
    load_demo_packs,
    render_demo_pack_text,
)


def test_shipped_demo_packs_load() -> None:
    packs = load_demo_packs()
    names = {p.name for p in packs}
    assert {"support-triage", "json-extractor"} <= names


def test_all_shipped_demo_packs_are_certified() -> None:
    certs = certify_all_demo_packs()
    assert certs
    for cert in certs:
        assert cert.certified, (cert.pack, cert.reasons)


def test_certificate_has_pack_metadata() -> None:
    packs = {p.name: p for p in load_demo_packs()}
    cert = certify_demo_pack(packs["support-triage"])
    assert cert.pack == "support-triage"
    assert cert.pack_version == "1.2.0"


def test_tampered_pack_rejected(tmp_path) -> None:
    import json

    src = {
        "name": "broken",
        "version": "0.1.0",
        "template": "<|im_start|>system\n{{docs}}<|im_end|>",
        "control_markers": ["<|im_start|>", "<|im_end|>"],
        "extension_points": [{"name": "docs", "placeholder": "{{docs}}"}],
        "stop_sequences": [],
        "sanitizers": [],
        "schemas": {},
        "exported_roles": ["system"],
        "models": [],
    }
    p = tmp_path / "broken.json"
    p.write_text(json.dumps(src), encoding="utf-8")
    cert = certify_demo_pack(load_demo_pack(p))
    assert not cert.certified
    assert any(r.startswith("rag:") for r in cert.reasons)
    assert any(r.startswith("stop:") for r in cert.reasons)


def test_render_text_smoke() -> None:
    cert = certify_all_demo_packs()[0]
    assert "demo pack" in render_demo_pack_text(cert)
