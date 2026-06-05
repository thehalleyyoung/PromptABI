import json

import promptabi
from promptabi.bundles import (
    VerificationBundleError,
    create_signed_verification_bundle,
    load_signed_verification_bundle,
    verify_signed_verification_bundle,
)
from promptabi.cli import main


def test_signed_verification_bundle_captures_real_audit_trail() -> None:
    bundle = create_signed_verification_bundle(
        "examples/role-boundary/unsafe.promptabi.json",
        key="test-secret",
        key_id="unit-test",
        excerpt_bytes=128,
    )

    payload = bundle.payload

    assert payload["bundle_version"] == 1
    assert payload["config"]["name"] == "role-boundary-unsafe-chatml"
    assert payload["reproducibility"]["ok"] is False
    assert payload["reproducibility"]["error_count"] >= 1
    assert payload["reproducibility"]["witness_hashes"]
    assert payload["lockfile"]["diagnostic_baseline"]
    assert any(
        diagnostic["rule_id"] == "role-boundary-nonforgeability"
        and diagnostic["witness"]["steps"][0]["action"] == "build bounded role-region model"
        for diagnostic in payload["diagnostics"]
    )
    assert all("metadata_hash" in artifact for artifact in payload["artifacts"])
    assert bundle.to_dict()["bundle_hash"] == bundle.bundle_hash

    verification = verify_signed_verification_bundle(bundle, key="test-secret")
    assert verification.ok
    assert verification.signing_key_id == "unit-test"


def test_signed_verification_bundle_rejects_tampering() -> None:
    bundle = create_signed_verification_bundle("examples/minimal/promptabi.json", key="test-secret")
    tampered = bundle.to_dict()
    tampered["payload"]["reproducibility"]["ok"] = False

    verification = verify_signed_verification_bundle(tampered, key="test-secret")

    assert verification.ok is False
    assert verification.reason == "signature mismatch"


def test_bundle_cli_writes_and_verifies_roundtrippable_bundle(tmp_path, capsys) -> None:
    output = tmp_path / "promptabi.bundle.json"

    exit_code = main(
        [
            "bundle",
            "create",
            "--config",
            "examples/minimal/promptabi.json",
            "--output",
            str(output),
            "--key",
            "test-secret",
            "--key-id",
            "cli-test",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "wrote signed verification bundle" in captured.out
    assert load_signed_verification_bundle(output).signing_key_id == "cli-test"

    exit_code = main(["bundle", "verify", str(output), "--key", "test-secret", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["signing_key_id"] == "cli-test"
    assert captured.err == ""


def test_bundle_requires_explicit_signing_key(monkeypatch) -> None:
    monkeypatch.delenv("PROMPTABI_BUNDLE_KEY", raising=False)

    try:
        create_signed_verification_bundle("examples/minimal/promptabi.json")
    except VerificationBundleError as exc:
        assert "signing key is required" in str(exc)
    else:
        raise AssertionError("expected signing key requirement")


def test_public_api_creates_and_verifies_bundle(tmp_path) -> None:
    output = tmp_path / "api.bundle.json"

    bundle = promptabi.create_verification_bundle(
        "examples/minimal/promptabi.json",
        key="test-secret",
        output=output,
    )
    verification = promptabi.verify_verification_bundle(output, key="test-secret")

    assert isinstance(bundle, promptabi.VerificationBundle)
    assert isinstance(verification, promptabi.VerificationBundleVerification)
    assert verification.ok is True
    assert output.is_file()
