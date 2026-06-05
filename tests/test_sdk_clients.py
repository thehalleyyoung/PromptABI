import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from promptabi.bundles import VERIFICATION_BUNDLE_VERSION, create_signed_verification_bundle, render_verification_bundle_json
from promptabi.config import load_config
from promptabi.integration_api import INTEGRATION_API_VERSION, build_integration_report, render_integration_report_json
from promptabi.lockfiles import LOCKFILE_VERSION, build_lockfile, lockfile_to_json
from promptabi.session import VerificationSession


REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = REPO_ROOT / "sdk"


@pytest.fixture()
def sdk_fixtures(tmp_path: Path) -> Path:
    config_path = REPO_ROOT / "examples" / "role-boundary" / "unsafe.promptabi.json"
    report = build_integration_report(
        config_path,
        surfaces=["ci-provider", "ide-extension", "model-registry", "internal-platform"],
        fail_on="never",
        bundle_key="sdk-test-key",
        bundle_key_id="sdk-test",
    )
    config = load_config(config_path)
    session = VerificationSession(config)
    result = session.run()
    loaded, load_diagnostics = session.load_artifacts_with_diagnostics()
    assert load_diagnostics == ()
    lockfile = build_lockfile(config, loaded, result.diagnostics, base_dir=config_path.parent)
    bundle = create_signed_verification_bundle(
        config_path,
        key="sdk-test-key",
        key_id="sdk-test",
        excerpt_bytes=128,
    )
    diagnostics_payload = {
        "diagnostics": [diagnostic.to_dict() for diagnostic in result.diagnostics],
        "ok": result.ok,
    }

    (tmp_path / "integration-report.json").write_text(render_integration_report_json(report), encoding="utf-8")
    (tmp_path / "lockfile.json").write_text(lockfile_to_json(lockfile), encoding="utf-8")
    (tmp_path / "bundle.json").write_text(render_verification_bundle_json(bundle), encoding="utf-8")
    (tmp_path / "diagnostics.json").write_text(
        json.dumps(diagnostics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_sdk_sources_track_stable_promptabi_payload_versions() -> None:
    sources = [
        SDK_ROOT / "typescript" / "src" / "index.ts",
        SDK_ROOT / "go" / "promptabi.go",
        SDK_ROOT / "rust" / "src" / "lib.rs",
    ]

    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert INTEGRATION_API_VERSION in text
        assert f"= {LOCKFILE_VERSION}" in text or f": {LOCKFILE_VERSION}" in text
        assert f"= {VERIFICATION_BUNDLE_VERSION}" in text or f": {VERIFICATION_BUNDLE_VERSION}" in text
        assert "ReadVerificationBundle" in text or "readVerificationBundle" in text or "read_verification_bundle" in text
        assert "ReadLockfile" in text or "readLockfile" in text or "read_lockfile" in text
        assert "Diagnostic" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to execute the TypeScript thin client")
def test_typescript_thin_client_reads_generated_promptabi_payloads(sdk_fixtures: Path, tmp_path: Path) -> None:
    smoke = tmp_path / "smoke.ts"
    sdk_entry = (SDK_ROOT / "typescript" / "src" / "index.ts").as_posix()
    smoke.write_text(
        f"""
        import {{
          diagnosticsFromBundle,
          readDiagnosticPayload,
          readIntegrationReport,
          readLockfile,
          readVerificationBundle,
          summarizeDiagnostics,
          summarizeReport,
        }} from {json.dumps(sdk_entry)};

        const fixtures = {json.dumps(sdk_fixtures.as_posix())};
        const report = readIntegrationReport(`${{fixtures}}/integration-report.json`);
        const lockfile = readLockfile(`${{fixtures}}/lockfile.json`);
        const bundle = readVerificationBundle(`${{fixtures}}/bundle.json`);
        const payload = readDiagnosticPayload(`${{fixtures}}/diagnostics.json`);
        const reportSummary = summarizeReport(report);
        const diagnosticSummary = summarizeDiagnostics(payload.diagnostics, payload.ok === true);

        if (report.protocol !== {json.dumps(INTEGRATION_API_VERSION)}) throw new Error("bad protocol");
        if (lockfile.lockfile_version !== {LOCKFILE_VERSION}) throw new Error("bad lockfile version");
        if (bundle.payload.bundle_version !== {VERIFICATION_BUNDLE_VERSION}) throw new Error("bad bundle version");
        if (diagnosticsFromBundle(bundle).length === 0) throw new Error("bundle diagnostics missing");
        if (reportSummary.diagnosticCount === 0 || diagnosticSummary.fingerprints.length === 0) {{
          throw new Error("summaries did not read generated diagnostics");
        }}
        """,
        encoding="utf-8",
    )

    subprocess.run(["node", "--experimental-strip-types", str(smoke)], check=True, cwd=REPO_ROOT)


@pytest.mark.skipif(shutil.which("go") is None, reason="go is required to test the Go thin client")
def test_go_thin_client_reads_generated_promptabi_payloads(sdk_fixtures: Path) -> None:
    env = {**os.environ, "PROMPTABI_SDK_FIXTURES": str(sdk_fixtures)}
    subprocess.run(["go", "test", "./..."], check=True, cwd=SDK_ROOT / "go", env=env)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is required to test the Rust thin client")
def test_rust_thin_client_reads_generated_promptabi_payloads(sdk_fixtures: Path) -> None:
    env = {**os.environ, "PROMPTABI_SDK_FIXTURES": str(sdk_fixtures)}
    subprocess.run(["cargo", "test", "--quiet"], check=True, cwd=SDK_ROOT / "rust", env=env)
