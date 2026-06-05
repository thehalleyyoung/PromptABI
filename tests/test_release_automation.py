from pathlib import Path


RELEASE_WORKFLOW = Path(".github/workflows/release.yml")


def test_release_workflow_builds_publishable_python_artifacts() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert 'tags:\n      - "v*.*.*"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "python -m build" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "INPUT_VERSION: ${{ inputs.version || '' }}" in workflow
    assert r"(?:0|[1-9]\d*)\." in workflow
    assert "does not match pyproject/_version metadata" in workflow
    assert "is not a semantic version tag" in workflow


def test_release_workflow_builds_signed_multiplatform_assets_and_container() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    for os_name in ("ubuntu-latest", "macos-latest", "windows-latest"):
        assert f"os: {os_name}" in workflow
    for artifact in ("promptabi-linux-x86_64", "promptabi-macos-arm64", "promptabi-windows-x86_64"):
        assert f"artifact: {artifact}" in workflow
    assert "pyinstaller --clean --onefile --name promptabi" in workflow
    assert "shell: bash" in workflow
    assert "docker/build-push-action@v6" in workflow
    assert "ghcr.io/${{ github.repository_owner }}/promptabi" in workflow
    assert "actions/attest-build-provenance@v2" in workflow
    assert "sigstore/gh-action-sigstore-python@v3.0.1" in workflow
    assert "softprops/action-gh-release@v2" in workflow


def test_dockerfile_installs_promptabi_cli_without_project_state() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim AS runtime" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "python -m pip install --no-cache-dir ." in dockerfile
    assert 'ENTRYPOINT ["promptabi"]' in dockerfile


def test_changelog_documents_semver_release_contract() -> None:
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "semantic versioning" in changelog
    assert "## 0.1.0" in changelog
    assert "static verification" in changelog
