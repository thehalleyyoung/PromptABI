from pathlib import Path


CI_WORKFLOW = Path(".github/workflows/ci.yml")


def test_cross_version_ci_matrix_covers_declared_support_surface() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "fail-fast: false" in workflow
    for python_version in ('python-version: "3.11"', 'python-version: "3.12"', 'python-version: "3.13"', 'python-version: "3.14"'):
        assert python_version in workflow
    for os_name in ("ubuntu-latest", "macos-latest", "windows-latest"):
        assert os_name in workflow
    for optional_set in (
        "core",
        "pinned-older-tokenizers",
        "grammars-and-z3",
        "action-and-local-workflows",
        "latest-all-backends",
    ):
        assert f"optional-set: {optional_set}" in workflow


def test_cross_version_ci_installs_real_optional_backend_combinations() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert 'python -m pip install -e ".[dev]"' in workflow
    assert 'python -m pip install -e ".[dev,grammars,solver]"' in workflow
    assert 'python -m pip install -e ".[dev,grammars,solver,tokenizers]"' in workflow
    assert "jsonschema==4.22.0" in workflow
    assert "tokenizers==0.15.2" in workflow
    assert "tiktoken==0.7.0" in workflow
    assert "sentencepiece==0.2.0" in workflow
    assert "transformers==4.57.3" in workflow
    for package in ("jsonschema", "sentencepiece", "tiktoken", "tokenizers", "transformers", "z3-solver"):
        assert package in workflow


def test_cross_version_ci_runs_targeted_real_test_slices() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    for test_file in (
        "tests/test_cli.py",
        "tests/test_tokenizers.py",
        "tests/test_stop_analysis.py",
        "tests/test_formal.py",
        "tests/test_static_contracts.py",
        "tests/test_grammar_ambiguity.py",
        "tests/test_github_action.py",
        "tests/test_local_workflows.py",
        "tests/test_chat_templates.py",
        "tests/test_parser_compatibility.py",
        "tests/test_checker_properties.py",
    ):
        assert test_file in workflow
