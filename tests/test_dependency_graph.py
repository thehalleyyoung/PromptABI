import json

from promptabi import (
    ArtifactKind,
    DependencyGraphReport,
    dependency_graph,
)
from promptabi.cli import main
from promptabi.config import load_config
from promptabi.dependency_graph import build_dependency_graph, render_dependency_graph_mermaid


def test_dependency_graph_maps_real_rag_artifacts_to_dependent_checks() -> None:
    config = load_config("examples/rag-chunking/promptabi.json")

    report = build_dependency_graph(config, include_all_checks=True)
    by_id = {node.id: node for node in report.nodes}
    edges = {(edge.source, edge.target, edge.relation) for edge in report.edges}

    assert isinstance(report, DependencyGraphReport)
    assert by_id["artifact:serving-byte-tokenizer"].metadata[0] == ("artifact_kind", ArtifactKind.TOKENIZER.value)
    assert ("artifact:serving-byte-tokenizer", "check:token-budget-model", "feeds") in edges
    assert ("artifact:rag-budget", "check:rag-chunking-compatibility", "feeds") in edges
    assert ("check:token-budget-model", "check:rag-chunking-compatibility", "runs-before") in edges
    assert ("kind:training-manifest", "check:static-contracts", "requires-kind") in edges
    assert by_id["kind:training-manifest"].configured is False


def test_dependency_graph_renderers_are_deterministic_and_cli_visible(capsys) -> None:
    exit_code = main(
        [
            "graph",
            "--config",
            "examples/rag-chunking/promptabi.json",
            "--all-checks",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["config_name"] == "rag-chunking-demo"
    assert any(edge["relation"] == "runs-before" for edge in payload["edges"])
    assert any(node["id"] == "kind:training-manifest" and not node["configured"] for node in payload["nodes"])
    assert captured.err == ""

    mermaid = render_dependency_graph_mermaid(build_dependency_graph(load_config("examples/minimal/promptabi.json")))
    assert mermaid.startswith("flowchart LR\n")
    assert "artifact:messages" not in mermaid
    assert "repository_skeleton" in mermaid


def test_dependency_graph_public_api_can_render_text_and_mermaid() -> None:
    report = dependency_graph("examples/rag-chunking/promptabi.json", include_all_checks=True)
    text = dependency_graph("examples/rag-chunking/promptabi.json", output_format="text")
    mermaid = dependency_graph("examples/rag-chunking/promptabi.json", output_format="mermaid")

    assert isinstance(report, DependencyGraphReport)
    assert "PromptABI dependency graph: rag-chunking-demo" in text
    assert "feeds: rag-segments (prompt-segment)" in text
    assert "flowchart LR" in mermaid
