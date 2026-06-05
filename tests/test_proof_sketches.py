import json

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ChatTemplateArtifact,
    ChatTemplateSymbolicBounds,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    SchemaArtifact,
    SpecialToken,
    SpecialTokenMapArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    TruncationStrategy,
    VerificationConfig,
    analyze_role_boundary_nonforgeability,
    analyze_static_contracts,
    analyze_stop_overreachability,
    analyze_token_budget,
    analyze_tokenizer_grammar_emptiness,
    build_proof_sketch_notebooks,
    parse_hf_chat_template_config,
    proof_sketch_notebooks,
    proof_sketches,
    prove_grammar_emptiness,
    prove_must_survive_budget,
    prove_role_boundary_nonforgeability,
    prove_static_contract,
    prove_stop_reachability,
    render_proof_sketch_notebook_report_text,
    render_proof_sketch_report_text,
    write_proof_sketch_notebooks,
)
from promptabi.cli import main
from promptabi.formal import SolverStatus
from promptabi.json_schema import compile_json_schema_mapping
from promptabi.loaders import ArtifactLoader, LoadedArtifact
from promptabi.proof_sketches import ProofOutcome
from promptabi.tokenizers import ByteLevelTokenizer


def _loaded(artifact):
    return LoadedArtifact(artifact=artifact, source_type="memory", pinned=True, resolved=True)


def test_role_boundary_proof_validates_real_forgery_witness() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{{ message['content'] }}<|im_end|>\n"
                "{% endfor %}"
            ),
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        }
    )
    report = analyze_role_boundary_nonforgeability(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=8),
    )

    sketch = prove_role_boundary_nonforgeability(report)

    assert sketch.outcome is ProofOutcome.COUNTEREXAMPLE
    assert sketch.passed
    assert dict(sketch.evidence)["finding_count"] == str(len(report.findings))
    assert any(check.name.endswith("marker-visible-in-rendered-excerpt") for check in sketch.checks)


def test_stop_proof_validates_prefix_split_for_real_overreach_witness() -> None:
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="xml-stop",
        location=ArtifactLocation(uri="memory://stop"),
        stop_sequences=("</arguments>",),
    )
    report = analyze_stop_overreachability(policy)

    sketch = prove_stop_reachability(report)

    assert sketch.outcome is ProofOutcome.COUNTEREXAMPLE
    assert sketch.passed
    assert dict(sketch.evidence)["finding_count"] == str(len(report.findings))
    assert any(check.name.endswith("valid-prefix-includes-stop") for check in sketch.checks)


def test_grammar_proof_uses_compiled_dfa_for_real_tokenizer_product(tmp_path) -> None:
    raw_schema = {"type": "object", "additionalProperties": False}
    compiled = compile_json_schema_mapping(raw_schema)
    schema_file = tmp_path / "object.schema.json"
    schema_file.write_text(json.dumps(raw_schema), encoding="utf-8")
    tokenizer_artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="byte",
        location=ArtifactLocation(uri="memory://byte"),
        family="byte-level",
    )
    schema_artifact = SchemaArtifact(kind=ArtifactKind.SCHEMA, name="object", location=ArtifactLocation(path=str(schema_file)))
    report = analyze_tokenizer_grammar_emptiness(tokenizer_artifact, schema_artifact, ByteLevelTokenizer())

    sketch_without_automaton = prove_grammar_emptiness(report)
    sketch_with_automaton = prove_grammar_emptiness(report, automaton=compiled.grammar.automaton)

    assert sketch_without_automaton.outcome is ProofOutcome.SKETCH
    assert sketch_with_automaton.outcome is ProofOutcome.PROVEN
    assert sketch_with_automaton.passed
    assert any(check.name == "decoded-text-accepted-by-grammar" for check in sketch_with_automaton.checks)


def test_must_survive_budget_proof_checks_real_truncation_partition() -> None:
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="conversation",
        location=ArtifactLocation(uri="memory://segments"),
        segments=(
            PromptSegment("system", role="system", required=True, token_count=18),
            PromptSegment("old-user", role="user", required=True, token_count=26),
            PromptSegment("latest-user", role="user", required=True, token_count=20),
        ),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="langchain-budget",
        location=ArtifactLocation(uri="memory://budget"),
        framework="langchain",
        strategy=TruncationStrategy.OLDEST_MESSAGE,
        max_context_tokens=50,
    )
    report = analyze_token_budget(
        VerificationConfig(name="budget"),
        (ArtifactLoader().load(segments), ArtifactLoader().load(budget)),
    )
    assert report.must_survive_proof is not None

    sketch = prove_must_survive_budget(report.must_survive_proof)

    assert sketch.outcome is ProofOutcome.COUNTEREXAMPLE
    assert sketch.passed
    assert dict(sketch.evidence)["dropped_required"] == "old-user"


def test_static_contract_proof_rechecks_solver_assignment_and_unsat_core() -> None:
    location = ArtifactLocation(uri="memory://static-contract")
    loaded = (
        _loaded(
            PromptSegmentArtifact(
                kind=ArtifactKind.PROMPT_SEGMENT,
                name="segments",
                location=location,
                segments=(PromptSegment("must", role="system", required=True, token_count=80),),
            )
        ),
        _loaded(
            FrameworkTruncationConfigArtifact(
                kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
                name="budget",
                location=location,
                framework="vllm",
                max_context_tokens=64,
                reserve_output_tokens=8,
            )
        ),
        _loaded(
            StopPolicyArtifact(
                kind=ArtifactKind.STOP_POLICY,
                name="stops",
                location=location,
                stop_sequences=("</s>",),
            )
        ),
        _loaded(
            SpecialTokenMapArtifact(
                kind=ArtifactKind.SPECIAL_TOKEN_MAP,
                name="specials",
                location=location,
                tokens=(SpecialToken("eos", "</s>", 2),),
            )
        ),
        _loaded(
            PromptSegmentArtifact(
                kind=ArtifactKind.PROMPT_SEGMENT,
                name="conversation",
                location=location,
                segments=(PromptSegment("user", role="user", content="hello"),),
            )
        ),
        _loaded(
            ChatTemplateArtifact(
                kind=ArtifactKind.CHAT_TEMPLATE,
                name="chatml",
                location=location,
                roles=("user", "assistant"),
            )
        ),
    )
    report = analyze_static_contracts(VerificationConfig(name="static"), loaded, prefer_z3=False)
    budget_violation = next(finding for finding in report.findings if finding.name == "prompt-segment-survival-violation")
    role_proof = next(finding for finding in report.findings if finding.name == "role-region-nonforgeability")

    budget_sketch = prove_static_contract(budget_violation)
    role_sketch = prove_static_contract(role_proof)

    assert budget_violation.status is SolverStatus.SAT
    assert budget_sketch.outcome is ProofOutcome.COUNTEREXAMPLE
    assert budget_sketch.passed
    assert role_proof.status is SolverStatus.UNSAT
    assert role_sketch.outcome is ProofOutcome.PROVEN
    assert role_sketch.passed
    assert any(check.name == "unsat-core-is-deletion-minimal" for check in role_sketch.checks)


def test_proof_catalog_api_and_cli_render_supported_families(capsys) -> None:
    report = proof_sketches()
    rendered = render_proof_sketch_report_text(report)

    assert report.passed
    assert "role-boundary-nonforgeability" in rendered
    assert "z3-backed-finite-contract" in rendered

    exit_code = main(["proofs", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["passed"] is True
    assert {item["property_id"] for item in payload["sketches"]} >= {
        "role-boundary-nonforgeability",
        "stop-overreachability",
        "grammar-tokenizer-emptiness",
        "must-survive-budget",
        "z3-backed-finite-contract",
    }
    assert captured.err == ""


def test_proof_sketch_notebooks_are_deterministic_and_execute_real_code(tmp_path) -> None:
    notebooks = build_proof_sketch_notebooks()

    assert {notebook.property_id for notebook in notebooks} == {
        "role-boundary-nonforgeability",
        "stop-overreachability",
        "grammar-tokenizer-emptiness",
        "must-survive-budget",
        "training-mask-alignment",
    }
    for notebook in notebooks:
        payload = notebook.to_dict()
        assert payload["nbformat"] == 4
        assert payload["metadata"]["promptabi"]["executes_real_promptabi_code"] is True
        code_sources = [
            "".join(cell["source"])
            for cell in payload["cells"]
            if cell["cell_type"] == "code"
        ]
        assert code_sources
        namespace: dict[str, object] = {}
        for source in code_sources:
            exec(source, namespace)

    report = write_proof_sketch_notebooks(tmp_path)
    rendered = render_proof_sketch_notebook_report_text(report)
    paths = sorted(tmp_path.glob("*.ipynb"))

    assert report.passed
    assert len(paths) == 5
    assert "training-mask-alignment" in rendered
    assert [path.name for path in paths] == sorted(notebook.filename for notebook in notebooks)

    api_report = proof_sketch_notebooks(tmp_path, force=True)
    assert api_report.passed


def test_cli_writes_executable_proof_sketch_notebooks(tmp_path, capsys) -> None:
    exit_code = main(["proofs", "--write-notebooks", str(tmp_path), "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["passed"] is True
    assert len(payload["notebooks"]) == 5
    assert (tmp_path / "01-role-boundary-nonforgeability.ipynb").exists()

    exit_code = main(["proofs", "--write-notebooks", str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "already exists" in captured.err
