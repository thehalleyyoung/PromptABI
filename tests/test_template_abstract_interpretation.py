import json

from promptabi import (
    AbstractCount,
    TemplateInvariantKind,
    interpret_chat_template_file,
    render_template_abstract_interpretation_text,
)
from promptabi.cli import main


QWEN = "fixtures/seed_corpus/qwen/tokenizer_config.json"
BALANCED = "examples/template-abstract-interpretation/balanced-chatml.json"
IMBALANCED = "examples/template-abstract-interpretation/imbalanced-chatml.json"


def test_abstract_count_lattice_join_add_star() -> None:
    one = AbstractCount(1, 1)
    assert one.add(one) == AbstractCount(2, 2)
    assert one.join(AbstractCount(0, 0)) == AbstractCount(0, 1)
    assert one.star() == AbstractCount(0, None)
    assert AbstractCount(0, 0).star() == AbstractCount(0, 0)
    assert not AbstractCount(0, None).bounded


def test_chatml_template_proves_balanced_frames_and_bounded_generation_prompt() -> None:
    report = interpret_chat_template_file(BALANCED, name="balanced")

    assert report.supported
    assert report.ok
    # im_start and im_end are both unbounded across messages but balanced per frame.
    assert not report.count_for("<|im_start|>").bounded
    assert not report.count_for("<|im_end|>").bounded
    # the assistant generation header is provably emitted at most once.
    assert report.count_for("<|im_start|>assistant") == AbstractCount(0, 1)
    assert "all proven" in render_template_abstract_interpretation_text(report)


def test_imbalanced_loop_is_refuted_for_any_message_count() -> None:
    report = interpret_chat_template_file(IMBALANCED, name="imbalanced")

    assert report.supported
    assert not report.ok
    kinds = {violation.kind for violation in report.violations}
    assert TemplateInvariantKind.MARKER_IMBALANCE in kinds
    violation = next(v for v in report.violations if v.kind == TemplateInvariantKind.MARKER_IMBALANCE)
    assert violation.witness.minimal_fixes


def test_real_qwen_template_is_proven() -> None:
    report = interpret_chat_template_file(QWEN, name="qwen")

    assert report.supported
    assert report.ok
    assert report.marker_pairs == (("<|im_start|>", "<|im_end|>"),)


def test_template_ai_cli_proves_and_refutes(capsys) -> None:
    proven = main(["template-ai", "--tokenizer-config", BALANCED, "--format", "json"])
    proven_payload = json.loads(capsys.readouterr().out)
    assert proven == 0
    assert proven_payload["ok"] is True
    assert proven_payload["version"] == "promptabi.template-abstract-interpretation.v1"

    refuted = main(["template-ai", "--tokenizer-config", IMBALANCED, "--format", "json"])
    refuted_payload = json.loads(capsys.readouterr().out)
    assert refuted == 1
    assert refuted_payload["ok"] is False
    assert any(v["kind"] == "marker-imbalance" for v in refuted_payload["violations"])
