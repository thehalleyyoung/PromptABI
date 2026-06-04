import json
from pathlib import Path

from promptabi import (
    GrammarDifferentialStatus,
    analyze_grammar_differential_corpus,
    analyze_grammar_differential_mapping,
)
from promptabi.cli import main


CORPUS_PATH = Path("fixtures/grammar_differential/corpus.json")


def test_grammar_differential_corpus_covers_backend_families() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    families = {case["backend_family"] for case in corpus["cases"]}

    assert corpus["version"] == 1
    assert {
        "outlines",
        "xgrammar",
        "llguidance",
        "lm-format-enforcer",
        "guidance",
        "instructor",
        "pydantic",
    } <= families
    for case in corpus["cases"]:
        assert case["accepts"]
        assert case["rejects"]
        assert case["declared_type"]


def test_grammar_differential_replays_recorded_backend_semantics() -> None:
    report = analyze_grammar_differential_corpus(CORPUS_PATH)

    assert not report.mismatches
    assert not report.abstentions
    assert len(report.agreements) >= 7
    xgrammar = next(case for case in report.cases if case.case_id == "xgrammar-finite-tool-call")
    assert xgrammar.status is GrammarDifferentialStatus.AGREEMENT
    assert xgrammar.terminals == ("call", " ", "refund", "search")


def test_grammar_differential_reports_intentional_mismatch() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus["cases"] = [
        {
            "id": "bad-label",
            "backend_family": "outlines",
            "declared_type": "outlines",
            "artifact": {"choices": ["yes"]},
            "accepts": ["yes"],
            "rejects": ["yes"],
        }
    ]

    report = analyze_grammar_differential_mapping(corpus)

    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0].mismatches[0]
    assert mismatch.sample.text == "yes"
    assert mismatch.sample.expected_accepts is False
    assert mismatch.promptabi_accepts is True


def test_verify_reports_grammar_differential_mismatch(tmp_path: Path, capsys) -> None:
    corpus_path = tmp_path / "grammar-differential.json"
    corpus_path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "mismatched-choice",
                        "backend_family": "outlines",
                        "declared_type": "outlines",
                        "artifact": {"choices": ["safe"]},
                        "accepts": ["safe"],
                        "rejects": ["safe"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "grammar-diff",
                "checks": ["grammar-differential"],
                "artifacts": {
                    "grammar-fixtures": {
                        "kind": "grammar",
                        "path": str(corpus_path),
                        "grammar_type": "grammar-differential",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    diagnostic = payload["diagnostics"][0]
    assert exit_code == 1
    assert diagnostic["rule_id"] == "grammar-differential-mismatch"
    assert diagnostic["check_modes"] == ["heuristic"]
    assert "mismatched-choice" in diagnostic["message"]
