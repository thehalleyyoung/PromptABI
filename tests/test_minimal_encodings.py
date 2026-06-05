"""Tests for mechanized smallest solver encodings (step 239)."""

from __future__ import annotations

import json

from promptabi.formal import (
    BoolDomain,
    Eq,
    NamedConstraint,
    Value,
    Var,
)
from promptabi.minimal_encodings import (
    EncodingFindingKind,
    MinimalEncoding,
    mechanize_minimal_encodings,
    render_mechanization_json,
    render_mechanization_text,
    standard_minimal_encodings,
)


def test_standard_corpus_is_mechanized() -> None:
    report = mechanize_minimal_encodings()
    assert report.mechanized
    assert all(r.passed for r in report.results)
    constructs = {r.construct for r in report.results}
    assert {"eq", "le", "sum", "and", "or", "not", "implies", "inset"} <= constructs


def test_every_construct_covered_once() -> None:
    corpus = standard_minimal_encodings()
    assert len(corpus) == len({e.construct for e in corpus})


def test_backends_agree_on_each_encoding() -> None:
    report = mechanize_minimal_encodings()
    for result in report.results:
        assert result.z3_verdict == result.enum_verdict
        assert result.z3_verdict == result.expected


def test_wrong_label_is_caught() -> None:
    b = BoolDomain(name="b")
    bad = MinimalEncoding(
        "bad",
        (b,),
        NamedConstraint(name="b=true", expression=Eq(Var("b"), Value(True))),
        "unsat",  # wrong: this is satisfiable
    )
    report = mechanize_minimal_encodings([bad])
    assert not report.mechanized
    assert any(
        f.kind is EncodingFindingKind.WRONG_VERDICT for f in report.findings
    )


def test_render_round_trips() -> None:
    report = mechanize_minimal_encodings()
    payload = json.loads(render_mechanization_json(report))
    assert payload["mechanized"] is True
    assert "mechanization" in render_mechanization_text(report)
