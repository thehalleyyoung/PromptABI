import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PAPER_TEX = REPO_ROOT / "tool_paper.tex"
TOOL_PAPER_PDF = REPO_ROOT / "tool_paper.pdf"


def _paper_text() -> str:
    return TOOL_PAPER_TEX.read_text(encoding="utf-8")


def test_tool_paper_contains_step_91_research_framing() -> None:
    paper = _paper_text()

    required_sections = (
        r"\section{Introduction}",
        r"\section{Motivation}",
        r"\section{Core Abstraction}",
        r"\section{Formal Verification Core}",
        r"\section{Evaluation Design}",
        r"\section{Threat Model, Limitations, and Non-Goals}",
        r"\section{Related Systems}",
    )
    for section in required_sections:
        assert section in paper

    required_terms = (
        r"\begin{theorem}[Bounded counterexample soundness]",
        r"\begin{theorem}[Honest supported-fragment boundary]",
        "supported fragment",
        "abstention",
        "threat model",
        "non-goals",
        "limitations",
    )
    lower_paper = paper.lower()
    for term in required_terms:
        assert term.lower() in lower_paper


def test_tool_paper_is_not_structured_as_the_internal_checklist() -> None:
    paper = _paper_text()

    assert "100_STEPS" not in paper
    assert "Step 91" not in paper
    assert not re.search(r"^\s*\d+\.\s+\[[ x]\]", paper, flags=re.MULTILINE)


def test_tool_paper_names_real_deployed_benchlines_and_commands() -> None:
    paper = _paper_text()

    for phrase in (
        r"benchmarks/benchmark\_smoke.py",
        "promptabi corpus verify",
        "promptabi corpus real-bug-benchmark",
        "promptabi corpus evaluation",
        "promptabi maintain refresh",
        "fixtures/evaluation/labeled\\_corpus.json",
    ):
        assert phrase in paper


@pytest.mark.skipif(shutil.which("pdfinfo") is None, reason="pdfinfo is required to inspect PDF page count")
def test_tool_paper_pdf_stays_within_requested_page_window() -> None:
    assert TOOL_PAPER_PDF.exists()

    completed = subprocess.run(
        ["pdfinfo", str(TOOL_PAPER_PDF)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", completed.stdout, flags=re.MULTILINE)
    assert match is not None
    page_count = int(match.group(1))

    assert 20 < page_count < 40
