"""Replay public real-world bugs from pinned production-code excerpts."""

from __future__ import annotations

from promptabi.production_code_bugs import load_production_code_bug_corpus


def main() -> None:
    corpus = load_production_code_bug_corpus()
    for replay in corpus.replay():
        status = "PASS" if replay.passed else "FAIL"
        rules = ", ".join(replay.rule_ids) or "no rules"
        print(f"{status} {replay.case_id}: {rules}")
        print(f"  {replay.evidence_summary}")


if __name__ == "__main__":
    main()
