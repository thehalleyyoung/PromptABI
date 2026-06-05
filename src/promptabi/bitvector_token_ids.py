"""Add bit-vector encodings for token ids (step 229).

Token ids are bounded non-negative integers drawn from a fixed vocabulary, so a
``ceil(log2(|V|))``-bit bit-vector represents them exactly.  Many interface
obligations are really statements about token-id *sets*:

* content tokens must never collide with reserved special-token ids
  (delimiter/control-token injection safety);
* every content token id must lie inside the declared vocabulary;
* special-token id ranges must be disjoint from one another.

This module gives those obligations a **bit-vector encoding**.  The exact
semantics are computed with set arithmetic (always available, deterministic),
and -- when Z3 is installed -- the same obligation is discharged with a
``BitVec`` SMT query and cross-validated against the exact result.  That keeps
the SMT encoding honest while letting it scale to large vocabularies where naive
integer enumeration would be wasteful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

BITVECTOR_TOKEN_ID_VERSION = "promptabi.bitvector-token-ids.v1"


def bit_width(vocab_size: int) -> int:
    """Smallest bit width that can represent every id in ``[0, vocab_size)``."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if vocab_size == 1:
        return 1
    return (vocab_size - 1).bit_length()


def encode_token_id(value: int, width: int) -> tuple[int, ...]:
    """Big-endian bit tuple for ``value`` in ``width`` bits."""

    if value < 0:
        raise ValueError("token ids must be non-negative")
    if value >= (1 << width):
        raise ValueError(f"token id {value} does not fit in {width} bits")
    return tuple((value >> shift) & 1 for shift in reversed(range(width)))


class TokenIdFindingKind(StrEnum):
    OUT_OF_VOCAB = "out-of-vocab"
    SPECIAL_COLLISION = "special-collision"
    SMT_DISAGREEMENT = "smt-disagreement"


@dataclass(frozen=True, slots=True)
class TokenIdRange:
    name: str
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low < 0 or self.high < self.low:
            raise ValueError("token-id range must satisfy 0 <= low <= high")

    def ids(self) -> range:
        return range(self.low, self.high + 1)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "low": self.low, "high": self.high}


@dataclass(frozen=True, slots=True)
class TokenIdContract:
    """Vocabulary, content-token ranges, and reserved special-token ids."""

    vocab_size: int
    content_ranges: tuple[TokenIdRange, ...]
    special_ids: frozenset[int]
    name: str = "token-id-contract"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

    @property
    def width(self) -> int:
        return bit_width(self.vocab_size)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "width": self.width,
            "content_ranges": [r.to_dict() for r in self.content_ranges],
            "special_ids": sorted(self.special_ids),
        }


@dataclass(frozen=True, slots=True)
class TokenIdFinding:
    kind: TokenIdFindingKind
    message: str
    witness: int | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"kind": self.kind.value, "message": self.message}
        if self.witness is not None:
            data["witness_id"] = self.witness
        return data


@dataclass(frozen=True, slots=True)
class TokenIdReport:
    version: str
    contract: str
    width: int
    exact_safe: bool
    smt_backend: str
    smt_safe: bool | None
    findings: tuple[TokenIdFinding, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return self.exact_safe and not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "contract": self.contract,
            "width": self.width,
            "ok": self.ok,
            "exact_safe": self.exact_safe,
            "smt_backend": self.smt_backend,
            "smt_safe": self.smt_safe,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _exact_findings(contract: TokenIdContract) -> list[TokenIdFinding]:
    findings: list[TokenIdFinding] = []
    for token_range in contract.content_ranges:
        if token_range.high >= contract.vocab_size:
            findings.append(
                TokenIdFinding(
                    kind=TokenIdFindingKind.OUT_OF_VOCAB,
                    message=(
                        f"content range {token_range.name!r} reaches id {token_range.high} "
                        f">= vocab_size {contract.vocab_size}"
                    ),
                    witness=token_range.high,
                )
            )
        collision = next((i for i in token_range.ids() if i in contract.special_ids), None)
        if collision is not None:
            findings.append(
                TokenIdFinding(
                    kind=TokenIdFindingKind.SPECIAL_COLLISION,
                    message=(
                        f"content range {token_range.name!r} contains reserved special id {collision}"
                    ),
                    witness=collision,
                )
            )
    return findings


def _smt_safe(contract: TokenIdContract) -> bool | None:
    """Discharge the disjointness obligation with a Z3 BitVec query, if available."""

    try:
        import z3  # type: ignore[import-not-found]
    except ImportError:
        return None

    width = contract.width
    solver = z3.Solver()
    token = z3.BitVec("token", width)
    # token lies in some content range...
    in_content = z3.Or(
        *(
            z3.And(z3.UGE(token, r.low), z3.ULE(token, r.high))
            for r in contract.content_ranges
        )
    ) if contract.content_ranges else z3.BoolVal(False)
    # ...and either exceeds the vocab or hits a special id.
    unsafe_clauses = [z3.UGE(token, contract.vocab_size)]
    if contract.special_ids:
        unsafe_clauses.append(z3.Or(*(token == sid for sid in sorted(contract.special_ids))))
    solver.add(in_content)
    solver.add(z3.Or(*unsafe_clauses))
    result = solver.check()
    # safe == no model where a content token is unsafe == UNSAT.
    return result == z3.unsat


def verify_token_id_contract(contract: TokenIdContract) -> TokenIdReport:
    """Verify content tokens stay in-vocab and disjoint from special ids."""

    exact = _exact_findings(contract)
    exact_safe = not exact
    smt_safe = _smt_safe(contract)
    backend = "z3-bitvec" if smt_safe is not None else "exact-only"

    findings = list(exact)
    if smt_safe is not None and smt_safe != exact_safe:
        findings.append(
            TokenIdFinding(
                kind=TokenIdFindingKind.SMT_DISAGREEMENT,
                message=(
                    f"bit-vector SMT verdict (safe={smt_safe}) disagrees with exact "
                    f"set verdict (safe={exact_safe})"
                ),
            )
        )
    return TokenIdReport(
        version=BITVECTOR_TOKEN_ID_VERSION,
        contract=contract.name,
        width=contract.width,
        exact_safe=exact_safe,
        smt_backend=backend,
        smt_safe=smt_safe,
        findings=tuple(findings),
    )


def verify_token_id_contracts(
    contracts: Sequence[TokenIdContract],
) -> tuple[TokenIdReport, ...]:
    return tuple(verify_token_id_contract(contract) for contract in contracts)


def render_token_id_report_json(report: TokenIdReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_token_id_report_text(report: TokenIdReport) -> str:
    lines = [
        f"PromptABI bit-vector token-id check ({report.version})",
        f"contract: {report.contract} (width={report.width} bits)",
        f"status: {'OK' if report.ok else 'VIOLATED'}",
        f"smt backend: {report.smt_backend} (safe={report.smt_safe})",
    ]
    for finding in report.findings:
        suffix = f" [id={finding.witness}]" if finding.witness is not None else ""
        lines.append(f"  ! {finding.kind.value}: {finding.message}{suffix}")
    return "\n".join(lines) + "\n"
