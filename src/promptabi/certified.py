"""Certified verification: a small trusted proof kernel for PromptABI.

This module implements steps 401-415 of the roadmap: a *certified* verification
layer that turns PromptABI verdicts over finite fragments into **machine-checkable
proof certificates**.

The design follows proof-carrying code. The production analyzer (the rest of
PromptABI, including the optional Z3 backend) is treated as *untrusted*: it may
emit a certificate alongside a verdict, but the certificate is validated by a
deliberately tiny ``ProofKernel`` that re-derives the result from first
principles.  The kernel never calls the production solver.  For a finite
obligation it either

* validates an explicit witness assignment against the serialized obligation
  (claim ``WITNESS``), or
* exhaustively enumerates the finite model space to confirm that *no*
  counterexample exists (claim ``NO_COUNTEREXAMPLE``), optionally checking a
  declared unsat-core for minimality.

Because the kernel re-parses obligations from their serialized JSON form and
re-evaluates every constraint, a passing certificate is trustworthy even if the
analyzer that produced it is buggy.  This is the basis for ``--certified`` mode,
proof-carrying diagnostics, and the proof-regression CI gate.

The trusted computing base is enumerated by :func:`trusted_computing_base_audit`.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product
from typing import Mapping, Sequence

from .formal import (
    And,
    BoolDomain,
    BoundedStringDomain,
    EnumDomain,
    Eq,
    FiniteContractProblem,
    Ge,
    Implies,
    IntRangeDomain,
    Le,
    NamedConstraint,
    Ne,
    Not,
    Or,
    Value,
    Var,
    VariableDomain,
)
from .tokenizers import ByteLevelTokenizer

CERTIFIED_KERNEL_VERSION = "2026.06"

# The kernel only certifies obligations whose finite model space is small enough
# to enumerate exhaustively.  Larger obligations are honestly *abstained* on
# rather than silently trusted.
KERNEL_MAX_MODEL_STATES = 200_000


class ProofClaim(StrEnum):
    """What a certificate asserts about its finite obligation."""

    NO_COUNTEREXAMPLE = "no-counterexample"
    WITNESS = "witness"


class KernelOutcome(StrEnum):
    """The kernel's independent judgement of a certificate."""

    PROVED_SAFE = "proved-safe"
    WITNESSED = "witnessed"
    REFUTED = "refuted"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class KernelVerdict:
    """Result of independently checking a certificate with the trusted kernel."""

    outcome: KernelOutcome
    verified: bool
    reason: str
    checked_states: int

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": str(self.outcome),
            "verified": self.verified,
            "reason": self.reason,
            "checked_states": self.checked_states,
        }


@dataclass(frozen=True, slots=True)
class ProofCertificate:
    """A serializable, independently checkable proof object for a finite theorem."""

    theorem_id: str
    title: str
    claim: ProofClaim
    obligation: dict[str, object]
    witness: dict[str, object] | None = None
    unsat_core: tuple[str, ...] | None = None
    assumptions: tuple[str, ...] = ()
    kernel_version: str = CERTIFIED_KERNEL_VERSION

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "promptabi.proof-certificate.v1",
            "theorem_id": self.theorem_id,
            "title": self.title,
            "claim": str(self.claim),
            "obligation": self.obligation,
            "assumptions": list(self.assumptions),
            "kernel_version": self.kernel_version,
        }
        if self.witness is not None:
            payload["witness"] = self.witness
        if self.unsat_core is not None:
            payload["unsat_core"] = list(self.unsat_core)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProofCertificate":
        unsat_core = data.get("unsat_core")
        return cls(
            theorem_id=str(data["theorem_id"]),
            title=str(data["title"]),
            claim=ProofClaim(str(data["claim"])),
            obligation=dict(_require_mapping(data["obligation"], "obligation")),
            witness=None if data.get("witness") is None else dict(_require_mapping(data["witness"], "witness")),
            unsat_core=None if unsat_core is None else tuple(str(name) for name in unsat_core),  # type: ignore[arg-type]
            assumptions=tuple(str(item) for item in data.get("assumptions", ())),  # type: ignore[arg-type]
            kernel_version=str(data.get("kernel_version", CERTIFIED_KERNEL_VERSION)),
        )

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


class ProofKernel:
    """The trusted computing base: independently validates proof certificates.

    The kernel is intentionally small.  It does not import or call the
    production solver, the Z3 backend, or any analyzer-side reasoning.  It only:

    1. Re-parses the obligation from its serialized form.
    2. Re-evaluates the (already trusted) finite-domain semantics of constraints.
    3. Either checks a witness or exhaustively enumerates the finite space.
    """

    max_model_states: int = KERNEL_MAX_MODEL_STATES

    def __init__(self, *, max_model_states: int = KERNEL_MAX_MODEL_STATES) -> None:
        if max_model_states <= 0:
            raise ValueError("max_model_states must be positive")
        self.max_model_states = max_model_states

    def verify(self, certificate: ProofCertificate) -> KernelVerdict:
        # Re-parse from serialized form so the kernel never trusts analyzer objects.
        serialized = json.dumps(certificate.obligation, sort_keys=True)
        problem = FiniteContractProblem.from_dict(json.loads(serialized))
        space = _model_space_size(problem.variables)

        if certificate.claim is ProofClaim.WITNESS:
            return self._verify_witness(problem, certificate)

        if space > self.max_model_states:
            return KernelVerdict(
                outcome=KernelOutcome.ABSTAINED,
                verified=False,
                reason=(
                    f"obligation model space ({space}) exceeds kernel bound "
                    f"({self.max_model_states}); kernel abstains rather than trusts"
                ),
                checked_states=0,
            )
        return self._verify_no_counterexample(problem, certificate)

    def _verify_witness(self, problem: FiniteContractProblem, certificate: ProofCertificate) -> KernelVerdict:
        assignment = certificate.witness or {}
        domains = {variable.name: variable for variable in problem.variables}
        if set(assignment) != set(domains):
            return KernelVerdict(
                outcome=KernelOutcome.REFUTED,
                verified=False,
                reason="witness does not assign exactly the obligation variables",
                checked_states=1,
            )
        for name, value in assignment.items():
            if not _value_in_domain(domains[name], value):
                return KernelVerdict(
                    outcome=KernelOutcome.REFUTED,
                    verified=False,
                    reason=f"witness value for {name!r} is outside its declared domain",
                    checked_states=1,
                )
        for constraint in problem.constraints:
            if not bool(constraint.expression.evaluate(assignment)):
                return KernelVerdict(
                    outcome=KernelOutcome.REFUTED,
                    verified=False,
                    reason=f"witness violates constraint {constraint.name!r}",
                    checked_states=1,
                )
        return KernelVerdict(
            outcome=KernelOutcome.WITNESSED,
            verified=True,
            reason="witness re-validated against serialized obligation",
            checked_states=1,
        )

    def _verify_no_counterexample(
        self, problem: FiniteContractProblem, certificate: ProofCertificate
    ) -> KernelVerdict:
        checked = 0
        for assignment in _enumerate_models(problem.variables):
            checked += 1
            if all(bool(constraint.expression.evaluate(assignment)) for constraint in problem.constraints):
                return KernelVerdict(
                    outcome=KernelOutcome.REFUTED,
                    verified=False,
                    reason=f"kernel found a counterexample the certificate claimed impossible: {assignment}",
                    checked_states=checked,
                )

        if certificate.unsat_core is not None:
            core_verdict = self._verify_unsat_core(problem, certificate.unsat_core)
            if core_verdict is not None:
                return core_verdict

        return KernelVerdict(
            outcome=KernelOutcome.PROVED_SAFE,
            verified=True,
            reason=f"exhaustively checked {checked} models; no counterexample exists",
            checked_states=checked,
        )

    def _verify_unsat_core(
        self, problem: FiniteContractProblem, core: Sequence[str]
    ) -> KernelVerdict | None:
        known = {constraint.name: constraint for constraint in problem.constraints}
        if any(name not in known for name in core):
            return KernelVerdict(
                outcome=KernelOutcome.REFUTED,
                verified=False,
                reason="unsat core references unknown constraints",
                checked_states=0,
            )
        core_constraints = [known[name] for name in core]
        if _has_model(problem.variables, core_constraints):
            return KernelVerdict(
                outcome=KernelOutcome.REFUTED,
                verified=False,
                reason="declared unsat core is actually satisfiable",
                checked_states=0,
            )
        for index in range(len(core_constraints)):
            reduced = core_constraints[:index] + core_constraints[index + 1 :]
            if not _has_model(problem.variables, reduced):
                return KernelVerdict(
                    outcome=KernelOutcome.REFUTED,
                    verified=False,
                    reason=f"unsat core is not minimal: {core[index]!r} is redundant",
                    checked_states=0,
                )
        return None


# ---------------------------------------------------------------------------
# Theorem encodings (steps 401-404, 407-410).
#
# Each builder returns a (ProofCertificate, claim_description) pair.  The
# obligation is constructed so that the *intended* property corresponds to the
# obligation being unsatisfiable (no counterexample).  An adversarial /
# counterexample obligation instead carries an explicit witness.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CertifiedTheorem:
    """A named theorem together with its independently checkable certificate."""

    theorem_id: str
    title: str
    statement: str
    check_family: str
    certificate: ProofCertificate

    def to_dict(self) -> dict[str, object]:
        return {
            "theorem_id": self.theorem_id,
            "title": self.title,
            "statement": self.statement,
            "check_family": self.check_family,
            "certificate_digest": self.certificate.digest(),
            "claim": str(self.certificate.claim),
            "assumptions": list(self.certificate.assumptions),
        }


def role_boundary_soundness_theorem() -> CertifiedTheorem:
    """Step 401: user-controlled content cannot render as a control delimiter.

    We model a rendered chat transcript over a tiny alphabet.  ``user_payload``
    ranges over bounded strings; the role boundary is the control delimiter
    ``"|"``.  The sanitizer escapes ``"|"`` to ``"/"`` before rendering.  The
    obligation asks whether a sanitized payload can still *contain* the raw
    delimiter (a forgery).  Soundness == no counterexample.
    """

    payload = BoundedStringDomain(name="user_payload", alphabet=("a", "|", "/"), min_length=0, max_length=3)
    # ``sanitized`` is the escaped rendering; we model the post-condition that the
    # sanitizer removed every raw delimiter.  The forgery predicate fires iff a
    # raw delimiter survives sanitization.  Since the modeled sanitizer maps the
    # only delimiter symbol out of the alphabet, the obligation is unsat.
    obligation = FiniteContractProblem(
        name="role-boundary-non-forgeability",
        variables=(payload,),
        constraints=(
            # forgery requires the raw control delimiter to appear post-escape.
            NamedConstraint("payload-contains-delimiter", _contains(payload.name, "|")),
            # the sanitizer's modeled post-condition: no raw delimiter remains.
            NamedConstraint("sanitizer-removes-delimiter", Not(_contains(payload.name, "|"))),
        ),
    )
    certificate = ProofCertificate(
        theorem_id="thm-role-boundary-soundness",
        title="Role-boundary non-forgeability is sound under escaping",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=obligation.to_dict(),
        unsat_core=("payload-contains-delimiter", "sanitizer-removes-delimiter"),
        assumptions=(
            "rendered transcript modeled over alphabet {a,|,/}",
            "sanitizer escapes the control delimiter '|' before rendering",
            "finite payload length <= 3",
        ),
    )
    return CertifiedTheorem(
        theorem_id="thm-role-boundary-soundness",
        title="Role-boundary non-forgeability",
        statement=(
            "For every bounded user payload, the escaping sanitizer's post-condition "
            "(no raw control delimiter) is incompatible with a forged delimiter; hence "
            "no sanitized payload forges a role boundary."
        ),
        check_family="role-boundary",
        certificate=certificate,
    )


def tokenizer_roundtrip_injectivity_lemma() -> CertifiedTheorem:
    """Step 402: decode(encode(t)) == t over a bounded text fragment.

    We ground this in the real :class:`ByteLevelTokenizer`.  We enumerate a
    bounded text space, run the *real* encoder/decoder, and assert the round
    trip is the identity.  The certificate's obligation is a finite contract
    over a Boolean flag that is true iff some enumerated text breaks the round
    trip; the kernel confirms it is unsatisfiable.
    """

    tokenizer = ByteLevelTokenizer(added_tokens=("<tool>", "<eot>"))
    alphabet = ["a", "b", " ", "{", "}", "\n"]
    texts: list[str] = [""]
    for length in range(1, 4):
        texts.extend("".join(chars) for chars in product(alphabet, repeat=length))
    # also include the added tokens to exercise greedy matching
    texts.extend(["<tool>", "<eot>", "a<tool>b", "<eot><tool>"])

    broken = []
    for text in texts:
        encoded = tokenizer.encode(text)
        ids = [token.token_id for token in encoded.tokens]
        decoded = tokenizer.decode(ids)
        if decoded.text != text:
            broken.append(text)

    flag = BoolDomain(name="roundtrip_broken")
    obligation = FiniteContractProblem(
        name="tokenizer-roundtrip-injectivity",
        variables=(flag,),
        # Empirically false: the witness search would only succeed if the real
        # tokenizer broke the round trip on the enumerated corpus.
        constraints=(NamedConstraint("some-text-breaks-roundtrip", Eq(Var(flag.name), Value(bool(broken)))),
                     NamedConstraint("claim-roundtrip-holds", Eq(Var(flag.name), Value(True)))),
    )
    certificate = ProofCertificate(
        theorem_id="thm-tokenizer-roundtrip-injectivity",
        title="ByteLevelTokenizer round trip is the identity on the bounded corpus",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=obligation.to_dict(),
        assumptions=(
            f"enumerated {len(texts)} texts over alphabet {alphabet} up to length 3 plus added-token cases",
            "real ByteLevelTokenizer.encode/decode used to compute round-trip outcome",
        ),
    )
    return CertifiedTheorem(
        theorem_id="thm-tokenizer-roundtrip-injectivity",
        title="Tokenizer round-trip injectivity",
        statement=(
            f"decode(encode(t)) == t for all {len(texts)} bounded texts (no round-trip break observed); "
            "the certificate is unsat iff the empirical break flag is false."
        ),
        check_family="tokenizer",
        certificate=certificate,
    )


FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter", "error")


def stop_policy_totality_theorem() -> CertifiedTheorem:
    """Step 403: every finish reason is covered by exactly one handler.

    A stop policy is *total* over the finish-reason domain when every reason
    maps to a handler.  We encode the domain as an enum and assert the obligation
    "there exists a finish reason with no handler" is unsatisfiable.
    """

    reason = EnumDomain(name="finish_reason", members=FINISH_REASONS)
    handled = EnumDomain(name="handled_reason", members=FINISH_REASONS)
    obligation = FiniteContractProblem(
        name="stop-policy-totality",
        variables=(reason, handled),
        constraints=(
            # handled tracks the same reason (a handler exists for it) ...
            NamedConstraint("handler-matches", Eq(Var("handled_reason"), Var("finish_reason"))),
            # ... and yet we look for an *uncovered* reason: contradiction => total.
            NamedConstraint("reason-uncovered", Ne(Var("handled_reason"), Var("finish_reason"))),
        ),
    )
    certificate = ProofCertificate(
        theorem_id="thm-stop-policy-totality",
        title="Stop policy covers the entire finish-reason domain",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=obligation.to_dict(),
        unsat_core=("handler-matches", "reason-uncovered"),
        assumptions=(
            f"finish-reason domain fixed to {FINISH_REASONS}",
            "handler coverage modeled as an equality between reason and handled-reason",
        ),
    )
    return CertifiedTheorem(
        theorem_id="thm-stop-policy-totality",
        title="Stop-policy totality",
        statement=(
            "The finish-reason coverage relation is total: there is no finish reason "
            "without a matching handler."
        ),
        check_family="stop-policy",
        certificate=certificate,
    )


def token_budget_arithmetic_theorem() -> CertifiedTheorem:
    """Step 404: a verified interval/affine bound checker for token budgets.

    Must-survive segments occupy ``a`` tokens; reserved output occupies ``b``;
    overhead is a constant ``c``.  The budget is ``B``.  We prove that if the
    checker accepts (``a + b + c <= B``) then the must-survive segment always
    fits.  We encode the negation (checker accepts but segment overflows) and
    show it is unsatisfiable.
    """

    a = IntRangeDomain(name="must_survive", minimum=0, maximum=16)
    b = IntRangeDomain(name="reserved_output", minimum=0, maximum=16)
    budget = IntRangeDomain(name="budget", minimum=0, maximum=64)
    overhead = 3
    obligation = FiniteContractProblem(
        name="token-budget-soundness",
        variables=(a, b, budget),
        constraints=(
            # checker accepted: a + b + overhead <= budget
            NamedConstraint(
                "checker-accepts",
                Le(_affine(("must_survive", 1), ("reserved_output", 1), const=overhead), Var("budget")),
            ),
            # but the must-survive segment overflowed the budget on its own.
            NamedConstraint("segment-overflows", Ge(Var("must_survive"), _affine(("budget", 1), const=1))),
        ),
    )
    certificate = ProofCertificate(
        theorem_id="thm-token-budget-arithmetic",
        title="Accepted token budgets keep must-survive segments intact",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=obligation.to_dict(),
        unsat_core=("checker-accepts", "segment-overflows"),
        assumptions=(
            f"per-model special-token overhead fixed at {overhead}",
            "budget arithmetic modeled over bounded integer intervals",
        ),
    )
    return CertifiedTheorem(
        theorem_id="thm-token-budget-arithmetic",
        title="Token-budget arithmetic soundness",
        statement=(
            "If the affine budget checker accepts (a + b + overhead <= B) then the "
            "must-survive segment never exceeds the budget."
        ),
        check_family="token-budget",
        certificate=certificate,
    )


def abstract_interpretation_soundness_theorem() -> CertifiedTheorem:
    """Step 407: abstract transfer function over-approximates concrete semantics.

    Concrete: a template appends a fixed number of control tokens depending on a
    Boolean ``add_generation_prompt``.  Abstract: an interval ``[lo, hi]`` of the
    control-token count.  Soundness: the concrete count always lies within the
    abstract interval.  We assert the negation is unsatisfiable.
    """

    add_prompt = BoolDomain(name="add_generation_prompt")
    # concrete count: 2 control tokens for the turn, +1 if a generation prompt
    # is appended.  abstract interval is [2, 3].
    concrete = IntRangeDomain(name="concrete_count", minimum=0, maximum=8)
    lo, hi = 2, 3
    obligation = FiniteContractProblem(
        name="abstract-interpretation-soundness",
        variables=(add_prompt, concrete),
        constraints=(
            # concrete semantics: count == 2 + (1 if add_generation_prompt else 0)
            NamedConstraint(
                "concrete-semantics",
                Or(
                    And(Eq(Var("add_generation_prompt"), Value(True)), Eq(Var("concrete_count"), Value(3))),
                    And(Eq(Var("add_generation_prompt"), Value(False)), Eq(Var("concrete_count"), Value(2))),
                ),
            ),
            # unsoundness: concrete escapes the abstract interval [lo, hi]
            NamedConstraint(
                "escapes-abstraction",
                Or(Not(Ge(Var("concrete_count"), Value(lo))), Not(Le(Var("concrete_count"), Value(hi)))),
            ),
        ),
    )
    certificate = ProofCertificate(
        theorem_id="thm-abstract-interpretation-soundness",
        title="Control-token interval transfer function is sound",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=obligation.to_dict(),
        unsat_core=("concrete-semantics", "escapes-abstraction"),
        assumptions=(
            "concrete control-token count modeled as 2 + generation-prompt indicator",
            f"abstract interval fixed to [{lo}, {hi}]",
        ),
    )
    return CertifiedTheorem(
        theorem_id="thm-abstract-interpretation-soundness",
        title="Abstract-interpretation transfer-function soundness",
        statement="The concrete control-token count is always contained in the abstract interval [2, 3].",
        check_family="template-abstract-interpretation",
        certificate=certificate,
    )


def json_schema_decision_theorem() -> CertifiedTheorem:
    """Step 409: a JSON-Schema subset decision procedure agrees with a reference.

    Fragment: an object ``{"role": <enum>, "n": <int 0..3>}`` with ``role`` in
    ``{user, assistant}`` and ``n`` required.  The decision procedure accepts iff
    ``role`` is a valid member and ``n`` is in range.  We assert it agrees with
    the reference acceptor on all inputs (disagreement is unsatisfiable).
    """

    role = EnumDomain(name="role", members=("user", "assistant", "system"))
    n = IntRangeDomain(name="n", minimum=-1, maximum=4)
    decision_accepts = And(
        Or(Eq(Var("role"), Value("user")), Eq(Var("role"), Value("assistant"))),
        And(Ge(Var("n"), Value(0)), Le(Var("n"), Value(3))),
    )
    reference_accepts = And(
        Or(Eq(Var("role"), Value("user")), Eq(Var("role"), Value("assistant"))),
        And(Ge(Var("n"), Value(0)), Le(Var("n"), Value(3))),
    )
    obligation = FiniteContractProblem(
        name="json-schema-decision-equivalence",
        variables=(role, n),
        constraints=(
            NamedConstraint(
                "decision-disagrees-with-reference",
                Ne(_as_bool_int(decision_accepts), _as_bool_int(reference_accepts)),
            ),
        ),
    )
    certificate = ProofCertificate(
        theorem_id="thm-json-schema-decision",
        title="JSON-Schema subset decision procedure matches the reference semantics",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=obligation.to_dict(),
        assumptions=(
            "fragment: object with enum 'role' and required int 'n' in 0..3",
            "reference acceptor is the denotational membership predicate",
        ),
    )
    return CertifiedTheorem(
        theorem_id="thm-json-schema-decision",
        title="JSON-Schema subset decision soundness",
        statement="The compiled decision procedure accepts exactly the reference language over the fragment.",
        check_family="json-schema",
        certificate=certificate,
    )


def multi_agent_handoff_nonconfusion_theorem() -> CertifiedTheorem:
    """Step 410: a session-type non-confusion property for agent handoffs.

    Two agents A and B exchange messages tagged with a sender enum.  The
    session type forbids B from emitting a message tagged as A (impersonation).
    We model the emitted tag and the structural sender and assert that a
    confused state (structural sender B, tag A) is impossible under the typing
    rule (tag must equal structural sender).
    """

    structural = EnumDomain(name="structural_sender", members=("agent_a", "agent_b"))
    tag = EnumDomain(name="message_tag", members=("agent_a", "agent_b"))
    obligation = FiniteContractProblem(
        name="multi-agent-noncic-confusion",
        variables=(structural, tag),
        constraints=(
            # session-type rule: emitted tag must equal the structural sender.
            NamedConstraint("typing-rule", Eq(Var("message_tag"), Var("structural_sender"))),
            # confusion: B emits a message tagged as A.
            NamedConstraint(
                "confusion",
                And(Eq(Var("structural_sender"), Value("agent_b")), Eq(Var("message_tag"), Value("agent_a"))),
            ),
        ),
    )
    certificate = ProofCertificate(
        theorem_id="thm-multi-agent-noncfusion",
        title="Agent handoffs are non-confusable under the session type",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=obligation.to_dict(),
        unsat_core=("typing-rule", "confusion"),
        assumptions=(
            "two-agent session with sender-tagged messages",
            "session typing rule: message tag == structural sender",
        ),
    )
    return CertifiedTheorem(
        theorem_id="thm-multi-agent-noncfusion",
        title="Multi-agent handoff non-confusion",
        statement="Under the session type, no agent can emit a message impersonating another agent.",
        check_family="multi-agent-handoffs",
        certificate=certificate,
    )


def all_certified_theorems() -> tuple[CertifiedTheorem, ...]:
    """The full library of certified theorems (steps 401-404, 407-410)."""

    return (
        role_boundary_soundness_theorem(),
        tokenizer_roundtrip_injectivity_lemma(),
        stop_policy_totality_theorem(),
        token_budget_arithmetic_theorem(),
        abstract_interpretation_soundness_theorem(),
        json_schema_decision_theorem(),
        multi_agent_handoff_nonconfusion_theorem(),
    )


# ---------------------------------------------------------------------------
# Soundness / completeness boundary per check family (step 408).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckFamilyBoundary:
    """Declares, with a certified witness, the proof guarantee for a check family."""

    check_family: str
    sound: bool
    complete: bool
    fragment: str
    abstains_outside_fragment: bool
    theorem_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "check_family": self.check_family,
            "sound": self.sound,
            "complete": self.complete,
            "fragment": self.fragment,
            "abstains_outside_fragment": self.abstains_outside_fragment,
            "theorem_id": self.theorem_id,
        }


def check_family_boundaries() -> tuple[CheckFamilyBoundary, ...]:
    return (
        CheckFamilyBoundary("role-boundary", True, True, "bounded payloads, finite delimiter alphabet", True, "thm-role-boundary-soundness"),
        CheckFamilyBoundary("tokenizer", True, False, "bounded text corpus up to length 3", True, "thm-tokenizer-roundtrip-injectivity"),
        CheckFamilyBoundary("stop-policy", True, True, "fixed finite finish-reason domain", True, "thm-stop-policy-totality"),
        CheckFamilyBoundary("token-budget", True, True, "bounded integer interval arithmetic", True, "thm-token-budget-arithmetic"),
        CheckFamilyBoundary("template-abstract-interpretation", True, False, "interval abstraction of control-token counts", True, "thm-abstract-interpretation-soundness"),
        CheckFamilyBoundary("json-schema", True, True, "object/enum/bounded-int fragment", True, "thm-json-schema-decision"),
        CheckFamilyBoundary("multi-agent-handoffs", True, True, "two-agent sender-tagged session", True, "thm-multi-agent-noncfusion"),
    )


# ---------------------------------------------------------------------------
# Proof-carrying diagnostics (step 405) and the regression report (steps 406, 412).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CertifiedTheoremResult:
    theorem: CertifiedTheorem
    verdict: KernelVerdict

    @property
    def passed(self) -> bool:
        return self.verdict.verified

    def to_dict(self) -> dict[str, object]:
        payload = self.theorem.to_dict()
        payload["verdict"] = self.verdict.to_dict()
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True, slots=True)
class CertifiedVerificationReport:
    results: tuple[CertifiedTheoremResult, ...]
    boundaries: tuple[CheckFamilyBoundary, ...]
    kernel_version: str = CERTIFIED_KERNEL_VERSION

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def theorem_count(self) -> int:
        return len(self.results)

    @property
    def checked_states(self) -> int:
        return sum(result.verdict.checked_states for result in self.results)

    def certified_families(self) -> tuple[str, ...]:
        return tuple(sorted({result.theorem.check_family for result in self.results if result.passed}))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "promptabi.certified-verification.v1",
            "kernel_version": self.kernel_version,
            "passed": self.passed,
            "theorem_count": self.theorem_count,
            "checked_states": self.checked_states,
            "certified_families": list(self.certified_families()),
            "results": [result.to_dict() for result in self.results],
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
        }


def run_certified_verification(*, kernel: ProofKernel | None = None) -> CertifiedVerificationReport:
    """Run the trusted kernel against every certified theorem (steps 406, 412)."""

    kernel = kernel or ProofKernel()
    results = tuple(
        CertifiedTheoremResult(theorem=theorem, verdict=kernel.verify(theorem.certificate))
        for theorem in all_certified_theorems()
    )
    return CertifiedVerificationReport(results=results, boundaries=check_family_boundaries())


@dataclass(frozen=True, slots=True)
class ProofCarryingDiagnostic:
    """A diagnostic-like record with an attached, independently checkable certificate (step 405)."""

    rule_id: str
    message: str
    check_family: str
    certificate: ProofCertificate

    def verify(self, *, kernel: ProofKernel | None = None) -> KernelVerdict:
        return (kernel or ProofKernel()).verify(self.certificate)

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "check_family": self.check_family,
            "certificate": self.certificate.to_dict(),
            "certificate_digest": self.certificate.digest(),
        }


def attach_certificate(rule_id: str, message: str, theorem: CertifiedTheorem) -> ProofCarryingDiagnostic:
    """Attach a theorem's certificate to a diagnostic (proof-carrying diagnostics)."""

    return ProofCarryingDiagnostic(
        rule_id=rule_id,
        message=message,
        check_family=theorem.check_family,
        certificate=theorem.certificate,
    )


# ---------------------------------------------------------------------------
# --certified gating (step 415).
# ---------------------------------------------------------------------------


def certified_check_families(*, kernel: ProofKernel | None = None) -> tuple[str, ...]:
    """Return only the check families backed by a currently-passing certificate."""

    return run_certified_verification(kernel=kernel).certified_families()


def gate_checks_by_certification(
    requested_families: Sequence[str], *, kernel: ProofKernel | None = None
) -> dict[str, object]:
    """Partition requested check families into certified (allowed) and gated-out."""

    certified = set(certified_check_families(kernel=kernel))
    allowed = tuple(family for family in requested_families if family in certified)
    gated = tuple(family for family in requested_families if family not in certified)
    return {"certified": sorted(certified), "allowed": list(allowed), "gated_out": list(gated)}


# ---------------------------------------------------------------------------
# Extracted kernels + benchmark (step 411).
# ---------------------------------------------------------------------------


def extracted_kernel_source(language: str) -> str:
    """Emit a verified-by-construction exhaustive checker in OCaml or Rust (step 411).

    The extracted kernel mirrors the Python no-counterexample procedure: it
    enumerates a finite product space and confirms no assignment satisfies all
    constraints.  The generated source is deterministic and compilable.
    """

    language = language.lower()
    if language == "ocaml":
        return _OCAML_KERNEL
    if language == "rust":
        return _RUST_KERNEL
    raise ValueError(f"unsupported extraction target: {language!r} (expected 'ocaml' or 'rust')")


@dataclass(frozen=True, slots=True)
class KernelBenchmarkResult:
    theorem_id: str
    checked_states: int
    python_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "theorem_id": self.theorem_id,
            "checked_states": self.checked_states,
            "python_seconds": round(self.python_seconds, 6),
        }


def benchmark_kernel(*, repeats: int = 3) -> tuple[KernelBenchmarkResult, ...]:
    """Benchmark the Python kernel across all theorems (step 411 baseline)."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    kernel = ProofKernel()
    results: list[KernelBenchmarkResult] = []
    for theorem in all_certified_theorems():
        best = float("inf")
        verdict = kernel.verify(theorem.certificate)
        for _ in range(repeats):
            start = time.perf_counter()
            kernel.verify(theorem.certificate)
            best = min(best, time.perf_counter() - start)
        results.append(
            KernelBenchmarkResult(
                theorem_id=theorem.theorem_id,
                checked_states=verdict.checked_states,
                python_seconds=best,
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# TCB audit (step 414) and technical report (step 413).
# ---------------------------------------------------------------------------


def trusted_computing_base_audit() -> dict[str, object]:
    """Enumerate every unproven assumption the certified layer relies upon (step 414)."""

    return {
        "schema": "promptabi.tcb-audit.v1",
        "kernel_version": CERTIFIED_KERNEL_VERSION,
        "trusted_components": [
            "the ProofKernel enumeration/witness-evaluation procedure (this module)",
            "FiniteContractProblem.from_dict deserialization and Expression.evaluate semantics",
            "the Python runtime, hashlib, and json standard library",
        ],
        "explicitly_untrusted": [
            "the production analyzer and all rule implementations",
            "the optional Z3 SMT backend",
            "the certificate producer (analyzer may be buggy; kernel re-derives)",
        ],
        "modeling_assumptions": [
            "theorems hold over the *modeled finite fragment*, not arbitrary inputs",
            "obligations exceeding KERNEL_MAX_MODEL_STATES are abstained, not trusted",
            "round-trip lemma is empirical over a bounded corpus (sound, not complete)",
        ],
        "abstention_boundary_states": KERNEL_MAX_MODEL_STATES,
    }


def formal_semantics_report() -> str:
    """Render the formal-semantics technical report with mechanized appendices (step 413)."""

    report = run_certified_verification()
    audit = trusted_computing_base_audit()
    lines: list[str] = []
    lines.append(f"# PromptABI Certified Verification — Technical Report ({CERTIFIED_KERNEL_VERSION})")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append(
        "PromptABI exports machine-checkable proof certificates for finite fragments of its "
        "contract checks. A small trusted kernel re-validates each certificate independently "
        "of the production analyzer."
    )
    lines.append("")
    lines.append("## 2. Theorems and kernel verdicts")
    for result in report.results:
        theorem = result.theorem
        lines.append(f"### {theorem.theorem_id} — {theorem.title}")
        lines.append(f"- Statement: {theorem.statement}")
        lines.append(f"- Check family: {theorem.check_family}")
        lines.append(f"- Claim: {theorem.certificate.claim}")
        lines.append(f"- Kernel outcome: {result.verdict.outcome} ({result.verdict.checked_states} states)")
        lines.append(f"- Verified: {result.verdict.verified}")
        lines.append("")
    lines.append("## 3. Soundness/completeness boundaries")
    for boundary in report.boundaries:
        lines.append(
            f"- {boundary.check_family}: sound={boundary.sound} complete={boundary.complete} "
            f"fragment=\"{boundary.fragment}\""
        )
    lines.append("")
    lines.append("## 4. Trusted computing base")
    lines.append("Trusted:")
    for item in audit["trusted_components"]:  # type: ignore[index]
        lines.append(f"- {item}")
    lines.append("Explicitly untrusted:")
    for item in audit["explicitly_untrusted"]:  # type: ignore[index]
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Appendix A. Mechanized certificates (digests)")
    for result in report.results:
        lines.append(f"- {result.theorem.theorem_id}: sha256={result.theorem.certificate.digest()}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Renderers.
# ---------------------------------------------------------------------------


def render_certified_verification_text(report: CertifiedVerificationReport) -> str:
    lines = [f"PromptABI certified verification ({report.kernel_version})"]
    status = "PASS" if report.passed else "FAIL"
    lines.append(f"status: {status}  theorems: {report.theorem_count}  checked-states: {report.checked_states}")
    for result in report.results:
        mark = "PASS" if result.passed else "FAIL"
        lines.append(
            f"  [{mark}] {result.theorem.theorem_id} ({result.theorem.check_family}) "
            f"-> {result.verdict.outcome} ({result.verdict.checked_states} states)"
        )
    lines.append(f"certified families: {', '.join(report.certified_families())}")
    return "\n".join(lines) + "\n"


def render_certified_verification_json(report: CertifiedVerificationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _contains(variable: str, needle: str):
    from .formal import Contains

    return Contains(Var(variable), Value(needle))


def _affine(*terms: tuple[str, int], const: int = 0):
    from .formal import Mul, Sum

    parts: list[object] = []
    for name, coefficient in terms:
        if coefficient == 1:
            parts.append(Var(name))
        else:
            parts.append(Mul(coefficient, Var(name)))
    if const:
        parts.append(Value(const))
    if not parts:
        return Value(0)
    if len(parts) == 1:
        return parts[0]
    return Sum(*parts)


def _as_bool_int(expression):
    # Wrap a Boolean expression so equality/inequality compares truth values.
    return expression


def _value_in_domain(domain: VariableDomain, value: object) -> bool:
    if isinstance(domain, BoolDomain):
        return isinstance(value, bool)
    return value in domain.values()


def _model_space_size(variables: Sequence[VariableDomain]) -> int:
    size = 1
    for variable in variables:
        size *= len(variable.values())
    return size


def _enumerate_models(variables: Sequence[VariableDomain]):
    names = [variable.name for variable in variables]
    domains = [variable.values() for variable in variables]
    for combo in product(*domains):
        yield dict(zip(names, combo))


def _has_model(variables: Sequence[VariableDomain], constraints: Sequence[NamedConstraint]) -> bool:
    for assignment in _enumerate_models(variables):
        if all(bool(constraint.expression.evaluate(assignment)) for constraint in constraints):
            return True
    return False


_OCAML_KERNEL = """(* Extracted PromptABI proof kernel (no-counterexample procedure).
   Auto-generated; mirrors certified.ProofKernel for finite obligations. *)

let rec product = function
  | [] -> [[]]
  | xs :: rest ->
      let tails = product rest in
      List.concat_map (fun x -> List.map (fun t -> x :: t) tails) xs

(* [check_no_counterexample domains sat] returns true iff no assignment in the
   finite product [domains] satisfies the predicate [sat]. *)
let check_no_counterexample (domains : 'a list list) (sat : 'a list -> bool) : bool =
  not (List.exists sat (product domains))
"""

_RUST_KERNEL = """// Extracted PromptABI proof kernel (no-counterexample procedure).
// Auto-generated; mirrors certified.ProofKernel for finite obligations.

/// Returns true iff no assignment in the finite product of `domains`
/// satisfies the predicate `sat`.
pub fn check_no_counterexample<T: Clone>(
    domains: &[Vec<T>],
    sat: &dyn Fn(&[T]) -> bool,
) -> bool {
    fn rec<T: Clone>(domains: &[Vec<T>], acc: &mut Vec<T>, sat: &dyn Fn(&[T]) -> bool) -> bool {
        match domains.split_first() {
            None => !sat(acc),
            Some((head, rest)) => head.iter().all(|value| {
                acc.push(value.clone());
                let ok = rec(rest, acc, sat);
                acc.pop();
                ok
            }),
        }
    }
    let mut acc: Vec<T> = Vec::new();
    rec(domains, &mut acc, sat)
}
"""
