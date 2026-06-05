"""An open provider-contract test-vector format (step 280).

Interoperability across LLM providers needs a *shared, declarative* test-vector
format: a request, the provider-agnostic obligations it must satisfy, and the
salient features of an acceptable response.  This module defines that format
(:class:`ProviderTestVector`), a canonical serialization (stable digest), and a
loader/validator so vectors can be authored once and replayed against any
provider adapter.  It is the substrate the rest of the provider-conformance
steps build on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum

PROVIDER_TEST_VECTOR_VERSION = "promptabi.provider-test-vector.v1"


class Obligation(StrEnum):
    ROLE_NON_FORGEABLE = "role-non-forgeable"
    STOP_TERMINATES = "stop-terminates"
    STRUCTURED_OUTPUT_VALID = "structured-output-valid"
    TOOL_CALL_WELL_FORMED = "tool-call-well-formed"
    ERROR_ENVELOPE_CONFORMANT = "error-envelope-conformant"


@dataclass(frozen=True, slots=True)
class VectorMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ProviderTestVector:
    vector_id: str
    messages: tuple[VectorMessage, ...]
    obligations: tuple[Obligation, ...]
    params: dict[str, object] = field(default_factory=dict)
    expected_features: dict[str, object] = field(default_factory=dict)

    def canonical_bytes(self) -> bytes:
        payload = {
            "version": PROVIDER_TEST_VECTOR_VERSION,
            "vector_id": self.vector_id,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "obligations": sorted(o.value for o in self.obligations),
            "params": self.params,
            "expected_features": self.expected_features,
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return json.loads(self.canonical_bytes())


class TestVectorError(ValueError):
    pass


def load_test_vector(raw: dict[str, object]) -> ProviderTestVector:
    try:
        messages = tuple(
            VectorMessage(role=str(m["role"]), content=str(m["content"]))
            for m in raw["messages"]  # type: ignore[union-attr]
        )
        obligations = tuple(Obligation(o) for o in raw.get("obligations", []))  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise TestVectorError(f"malformed test vector: {exc}") from exc
    if not messages:
        raise TestVectorError("test vector must contain at least one message")
    return ProviderTestVector(
        vector_id=str(raw["vector_id"]),
        messages=messages,
        obligations=obligations,
        params=dict(raw.get("params", {})),  # type: ignore[arg-type]
        expected_features=dict(raw.get("expected_features", {})),  # type: ignore[arg-type]
    )


def render_test_vector_text(vector: ProviderTestVector) -> str:
    lines = [
        f"PromptABI provider test vector {vector.vector_id} "
        f"({PROVIDER_TEST_VECTOR_VERSION})",
        f"digest: {vector.digest()}",
        f"messages: {len(vector.messages)}",
        "obligations: " + ", ".join(o.value for o in vector.obligations),
    ]
    return "\n".join(lines) + "\n"
