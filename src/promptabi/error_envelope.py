"""Certify error-envelope compatibility (step 286).

Clients written against the OpenAI error envelope (``{"error": {"message",
"type", "code", "param"}}``) break when a "compatible" provider returns a
different shape -- a bare string, a different key, a missing ``type``, or an HTTP
status that disagrees with the body.  This module certifies a provider's error
responses against the canonical envelope and a status/type consistency table,
reporting exactly which field diverges so a client adapter can normalize it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

ERROR_ENVELOPE_VERSION = "promptabi.error-envelope.v1"

_REQUIRED_FIELDS = ("message", "type")
# Canonical (status -> expected error type) pairs.
_STATUS_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    500: "server_error",
}


class ErrorEnvelopeFindingKind(StrEnum):
    NOT_AN_OBJECT = "not-an-object"
    MISSING_ERROR_KEY = "missing-error-key"
    MISSING_FIELD = "missing-field"
    STATUS_TYPE_MISMATCH = "status-type-mismatch"


@dataclass(frozen=True, slots=True)
class ProviderErrorResponse:
    http_status: int
    body: object


@dataclass(frozen=True, slots=True)
class ErrorEnvelopeFinding:
    kind: ErrorEnvelopeFindingKind
    field: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "field": self.field, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ErrorEnvelopeResult:
    version: str
    conformant: bool
    findings: tuple[ErrorEnvelopeFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "conformant": self.conformant,
            "findings": [f.to_dict() for f in self.findings],
        }


def certify_error_envelope(response: ProviderErrorResponse) -> ErrorEnvelopeResult:
    findings: list[ErrorEnvelopeFinding] = []
    body = response.body

    if not isinstance(body, dict):
        findings.append(
            ErrorEnvelopeFinding(
                ErrorEnvelopeFindingKind.NOT_AN_OBJECT,
                "body",
                f"error body is {type(body).__name__}, expected object",
            )
        )
        return ErrorEnvelopeResult(ERROR_ENVELOPE_VERSION, False, tuple(findings))

    error = body.get("error")
    if not isinstance(error, dict):
        findings.append(
            ErrorEnvelopeFinding(
                ErrorEnvelopeFindingKind.MISSING_ERROR_KEY,
                "error",
                "top-level 'error' object is missing or not an object",
            )
        )
        return ErrorEnvelopeResult(ERROR_ENVELOPE_VERSION, False, tuple(findings))

    for fld in _REQUIRED_FIELDS:
        if not error.get(fld):
            findings.append(
                ErrorEnvelopeFinding(
                    ErrorEnvelopeFindingKind.MISSING_FIELD,
                    fld,
                    f"error.{fld} is missing or empty",
                )
            )

    expected_type = _STATUS_TYPE.get(response.http_status)
    actual_type = error.get("type")
    if expected_type is not None and actual_type and actual_type != expected_type:
        findings.append(
            ErrorEnvelopeFinding(
                ErrorEnvelopeFindingKind.STATUS_TYPE_MISMATCH,
                "type",
                f"HTTP {response.http_status} should carry type "
                f"{expected_type!r}, got {actual_type!r}",
            )
        )

    return ErrorEnvelopeResult(
        version=ERROR_ENVELOPE_VERSION,
        conformant=not findings,
        findings=tuple(findings),
    )


def render_error_envelope_text(result: ErrorEnvelopeResult) -> str:
    lines = [
        f"PromptABI error-envelope certification ({result.version})",
        f"result: {'CONFORMANT' if result.conformant else 'NONCONFORMANT'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value} [{f.field}]: {f.detail}")
    return "\n".join(lines) + "\n"
