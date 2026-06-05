"""Privacy-preserving transforms for diagnostic witnesses."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from enum import StrEnum
from typing import Any

from .diagnostics import ArtifactRef, Diagnostic, WitnessStep, WitnessTrace
from .session import VerificationResult


class WitnessPrivacyMode(StrEnum):
    """Output modes for witness fields that may contain prompt or dataset text."""

    RAW = "raw"
    REDACTED = "redacted"
    HASH_ONLY = "hash-only"
    STRUCTURAL = "structural"


def apply_witness_privacy(
    result: VerificationResult,
    mode: WitnessPrivacyMode | str,
) -> VerificationResult:
    """Return a result whose witness payloads obey the requested privacy mode.

    Diagnostic messages, source spans, fingerprints, rule IDs, severities, token
    IDs, and structural offsets are intentionally left unchanged so CI baselines
    and proofs remain reproducible. Only witness payload fields that can carry
    rendered prompts, solver literals, parser excerpts, or suggested raw fixes
    are transformed.
    """

    privacy_mode = WitnessPrivacyMode(mode)
    if privacy_mode is WitnessPrivacyMode.RAW:
        return result
    return replace(
        result,
        diagnostics=tuple(_private_diagnostic(diagnostic, privacy_mode) for diagnostic in result.diagnostics),
    )


def private_witness(witness: WitnessTrace, mode: WitnessPrivacyMode | str) -> WitnessTrace:
    """Return a privacy-preserving copy of one witness trace."""

    privacy_mode = WitnessPrivacyMode(mode)
    if privacy_mode is WitnessPrivacyMode.RAW:
        return witness
    return WitnessTrace(
        summary=witness.summary,
        steps=tuple(
            WitnessStep(
                action=step.action,
                input=_private_string(step.input, privacy_mode) if step.input is not None else None,
                output=_private_string(step.output, privacy_mode) if step.output is not None else None,
            )
            for step in witness.steps
        ),
        artifacts=tuple(_private_artifact_ref(artifact, privacy_mode) for artifact in witness.artifacts),
        rendered_strings=tuple(_private_string(item, privacy_mode) for item in witness.rendered_strings),
        token_ids=witness.token_ids,
        role_regions=tuple(_private_role_region(region, privacy_mode) for region in witness.role_regions),
        parser_states=tuple(_private_string(item, privacy_mode) for item in witness.parser_states),
        solver_assignments=tuple(_private_mapping(item, privacy_mode) for item in witness.solver_assignments),
        truncation_decisions=tuple(_private_mapping(item, privacy_mode) for item in witness.truncation_decisions),
        minimal_fixes=tuple(_private_string(item, privacy_mode) for item in witness.minimal_fixes),
    )


def _private_diagnostic(diagnostic: Diagnostic, mode: WitnessPrivacyMode) -> Diagnostic:
    if diagnostic.witness is None:
        return diagnostic
    properties = dict(diagnostic.properties)
    properties["witness_privacy"] = {
        "mode": mode.value,
        "guarantee": "witness payload strings are transformed; structural offsets, token IDs, spans, and fingerprints are preserved",
    }
    return replace(
        diagnostic,
        witness=private_witness(diagnostic.witness, mode),
        properties=tuple(properties.items()),
    )


def _private_artifact_ref(artifact: ArtifactRef, mode: WitnessPrivacyMode) -> ArtifactRef:
    return ArtifactRef(
        kind=artifact.kind,
        name=artifact.name,
        path=_private_string(artifact.path, mode) if artifact.path is not None else None,
        uri=_private_string(artifact.uri, mode) if artifact.uri is not None else None,
        version=artifact.version,
        revision=artifact.revision,
        sha256=artifact.sha256,
        license=artifact.license,
        source=artifact.source,
    )


def _private_value(value: Any, mode: WitnessPrivacyMode) -> Any:
    if isinstance(value, str):
        return _private_string(value, mode)
    if isinstance(value, dict):
        return _private_mapping(value, mode)
    if isinstance(value, list):
        return [_private_value(item, mode) for item in value]
    if isinstance(value, tuple):
        return tuple(_private_value(item, mode) for item in value)
    return value


def _private_mapping(value: dict[str, Any], mode: WitnessPrivacyMode) -> dict[str, Any]:
    return {str(key): _private_value(item, mode) for key, item in value.items()}


def _private_role_region(value: dict[str, Any], mode: WitnessPrivacyMode) -> dict[str, Any]:
    structural_keys = {
        "path_index",
        "region_index",
        "role",
        "role_source",
        "start_offset",
        "end_offset",
        "segment_indexes",
        "message_index",
        "content_expressions",
        "excluded_roles",
    }
    return {
        str(key): item if key in structural_keys else _private_value(item, mode)
        for key, item in value.items()
    }


def _private_string(value: str, mode: WitnessPrivacyMode) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if mode is WitnessPrivacyMode.HASH_ONLY:
        return f"sha256:{digest};bytes:{len(value.encode('utf-8'))};chars:{len(value)}"
    if mode is WitnessPrivacyMode.STRUCTURAL:
        return (
            f"<structural chars={len(value)} bytes={len(value.encode('utf-8'))} "
            f"lines={value.count(chr(10)) + 1} sha256={digest[:16]}>"
        )
    if mode is WitnessPrivacyMode.REDACTED:
        return _position_preserving_redaction(value, digest)
    raise AssertionError(f"unhandled witness privacy mode: {mode}")


def _position_preserving_redaction(value: str, digest: str) -> str:
    masked = "".join(char if char.isspace() else "x" for char in value)
    if not masked:
        return masked
    marker = f"sha256:{digest[:16]}"
    if len(masked) <= len(marker) + 2:
        return masked
    start = max(0, len(masked) - len(marker) - 1)
    return f"{masked[:start]}#{marker}"
