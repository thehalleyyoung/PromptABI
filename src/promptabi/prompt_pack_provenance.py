"""Integrate prompt-pack provenance with model registries (step 254).

A prompt pack is only meaningful relative to the *models* it was validated
against.  When a pack is published it should carry a provenance chain that ties
its content digest to one or more model-registry entries (e.g. a registry name,
model id, and the artifact revision/digest the pack was certified against).  A
consumer resolving the pack against a registry must be able to prove:

* the pack's recorded content digest matches what they actually downloaded,
* the model the pack claims compatibility with exists in the registry snapshot
  they trust, and
* the model revision the pack pins matches the registry's current digest (or is
  explicitly recorded as a known-good prior revision).

This module models the registry snapshot and the provenance link, and verifies
the chain end-to-end with plain content-digest comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

PROMPT_PACK_PROVENANCE_VERSION = "promptabi.prompt-pack-provenance.v1"


class ProvenanceFindingKind(StrEnum):
    PACK_DIGEST_MISMATCH = "pack-digest-mismatch"
    MODEL_NOT_IN_REGISTRY = "model-not-in-registry"
    MODEL_REVISION_MISMATCH = "model-revision-mismatch"
    NO_MODELS_PINNED = "no-models-pinned"


@dataclass(frozen=True, slots=True)
class RegistryModel:
    model_id: str
    revision: str
    digest: str


@dataclass(frozen=True, slots=True)
class ModelRegistrySnapshot:
    registry: str
    models: tuple[RegistryModel, ...]

    def find(self, model_id: str) -> RegistryModel | None:
        return next((m for m in self.models if m.model_id == model_id), None)


@dataclass(frozen=True, slots=True)
class PinnedModel:
    model_id: str
    revision: str
    digest: str
    known_good_prior: bool = False


@dataclass(frozen=True, slots=True)
class PackProvenance:
    pack: str
    version: str
    pack_digest: str
    registry: str
    models: tuple[PinnedModel, ...]


@dataclass(frozen=True, slots=True)
class ProvenanceFinding:
    kind: ProvenanceFindingKind
    subject: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "subject": self.subject, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ProvenanceVerification:
    version: str
    valid: bool
    findings: tuple[ProvenanceFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "findings": [f.to_dict() for f in self.findings],
        }


def verify_provenance(
    provenance: PackProvenance,
    snapshot: ModelRegistrySnapshot,
    actual_pack_digest: str,
) -> ProvenanceVerification:
    findings: list[ProvenanceFinding] = []

    if provenance.pack_digest != actual_pack_digest:
        findings.append(
            ProvenanceFinding(
                ProvenanceFindingKind.PACK_DIGEST_MISMATCH,
                provenance.pack,
                f"recorded {provenance.pack_digest} != actual {actual_pack_digest}",
            )
        )

    if not provenance.models:
        findings.append(
            ProvenanceFinding(
                ProvenanceFindingKind.NO_MODELS_PINNED,
                provenance.pack,
                "provenance pins no model-registry entries",
            )
        )

    for pinned in provenance.models:
        registry_model = snapshot.find(pinned.model_id)
        if registry_model is None:
            findings.append(
                ProvenanceFinding(
                    ProvenanceFindingKind.MODEL_NOT_IN_REGISTRY,
                    pinned.model_id,
                    f"not present in registry snapshot {snapshot.registry!r}",
                )
            )
            continue
        if registry_model.digest != pinned.digest and not pinned.known_good_prior:
            findings.append(
                ProvenanceFinding(
                    ProvenanceFindingKind.MODEL_REVISION_MISMATCH,
                    pinned.model_id,
                    f"pinned digest {pinned.digest} != registry "
                    f"{registry_model.digest} (rev {registry_model.revision})",
                )
            )

    return ProvenanceVerification(
        version=PROMPT_PACK_PROVENANCE_VERSION,
        valid=not findings,
        findings=tuple(findings),
    )


def render_provenance_text(result: ProvenanceVerification) -> str:
    lines = [
        f"PromptABI prompt-pack provenance ({result.version})",
        f"result: {'VALID' if result.valid else 'INVALID'}",
    ]
    for finding in result.findings:
        lines.append(f"  ! {finding.kind.value} [{finding.subject}]: {finding.detail}")
    return "\n".join(lines) + "\n"
