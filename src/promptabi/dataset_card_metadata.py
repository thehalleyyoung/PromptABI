"""Dataset-card PromptABI metadata (step 272).

A dataset card (the README front-matter of a HF dataset) can carry a structured
``promptabi:`` metadata block declaring the prompt interface the data targets:
the chat-template digest, the tokenizer it was rendered with, the supervised
role, and the special tokens it relies on.  Tooling needs this block to verify a
dataset against a model before training.

This module parses the metadata block (from an already-decoded mapping) and
validates that the required fields are present and well-typed, abstaining
gracefully when the block is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

DATASET_CARD_VERSION = "promptabi.dataset-card.v1"

_REQUIRED_FIELDS = ("template_digest", "tokenizer", "supervised_role")


class DatasetCardFindingKind(StrEnum):
    MISSING_BLOCK = "missing-block"
    MISSING_FIELD = "missing-field"
    WRONG_TYPE = "wrong-type"
    UNKNOWN_SCHEMA_VERSION = "unknown-schema-version"


@dataclass(frozen=True, slots=True)
class DatasetCardMetadata:
    schema_version: str
    template_digest: str
    tokenizer: str
    supervised_role: str
    special_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "template_digest": self.template_digest,
            "tokenizer": self.tokenizer,
            "supervised_role": self.supervised_role,
            "special_tokens": list(self.special_tokens),
        }


@dataclass(frozen=True, slots=True)
class DatasetCardFinding:
    kind: DatasetCardFindingKind
    field: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "field": self.field, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class DatasetCardResult:
    version: str
    valid: bool
    metadata: DatasetCardMetadata | None
    findings: tuple[DatasetCardFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "findings": [f.to_dict() for f in self.findings],
        }


def parse_dataset_card(card: dict[str, object]) -> DatasetCardResult:
    block = card.get("promptabi")
    if not isinstance(block, dict):
        return DatasetCardResult(
            version=DATASET_CARD_VERSION,
            valid=False,
            metadata=None,
            findings=(
                DatasetCardFinding(
                    DatasetCardFindingKind.MISSING_BLOCK,
                    "promptabi",
                    "dataset card has no 'promptabi:' metadata block",
                ),
            ),
        )

    findings: list[DatasetCardFinding] = []
    schema_version = block.get("schema_version")
    if schema_version not in (None, "1", 1, "v1"):
        findings.append(
            DatasetCardFinding(
                DatasetCardFindingKind.UNKNOWN_SCHEMA_VERSION,
                "schema_version",
                f"unsupported schema version {schema_version!r}",
            )
        )

    for fld in _REQUIRED_FIELDS:
        if fld not in block:
            findings.append(
                DatasetCardFinding(
                    DatasetCardFindingKind.MISSING_FIELD,
                    fld,
                    f"required field {fld!r} is missing",
                )
            )
        elif not isinstance(block[fld], str):
            findings.append(
                DatasetCardFinding(
                    DatasetCardFindingKind.WRONG_TYPE,
                    fld,
                    f"field {fld!r} must be a string",
                )
            )

    special = block.get("special_tokens", [])
    if not isinstance(special, (list, tuple)):
        findings.append(
            DatasetCardFinding(
                DatasetCardFindingKind.WRONG_TYPE,
                "special_tokens",
                "special_tokens must be a list",
            )
        )
        special = []

    metadata: DatasetCardMetadata | None = None
    if not findings:
        metadata = DatasetCardMetadata(
            schema_version=str(schema_version or "1"),
            template_digest=str(block["template_digest"]),
            tokenizer=str(block["tokenizer"]),
            supervised_role=str(block["supervised_role"]),
            special_tokens=tuple(str(s) for s in special),
        )

    return DatasetCardResult(
        version=DATASET_CARD_VERSION,
        valid=not findings,
        metadata=metadata,
        findings=tuple(findings),
    )


def render_dataset_card_text(result: DatasetCardResult) -> str:
    lines = [
        f"PromptABI dataset-card metadata ({result.version})",
        f"result: {'VALID' if result.valid else 'INVALID'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value} [{f.field}]: {f.detail}")
    return "\n".join(lines) + "\n"
