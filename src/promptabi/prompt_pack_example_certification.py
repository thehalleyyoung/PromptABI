"""Certify prompt-pack examples against supported model families (step 246).

A reusable pack ships *examples* -- sample conversations that demonstrate how to
drive the pack against a particular model family.  An example is only honest if
it stays inside the ABI the pack actually promises: it must target a model family
the pack supports, reference a template the pack exports, and only use roles and
tools the pack declares.  An example that calls an undeclared tool, uses a role
the templates never emit, or targets a family the chosen template does not list
is a latent break that will surface in a consumer's project, not the author's.

:func:`certify_examples` checks every example against the pack's real
:class:`~promptabi.artifacts.PromptPackArtifact` (via its capability signature
and its per-template family declarations) and reports each inconsistency with the
exact example and offending name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from .artifacts import PromptPackArtifact
from .prompt_pack_capability import derive_capability_signature

PROMPT_PACK_EXAMPLE_CERTIFICATION_VERSION = "promptabi.prompt-pack-example-certification.v1"


class ExampleFindingKind(StrEnum):
    UNSUPPORTED_MODEL_FAMILY = "unsupported-model-family"
    UNKNOWN_TEMPLATE = "unknown-template"
    TEMPLATE_FAMILY_MISMATCH = "template-family-mismatch"
    UNDECLARED_ROLE = "undeclared-role"
    UNDECLARED_TOOL = "undeclared-tool"


@dataclass(frozen=True, slots=True)
class PackExample:
    name: str
    model_family: str
    template: str
    roles_used: tuple[str, ...] = ()
    tools_called: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_family": self.model_family,
            "template": self.template,
            "roles_used": list(self.roles_used),
            "tools_called": list(self.tools_called),
        }


@dataclass(frozen=True, slots=True)
class ExampleFinding:
    kind: ExampleFindingKind
    example: str
    name: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "example": self.example, "name": self.name}


@dataclass(frozen=True, slots=True)
class ExampleCertificationReport:
    version: str
    pack_name: str
    certified: bool
    examples_checked: int
    findings: tuple[ExampleFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "pack_name": self.pack_name,
            "certified": self.certified,
            "examples_checked": self.examples_checked,
            "findings": [f.to_dict() for f in self.findings],
        }


def certify_examples(
    pack: PromptPackArtifact, examples: "tuple[PackExample, ...] | list[PackExample]"
) -> ExampleCertificationReport:
    """Certify that every example is consistent with the pack's declared ABI."""

    signature = derive_capability_signature(pack)
    supported_families = set(signature.model_families)
    roles = set(signature.roles)
    tools = set(signature.tools)
    templates_by_name = {t.name: t for t in pack.exported_templates}

    findings: list[ExampleFinding] = []
    for example in examples:
        if example.model_family not in supported_families:
            findings.append(
                ExampleFinding(
                    ExampleFindingKind.UNSUPPORTED_MODEL_FAMILY,
                    example.name,
                    example.model_family,
                )
            )

        template = templates_by_name.get(example.template)
        if template is None:
            findings.append(
                ExampleFinding(
                    ExampleFindingKind.UNKNOWN_TEMPLATE,
                    example.name,
                    example.template,
                )
            )
        elif (
            template.supported_model_families
            and example.model_family not in template.supported_model_families
        ):
            # The template restricts its families and this one isn't listed.
            findings.append(
                ExampleFinding(
                    ExampleFindingKind.TEMPLATE_FAMILY_MISMATCH,
                    example.name,
                    f"{example.template}:{example.model_family}",
                )
            )

        for role in example.roles_used:
            if role not in roles:
                findings.append(
                    ExampleFinding(
                        ExampleFindingKind.UNDECLARED_ROLE, example.name, role
                    )
                )
        for tool in example.tools_called:
            if tool not in tools:
                findings.append(
                    ExampleFinding(
                        ExampleFindingKind.UNDECLARED_TOOL, example.name, tool
                    )
                )

    return ExampleCertificationReport(
        version=PROMPT_PACK_EXAMPLE_CERTIFICATION_VERSION,
        pack_name=pack.pack_name,
        certified=not findings,
        examples_checked=len(examples),
        findings=tuple(findings),
    )


def render_example_report_json(report: ExampleCertificationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_example_report_text(report: ExampleCertificationReport) -> str:
    lines = [
        f"PromptABI prompt-pack example certification ({report.version})",
        f"pack: {report.pack_name}",
        f"examples checked: {report.examples_checked}",
        f"result: {'CERTIFIED' if report.certified else 'REJECTED'}",
    ]
    for finding in report.findings:
        lines.append(f"  ! [{finding.example}] {finding.kind.value}: {finding.name}")
    return "\n".join(lines) + "\n"
