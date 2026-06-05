"""Deterministic auto-fix planning for PromptABI projects."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config
from .diagnostics import Diagnostic, DiagnosticSeverity, WitnessStep, WitnessTrace
from .fix_suggestions import RankedFixSuggestion, rank_fix_suggestions
from .lockfiles import build_lockfile, write_lockfile
from .plugins import PluginRegistry
from .session import VerificationSession


class AutoFixKind(StrEnum):
    """Supported low-risk auto-fix families."""

    LOCKFILE = "lockfile"
    SPECIAL_TOKENS = "special-tokens"
    UNSUPPORTED_FRAGMENTS = "unsupported-fragments"
    DOCS_STUB = "docs-stub"


class AutoFixStatus(StrEnum):
    """Lifecycle state for one planned auto-fix."""

    PLANNED = "planned"
    APPLIED = "applied"
    SKIPPED = "skipped"


class AutoFixError(ValueError):
    """Raised when a low-risk auto-fix cannot be planned or applied safely."""


class GuardedPreviewRisk(StrEnum):
    """Risk bands supported by guarded preview planning."""

    HIGH = "high"


@dataclass(frozen=True, slots=True)
class AutoFixChange:
    """One file-level change produced by auto-fix planning or application."""

    kind: AutoFixKind
    path: str
    action: str
    status: AutoFixStatus
    message: str
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "action": self.action,
            "status": self.status.value,
            "message": self.message,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class GuardedAutoFixPreview:
    """A non-mutating preview for a high-risk interface change."""

    risk: GuardedPreviewRisk
    suggestion: RankedFixSuggestion
    diagnostics: tuple[Diagnostic, ...]
    before_witnesses: tuple[WitnessTrace, ...]
    after_witnesses: tuple[WitnessTrace, ...]
    guardrails: tuple[str, ...]

    @property
    def changes_user_visible_prompt_behavior(self) -> bool:
        return self.suggestion.changes_user_visible_prompt_behavior

    def to_dict(self) -> dict[str, object]:
        return {
            "risk": self.risk.value,
            "suggestion": self.suggestion.to_dict(),
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "before_witnesses": [witness.to_dict() for witness in self.before_witnesses],
            "after_witnesses": [witness.to_dict() for witness in self.after_witnesses],
            "guardrails": list(self.guardrails),
            "changes_user_visible_prompt_behavior": self.changes_user_visible_prompt_behavior,
        }


@dataclass(frozen=True, slots=True)
class AutoFixReport:
    """A deterministic summary of low-risk fixes that were planned or applied."""

    config_path: str
    applied: bool
    changes: tuple[AutoFixChange, ...]
    diagnostics_before: tuple[Diagnostic, ...]
    diagnostics_after: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return all(change.status is not AutoFixStatus.SKIPPED for change in self.changes)

    @property
    def applied_count(self) -> int:
        return sum(1 for change in self.changes if change.status is AutoFixStatus.APPLIED)

    @property
    def planned_count(self) -> int:
        return sum(1 for change in self.changes if change.status is AutoFixStatus.PLANNED)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "applied": self.applied,
            "applied_count": self.applied_count,
            "changes": [change.to_dict() for change in self.changes],
            "config_path": self.config_path,
            "diagnostics_before": [diagnostic.to_dict() for diagnostic in self.diagnostics_before],
            "ok": self.ok,
            "planned_count": self.planned_count,
        }
        if self.diagnostics_after:
            data["diagnostics_after"] = [diagnostic.to_dict() for diagnostic in self.diagnostics_after]
        return data


@dataclass(frozen=True, slots=True)
class GuardedAutoFixPreviewReport:
    """A deterministic summary of guarded high-risk fix previews."""

    config_path: str
    risk: GuardedPreviewRisk
    previews: tuple[GuardedAutoFixPreview, ...]
    diagnostics_before: tuple[Diagnostic, ...]

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "applied": False,
            "config_path": self.config_path,
            "diagnostics_before": [diagnostic.to_dict() for diagnostic in self.diagnostics_before],
            "ok": self.ok,
            "preview_count": len(self.previews),
            "previews": [preview.to_dict() for preview in self.previews],
            "risk": self.risk.value,
        }


def run_low_risk_autofix(
    config_path: str | Path,
    *,
    kinds: Sequence[str | AutoFixKind] | None = None,
    write: bool = False,
    lockfile_path: str | Path | None = None,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    plugin_registry: PluginRegistry | None = None,
) -> AutoFixReport:
    """Plan or apply low-risk fixes that do not change prompt rendering behavior."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    selected_kinds = _selected_kinds(kinds)
    config = load_config(resolved_config_path)
    if artifact_overrides:
        base_dir = Path(override_base_dir) if override_base_dir is not None else Path.cwd()
        config = config.with_artifact_overrides(dict(artifact_overrides), base_dir=base_dir)
    session = VerificationSession(config, plugin_registry=plugin_registry)
    result = session.run()
    loaded_artifacts, load_diagnostics = session.load_artifacts_with_diagnostics()
    diagnostics_before = result.diagnostics

    config_payload = _read_config_payload(resolved_config_path)
    changes: list[AutoFixChange] = []

    if AutoFixKind.LOCKFILE in selected_kinds:
        lock_path = _resolve_lockfile(lockfile_path, resolved_config_path)
        if _has_load_error(load_diagnostics):
            changes.append(
                AutoFixChange(
                    kind=AutoFixKind.LOCKFILE,
                    path=str(lock_path),
                    action="write-lockfile",
                    status=AutoFixStatus.SKIPPED,
                    message="artifact loading has errors; lockfile would not be reproducible",
                    diagnostics=_diagnostic_fingerprints(load_diagnostics),
                )
            )
        else:
            status = AutoFixStatus.APPLIED if write else AutoFixStatus.PLANNED
            if write:
                write_lockfile(lock_path, build_lockfile(config, loaded_artifacts, result.diagnostics, base_dir=lock_path.parent))
            changes.append(
                AutoFixChange(
                    kind=AutoFixKind.LOCKFILE,
                    path=str(lock_path),
                    action="write-lockfile",
                    status=status,
                    message="pin current artifacts, tool versions, and diagnostic baseline",
                    diagnostics=_diagnostic_fingerprints(diagnostics_before, rule_ids=_LOCKFILE_TRIGGER_RULES),
                )
            )

    if AutoFixKind.SPECIAL_TOKENS in selected_kinds:
        changes.extend(
            _special_token_changes(
                config_payload,
                loaded_artifacts,
                config_path=resolved_config_path,
                write=write,
            )
        )

    if AutoFixKind.UNSUPPORTED_FRAGMENTS in selected_kinds:
        changes.extend(
            _unsupported_fragment_changes(
                loaded_artifacts,
                config_path=resolved_config_path,
                write=write,
            )
        )

    if AutoFixKind.DOCS_STUB in selected_kinds:
        changes.extend(
            _docs_stub_changes(
                diagnostics_before,
                config_path=resolved_config_path,
                write=write,
            )
        )

    if write and _config_payload_changed(config_payload, resolved_config_path):
        _write_config_payload(resolved_config_path, config_payload)

    diagnostics_after: tuple[Diagnostic, ...] = ()
    if write:
        after_config = load_config(resolved_config_path)
        after_session = VerificationSession(after_config, plugin_registry=plugin_registry)
        after_result = after_session.run()
        _after_loaded, after_load_diagnostics = after_session.load_artifacts_with_diagnostics()
        diagnostics_after = tuple(sorted((*after_result.diagnostics, *after_load_diagnostics), key=lambda item: item.sort_key))

    return AutoFixReport(
        config_path=str(resolved_config_path),
        applied=write,
        changes=tuple(changes),
        diagnostics_before=diagnostics_before,
        diagnostics_after=diagnostics_after,
    )


def run_guarded_autofix_preview(
    config_path: str | Path,
    *,
    risk: str | GuardedPreviewRisk = GuardedPreviewRisk.HIGH,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    plugin_registry: PluginRegistry | None = None,
) -> GuardedAutoFixPreviewReport:
    """Preview high-risk template/schema/stop/truncation fixes without mutating files."""

    selected_risk = GuardedPreviewRisk(str(risk))
    resolved_config_path = Path(config_path).expanduser().resolve()
    config = load_config(resolved_config_path)
    if artifact_overrides:
        base_dir = Path(override_base_dir) if override_base_dir is not None else Path.cwd()
        config = config.with_artifact_overrides(dict(artifact_overrides), base_dir=base_dir)
    session = VerificationSession(config, plugin_registry=plugin_registry)
    result = session.run()
    diagnostics_before = result.diagnostics
    previews = _guarded_previews_for_diagnostics(diagnostics_before, selected_risk)
    return GuardedAutoFixPreviewReport(
        config_path=str(resolved_config_path),
        risk=selected_risk,
        previews=previews,
        diagnostics_before=diagnostics_before,
    )


def render_autofix_text(report: AutoFixReport) -> str:
    """Render an auto-fix report for terminal workflows."""

    mode = "applied" if report.applied else "preview"
    lines = [
        f"PromptABI low-risk auto-fix {mode}: {report.config_path}",
        f"changes: {len(report.changes)} ({report.applied_count} applied, {report.planned_count} planned)",
    ]
    if not report.changes:
        lines.append("status: no low-risk fixes found")
    else:
        lines.append(f"status: {'OK' if report.ok else 'PARTIAL'}")
    for change in report.changes:
        lines.append(f"{change.status.value.upper()} {change.kind.value}: {change.action} {change.path}")
        lines.append(f"  {change.message}")
        if change.diagnostics:
            lines.append(f"  diagnostics: {', '.join(change.diagnostics)}")
    return "\n".join(lines) + "\n"


def render_guarded_autofix_preview_text(report: GuardedAutoFixPreviewReport) -> str:
    """Render a guarded auto-fix preview report for terminal workflows."""

    lines = [
        f"PromptABI guarded auto-fix preview ({report.risk.value} risk): {report.config_path}",
        f"previews: {len(report.previews)}",
    ]
    if not report.previews:
        lines.append("status: no high-risk fix previews found")
    else:
        lines.append("status: REVIEW REQUIRED")
    for preview in report.previews:
        suggestion = preview.suggestion
        lines.append(
            f"PREVIEW {preview.risk.value}: rank={suggestion.rank} score={suggestion.score} "
            f"rules={', '.join(suggestion.rules) or '(none)'}"
        )
        lines.append(f"  suggestion: {suggestion.text}")
        lines.append(
            "  risk: "
            f"safety={suggestion.safety.value}, compatibility={suggestion.compatibility.value}, "
            f"blast_radius={suggestion.blast_radius.value}, "
            f"user_visible_prompt_change={str(suggestion.changes_user_visible_prompt_behavior).lower()}"
        )
        lines.append(f"  diagnostics: {', '.join(diagnostic.fingerprint for diagnostic in preview.diagnostics)}")
        for witness in preview.before_witnesses:
            lines.append(f"  before witness: {witness.summary}")
        for witness in preview.after_witnesses:
            lines.append(f"  after witness: {witness.summary}")
        for guardrail in preview.guardrails:
            lines.append(f"  guardrail: {guardrail}")
    return "\n".join(lines) + "\n"


def render_autofix_json(report: AutoFixReport) -> str:
    """Render an auto-fix report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_guarded_autofix_preview_json(report: GuardedAutoFixPreviewReport) -> str:
    """Render a guarded auto-fix preview report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


_LOCKFILE_TRIGGER_RULES = frozenset(
    {
        "artifact-unpinned",
        "artifact-weak-pin",
        "lockfile-artifact-added",
        "lockfile-artifact-drift",
        "lockfile-artifact-missing",
        "lockfile-config-drift",
        "lockfile-diagnostic-baseline-drift",
        "lockfile-library-version-drift",
        "lockfile-provider-fixture-drift",
        "lockfile-load-failed",
    }
)

_HIGH_RISK_TERMS = frozenset(
    {
        "template",
        "schema",
        "stop",
        "truncation",
        "context",
        "prompt",
        "parser",
        "grammar",
        "tool",
        "provider",
        "role",
        "delimiter",
    }
)


def _selected_kinds(values: Sequence[str | AutoFixKind] | None) -> frozenset[AutoFixKind]:
    if not values:
        return frozenset(AutoFixKind)
    try:
        return frozenset(AutoFixKind(str(value)) for value in values)
    except ValueError as exc:
        allowed = ", ".join(kind.value for kind in AutoFixKind)
        raise AutoFixError(f"unsupported auto-fix kind (expected one of {allowed})") from exc


def _read_config_payload(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file is not valid JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    raw.setdefault("artifacts", {})
    if not isinstance(raw["artifacts"], dict):
        raise ConfigError("config field 'artifacts' must be an object")
    return raw


def _write_config_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_payload_changed(payload: dict[str, Any], path: Path) -> bool:
    current = json.loads(path.read_text(encoding="utf-8"))
    return current != payload


def _resolve_lockfile(value: str | Path | None, config_path: Path) -> Path:
    return Path(value).expanduser().resolve() if value is not None else config_path.with_name("promptabi.lock.json")


def _has_load_error(diagnostics: Iterable[Diagnostic]) -> bool:
    return any(diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in diagnostics)


def _diagnostic_fingerprints(
    diagnostics: Iterable[Diagnostic],
    *,
    rule_ids: frozenset[str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        diagnostic.fingerprint
        for diagnostic in diagnostics
        if rule_ids is None or diagnostic.rule_id in rule_ids
    )


def _special_token_changes(config_payload: dict[str, Any], loaded_artifacts, *, config_path: Path, write: bool) -> tuple[AutoFixChange, ...]:
    artifacts = config_payload["artifacts"]
    assert isinstance(artifacts, dict)
    if any(isinstance(value, dict) and value.get("kind") == "special-token-map" for value in artifacts.values()):
        return ()

    tokens = _collect_special_tokens(loaded_artifacts)
    if not tokens:
        return ()

    token_path = config_path.with_name("promptabi.special-tokens.json")
    token_payload = {"tokens": [{"name": name, "text": text, **({"token_id": token_id} if token_id is not None else {})} for name, text, token_id in tokens]}
    if write:
        token_path.write_text(json.dumps(token_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts["promptabi-special-tokens"] = {
            "kind": "special-token-map",
            "path": token_path.name,
            "tokens": token_payload["tokens"],
        }
    return (
        AutoFixChange(
            kind=AutoFixKind.SPECIAL_TOKENS,
            path=str(token_path),
            action="write-special-token-map",
            status=AutoFixStatus.APPLIED if write else AutoFixStatus.PLANNED,
            message=f"declare {len(tokens)} tokenizer/chat-template special tokens without changing rendering",
        ),
    )


def _collect_special_tokens(loaded_artifacts) -> tuple[tuple[str, str, int | None], ...]:
    collected: dict[tuple[str, str], int | None] = {}
    for loaded in loaded_artifacts:
        for key, value in loaded.metadata:
            if key == "special_tokens":
                for index, token in enumerate(_tuple_value(value)):
                    if isinstance(token, str) and token:
                        collected.setdefault((f"special_{index}", token), None)
            if key in {"bos_token", "eos_token"} and isinstance(value, str) and value:
                collected.setdefault((key, value), _metadata_int(loaded.metadata, f"{key}_id"))
    return tuple((name, text, token_id) for (name, text), token_id in sorted(collected.items()))


def _unsupported_fragment_changes(loaded_artifacts, *, config_path: Path, write: bool) -> tuple[AutoFixChange, ...]:
    annotations: list[dict[str, object]] = []
    for loaded in loaded_artifacts:
        metadata = dict(loaded.metadata)
        fragments = []
        for key in ("unsupported_constructs", "symbolic_abstentions"):
            for item in _tuple_value(metadata.get(key, ())):
                if item:
                    fragments.append({"source": key, "value": str(item)})
        supported = metadata.get("supported_fragment")
        if supported is False:
            fragments.append({"source": "supported_fragment", "value": "false"})
        if fragments:
            annotations.append(
                {
                    "artifact": loaded.artifact.name,
                    "kind": loaded.artifact.kind.value,
                    "fragments": fragments,
                }
            )
    if not annotations:
        return ()

    path = config_path.with_name("promptabi.unsupported-fragments.json")
    if write:
        path.write_text(
            json.dumps({"annotations": annotations, "schema": "promptabi.unsupported-fragments/v1"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return (
        AutoFixChange(
            kind=AutoFixKind.UNSUPPORTED_FRAGMENTS,
            path=str(path),
            action="write-unsupported-fragment-annotations",
            status=AutoFixStatus.APPLIED if write else AutoFixStatus.PLANNED,
            message=f"record {sum(len(item['fragments']) for item in annotations)} unsupported-fragment annotations for review",
        ),
    )


def _docs_stub_changes(diagnostics: Sequence[Diagnostic], *, config_path: Path, write: bool) -> tuple[AutoFixChange, ...]:
    actionable = [diagnostic for diagnostic in diagnostics if diagnostic.severity is not DiagnosticSeverity.INFO and diagnostic.suggestions]
    if not actionable:
        return ()
    path = config_path.with_name("promptabi-fix-notes.md")
    lines = [
        "# PromptABI fix notes",
        "",
        "Low-risk auto-fix generated this stub so reviewers can document decisions without changing prompt behavior.",
        "",
    ]
    for diagnostic in actionable:
        lines.extend(
            [
                f"## {diagnostic.rule_id}",
                "",
                f"- Fingerprint: `{diagnostic.fingerprint}`",
                f"- Artifact: `{diagnostic.artifact.kind}:{diagnostic.artifact.name}`" if diagnostic.artifact else "- Artifact: `(none)`",
                f"- Suggested next step: {diagnostic.suggestions[0]}",
                "",
            ]
        )
    if write:
        path.write_text("\n".join(lines), encoding="utf-8")
    return (
        AutoFixChange(
            kind=AutoFixKind.DOCS_STUB,
            path=str(path),
            action="write-docs-stub",
            status=AutoFixStatus.APPLIED if write else AutoFixStatus.PLANNED,
            message=f"create reviewer notes for {len(actionable)} non-info diagnostics",
            diagnostics=tuple(diagnostic.fingerprint for diagnostic in actionable),
        ),
    )


def _tuple_value(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _metadata_int(metadata: tuple[tuple[str, object], ...], key: str) -> int | None:
    value = dict(metadata).get(key)
    return value if isinstance(value, int) and value >= 0 else None


def _guarded_previews_for_diagnostics(
    diagnostics: Sequence[Diagnostic],
    risk: GuardedPreviewRisk,
) -> tuple[GuardedAutoFixPreview, ...]:
    if risk is not GuardedPreviewRisk.HIGH:
        raise AutoFixError(f"unsupported guarded preview risk: {risk.value}")
    ranked = rank_fix_suggestions(diagnostics)
    previews: list[GuardedAutoFixPreview] = []
    for suggestion in ranked:
        related = tuple(
            diagnostic
            for diagnostic in diagnostics
            if suggestion.text in diagnostic.suggestions and _is_high_risk_fix_candidate(diagnostic, suggestion)
        )
        if not related:
            continue
        before_witnesses = tuple(
            diagnostic.witness if diagnostic.witness is not None else _diagnostic_summary_witness(diagnostic)
            for diagnostic in related
        )
        previews.append(
            GuardedAutoFixPreview(
                risk=risk,
                suggestion=suggestion,
                diagnostics=related,
                before_witnesses=before_witnesses,
                after_witnesses=tuple(_after_preview_witness(diagnostic, suggestion) for diagnostic in related),
                guardrails=_guardrails_for_suggestion(suggestion),
            )
        )
    return tuple(previews)


def _is_high_risk_fix_candidate(diagnostic: Diagnostic, suggestion: RankedFixSuggestion) -> bool:
    properties = dict(diagnostic.properties)
    explicit_safety = str(properties.get("fix_safety", "")).lower()
    if explicit_safety == "low":
        return False
    explicit_change = str(
        properties.get(
            "fix_changes_user_visible_prompt_behavior",
            properties.get("fix_user_visible_prompt_behavior", ""),
        )
    ).lower()
    if explicit_change in {"true", "1", "yes"}:
        return True
    if suggestion.changes_user_visible_prompt_behavior:
        return True
    haystack = " ".join(
        (
            diagnostic.rule_id,
            diagnostic.message,
            " ".join(diagnostic.suggestions),
            diagnostic.artifact.kind if diagnostic.artifact is not None else "",
        )
    ).lower()
    return any(term in haystack for term in _HIGH_RISK_TERMS)


def _diagnostic_summary_witness(diagnostic: Diagnostic) -> WitnessTrace:
    artifacts = (diagnostic.artifact,) if diagnostic.artifact is not None else ()
    return WitnessTrace(
        summary=f"Current diagnostic {diagnostic.fingerprint}: {diagnostic.message}",
        steps=(
            WitnessStep(action="run PromptABI verification", output=diagnostic.rule_id),
            WitnessStep(action="observe current failing interface condition", output=diagnostic.message),
        ),
        artifacts=artifacts,
    )


def _after_preview_witness(diagnostic: Diagnostic, suggestion: RankedFixSuggestion) -> WitnessTrace:
    artifacts = (diagnostic.artifact,) if diagnostic.artifact is not None else ()
    return WitnessTrace(
        summary=(
            "Preview only: applying this high-risk fix must remove or intentionally replace "
            f"diagnostic {diagnostic.fingerprint} before merge"
        ),
        steps=(
            WitnessStep(action="review proposed high-risk change", output=suggestion.text),
            WitnessStep(action="apply change in a separate review branch"),
            WitnessStep(action="rerun PromptABI verification and compare witness", output=diagnostic.rule_id),
        ),
        artifacts=artifacts,
        minimal_fixes=(suggestion.text,),
    )


def _guardrails_for_suggestion(suggestion: RankedFixSuggestion) -> tuple[str, ...]:
    guardrails = [
        "does not write files; reviewer must apply the change explicitly",
        "requires before/after verification on the same PromptABI config",
        "requires reviewer approval because prompt-interface behavior may change",
    ]
    if suggestion.changes_user_visible_prompt_behavior:
        guardrails.append("requires product/owner signoff for user-visible prompt behavior changes")
    if suggestion.blast_radius.value == "high":
        guardrails.append("requires compatibility review for affected templates, providers, or truncation policy")
    return tuple(guardrails)
