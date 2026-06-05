"""Localization catalog helpers for PromptABI diagnostics."""

from __future__ import annotations

import json
import string
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .diagnostics import Diagnostic, LOCALIZATION_ARG_PATTERN

DEFAULT_LOCALE = "en"


class LocalizationError(ValueError):
    """Raised when a diagnostic message catalog is inconsistent."""


@dataclass(frozen=True, slots=True)
class DiagnosticCatalogEntry:
    """One translation-ready diagnostic message template."""

    message_id: str
    locale: str
    default_message: str
    rule_ids: tuple[str, ...]
    severities: tuple[str, ...]
    placeholders: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "locale": self.locale,
            "default_message": self.default_message,
            "rule_ids": list(self.rule_ids),
            "severities": list(self.severities),
            "placeholders": list(self.placeholders),
        }


def render_localized_message(template: str, arguments: Mapping[str, object] | None = None) -> str:
    """Render a diagnostic template with strict placeholder validation."""

    arguments = arguments or {}
    placeholders = _template_placeholders(template)
    missing = sorted(placeholders.difference(arguments))
    if missing:
        raise LocalizationError(f"missing localization argument(s): {', '.join(missing)}")
    extra = sorted(set(arguments).difference(placeholders))
    if extra:
        raise LocalizationError(f"unused localization argument(s): {', '.join(extra)}")
    return template.format_map({key: str(arguments[key]) for key in placeholders})


def build_diagnostic_catalog(
    diagnostics: Iterable[Diagnostic],
    *,
    locale: str = DEFAULT_LOCALE,
) -> tuple[DiagnosticCatalogEntry, ...]:
    """Build a deterministic translation catalog from real diagnostics."""

    grouped: OrderedDict[str, list[Diagnostic]] = OrderedDict()
    for diagnostic in sorted(diagnostics, key=lambda item: (item.localization_key, item.rule_id, item.message)):
        grouped.setdefault(diagnostic.localization_key, []).append(diagnostic)

    entries: list[DiagnosticCatalogEntry] = []
    for message_id, items in grouped.items():
        defaults = {item.message for item in items}
        default_message = items[0].message
        if len(defaults) > 1 and any(item.message_id is not None for item in items):
            raise LocalizationError(f"message_id {message_id!r} maps to multiple English messages")
        declared_args = {key for item in items for key, _value in item.message_args}
        template_args = _template_placeholders(default_message)
        if declared_args and declared_args != template_args:
            missing = sorted(template_args.difference(declared_args))
            extra = sorted(declared_args.difference(template_args))
            detail = []
            if missing:
                detail.append(f"missing declared args: {', '.join(missing)}")
            if extra:
                detail.append(f"unused declared args: {', '.join(extra)}")
            raise LocalizationError(f"message_id {message_id!r} has inconsistent placeholders ({'; '.join(detail)})")
        entries.append(
            DiagnosticCatalogEntry(
                message_id=message_id,
                locale=locale,
                default_message=default_message,
                rule_ids=tuple(sorted({item.rule_id for item in items})),
                severities=tuple(sorted({item.severity.value for item in items})),
                placeholders=tuple(sorted(template_args or declared_args)),
            )
        )
    return tuple(entries)


def render_diagnostic_catalog_json(entries: Iterable[DiagnosticCatalogEntry]) -> str:
    """Render a stable machine-readable message catalog."""

    entries = tuple(entries)
    locales = sorted({entry.locale for entry in entries})
    payload = {
        "locale": locales[0] if len(locales) == 1 else None,
        "entries": [entry.to_dict() for entry in entries],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_diagnostic_catalog_text(entries: Iterable[DiagnosticCatalogEntry]) -> str:
    """Render a compact human-readable message catalog."""

    lines = ["PromptABI diagnostic message catalog"]
    for entry in entries:
        lines.append(f"- {entry.message_id} [{entry.locale}]")
        lines.append(f"  rules: {', '.join(entry.rule_ids)}")
        lines.append(f"  severities: {', '.join(entry.severities)}")
        if entry.placeholders:
            lines.append(f"  placeholders: {', '.join(entry.placeholders)}")
        lines.append(f"  default: {entry.default_message}")
    return "\n".join(lines) + "\n"


def _template_placeholders(template: str) -> set[str]:
    placeholders: set[str] = set()
    formatter = string.Formatter()
    for _literal, field_name, _format_spec, _conversion in formatter.parse(template):
        if field_name is None:
            continue
        if not LOCALIZATION_ARG_PATTERN.fullmatch(field_name):
            raise LocalizationError(f"unsupported localization placeholder: {field_name!r}")
        placeholders.add(field_name)
    return placeholders
