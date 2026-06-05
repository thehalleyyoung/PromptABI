"""Environment and setup inspection for PromptABI."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ._version import __version__
from .config import ConfigError, discover_config, load_config
from .diagnostics import DiagnosticSeverity
from .first_party_plugins import create_first_party_plugin_registry
from .plugins import PluginError, PluginRegistry, load_plugin_modules
from .session import VerificationSession


class DoctorStatus(StrEnum):
    """Status values emitted by ``promptabi doctor``."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One deterministic doctor check result."""

    name: str
    status: DoctorStatus
    summary: str
    details: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("doctor check name must be non-empty")
        if not self.summary:
            raise ValueError("doctor check summary must be non-empty")
        object.__setattr__(self, "details", tuple(sorted(self.details, key=lambda item: item[0])))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.details:
            data["details"] = dict(self.details)
        return data


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Complete PromptABI setup report."""

    promptabi_version: str
    python_version: str
    python_executable: str
    platform: str
    cwd: str
    cache_dir: str
    config_path: str | None
    checks: tuple[DoctorCheck, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not any(check.status is DoctorStatus.ERROR for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "promptabi_version": self.promptabi_version,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "platform": self.platform,
            "cwd": self.cwd,
            "cache_dir": self.cache_dir,
            "config_path": self.config_path,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
        }


def run_doctor(
    *,
    config_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    plugin_specs: tuple[str, ...] = (),
    cwd: str | Path | None = None,
) -> DoctorReport:
    """Inspect the local PromptABI environment, config, artifacts, and backends."""

    working_dir = Path(cwd or Path.cwd()).resolve()
    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    checks: list[DoctorCheck] = [_environment_check()]
    checks.append(_cache_check(resolved_cache_dir))
    registry, plugin_check = _plugin_registry_check(plugin_specs)
    checks.append(plugin_check)
    checks.append(_optional_dependency_check())
    checks.append(_supported_backend_check(registry))

    resolved_config_path: Path | None
    if config_path is not None:
        resolved_config_path = Path(config_path).expanduser().resolve()
    else:
        try:
            resolved_config_path = discover_config(working_dir)
        except ConfigError:
            resolved_config_path = None
    config_checks, report_config_path = _config_checks(resolved_config_path, registry, explicit=config_path is not None)
    checks.extend(config_checks)
    checks.append(_setup_summary_check(checks))

    return DoctorReport(
        promptabi_version=__version__,
        python_version=platform.python_version(),
        python_executable=sys.executable,
        platform=platform.platform(),
        cwd=str(working_dir),
        cache_dir=str(resolved_cache_dir),
        config_path=report_config_path,
        checks=tuple(checks),
    )


def render_doctor_json(report: DoctorReport) -> str:
    """Render a doctor report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_doctor_text(report: DoctorReport) -> str:
    """Render a concise human-readable doctor report."""

    lines = [
        "PromptABI doctor:",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"version: {report.promptabi_version}",
        f"python: {report.python_version} ({report.python_executable})",
        f"platform: {report.platform}",
        f"cwd: {report.cwd}",
        f"cache: {report.cache_dir}",
    ]
    if report.config_path is not None:
        lines.append(f"config: {report.config_path}")
    lines.append("checks:")
    for check in report.checks:
        lines.append(f"  {check.status.value.upper()} {check.name}: {check.summary}")
        for key, value in check.details:
            lines.append(f"    {key}: {_format_detail(value)}")
    return "\n".join(lines) + "\n"


def _environment_check() -> DoctorCheck:
    minimum = (3, 11)
    current = sys.version_info[:2]
    if current < minimum:
        return DoctorCheck(
            "environment",
            DoctorStatus.ERROR,
            "Python is older than PromptABI's supported runtime.",
            (("required", ">=3.11"), ("actual", platform.python_version())),
        )
    return DoctorCheck(
        "environment",
        DoctorStatus.OK,
        "Python and PromptABI version metadata are readable.",
        (("required_python", ">=3.11"), ("promptabi", __version__)),
    )


def _cache_check(cache_dir: Path) -> DoctorCheck:
    probe = cache_dir / ".promptabi-doctor-probe"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return DoctorCheck(
            "cache-health",
            DoctorStatus.ERROR,
            "Cache directory is not writable.",
            (("path", str(cache_dir)), ("error", str(exc))),
        )
    return DoctorCheck(
        "cache-health",
        DoctorStatus.OK,
        "Cache directory exists and accepts writes.",
        (("path", str(cache_dir)),),
    )


def _plugin_registry_check(plugin_specs: tuple[str, ...]) -> tuple[PluginRegistry, DoctorCheck]:
    registry = create_first_party_plugin_registry()
    if not plugin_specs:
        return registry, DoctorCheck(
            "plugins",
            DoctorStatus.OK,
            "First-party plugin registry loaded.",
            _plugin_registry_details(registry),
        )
    try:
        load_plugin_modules(plugin_specs, registry=registry)
    except PluginError as exc:
        return registry, DoctorCheck(
            "plugins",
            DoctorStatus.ERROR,
            "One or more requested plugins could not be loaded.",
            (("requested", list(plugin_specs)), ("error", str(exc)), *_plugin_registry_details(registry)),
        )
    return registry, DoctorCheck(
        "plugins",
        DoctorStatus.OK,
        "First-party and requested plugin registries loaded.",
        (("requested", list(plugin_specs)), *_plugin_registry_details(registry)),
    )


def _optional_dependency_check() -> DoctorCheck:
    dependencies = (
        ("jsonschema", "jsonschema", "JSON Schema replay and parser compatibility"),
        ("sentencepiece", "sentencepiece", "SentencePiece tokenizer adapters"),
        ("tiktoken", "tiktoken", "tiktoken tokenizer adapters"),
        ("tokenizers", "tokenizers", "Hugging Face tokenizers adapters"),
        ("z3-solver", "z3", "Z3-backed finite contract checks"),
    )
    installed: list[str] = []
    missing: list[str] = []
    versions: dict[str, str] = {}
    for distribution, import_name, _purpose in dependencies:
        if importlib.util.find_spec(import_name) is None:
            missing.append(distribution)
            versions[distribution] = "not-installed"
            continue
        installed.append(distribution)
        versions[distribution] = _distribution_version(distribution)

    z3_result = _z3_probe()
    details: list[tuple[str, object]] = [
        ("installed", installed),
        ("missing", missing),
        ("versions", versions),
        ("z3_probe", z3_result),
    ]
    if missing:
        return DoctorCheck(
            "optional-dependencies",
            DoctorStatus.WARNING,
            "Some optional backends are unavailable; related checks will abstain or use narrower fallbacks.",
            tuple(details),
        )
    if not z3_result.startswith("sat:"):
        return DoctorCheck(
            "optional-dependencies",
            DoctorStatus.WARNING,
            "Optional imports are present, but the Z3 smoke query did not complete normally.",
            tuple(details),
        )
    return DoctorCheck(
        "optional-dependencies",
        DoctorStatus.OK,
        "Optional tokenizer, grammar, and solver backends are importable.",
        tuple(details),
    )


def _supported_backend_check(registry: PluginRegistry) -> DoctorCheck:
    capability_counts: dict[str, int] = {}
    for capability in registry.capabilities:
        capability_counts[capability.kind.value] = capability_counts.get(capability.kind.value, 0) + 1
    return DoctorCheck(
        "supported-backends",
        DoctorStatus.OK,
        "Registered capabilities cover artifact loaders, checks, providers, grammars, truncation policies, solvers, and renderers.",
        (
            ("capability_counts", capability_counts),
            ("artifact_loaders", [loader.name for loader in registry.artifact_loaders]),
            ("checks", sorted(registry.checks)),
            ("renderers", sorted(registry.renderers)),
        ),
    )


def _config_checks(
    config_path: Path | None,
    registry: PluginRegistry,
    *,
    explicit: bool,
) -> tuple[tuple[DoctorCheck, ...], str | None]:
    if config_path is None:
        status = DoctorStatus.ERROR if explicit else DoctorStatus.WARNING
        return (
            (
                DoctorCheck(
                    "config-validity",
                    status,
                    "No PromptABI config was found.",
                    (("looked_for", ["promptabi.json", ".promptabi.json"]),),
                ),
                DoctorCheck(
                    "artifact-paths",
                    status,
                    "Artifact paths could not be inspected without a valid config.",
                ),
            ),
            None,
        )
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        return (
            (
                DoctorCheck(
                    "config-validity",
                    DoctorStatus.ERROR,
                    "PromptABI config could not be loaded.",
                    (("path", str(config_path)), ("error", str(exc))),
                ),
                DoctorCheck(
                    "artifact-paths",
                    DoctorStatus.ERROR,
                    "Artifact paths could not be inspected because config loading failed.",
                    (("path", str(config_path)),),
                ),
            ),
            str(config_path),
        )

    session = VerificationSession(config, plugin_registry=registry)
    loaded_artifacts, diagnostics = session.load_artifacts_with_diagnostics()
    severity_counts = _severity_counts(diagnostics)
    fatal_count = severity_counts.get("error", 0)
    warning_count = severity_counts.get("warning", 0)
    config_status = DoctorStatus.ERROR if fatal_count else DoctorStatus.WARNING if warning_count else DoctorStatus.OK
    config_check = DoctorCheck(
        "config-validity",
        config_status,
        "Config loaded and artifact preflight completed."
        if config_status is DoctorStatus.OK
        else "Config loaded, but artifact preflight reported setup issues.",
        (
            ("path", str(config_path)),
            ("name", config.name),
            ("checks", list(config.checks)),
            ("artifact_count", len(config.artifact_bundle.artifacts)),
            ("loaded_artifact_count", len(loaded_artifacts)),
            ("diagnostic_counts", severity_counts),
        ),
    )
    artifact_check = _artifact_path_check(config_path, config, loaded_artifacts, diagnostics)
    return ((config_check, artifact_check), str(config_path))


def _artifact_path_check(config_path: Path, config, loaded_artifacts, diagnostics) -> DoctorCheck:
    del loaded_artifacts
    local_paths: list[str] = []
    missing_paths: list[str] = []
    uri_refs: list[str] = []
    for artifact in config.artifact_bundle:
        if artifact.location.path is not None:
            local_paths.append(artifact.location.path)
            if not Path(artifact.location.path).exists():
                missing_paths.append(artifact.location.path)
        elif artifact.location.uri is not None:
            uri_refs.append(artifact.location.uri)

    load_errors = [diagnostic.to_dict() for diagnostic in diagnostics if diagnostic.severity is DiagnosticSeverity.ERROR]
    load_warnings = [diagnostic.to_dict() for diagnostic in diagnostics if diagnostic.severity is DiagnosticSeverity.WARNING]
    if missing_paths or load_errors:
        return DoctorCheck(
            "artifact-paths",
            DoctorStatus.ERROR,
            "One or more configured artifacts are missing or failed loader preflight.",
            (
                ("config_dir", str(config_path.parent)),
                ("local_paths", local_paths),
                ("missing_paths", missing_paths),
                ("uri_refs", uri_refs),
                ("load_errors", load_errors),
                ("load_warnings", load_warnings),
            ),
        )
    if load_warnings:
        return DoctorCheck(
            "artifact-paths",
            DoctorStatus.WARNING,
            "Artifacts are readable, but loader preflight found warnings.",
            (
                ("config_dir", str(config_path.parent)),
                ("local_paths", local_paths),
                ("uri_refs", uri_refs),
                ("load_warnings", load_warnings),
            ),
        )
    return DoctorCheck(
        "artifact-paths",
        DoctorStatus.OK,
        "All local artifact paths are readable or represented by supported offline references.",
        (("config_dir", str(config_path.parent)), ("local_paths", local_paths), ("uri_refs", uri_refs)),
    )


def _setup_summary_check(checks: list[DoctorCheck]) -> DoctorCheck:
    errors = [check.name for check in checks if check.status is DoctorStatus.ERROR]
    warnings = [check.name for check in checks if check.status is DoctorStatus.WARNING]
    if errors:
        return DoctorCheck(
            "setup-summary",
            DoctorStatus.ERROR,
            "Setup has errors that should be fixed before relying on verification.",
            (("errors", errors), ("warnings", warnings)),
        )
    if warnings:
        return DoctorCheck(
            "setup-summary",
            DoctorStatus.WARNING,
            "Setup is usable, but some optional or project-specific checks need attention.",
            (("warnings", warnings),),
        )
    return DoctorCheck(
        "setup-summary",
        DoctorStatus.OK,
        "Environment, cache, plugins, config, artifact paths, and optional backends are ready.",
    )


def _resolve_cache_dir(value: str | Path | None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("PROMPTABI_CACHE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return (Path(xdg_cache_home).expanduser() / "promptabi").resolve()
    return (Path.home() / ".cache" / "promptabi").resolve()


def _plugin_registry_details(registry: PluginRegistry) -> tuple[tuple[str, object], ...]:
    return (
        ("capabilities", len(registry.capabilities)),
        ("artifact_loaders", len(registry.artifact_loaders)),
        ("checks", len(registry.checks)),
        ("renderers", len(registry.renderers)),
    )


def _severity_counts(diagnostics) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for diagnostic in diagnostics:
        counts[diagnostic.severity.value] = counts.get(diagnostic.severity.value, 0) + 1
    return counts


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "installed-without-distribution-metadata"


def _z3_probe() -> str:
    if importlib.util.find_spec("z3") is None:
        return "missing"
    try:
        import z3  # type: ignore[import-not-found]

        x = z3.Bool("promptabi_doctor")
        solver = z3.Solver()
        solver.add(x)
        return f"sat:{solver.check()}"
    except Exception as exc:
        return f"error:{exc}"


def _format_detail(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return str(value)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        to_dict = getattr(value, "to_dict")
        if callable(to_dict):
            return _jsonable(to_dict())
    return str(value)
