"""Command line entrypoint for PromptABI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from ._version import __version__
from .config import ConfigError, discover_config, load_config
from .diff import diff_config_files
from .explain import ExplainError, explain_diagnostic, render_explanation_json, render_explanation_text
from .init import InitError, available_stacks, scaffold_promptabi_project
from .lockfiles import (
    LockfileError,
    build_lockfile,
    compare_lockfile,
    load_lockfile,
    lockfile_error_diagnostic,
    write_lockfile,
)
from .plugins import PluginError, PluginRegistry, load_plugin_modules
from .render import render_json, render_sarif, render_text
from .seed_corpus import SeedCorpusError, build_seed_corpus_manifest, write_seed_corpus_manifest
from .session import VerificationSession
from .structured_schema_corpus import (
    StructuredSchemaCorpusError,
    build_structured_schema_corpus_manifest,
    write_structured_schema_corpus_manifest,
)
from .provider_fixture_packs import (
    ProviderFixturePackError,
    build_provider_fixture_pack_manifest,
    write_provider_fixture_pack_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptabi",
        description="Verify tokenizer/template/tool-calling interface contracts for LLM apps.",
    )
    parser.add_argument("--version", action="version", version=f"promptabi {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="run PromptABI checks for a config")
    verify.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    verify.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    verify.add_argument(
        "--cache-dir",
        help="directory for reusable PromptABI analysis caches (default: PROMPTABI_CACHE_DIR or user cache)",
    )
    verify.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    verify.add_argument("-q", "--quiet", action="count", default=0, help="suppress informational text output")
    verify.add_argument("-v", "--verbose", action="count", default=0, help="include additional workflow metadata")
    verify.add_argument(
        "--format",
        default="text",
        help="output format: text, json, sarif, or a plugin renderer (default: text)",
    )
    verify.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import a PromptABI plugin module for this run; may be repeated",
    )
    verify.add_argument(
        "--lockfile",
        help="path to a PromptABI lockfile (default: promptabi.lock.json beside the config)",
    )
    verify.add_argument(
        "--write-lockfile",
        action="store_true",
        help="write a lockfile pinning artifact hashes, revisions, tool versions, and diagnostic baselines",
    )
    verify.add_argument(
        "--require-lockfile",
        action="store_true",
        help="fail verification if the current artifacts or diagnostic baseline drift from the lockfile",
    )

    explain = subparsers.add_parser("explain", help="expand one diagnostic into a tutorial-style explanation")
    explain.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    explain.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    explain.add_argument(
        "--cache-dir",
        help="directory for reusable PromptABI analysis caches (default: PROMPTABI_CACHE_DIR or user cache)",
    )
    explain_selector = explain.add_mutually_exclusive_group()
    explain_selector.add_argument("--fingerprint", help="explain the diagnostic with this stable fingerprint")
    explain_selector.add_argument("--rule-id", help="explain the only diagnostic with this rule id")
    explain_selector.add_argument("--index", type=int, help="explain the one-based diagnostic index after de-duplication")
    explain.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    diff = subparsers.add_parser("diff", help="compare two PromptABI configs for contract-breaking changes")
    diff.add_argument("baseline", help="baseline PromptABI JSON config")
    diff.add_argument("current", help="current PromptABI JSON config")
    diff.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    diff.add_argument("-q", "--quiet", action="count", default=0, help="suppress informational text output")
    diff.add_argument("-v", "--verbose", action="count", default=0, help="include compared config paths")
    diff.add_argument(
        "--format",
        default="text",
        help="output format: text, json, sarif, or a plugin renderer (default: text)",
    )
    diff.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import a PromptABI plugin module for renderer extensions; may be repeated",
    )

    init = subparsers.add_parser("init", help="scaffold a PromptABI config for a common LLM stack")
    init.add_argument(
        "--stack",
        choices=available_stacks(),
        default="openai-tools",
        help="application stack to scaffold (default: openai-tools)",
    )
    init.add_argument(
        "--output-dir",
        default=".",
        help="directory to write promptabi.json and local fixture stubs into (default: cwd)",
    )
    init.add_argument(
        "--name",
        help="verification config name (default: <stack>-promptabi)",
    )
    init.add_argument(
        "--config",
        default="promptabi.json",
        help="config file name to write inside output-dir (default: promptabi.json)",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing scaffold files",
    )

    corpus = subparsers.add_parser("corpus", help="seed corpus maintenance commands")
    corpus_subparsers = corpus.add_subparsers(dest="corpus_command", required=True)
    manifest = corpus_subparsers.add_parser("manifest", help="validate corpus and emit its manifest")
    manifest.add_argument(
        "--root",
        help="seed corpus root (default: repository fixtures/seed_corpus)",
    )
    manifest.add_argument(
        "--output",
        help="write manifest JSON to this path instead of stdout",
    )
    schema_manifest = corpus_subparsers.add_parser(
        "structured-schema-manifest",
        help="validate structured-output/tool schema corpus and emit its manifest",
    )
    schema_manifest.add_argument(
        "--root",
        help="structured schema corpus root (default: repository fixtures/structured_schemas)",
    )
    schema_manifest.add_argument(
        "--output",
        help="write manifest JSON to this path instead of stdout",
    )
    provider_fixture_manifest = corpus_subparsers.add_parser(
        "provider-fixture-manifest",
        help="validate recorded provider fixture packs and emit their manifest",
    )
    provider_fixture_manifest.add_argument(
        "--root",
        help="provider fixture pack root (default: repository fixtures/provider_fixture_packs)",
    )
    provider_fixture_manifest.add_argument(
        "--output",
        help="write manifest JSON to this path instead of stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify":
        try:
            if args.quiet and args.verbose:
                parser.error("--quiet and --verbose cannot be used together")
            config_path = Path(args.config).resolve() if args.config else discover_config()
            cache_dir = _resolve_cache_dir(args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            plugin_registry = _load_cli_plugins(args.plugin)
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            session = VerificationSession(config, plugin_registry=plugin_registry)
            result = session.run()
            lockfile_path = _resolve_lockfile_path(args.lockfile, config_path)
            if args.write_lockfile:
                loaded_artifacts, load_diagnostics = session.load_artifacts_with_diagnostics()
                load_failed = any(diagnostic.severity.value == "error" for diagnostic in load_diagnostics)
                if load_failed:
                    print("promptabi: cannot write lockfile while artifact loading has errors", file=sys.stderr)
                    return 2
                write_lockfile(lockfile_path, build_lockfile(config, loaded_artifacts, result.diagnostics))
            if args.require_lockfile:
                loaded_artifacts, _load_diagnostics = session.load_artifacts_with_diagnostics()
                try:
                    lockfile = load_lockfile(lockfile_path)
                    lock_diagnostics = compare_lockfile(
                        lockfile,
                        config,
                        loaded_artifacts,
                        result.diagnostics,
                        lockfile_path=lockfile_path,
                    )
                except LockfileError as exc:
                    lock_diagnostics = (lockfile_error_diagnostic(exc, lockfile_path=lockfile_path),)
                if lock_diagnostics:
                    result = type(result)(
                        config=result.config,
                        diagnostics=tuple(sorted((*result.diagnostics, *lock_diagnostics), key=lambda item: item.sort_key)),
                    )
        except ConfigError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot prepare cache directory: {exc}", file=sys.stderr)
            return 2
        try:
            output = _render_verification_output(
                result,
                args.format,
                plugin_registry=plugin_registry,
                text_kwargs={
                    "verbosity": args.verbose - args.quiet,
                    "config_path": config_path,
                    "cache_dir": cache_dir,
                },
            )
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return _exit_code(result, fail_on=args.fail_on)

    if args.command == "explain":
        try:
            config_path = Path(args.config).resolve() if args.config else discover_config()
            cache_dir = _resolve_cache_dir(args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            result = VerificationSession(config).run()
            explanation = explain_diagnostic(
                result,
                fingerprint=args.fingerprint,
                rule_id=args.rule_id,
                index=args.index,
                base_dir=config_path.parent,
            )
        except (ConfigError, ExplainError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot prepare cache directory: {exc}", file=sys.stderr)
            return 2
        if args.format == "json":
            output = render_explanation_json(explanation)
        else:
            output = render_explanation_text(explanation)
        print(output, end="")
        return 0

    if args.command == "diff":
        try:
            if args.quiet and args.verbose:
                parser.error("--quiet and --verbose cannot be used together")
            baseline_path = Path(args.baseline).resolve()
            current_path = Path(args.current).resolve()
            plugin_registry = _load_cli_plugins(args.plugin)
            result = diff_config_files(baseline_path, current_path)
        except ConfigError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        try:
            output = _render_verification_output(
                result,
                args.format,
                plugin_registry=plugin_registry,
                text_kwargs={
                    "verbosity": args.verbose - args.quiet,
                    "config_path": current_path if args.verbose else None,
                    "heading": "PromptABI diff",
                },
            )
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        if args.format == "text" and args.verbose:
            output = output.replace(f"config: {current_path}\n", f"baseline: {baseline_path}\ncurrent: {current_path}\n")
        print(output, end="")
        return _exit_code(result, fail_on=args.fail_on)

    if args.command == "init":
        try:
            written = scaffold_promptabi_project(
                stack=args.stack,
                output_dir=args.output_dir,
                name=args.name,
                config_filename=args.config,
                force=args.force,
            )
        except InitError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(f"wrote PromptABI {args.stack} scaffold:")
        for path in written:
            print(f"  {path}")
        print(f"next: promptabi verify --config {written[0]}")
        return 0

    if args.command == "corpus" and args.corpus_command == "manifest":
        try:
            if args.output:
                manifest = write_seed_corpus_manifest(args.output, root=args.root)
                print(f"wrote seed corpus manifest: {args.output} ({manifest['entry_count']} entries)")
            else:
                manifest = build_seed_corpus_manifest(args.root)
                print(json.dumps(manifest, indent=2, sort_keys=True))
        except SeedCorpusError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write corpus manifest: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "corpus" and args.corpus_command == "structured-schema-manifest":
        try:
            if args.output:
                manifest = write_structured_schema_corpus_manifest(args.output, root=args.root)
                print(f"wrote structured schema corpus manifest: {args.output} ({manifest['entry_count']} entries)")
            else:
                manifest = build_structured_schema_corpus_manifest(args.root)
                print(json.dumps(manifest, indent=2, sort_keys=True))
        except StructuredSchemaCorpusError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write corpus manifest: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "corpus" and args.corpus_command == "provider-fixture-manifest":
        try:
            if args.output:
                manifest = write_provider_fixture_pack_manifest(args.output, root=args.root)
                print(f"wrote provider fixture pack manifest: {args.output} ({manifest['entry_count']} entries)")
            else:
                manifest = build_provider_fixture_pack_manifest(args.root)
                print(json.dumps(manifest, indent=2, sort_keys=True))
        except ProviderFixturePackError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write corpus manifest: {exc}", file=sys.stderr)
            return 2
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _parse_artifact_overrides(values: Sequence[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        name, separator, location = value.partition("=")
        if not separator or not name or not location:
            parser.error("--artifact values must use NAME=PATH_OR_URI")
        overrides[name] = location
    return overrides


def _resolve_cache_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("PROMPTABI_CACHE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return (Path(xdg_cache_home).expanduser() / "promptabi").resolve()
    return (Path.home() / ".cache" / "promptabi").resolve()


def _resolve_lockfile_path(value: str | None, config_path: Path) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return config_path.with_name("promptabi.lock.json")


def _load_cli_plugins(values: Sequence[str]) -> PluginRegistry:
    registry = PluginRegistry()
    if values:
        load_plugin_modules(values, registry=registry)
    return registry


def _render_verification_output(
    result,
    output_format: str,
    *,
    plugin_registry: PluginRegistry,
    text_kwargs: dict[str, object],
) -> str:
    if output_format == "text":
        return render_text(result, **text_kwargs)
    if output_format == "json":
        return render_json(result)
    if output_format == "sarif":
        return render_sarif(result)
    return plugin_registry.render(output_format, result)


def _exit_code(result, *, fail_on: str) -> int:
    if fail_on == "never":
        return 0
    severities = {diagnostic.severity.value for diagnostic in result.diagnostics}
    if fail_on == "any":
        return 1 if severities else 0
    if fail_on == "warning":
        return 1 if severities.intersection({"error", "warning"}) else 0
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
