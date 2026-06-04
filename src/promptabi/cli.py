"""Command line entrypoint for PromptABI."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from ._version import __version__
from .config import ConfigError, discover_config, load_config
from .compatibility_matrix import (
    build_compatibility_matrix,
    render_compatibility_matrix_json,
    render_compatibility_matrix_text,
)
from .diff import diff_config_files
from .explain import ExplainError, explain_diagnostic, render_explanation_json, render_explanation_text
from .first_party_plugins import create_first_party_plugin_registry, render_plugin_capabilities
from .github_action import GitHubActionError, run_github_action
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
from .render import SarifRenderOptions, render_github_annotations, render_json, render_sarif, render_text
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
        help="output format: text, json, sarif, github-annotations, or a plugin renderer (default: text)",
    )
    _add_github_output_arguments(verify)
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
        help="output format: text, json, sarif, github-annotations, or a plugin renderer (default: text)",
    )
    _add_github_output_arguments(diff)
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

    plugins = subparsers.add_parser("plugins", help="inspect PromptABI plugin capabilities")
    plugins.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import an additional PromptABI plugin module before listing capabilities; may be repeated",
    )
    plugins.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    matrix = subparsers.add_parser("matrix", help="show check compatibility and guarantee modes")
    matrix.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import an additional PromptABI plugin before building the matrix; may be repeated",
    )
    matrix.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    github_action = subparsers.add_parser(
        "github-action",
        help="run PromptABI with GitHub Actions caching, SARIF, lockfile, summary, and changed-artifact behavior",
    )
    github_action.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json in the workspace",
    )
    github_action.add_argument(
        "--lockfile",
        help="path to a PromptABI lockfile (default: promptabi.lock.json beside the config)",
    )
    github_action.add_argument(
        "--cache-dir",
        help="directory for reusable PromptABI analysis caches (default: .promptabi/cache in the workspace)",
    )
    github_action.add_argument(
        "--sarif-output",
        default="promptabi.sarif",
        help="path for the SARIF log consumed by GitHub code scanning (default: promptabi.sarif)",
    )
    github_action.add_argument(
        "--summary-output",
        help="path for the markdown job summary (default: GITHUB_STEP_SUMMARY when set)",
    )
    github_action.add_argument(
        "--repo-root",
        help="repository checkout root (default: GITHUB_WORKSPACE or cwd)",
    )
    github_action.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    github_action.add_argument(
        "--require-lockfile",
        action="store_true",
        help="fail if artifacts or diagnostics drift from the PromptABI lockfile",
    )
    github_action.add_argument(
        "--changed-only",
        action="store_true",
        help="skip verification when git diff shows no configured PromptABI input changed",
    )
    github_action.add_argument("--base-ref", help="base git ref or SHA for changed-artifact detection")
    github_action.add_argument("--head-ref", help="head git ref or SHA for changed-artifact detection")
    github_action.add_argument(
        "--annotations",
        action="store_true",
        help="also emit GitHub workflow command annotations to stdout",
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
                write_lockfile(
                    lockfile_path,
                    build_lockfile(config, loaded_artifacts, result.diagnostics, base_dir=lockfile_path.parent),
                )
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
                sarif_options=_sarif_options(args, argv),
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
                sarif_options=_sarif_options(args, argv),
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

    if args.command == "plugins":
        try:
            registry = _load_cli_plugins(args.plugin)
            output = render_plugin_capabilities(registry, output_format=args.format)
        except (PluginError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0

    if args.command == "matrix":
        try:
            registry = _load_cli_plugins(args.plugin)
            matrix = build_compatibility_matrix(plugin_registry=registry)
            if args.format == "json":
                output = render_compatibility_matrix_json(matrix)
            else:
                output = render_compatibility_matrix_text(matrix)
        except (PluginError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0

    if args.command == "github-action":
        try:
            run = run_github_action(
                config_path=args.config,
                lockfile_path=args.lockfile,
                cache_dir=args.cache_dir,
                sarif_output=args.sarif_output,
                summary_output=args.summary_output,
                repo_root=args.repo_root,
                fail_on=args.fail_on,
                require_lockfile=args.require_lockfile,
                changed_only=args.changed_only,
                base_ref=args.base_ref,
                head_ref=args.head_ref,
                annotations=args.annotations,
                argv=argv,
            )
        except (ConfigError, GitHubActionError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write GitHub Action outputs: {exc}", file=sys.stderr)
            return 2
        if run.skipped:
            print("PromptABI GitHub Action: skipped (no configured PromptABI inputs changed)")
        else:
            assert run.result is not None
            errors = sum(1 for diagnostic in run.result.diagnostics if diagnostic.severity.value == "error")
            warnings = sum(1 for diagnostic in run.result.diagnostics if diagnostic.severity.value == "warning")
            print(
                "PromptABI GitHub Action: "
                f"{'PASS' if run.result.ok else 'FAIL'} "
                f"({errors} errors, {warnings} warnings, SARIF: {run.sarif_path})"
            )
        return run.exit_code

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


def _add_github_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sarif-category",
        help="stable GitHub code-scanning SARIF category for this analysis run",
    )
    parser.add_argument(
        "--sarif-checkout-uri-base",
        help="repository checkout root for SARIF uriBaseId locations; defaults to GITHUB_WORKSPACE when set",
    )
    parser.add_argument(
        "--sarif-include-invocation",
        action="store_true",
        help="include deterministic invocation metadata in SARIF output",
    )


def _sarif_options(args, argv: Sequence[str] | None) -> SarifRenderOptions:
    checkout_base = _resolve_checkout_uri_base(args.sarif_checkout_uri_base)
    command_line = None
    if args.sarif_include_invocation:
        words = ["promptabi", *(argv if argv is not None else sys.argv[1:])]
        command_line = " ".join(shlex.quote(str(word)) for word in words)
    return SarifRenderOptions(
        category=args.sarif_category,
        checkout_uri_base=checkout_base,
        include_invocation=args.sarif_include_invocation,
        command_line=command_line,
        working_directory=checkout_base if checkout_base is not None else None,
    )


def _resolve_checkout_uri_base(value: str | None) -> Path | None:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("GITHUB_WORKSPACE")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return None


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
    registry = create_first_party_plugin_registry()
    if values:
        load_plugin_modules(values, registry=registry)
    return registry


def _render_verification_output(
    result,
    output_format: str,
    *,
    plugin_registry: PluginRegistry,
    text_kwargs: dict[str, object],
    sarif_options: SarifRenderOptions | None = None,
) -> str:
    if output_format == "text":
        return render_text(result, **text_kwargs)
    if output_format == "json":
        return render_json(result)
    if output_format == "sarif":
        return render_sarif(result, options=sarif_options)
    if output_format == "github-annotations":
        checkout_base = sarif_options.checkout_uri_base if sarif_options is not None else None
        return render_github_annotations(result, checkout_uri_base=checkout_base)
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
