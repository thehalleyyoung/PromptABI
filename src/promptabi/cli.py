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
from .render import render_json, render_sarif, render_text
from .seed_corpus import SeedCorpusError, build_seed_corpus_manifest, write_seed_corpus_manifest
from .session import VerificationSession
from .structured_schema_corpus import (
    StructuredSchemaCorpusError,
    build_structured_schema_corpus_manifest,
    write_structured_schema_corpus_manifest,
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
        choices=("text", "json", "sarif"),
        default="text",
        help="output format (default: text)",
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
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            result = VerificationSession(config).run()
        except ConfigError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot prepare cache directory: {exc}", file=sys.stderr)
            return 2
        if args.format == "json":
            output = render_json(result)
        elif args.format == "sarif":
            output = render_sarif(result)
        else:
            output = render_text(
                result,
                verbosity=args.verbose - args.quiet,
                config_path=config_path,
                cache_dir=cache_dir,
            )
        print(output, end="")
        return _exit_code(result, fail_on=args.fail_on)

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
