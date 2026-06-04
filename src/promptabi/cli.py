"""Command line entrypoint for PromptABI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ._version import __version__
from .config import ConfigError
from .render import render_json, render_sarif, render_text
from .session import VerificationSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptabi",
        description="Verify tokenizer/template/tool-calling interface contracts for LLM apps.",
    )
    parser.add_argument("--version", action="version", version=f"promptabi {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="run PromptABI checks for a config")
    verify.add_argument("--config", required=True, help="path to a PromptABI JSON config")
    verify.add_argument(
        "--format",
        choices=("text", "json", "sarif"),
        default="text",
        help="output format (default: text)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify":
        try:
            result = VerificationSession.from_config_file(args.config).run()
        except ConfigError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        if args.format == "json":
            output = render_json(result)
        elif args.format == "sarif":
            output = render_sarif(result)
        else:
            output = render_text(result)
        print(output, end="")
        return 0 if result.ok else 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
