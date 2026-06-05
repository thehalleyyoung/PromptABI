"""Generate a PromptABI config from a dynamic prompt-pack agent spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from promptabi.agent_frameworks import (
    AgentFrameworkIntegrationError,
    load_agent_prompt_pack_assembly,
    render_agent_prompt_pack_plan,
    write_agent_promptabi_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="agent prompt-pack integration JSON")
    parser.add_argument("--write-config", help="write the generated PromptABI config to this path")
    parser.add_argument("--preview", action="store_true", help="print a deterministic prompt preview")
    args = parser.parse_args(argv)

    try:
        assembly = load_agent_prompt_pack_assembly(args.spec)
        if args.write_config:
            write_agent_promptabi_config(assembly, args.write_config)
        print(render_agent_prompt_pack_plan(assembly), end="")
        if args.preview:
            preview = assembly.render_prompt_preview({})
            print("\nPrompt preview:\n" + preview, end="")
    except (AgentFrameworkIntegrationError, OSError) as exc:
        print(f"dynamic_support_agent: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
