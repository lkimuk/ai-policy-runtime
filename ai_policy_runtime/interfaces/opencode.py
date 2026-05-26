from __future__ import annotations

import argparse
from pathlib import Path

from ai_policy_runtime.adapters.opencode import OpenCodeWrapperOptions, run_opencode_policy_wrapper
from ai_policy_runtime.interfaces.agent_cli import (
    AgentCliSpec,
    optional_path,
    post_refine_mode,
    post_refine_packs,
    run_agent_cli,
)


SPEC = AgentCliSpec(
    prog="policy-opencode",
    description="Resolve Effective Rules, inject AGENTS.md, then invoke OpenCode.",
    command_option="--opencode-command",
    command_default="opencode",
    command_help="OpenCode executable path. Use quotes when the path contains spaces.",
    arg_option="--opencode-arg",
    arg_help="Extra argument passed to `opencode run` before the task. Repeat for multiple args.",
    no_exec_help="Only resolve and inject AGENTS.md; do not invoke OpenCode.",
)


def main() -> None:
    """Policy-aware OpenCode entry point."""

    run_agent_cli(SPEC, _options_from_args, run_opencode_policy_wrapper)


def _options_from_args(args: argparse.Namespace) -> OpenCodeWrapperOptions:
    return OpenCodeWrapperOptions(
        task=args.task,
        root=Path(args.root),
        policy_root=optional_path(args.policy_root),
        skills_dir=args.skills,
        packs_dir=args.packs,
        pack_ids=tuple(args.pack),
        opencode_command=(args.opencode_command,),
        opencode_args=("run", *tuple(args.opencode_arg)),
        execute=not args.no_exec,
        verify_target=args.verify_target,
        post_refine_mode=post_refine_mode(args),
        post_refine_pack_ids=post_refine_packs(args),
    )


if __name__ == "__main__":
    main()
