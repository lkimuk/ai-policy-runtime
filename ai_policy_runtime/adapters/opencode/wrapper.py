from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ai_policy_runtime.adapters.agent import (
    AgentWrapperOptions,
    AgentWrapperResult,
    POST_REFINE_PACK_ID,
    build_agent_command,
    run_policy_agent_wrapper,
)


@dataclass(frozen=True)
class OpenCodeWrapperOptions:
    """Options for the OpenCode policy wrapper."""

    task: str
    root: Path = Path(".")
    policy_root: Path | None = None
    skills_dir: str = "skills"
    packs_dir: str = "packs"
    pack_ids: tuple[str, ...] = ()
    opencode_command: tuple[str, ...] = ("opencode",)
    opencode_args: tuple[str, ...] = ()
    execute: bool = True
    verify_target: str | Path | None = None
    post_refine_mode: str = "off"
    post_refine_pack_ids: tuple[str, ...] = (POST_REFINE_PACK_ID,)

    def to_agent_options(self) -> AgentWrapperOptions:
        """Return generic wrapper options for the OpenCode adapter."""

        return AgentWrapperOptions(
            task=self.task,
            agent="opencode",
            root=self.root,
            policy_root=self.policy_root,
            skills_dir=self.skills_dir,
            packs_dir=self.packs_dir,
            pack_ids=self.pack_ids,
            command=self.opencode_command,
            command_args=self.opencode_args,
            execute=self.execute,
            verify_target=self.verify_target,
            post_refine_mode=self.post_refine_mode,
            post_refine_pack_ids=self.post_refine_pack_ids,
        )


OpenCodeWrapperResult = AgentWrapperResult


def run_opencode_policy_wrapper(options: OpenCodeWrapperOptions) -> OpenCodeWrapperResult:
    """Resolve task policy, inject AGENTS.md, then optionally invoke OpenCode."""

    return run_policy_agent_wrapper(options.to_agent_options())


def _build_opencode_command(
    opencode_command: Sequence[str],
    opencode_args: Sequence[str],
    task: str,
) -> tuple[str, ...]:
    return build_agent_command(opencode_command, opencode_args, task)
