from __future__ import annotations

from hooks.opencode_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run("hooks.user_prompt_submit"))
