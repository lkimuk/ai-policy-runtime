from __future__ import annotations

from hooks.opencode_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run("hooks.stop_refinement"))
