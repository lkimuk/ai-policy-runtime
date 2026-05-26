from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap
from typing import Any

from tools.configure_common import (
    append_unique,
    configure_agent_policy,
    git_commit_style,
    read_json_object,
    remove_item,
    same_path,
    string_list,
    write_json,
)


AGENT = "opencode"
CONFIG_SCHEMA = "https://opencode.ai/config.json"
OPENCODE_CONFIG_FILE = Path("opencode.json")
OPENCODE_PLUGIN_FILE = Path(".opencode") / "plugins" / "ai-policy-runtime.js"
OPENCODE_PLUGIN_STATE_FILE = Path(".policy") / "current" / "opencode-plugin-state.json"
OPENCODE_POST_REFINE_PROMPT_FILE = Path(".policy") / "current" / "opencode-post-refine-prompt.md"
OPENCODE_INSTRUCTION = "AGENTS.md"
PLUGIN_TEMPLATE = Path("hooks") / "opencode-plugin.js"
PLUGIN_ROOT_PLACEHOLDER = "__AI_POLICY_RUNTIME_ROOT__"
PLUGIN_OWNERSHIP_MARKERS = ("ai-policy-runtime", "opencode-user-prompt-submit")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Configure OpenCode for AI Policy Runtime.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Common commands:
              Show status:
                python tools/configure_opencode.py --root C:\\work\\project --status

              Enable OpenCode for a workspace:
                python tools/configure_opencode.py --root C:\\work\\project --plugin-root D:\\MilesLi\\ai-policy-runtime

              Disable OpenCode for a workspace:
                python tools/configure_opencode.py --root C:\\work\\project --disable
            """
        ),
    )
    parser.add_argument("--root", default=".", help="Project root to configure.")
    parser.add_argument(
        "--plugin-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="AI Policy Runtime checkout or installed package root.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current policy and OpenCode status without modifying files.",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable the OpenCode agent in this workspace policy config.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    plugin_root = Path(args.plugin_root).resolve()

    if args.status:
        print(json.dumps(status(root, plugin_root), ensure_ascii=False, indent=2))
        return 0

    _validate_plugin_root(plugin_root)
    root.mkdir(parents=True, exist_ok=True)
    policy_path = configure_policy(root, plugin_root, enabled=not args.disable)
    opencode_config_path = configure_opencode_config(root, enabled=not args.disable)
    plugin_path = configure_opencode_plugin(root, plugin_root, enabled=not args.disable)
    print(f"Updated policy config: {policy_path}")
    print(f"Updated OpenCode config: {opencode_config_path}")
    print(f"Updated OpenCode plugin: {plugin_path}")
    print(f"{'Enabled' if not args.disable else 'Disabled'} OpenCode agent: {AGENT}")
    return 0


def configure_policy(root: Path, plugin_root: Path, *, enabled: bool = True) -> Path:
    path = root / ".policy" / "config.json"
    config = read_json_object(path)
    configure_agent_policy(config, AGENT, plugin_root, enabled=enabled)
    write_json(path, config)
    return path


def configure_opencode_config(root: Path, *, enabled: bool = True) -> Path:
    """Write project-local OpenCode config without overwriting user settings."""

    path = root / OPENCODE_CONFIG_FILE
    config = read_json_object(path)
    if enabled:
        config.setdefault("$schema", CONFIG_SCHEMA)
        config["instructions"] = append_unique(config.get("instructions"), OPENCODE_INSTRUCTION)
    else:
        _remove_instruction(config, OPENCODE_INSTRUCTION)
    write_json(path, config)
    return path


def configure_opencode_plugin(root: Path, plugin_root: Path, *, enabled: bool = True) -> Path:
    """Install or remove the project-local OpenCode plugin."""

    path = root / OPENCODE_PLUGIN_FILE
    if not enabled:
        if path.exists() and _is_ai_policy_opencode_plugin(path):
            path.unlink()
        return path

    content = _render_plugin_template(plugin_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def status(root: Path, plugin_root: Path) -> dict[str, Any]:
    """Return current project policy and OpenCode config status."""

    policy_path = root / ".policy" / "config.json"
    opencode_config_path = root / OPENCODE_CONFIG_FILE
    opencode_plugin_path = root / OPENCODE_PLUGIN_FILE
    opencode_state_path = root / OPENCODE_PLUGIN_STATE_FILE
    opencode_post_refine_path = root / OPENCODE_POST_REFINE_PROMPT_FILE
    policy = read_json_object(policy_path)
    opencode_config = read_json_object(opencode_config_path)
    policy_root = policy.get("policyRoot")
    instructions = string_list(opencode_config.get("instructions"))
    plugin_root_value = _plugin_runtime_root(opencode_plugin_path)
    return {
        "policy_config": str(policy_path),
        "opencode_config": str(opencode_config_path),
        "opencode_plugin": str(opencode_plugin_path),
        "opencode_plugin_state": str(opencode_state_path),
        "opencode_post_refine_prompt": str(opencode_post_refine_path),
        "runtime_enabled": bool(policy.get("enabled", False)),
        "opencode_agent_enabled": AGENT in string_list(policy.get("agents")),
        "packs": string_list(policy.get("packs")),
        "policy_root": policy_root,
        "policy_root_matches_expected": same_path(policy_root, plugin_root),
        "git_commit_style": git_commit_style(policy),
        "opencode_config_present": opencode_config_path.exists(),
        "instructions": instructions,
        "agents_instruction_configured": OPENCODE_INSTRUCTION in instructions,
        "project_plugin_present": opencode_plugin_path.exists(),
        "project_plugin_configured": _is_ai_policy_opencode_plugin(opencode_plugin_path),
        "project_plugin_state_present": opencode_state_path.exists(),
        "project_post_refine_prompt_present": opencode_post_refine_path.exists(),
        "project_plugin_runtime_root": plugin_root_value,
        "project_plugin_runtime_root_matches_expected": same_path(plugin_root_value, plugin_root),
        "expected_plugin_root": str(plugin_root),
    }


def _validate_plugin_root(plugin_root: Path) -> None:
    required = (
        plugin_root / "ai_policy_runtime" / "__init__.py",
        plugin_root / "bin" / "ai-policy-hook.js",
        plugin_root / PLUGIN_TEMPLATE,
        plugin_root / "packs",
        plugin_root / "skills",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"AI Policy Runtime files are missing:\n{formatted}")


def _is_ai_policy_opencode_plugin(path: Path) -> bool:
    text = _read_text_or_none(path)
    return text is not None and all(marker in text for marker in PLUGIN_OWNERSHIP_MARKERS)


def _plugin_runtime_root(path: Path) -> str | None:
    text = _read_text_or_none(path)
    if text is None:
        return None
    marker = 'const PACKAGE_ROOT = "'
    start = text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = text.find('";', start)
    if end < 0:
        return None
    try:
        return bytes(text[start:end], "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return text[start:end]


def _js_string(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _read_text_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _remove_instruction(config: dict[str, Any], instruction: str) -> None:
    instructions = remove_item(config.get("instructions"), instruction)
    if instructions:
        config["instructions"] = instructions
    else:
        config.pop("instructions", None)


def _render_plugin_template(plugin_root: Path) -> str:
    template = (plugin_root / PLUGIN_TEMPLATE).read_text(encoding="utf-8")
    return template.replace(PLUGIN_ROOT_PLACEHOLDER, _js_string(plugin_root))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
