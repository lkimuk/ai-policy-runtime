from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_unique(value: Any, item: str) -> list[str]:
    items = string_list(value)
    if item not in items:
        items.append(item)
    return items


def remove_item(value: Any, item: str) -> list[str]:
    return [current for current in string_list(value) if current != item]


def configure_agent_policy(
    config: dict[str, Any],
    agent: str,
    plugin_root: Path,
    *,
    enabled: bool = True,
) -> None:
    """Update shared workspace policy settings for an agent integration."""

    if enabled:
        config["enabled"] = True
        config["agents"] = append_unique(config.get("agents"), agent)
        config.setdefault("packs", [])
        config["policyRoot"] = str(plugin_root)
        ensure_git_commit_style(config)
        return

    agents = remove_item(config.get("agents"), agent)
    config["agents"] = agents
    if not agents:
        config["enabled"] = False


def string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def ensure_git_commit_style(config: dict[str, Any]) -> None:
    git_config = config.setdefault("git", {})
    if isinstance(git_config, dict):
        git_config.setdefault("commitStyle", "auto")


def git_commit_style(config: dict[str, Any]) -> str:
    git_config = config.get("git")
    if isinstance(git_config, dict):
        return str(git_config.get("commitStyle") or "auto")
    return str(config.get("gitCommitStyle") or "auto")


def same_path(left: Any, right: Path) -> bool:
    if not left:
        return False
    try:
        left_path = Path(str(left)).resolve()
    except OSError:
        return False
    if sys.platform == "win32":
        return str(left_path).lower() == str(right.resolve()).lower()
    return left_path == right.resolve()
