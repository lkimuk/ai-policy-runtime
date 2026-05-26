from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


CODEX_HOOKS_FILE = Path(".codex") / "hooks.json"
CODEX_CONFIG_FILE = Path(".codex") / "config.toml"
CLAUDE_SETTINGS_FILE = Path(".claude") / "settings.local.json"
OPENCODE_CONFIG_FILE = Path("opencode.json")
OPENCODE_PLUGIN_FILE = Path(".opencode") / "plugins" / "ai-policy-runtime.js"
OPENCODE_PLUGIN_STATE_FILE = Path(".policy") / "current" / "opencode-plugin-state.json"
OPENCODE_POST_REFINE_PROMPT_FILE = Path(".policy") / "current" / "opencode-post-refine-prompt.md"
OPENCODE_INSTRUCTION = "AGENTS.md"
POLICY_CONFIG_FILE = Path(".policy") / "config.json"
POLICY_CURRENT_DIR = Path(".policy") / "current"
CLAUDE_MARKETPLACE_NAME = "ai-policy-runtime"
CLAUDE_PLUGIN_ID = "ai-policy-runtime@ai-policy-runtime"
AI_POLICY_PLUGIN_MARKERS = ("ai-policy-runtime", "opencode-user-prompt-submit")


def clean_workspace(root: str | Path, *, remove_current: bool = True) -> dict[str, Any]:
    """Remove AI Policy Runtime workspace integration state.

    The cleanup is intentionally scoped to project-local files and entries owned
    by AI Policy Runtime. Caches and local model assets are left in place.
    """

    project_root = Path(root)
    result: dict[str, Any] = {
        "root": str(project_root),
        "removed": [],
        "updated": [],
        "skipped": [],
    }

    codex_hooks_remain = _clean_codex_hooks(project_root / CODEX_HOOKS_FILE, result)
    _clean_codex_config(
        project_root / CODEX_CONFIG_FILE,
        result,
        disable_hooks=not codex_hooks_remain,
    )
    _clean_claude_settings(project_root / CLAUDE_SETTINGS_FILE, result)
    _clean_opencode_config(project_root / OPENCODE_CONFIG_FILE, result)
    _clean_opencode_plugin(project_root / OPENCODE_PLUGIN_FILE, result)
    _remove_file(project_root / OPENCODE_PLUGIN_STATE_FILE, result)
    _remove_file(project_root / OPENCODE_POST_REFINE_PROMPT_FILE, result)
    _remove_file(project_root / POLICY_CONFIG_FILE, result)
    if remove_current:
        _remove_dir(project_root / POLICY_CURRENT_DIR, result)
    else:
        result["skipped"].append(str(project_root / POLICY_CURRENT_DIR))
    return result


def _clean_codex_hooks(path: Path, result: dict[str, Any]) -> bool:
    if not path.exists():
        result["skipped"].append(str(path))
        return False
    config = _read_json_object(path)
    hooks = config.get("hooks")
    if not isinstance(hooks, dict):
        result["skipped"].append(str(path))
        return False

    changed = False
    for event in ("UserPromptSubmit", "Stop"):
        entries = hooks.get(event)
        if not isinstance(entries, list):
            continue
        kept = [entry for entry in entries if not _is_ai_policy_hook_entry(entry)]
        if len(kept) != len(entries):
            changed = True
            if kept:
                hooks[event] = kept
            else:
                hooks.pop(event, None)

    if changed:
        _write_json(path, config)
        result["updated"].append(str(path))
    else:
        result["skipped"].append(str(path))
    return _has_codex_hook_entries(hooks)


def _clean_codex_config(path: Path, result: dict[str, Any], *, disable_hooks: bool) -> None:
    if not path.exists():
        result["skipped"].append(str(path))
        return
    text = path.read_text(encoding="utf-8")
    updated = _remove_toml_keys(text, "features", ("codex_hooks",))
    if disable_hooks:
        updated = _set_toml_bool(updated, "features", "hooks", False)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        result["updated"].append(str(path))
    else:
        result["skipped"].append(str(path))


def _clean_claude_settings(path: Path, result: dict[str, Any]) -> None:
    if not path.exists():
        result["skipped"].append(str(path))
        return
    settings = _read_json_object(path)
    changed = False
    enabled_plugins = settings.get("enabledPlugins")
    if isinstance(enabled_plugins, dict) and CLAUDE_PLUGIN_ID in enabled_plugins:
        enabled_plugins.pop(CLAUDE_PLUGIN_ID, None)
        changed = True
        if not enabled_plugins:
            settings.pop("enabledPlugins", None)
    marketplaces = settings.get("extraKnownMarketplaces")
    if isinstance(marketplaces, dict) and CLAUDE_MARKETPLACE_NAME in marketplaces:
        marketplaces.pop(CLAUDE_MARKETPLACE_NAME, None)
        changed = True
        if not marketplaces:
            settings.pop("extraKnownMarketplaces", None)
    if changed:
        _write_json(path, settings)
        result["updated"].append(str(path))
    else:
        result["skipped"].append(str(path))


def _clean_opencode_config(path: Path, result: dict[str, Any]) -> None:
    if not path.exists():
        result["skipped"].append(str(path))
        return
    config = _read_json_object(path)
    if not _remove_json_list_item(config, "instructions", OPENCODE_INSTRUCTION):
        result["skipped"].append(str(path))
        return
    _write_json(path, config)
    result["updated"].append(str(path))


def _clean_opencode_plugin(path: Path, result: dict[str, Any]) -> None:
    if not path.exists():
        result["skipped"].append(str(path))
        return
    if not _is_ai_policy_opencode_plugin(path):
        result["skipped"].append(str(path))
        return
    path.unlink()
    result["removed"].append(str(path))


def _remove_file(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        path.unlink()
        result["removed"].append(str(path))
    else:
        result["skipped"].append(str(path))


def _remove_dir(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        shutil.rmtree(path)
        result["removed"].append(str(path))
    else:
        result["skipped"].append(str(path))


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_ai_policy_hook_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    hooks = entry.get("hooks")
    if not isinstance(hooks, list):
        return False
    return any(
        isinstance(item, dict) and _is_ai_policy_hook_command(str(item.get("command", "")))
        for item in hooks
    )


def _is_ai_policy_hook_command(command: str) -> bool:
    normalized = command.replace("\\", "/")
    return (
        "ai-policy-hook.js" in normalized
        or "/hooks/user_prompt_submit.py" in normalized
        or "/hooks/stop_refinement.py" in normalized
    )


def _has_codex_hook_entries(hooks: dict[str, object]) -> bool:
    return any(isinstance(value, list) and bool(value) for value in hooks.values())


def _is_ai_policy_opencode_plugin(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return all(marker in text for marker in AI_POLICY_PLUGIN_MARKERS)


def _remove_json_list_item(config: dict[str, Any], key: str, item: str) -> bool:
    value = config.get(key)
    if not isinstance(value, list):
        return False
    kept = [current for current in value if str(current).strip() != item]
    if len(kept) == len(value):
        return False
    if kept:
        config[key] = kept
    else:
        config.pop(key, None)
    return True


def _remove_toml_keys(text: str, section: str, keys: tuple[str, ...]) -> str:
    lines = text.splitlines()
    section_header = f"[{section}]"
    in_section = False
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == section_header
            output.append(line)
            continue
        if in_section and any(stripped.startswith(f"{key} ") and "=" in stripped for key in keys):
            continue
        output.append(line)
    return "\n".join(output).rstrip() + "\n"


def _set_toml_bool(
    text: str,
    section: str,
    key: str,
    value: bool,
    *,
    remove_keys: tuple[str, ...] = (),
) -> str:
    lines = text.splitlines()
    target = f"{key} = {'true' if value else 'false'}"
    section_header = f"[{section}]"
    in_section = False
    section_found = False
    key_written = False
    output: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_section and not key_written:
                output.append(target)
                key_written = True
            in_section = stripped == section_header
            section_found = section_found or in_section
            output.append(line)
            continue
        if in_section and any(
            stripped.startswith(f"{remove_key} ") and "=" in stripped
            for remove_key in remove_keys
        ):
            continue
        if in_section and stripped.startswith(f"{key} ") and "=" in stripped:
            output.append(target)
            key_written = True
            continue
        output.append(line)

    if not section_found:
        if output and output[-1].strip():
            output.append("")
        output.extend([section_header, target])
    elif in_section and not key_written:
        output.append(target)

    return "\n".join(output).rstrip() + "\n"
