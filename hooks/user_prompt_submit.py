from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))
CONFIG_PATH = Path(".policy") / "config.json"
HOOK_STATE_PATH = Path(".policy") / "current" / "agent-hook-state.json"
FALSE_VALUES = {"0", "false", "no", "off"}
POST_REFINE_PACK_ID = "generic.production_refinement"
POST_REFINE_MODES = {"off", "light", "standard", "strict"}
DEFAULT_AGENT = "codex"
SUPPORTED_AGENTS = {"codex", "claude", "opencode"}


def _normalize_embedding_provider(value: str | None) -> str:
    return (value or "").strip().lower().replace("_", "-")


def main() -> int:
    payload = _read_payload()
    raw_prompt = str(payload.get("prompt", ""))
    prompt = _task_prompt(raw_prompt)
    if not prompt.strip():
        return 0

    project_root = Path(payload.get("cwd") or ".").resolve()
    agent = _current_agent()
    config = ProjectHookConfig.load(project_root)
    if not config.enabled_for(agent):
        return 0

    config.apply_environment()
    config.ensure_semantic_dependencies(project_root)
    _clear_agent_injection(project_root, agent)
    _write_turn_state(project_root, payload, prompt, config)

    try:
        additional_context = _resolve_effective_prompt(
            prompt,
            project_root,
            config.policy_root(project_root),
            config.packs,
        )
        _write_turn_state(
            project_root,
            payload,
            prompt,
            config,
            effective_rules_generated=bool(additional_context),
            effective_prompt_path=(
                project_root / ".policy" / "current" / "effective-prompt.md"
                if additional_context
                else None
            ),
            additional_context_chars=len(additional_context),
        )
    except Exception as exc:
        hook_error = f"{type(exc).__name__}: {exc}"
        _write_turn_state(
            project_root,
            payload,
            prompt,
            config,
            hook_error=hook_error,
        )
        additional_context = (
            "AI Policy Runtime hook could not generate Effective Rules for this turn. "
            f"Error: {hook_error}"
        )

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": additional_context,
                }
            },
            ensure_ascii=False,
        )
    )
    return 0


def _read_payload() -> dict[str, object]:
    raw = sys.stdin.buffer.read().decode("utf-8", "replace")
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        return {}
    return data


def _task_prompt(prompt: str) -> str:
    """Return the user's actual request from agent UI prompt wrappers."""

    text = prompt.strip()
    markers = (
        "## My request for Codex:",
        "## My request for Claude:",
        "## My request:",
    )
    for marker in markers:
        if marker not in text:
            continue
        request = text.rsplit(marker, 1)[-1].strip()
        return request or text
    return text


@dataclass(frozen=True)
class ProjectHookConfig:
    """Project-local agent hook configuration with environment overrides."""

    enabled: bool = True
    agents: tuple[str, ...] = (DEFAULT_AGENT,)
    packs: tuple[str, ...] = ()
    policy_root_value: str | Path = PLUGIN_ROOT
    auto_install: bool | None = None
    embedding_provider: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    embedding_timeout: str | None = None
    post_refine_mode: str = "off"
    post_refine_pack_ids: tuple[str, ...] = (POST_REFINE_PACK_ID,)
    verify_target: str | None = None

    @classmethod
    def load(cls, project_root: Path) -> "ProjectHookConfig":
        return cls.from_mapping(_load_project_config(project_root))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ProjectHookConfig":
        return cls(
            enabled=_coerce_enabled(data.get("enabled", True)),
            agents=_configured_agents(data),
            packs=_configured_packs(data),
            policy_root_value=os.environ.get("AI_POLICY_ROOT")
            or data.get("policyRoot")
            or PLUGIN_ROOT,
            auto_install=_optional_bool(data.get("autoInstall")),
            embedding_provider=_optional_string(data.get("embeddingProvider")),
            embedding_base_url=_optional_string(data.get("embeddingBaseUrl")),
            embedding_api_key=_optional_string(data.get("embeddingApiKey")),
            embedding_model=_optional_string(data.get("embeddingModel")),
            embedding_timeout=_optional_string(data.get("embeddingTimeout")),
            post_refine_mode=_configured_post_refine_mode(data),
            post_refine_pack_ids=_configured_post_refine_packs(data),
            verify_target=_optional_string(os.environ.get("AI_POLICY_VERIFY_TARGET"))
            or _optional_string(data.get("verifyTarget")),
        )

    def policy_root(self, project_root: Path) -> Path:
        path = Path(str(self.policy_root_value))
        if not path.is_absolute():
            path = project_root / path
        return path.resolve()

    def enabled_for(self, agent: str) -> bool:
        return self.enabled and agent in self.agents

    def apply_environment(self) -> None:
        self._apply_env("AI_POLICY_EMBEDDING_PROVIDER", self.embedding_provider)
        if _normalize_embedding_provider(self.embedding_provider) == "local":
            self._clear_env("AI_POLICY_EMBEDDING_BASE_URL")
            self._clear_env("AI_POLICY_EMBEDDING_API_KEY")
            self._clear_env("AI_POLICY_EMBEDDING_TIMEOUT")
            self._apply_or_clear_env("AI_POLICY_EMBEDDING_MODEL", self.embedding_model)
        else:
            self._apply_env("AI_POLICY_EMBEDDING_BASE_URL", self.embedding_base_url)
            self._apply_env("AI_POLICY_EMBEDDING_API_KEY", self.embedding_api_key)
            self._apply_env("AI_POLICY_EMBEDDING_MODEL", self.embedding_model)
            self._apply_env("AI_POLICY_EMBEDDING_TIMEOUT", self.embedding_timeout)
        if self.auto_install is not None and "AI_POLICY_AUTO_INSTALL" not in os.environ:
            os.environ["AI_POLICY_AUTO_INSTALL"] = "1" if self.auto_install else "0"

    def ensure_semantic_dependencies(self, project_root: Path) -> None:
        provider = _normalize_embedding_provider(
            self.embedding_provider or os.environ.get("AI_POLICY_EMBEDDING_PROVIDER")
        )
        if provider != "local" and not self._auto_uses_installed_local_model(project_root):
            return
        try:
            import sentence_transformers  # noqa: F401
        except ModuleNotFoundError:
            _bootstrap_package(semantic=True)

    def _auto_uses_installed_local_model(self, project_root: Path) -> bool:
        provider = _normalize_embedding_provider(
            self.embedding_provider or os.environ.get("AI_POLICY_EMBEDDING_PROVIDER")
        )
        if provider not in {"", "auto"}:
            return False
        if (
            self.embedding_base_url
            or self.embedding_api_key
            or os.environ.get("AI_POLICY_EMBEDDING_BASE_URL")
            or os.environ.get("AI_POLICY_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ):
            return False
        model = self.embedding_model or os.environ.get("AI_POLICY_EMBEDDING_MODEL")
        if model:
            path = Path(model)
            return path.is_absolute() and path.exists() or (project_root / path).exists()
        return (
            self.policy_root(project_root)
            / "models"
            / "paraphrase-multilingual-MiniLM-L12-v2"
        ).exists()

    @staticmethod
    def _apply_env(name: str, value: str | None) -> None:
        if value:
            os.environ[name] = value

    @staticmethod
    def _apply_or_clear_env(name: str, value: str | None) -> None:
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)

    @staticmethod
    def _clear_env(name: str) -> None:
        os.environ.pop(name, None)


def _resolve_effective_prompt(
    prompt: str,
    project_root: Path,
    policy_root: Path,
    packs: tuple[str, ...],
) -> str:
    _prepare_imports()

    from ai_policy_runtime import PolicyRuntime, RuntimeConfig

    runtime = PolicyRuntime(
        RuntimeConfig.from_values(
            root=project_root,
            policy_root=policy_root,
        )
    )
    result = runtime.resolve_if_applicable(prompt, packs)
    if not result.applicable or result.resolve_result is None:
        return ""
    return (result.resolve_result.current / "effective-prompt.md").read_text(encoding="utf-8")


def _clear_agent_injection(project_root: Path, agent: str) -> None:
    _prepare_imports()

    from ai_policy_runtime.services.injector import clear_injected_prompt

    clear_injected_prompt(project_root, agent)


def _prepare_imports() -> None:
    if str(PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(PLUGIN_ROOT))

    try:
        import ai_policy_runtime  # noqa: F401
        import jsonschema  # noqa: F401
        import yaml  # noqa: F401
    except ModuleNotFoundError:
        _bootstrap_package()
        import ai_policy_runtime  # noqa: F401
        import jsonschema  # noqa: F401
        import yaml  # noqa: F401


def _bootstrap_package(*, semantic: bool = False) -> None:
    if os.environ.get("AI_POLICY_AUTO_INSTALL", "1") in {"0", "false", "False"}:
        raise RuntimeError(
            "Python dependencies are missing and AI_POLICY_AUTO_INSTALL is disabled."
        )
    if not (PLUGIN_ROOT / "pyproject.toml").exists():
        raise RuntimeError(f"pyproject.toml not found under plugin root: {PLUGIN_ROOT}")

    package = f"{PLUGIN_ROOT}[semantic]" if semantic else str(PLUGIN_ROOT)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-e",
        package,
    ]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _configured_packs(config: dict[str, Any]) -> tuple[str, ...]:
    if "AI_POLICY_PACKS" in os.environ:
        return _split_csv(os.environ.get("AI_POLICY_PACKS", ""))

    packs = config.get("packs", ())
    if isinstance(packs, str):
        return _split_csv(packs)
    if isinstance(packs, list):
        return tuple(str(item).strip() for item in packs if str(item).strip())
    return ()


def _configured_agents(config: dict[str, Any]) -> tuple[str, ...]:
    configured = config.get("agents")
    if configured is None:
        return (DEFAULT_AGENT,)
    if isinstance(configured, str):
        agents = _split_csv(configured)
    elif isinstance(configured, list):
        agents = tuple(str(item).strip() for item in configured if str(item).strip())
    else:
        agents = ()
    filtered = tuple(dict.fromkeys(agent for agent in agents if agent in SUPPORTED_AGENTS))
    return filtered if filtered else (DEFAULT_AGENT,)


def _load_project_config(project_root: Path) -> dict[str, Any]:
    path = project_root / CONFIG_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Policy config must be a JSON object: {path}")
    return data


def _write_turn_state(
    project_root: Path,
    payload: dict[str, object],
    prompt: str,
    config: ProjectHookConfig,
    *,
    effective_rules_generated: bool = False,
    effective_prompt_path: Path | None = None,
    additional_context_chars: int = 0,
    hook_error: str | None = None,
) -> None:
    state_path = project_root / HOOK_STATE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "turn_id": payload.get("turn_id"),
        "session_id": payload.get("session_id"),
        "agent": _current_agent(),
        "prompt": prompt,
        "post_refine_mode": config.post_refine_mode,
        "post_refine_pack_ids": list(config.post_refine_pack_ids),
        "verify_target": config.verify_target,
        "effective_rules_generated": effective_rules_generated,
        "effective_prompt_path": (
            str(effective_prompt_path.resolve()) if effective_prompt_path else None
        ),
        "additional_context_chars": additional_context_chars,
        "hook_error": hook_error,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _coerce_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in FALSE_VALUES
    return bool(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in FALSE_VALUES
    return bool(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _configured_post_refine_mode(config: dict[str, Any]) -> str:
    value = os.environ.get("AI_POLICY_POST_REFINE", config.get("postRefine", "off"))
    if isinstance(value, bool):
        mode = "standard" if value else "off"
    else:
        mode = str(value).strip().lower() or "off"
    if mode not in POST_REFINE_MODES:
        allowed = ", ".join(sorted(POST_REFINE_MODES))
        raise ValueError(f"postRefine must be one of: {allowed}")
    return mode


def _configured_post_refine_packs(config: dict[str, Any]) -> tuple[str, ...]:
    if "AI_POLICY_POST_REFINE_PACKS" in os.environ:
        packs = _split_csv(os.environ.get("AI_POLICY_POST_REFINE_PACKS", ""))
    else:
        configured = config.get("postRefinePacks", ())
        if isinstance(configured, str):
            packs = _split_csv(configured)
        elif isinstance(configured, list):
            packs = tuple(str(item).strip() for item in configured if str(item).strip())
        else:
            packs = ()
    return packs if packs else (POST_REFINE_PACK_ID,)


def _enabled(config: dict[str, Any]) -> bool:
    """Compatibility helper for focused unit tests."""

    return ProjectHookConfig.from_mapping(config).enabled


def _enabled_for(config: dict[str, Any], agent: str) -> bool:
    """Compatibility helper for focused unit tests."""

    return ProjectHookConfig.from_mapping(config).enabled_for(agent)


def _current_agent() -> str:
    agent = os.environ.get("AI_POLICY_AGENT", DEFAULT_AGENT).strip().lower()
    return agent if agent in SUPPORTED_AGENTS else DEFAULT_AGENT


if __name__ == "__main__":
    raise SystemExit(main())
