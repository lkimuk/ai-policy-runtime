from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_policy_runtime.application.runtime import NonApplicableTaskError, PolicyRuntime
from ai_policy_runtime.domain.config import RuntimeConfig
from ai_policy_runtime.infrastructure.schema_loader import SchemaLoader
from ai_policy_runtime.services.embedding_health import (
    inspect_embedding_health,
    test_embedding_provider,
)
from ai_policy_runtime.services.local_models import LocalModelManager
from ai_policy_runtime.services.validator import validate_effective_rules_file
from ai_policy_runtime.services.workspace_cleanup import clean_workspace


def main() -> None:
    """Command-line entry point for the policy runtime MVP."""

    parser = argparse.ArgumentParser(prog="policy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve", help="Generate Effective Rules for a task.")
    _add_runtime_args(resolve)
    resolve.add_argument("task", help="Natural-language task description.")
    resolve.add_argument("--pack", action="append", default=[], help="Pack id to expand.")
    resolve.add_argument(
        "--format",
        choices=("json", "prompt"),
        default="prompt",
        help="Output JSON metadata or the agent-facing effective prompt.",
    )

    explain = subparsers.add_parser("explain", help="Explain Task Analysis for a task.")
    _add_runtime_args(explain)
    explain.add_argument("task", help="Natural-language task description.")

    validate = subparsers.add_parser("validate", help="Validate Skill DSL files.")
    _add_runtime_args(validate)

    validate_effective = subparsers.add_parser(
        "validate-effective", help="Validate an effective-rules.yaml file."
    )
    validate_effective.add_argument("path", help="Path to effective-rules.yaml or JSON.")

    schema = subparsers.add_parser("schema", help="Print a bundled JSON Schema.")
    schema.add_argument("name", choices=("skill", "pack", "effective-rules"))

    inspect = subparsers.add_parser("inspect", help="Inspect current runtime state.")
    inspect.add_argument("--root", default=".", help="Project root.")

    cache = subparsers.add_parser("cache", help="Inspect or clear runtime caches.")
    cache.add_argument("action", choices=("list", "clear"))
    cache.add_argument("--root", default=".", help="Project root.")

    cleanup = subparsers.add_parser(
        "cleanup", help="Remove AI Policy Runtime workspace configuration and generated state."
    )
    cleanup.add_argument("--root", default=".", help="Project root.")
    cleanup.add_argument(
        "--keep-current",
        action="store_true",
        help="Keep .policy/current generated state while removing integration config.",
    )

    model = subparsers.add_parser("model", help="Inspect or install local embedding models.")
    model.add_argument("action", choices=("list", "install"))
    model.add_argument(
        "--policy-root",
        default=".",
        help="Policy asset root where models/ should be inspected or installed.",
    )
    model.add_argument(
        "--model",
        default="default",
        help="Known model key, repo id, or directory name. Defaults to the multilingual model.",
    )

    embedding = subparsers.add_parser(
        "embedding", help="Configure project embedding provider settings."
    )
    embedding.add_argument("action", choices=("status", "configure", "test"))
    embedding.add_argument("--root", default=".", help="Project root.")
    embedding.add_argument(
        "--provider",
        choices=("auto", "openai-compatible", "local"),
        default=None,
        help="Embedding provider to save for command-line hooks.",
    )
    embedding.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    embedding.add_argument("--api-key", default=None, help="OpenAI-compatible API key.")
    embedding.add_argument("--model", default=None, help="Remote model or local model path/name.")
    embedding.add_argument("--timeout", type=float, default=None, help="Remote request timeout seconds.")
    embedding.add_argument(
        "--policy-root",
        default=None,
        help=(
            "Policy asset root used when installing the default local model. "
            "Defaults to the configured policyRoot or the project root."
        ),
    )
    embedding.add_argument(
        "--install",
        action="store_true",
        help="With --provider local, download the default local model and configure this project to use it.",
    )

    verify = subparsers.add_parser("verify", help="Verify files against current Effective Rules.")
    verify.add_argument("--root", default=".", help="Project root.")
    verify.add_argument("--target", default=".", help="File or directory to verify.")

    repair_plan = subparsers.add_parser(
        "repair-plan", help="Generate repair instructions from current violations."
    )
    repair_plan.add_argument("--root", default=".", help="Project root.")

    inject = subparsers.add_parser("inject", help="Inject current Effective Prompt into an agent file.")
    inject.add_argument("--root", default=".", help="Project root.")
    inject.add_argument(
        "--target",
        choices=("codex", "claude", "opencode", "custom"),
        default="codex",
        help="Injection target.",
    )

    run = subparsers.add_parser("run", help="Resolve, inject, and optionally verify a task.")
    _add_runtime_args(run)
    run.add_argument("task", help="Natural-language task description.")
    run.add_argument("--pack", action="append", default=[], help="Pack id to expand.")
    run.add_argument(
        "--agent",
        choices=("codex", "claude", "opencode", "custom"),
        default="custom",
        help="Agent injection target.",
    )
    run.add_argument("--verify-target", default=None, help="File or directory to verify after injection.")

    args = parser.parse_args()
    output, exit_code = _dispatch(args)
    if isinstance(output, str):
        print(output)
    else:
        print(json.dumps(output, indent=2))
    if exit_code:
        raise SystemExit(exit_code)


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="Target project root.")
    parser.add_argument(
        "--policy-root",
        default=None,
        help="Policy runtime asset root containing skills/ and packs/. Defaults to --root.",
    )
    parser.add_argument("--skills", default="skills", help="Skill directory relative to policy root.")
    parser.add_argument("--packs", default="packs", help="Pack directory relative to policy root.")
    parser.add_argument(
        "--extra-skills",
        action="append",
        default=[],
        help="Additional skill directory (relative to policy root or absolute). Repeatable.",
    )
    parser.add_argument(
        "--extra-packs",
        action="append",
        default=[],
        help="Additional pack directory (relative to policy root or absolute). Repeatable.",
    )
    parser.add_argument(
        "--on-duplicate",
        choices=("error", "first_wins", "last_wins"),
        default=None,
        help="Behavior when the same skill_id/pack_id is loaded from multiple directories.",
    )


def _runtime_from_args(args: argparse.Namespace) -> PolicyRuntime:
    root = Path(args.root)
    project_config = _read_project_config(root / ".policy" / "config.json")
    return PolicyRuntime(
        RuntimeConfig.from_values(
            root=root,
            policy_root=getattr(args, "policy_root", None)
            or _optional_string(project_config.get("policyRoot")),
            skills_dir=getattr(args, "skills", "skills"),
            packs_dir=getattr(args, "packs", "packs"),
            extra_skills_dirs=_combine_extra_dirs(
                getattr(args, "extra_skills", []),
                project_config.get("extraSkillsDirs"),
            ),
            extra_packs_dirs=_combine_extra_dirs(
                getattr(args, "extra_packs", []),
                project_config.get("extraPacksDirs"),
            ),
            on_duplicate=getattr(args, "on_duplicate", None)
            or _optional_string(project_config.get("onDuplicate"))
            or "error",
            embedding_provider=_project_embedding_provider(project_config),
            embedding_base_url=_optional_string(project_config.get("embeddingBaseUrl")),
            embedding_api_key=_optional_string(project_config.get("embeddingApiKey")),
            embedding_model=_project_embedding_model(root, project_config),
            embedding_timeout_seconds=_optional_float(project_config.get("embeddingTimeout")),
        )
    )


def _combine_extra_dirs(
    cli_values: list[str] | None, config_value: object
) -> tuple[str, ...]:
    """Merge CLI --extra-* (wins) with config arrays. Preserve order, dedupe."""

    seen: dict[str, None] = {}
    for value in cli_values or ():
        text = str(value).strip()
        if text:
            seen.setdefault(text, None)
    if isinstance(config_value, list):
        for value in config_value:
            text = str(value).strip()
            if text:
                seen.setdefault(text, None)
    elif isinstance(config_value, str):
        text = config_value.strip()
        if text:
            seen.setdefault(text, None)
    return tuple(seen)


def _read_project_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Project config must be a JSON object: {path}")
    return data


def _project_embedding_model(root: Path, config: dict[str, Any]) -> str | None:
    model = _optional_string(config.get("embeddingModel"))
    if not model or _project_embedding_provider(config) != "local":
        return model
    path = Path(model)
    if path.is_absolute():
        return model
    return str(root / path)


def _project_embedding_provider(config: dict[str, Any]) -> str | None:
    provider = _optional_string(config.get("embeddingProvider"))
    if provider is None:
        return None
    return provider.lower().replace("_", "-")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _configure_embedding(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    provider = "" if args.provider == "auto" else args.provider
    config["embeddingProvider"] = provider
    installed_model = None
    for key in (
        "embeddingBaseUrl",
        "embeddingApiKey",
        "embeddingModel",
        "embeddingTimeout",
    ):
        config.pop(key, None)
    if provider == "openai-compatible":
        if args.base_url:
            config["embeddingBaseUrl"] = args.base_url
        if args.api_key:
            config["embeddingApiKey"] = args.api_key
        if args.model:
            config["embeddingModel"] = args.model
        if args.timeout is not None:
            config["embeddingTimeout"] = str(args.timeout)
    elif provider == "local":
        if args.install:
            installed_model = LocalModelManager(_embedding_policy_root(config, args)).install(
                args.model or "default"
            )
            config["embeddingModel"] = installed_model["path"]
        elif args.model:
            config["embeddingModel"] = args.model
    return installed_model


def _embedding_policy_root(config: dict[str, Any], args: argparse.Namespace) -> Path:
    configured = getattr(args, "policy_root", None) or config.get("policyRoot")
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else Path(args.root) / path
    return Path(args.root)


def _dispatch(args: argparse.Namespace) -> tuple[dict[str, Any] | str, int]:
    return CommandDispatcher().dispatch(args)


class CommandDispatcher:
    """Dispatch parsed CLI arguments to application services."""

    def dispatch(self, args: argparse.Namespace) -> tuple[dict[str, Any] | str, int]:
        handlers = {
            "resolve": self._resolve,
            "explain": self._explain,
            "validate": self._validate,
            "validate-effective": self._validate_effective,
            "schema": self._schema,
            "inspect": self._inspect,
            "cache": self._cache,
            "cleanup": self._cleanup,
            "model": self._model,
            "embedding": self._embedding,
            "verify": self._verify,
            "repair-plan": self._repair_plan,
            "inject": self._inject,
            "run": self._run,
        }
        try:
            return handlers[args.command](args)
        except NonApplicableTaskError as exc:
            return exc.to_dict(), 0
        except KeyError as exc:
            raise ValueError(f"Unsupported command: {args.command}") from exc

    def _resolve(self, args: argparse.Namespace) -> tuple[dict[str, Any] | str, int]:
        result = _runtime_from_args(args).resolve(args.task, tuple(args.pack))
        if args.format == "prompt":
            prompt = result.current / "effective-prompt.md"
            return prompt.read_text(encoding="utf-8").rstrip(), 0
        return result.to_dict(), 0

    def _explain(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        return _runtime_from_args(args).explain(args.task).to_dict(), 0

    def _validate(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        diagnostics = _runtime_from_args(args).validate()
        return _diagnostics_output(diagnostics)

    def _validate_effective(
        self, args: argparse.Namespace
    ) -> tuple[dict[str, Any], int]:
        return _diagnostics_output(validate_effective_rules_file(args.path))

    def _schema(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        return SchemaLoader().load(args.name), 0

    def _inspect(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        current = Path(args.root) / ".policy" / "current"
        trace_path = current / "trace.json"
        if not trace_path.exists():
            return {"current": str(current), "exists": False}, 0
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        return {
            "current": str(current),
            "exists": True,
            "task": trace.get("task"),
            "active_skills": trace.get("active_skills", []),
            "task_analysis": trace.get("task_analysis", {}),
            "conflict_count": trace.get("conflict_count", 0),
        }, 0

    def _cache(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        cache_dir = Path(args.root) / ".policy" / "cache" / "semantic-index"
        if args.action == "list":
            files = [
                {"name": item.name, "size": item.stat().st_size}
                for item in sorted(cache_dir.glob("*.json"))
            ] if cache_dir.exists() else []
            return {"cache": str(cache_dir), "entries": files}, 0
        if cache_dir.exists():
            for item in cache_dir.glob("*.json"):
                item.unlink()
        return {"cache": str(cache_dir), "cleared": True}, 0

    def _cleanup(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        return {
            "cleanup": clean_workspace(
                args.root,
                remove_current=not args.keep_current,
            )
        }, 0

    def _model(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        manager = LocalModelManager(args.policy_root)
        if args.action == "list":
            return {"models": list(manager.list())}, 0
        return {"model": manager.install(args.model)}, 0

    def _embedding(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        path = Path(args.root) / ".policy" / "config.json"
        config = _read_project_config(path)
        if args.action == "status":
            status = inspect_embedding_health(
                root=args.root,
                policy_root=getattr(args, "policy_root", None),
                config=config,
                include_env=True,
                check_loadable=True,
            )
            return {"config": str(path), "embedding": status}, 0
        if args.action == "test":
            status = test_embedding_provider(
                root=args.root,
                policy_root=getattr(args, "policy_root", None),
                config=config,
                include_env=True,
            )
            return {"config": str(path), "embedding": status}, int(not status["probe_ok"])
        if args.provider is None:
            raise ValueError("--provider is required for embedding configure")
        if args.install and args.provider != "local":
            raise ValueError("--install can only be used with --provider local")
        installed_model = _configure_embedding(config, args)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        output = {
            "config": str(path),
            "embedding": inspect_embedding_health(
                root=args.root,
                policy_root=getattr(args, "policy_root", None),
                config=config,
                include_env=True,
                check_loadable=False,
            ),
        }
        if installed_model is not None:
            output["installed_model"] = installed_model
        return output, 0

    def _verify(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        violations = _runtime_from_args(args).verify(Path(args.target))
        return {"violations": [item.to_dict() for item in violations]}, int(bool(violations))

    def _repair_plan(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        instructions = _runtime_from_args(args).repair_plan()
        return {"repair_plan": [item.to_dict() for item in instructions]}, 0

    def _inject(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        return {"injected": str(_runtime_from_args(args).inject(args.target)), "target": args.target}, 0

    def _run(self, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
        result = _runtime_from_args(args).run(
            args.task,
            pack_ids=tuple(args.pack),
            agent=args.agent,
            verify_target=args.verify_target,
        )
        return result.to_dict(), int(bool(result.violations))


def _diagnostics_output(diagnostics: list[Any]) -> tuple[dict[str, Any], int]:
    return {"diagnostics": [item.to_dict() for item in diagnostics]}, int(bool(diagnostics))


if __name__ == "__main__":
    main()
