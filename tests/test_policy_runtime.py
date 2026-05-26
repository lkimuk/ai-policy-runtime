from __future__ import annotations

import os
import json
import argparse
import io
import math
import re
import shutil
import subprocess
import sys
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from ai_policy_runtime import PolicyEngine, Skill, SkillRegistry, TaskContext
from ai_policy_runtime.application.runtime import NonApplicableTaskError, PolicyRuntime
from ai_policy_runtime.domain.config import EmbeddingConfig, RuntimeConfig
from ai_policy_runtime.domain.pack import PackRegistry, SkillPack
from ai_policy_runtime.domain.rule import RuleAction
from ai_policy_runtime.task_analysis import TaskAnalyzer, TaskSignals
from ai_policy_runtime.task_analysis.analyzer import default_embedding_provider
from ai_policy_runtime.task_analysis.schema import TaskAnalysis
from ai_policy_runtime.task_analysis.embeddings import (
    OpenAICompatibleEmbeddingConfig,
    OpenAICompatibleEmbeddingProvider,
)
from ai_policy_runtime.task_analysis.lexicon import LexiconRule, TaskLexicon
from ai_policy_runtime.task_analysis.semantic_index import SemanticTaskIndex
from ai_policy_runtime.adapters.agent import build_post_refinement_task, merge_pack_ids
from ai_policy_runtime.adapters.codex.wrapper import _build_codex_command
from ai_policy_runtime.adapters.claude.wrapper import _build_claude_command
from ai_policy_runtime.adapters.opencode.wrapper import _build_opencode_command
from ai_policy_runtime.interfaces.cli import CommandDispatcher, _runtime_from_args
from ai_policy_runtime.services.project_context import (
    ProjectContextAnalyzer,
    merge_project_analysis,
)
import ai_policy_runtime.services.analyzer as analyzer_service
from ai_policy_runtime.services.analyzer import analyze
from ai_policy_runtime.services.effective_rules import EffectiveRulesRenderer
from ai_policy_runtime.services.engine import PolicyConflictError
from ai_policy_runtime.services.injector import (
    BEGIN,
    END,
    clear_injected_prompt,
    inject_current_prompt,
)
from ai_policy_runtime.services.local_models import LocalModelManager
from ai_policy_runtime.services.validator import validate_effective_rules_mapping
from ai_policy_runtime.services.workspace_cleanup import clean_workspace


NON_ENGLISH_DSL_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
from ai_policy_runtime.services.verification import FileVerifier, Violation, verify_rules
from hooks import stop_refinement, user_prompt_submit
from tools.configure_claude_desktop import (
    PLUGIN_ID,
    configure_claude_settings,
    DEFAULT_POST_REFINE_PACK,
    main as configure_claude_desktop_main,
    configure_policy,
    status as claude_desktop_status,
)
from tools.configure_codex import (
    configure_codex_config,
    configure_codex_hooks,
    configure_policy as configure_codex_policy,
    main as configure_codex_main,
    status as codex_status,
)
from tools.configure_opencode import (
    configure_opencode_config,
    configure_opencode_plugin,
    configure_policy as configure_opencode_policy,
    main as configure_opencode_main,
    status as opencode_status,
)


def _npm_test_env(**overrides: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.lower().startswith("npm_")
    }
    env.update(overrides)
    return env


def _node_package_root() -> Path:
    return (Path("bin") / "ai-policy.js").resolve().parents[1]


def _load_fixture(name: str) -> dict[str, object]:
    import yaml  # type: ignore

    path = Path("tests") / "fixtures" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_fixture(fixture: dict[str, object]) -> dict[str, object]:
    runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
    result = runtime.resolve(
        str(fixture["task"]),
        tuple(str(item) for item in fixture.get("packs", ())),
    )
    return result.structured["effective_rules"]


def _statements(effective: dict[str, object]) -> set[str]:
    statements: set[str] = set()
    for group in ("hard", "soft", "preference"):
        for rule in effective.get(group, ()):
            statements.add(str(rule.get("statement", "")))
    return statements


def _sources(effective: dict[str, object]) -> set[str]:
    sources: set[str] = set()
    for group in ("hard", "soft", "preference"):
        for rule in effective.get(group, ()):
            source = rule.get("source", {})
            if isinstance(source, dict):
                sources.add(str(source.get("skill", "")))
    for item in effective.get("exceptions", ()):
        source = item.get("source", {})
        if isinstance(source, dict):
            sources.add(str(source.get("skill", "")))
    return sources


def _has_statement_containing(effective: dict[str, object], text: str) -> bool:
    return any(text in statement for statement in _statements(effective))


def _section_bullet_count(prompt: str, title: str) -> int:
    marker = f"## {title}"
    start = prompt.find(marker)
    if start < 0:
        return 0
    next_section = prompt.find("\n## ", start + len(marker))
    section = prompt[start:] if next_section < 0 else prompt[start:next_section]
    return sum(1 for line in section.splitlines() if line.startswith("- "))


def _has_policy_content_for_test(effective) -> bool:
    return any(
        (
            effective.hard,
            effective.soft,
            effective.preferences,
            effective.exceptions,
        )
    )


def _git_prepare_commit_analysis() -> TaskAnalysis:
    return TaskAnalysis(
        task=TaskContext(
            domain="git",
            task_type="prepare_commit",
            capabilities=("git_workflow",),
            tags=("git", "commit"),
            context={"language": "git"},
        ),
        confidence=0.9,
        evidence=(),
        needs_review=False,
        activation_ready=True,
    )


class FakeEmbeddingProvider:
    """Test-only deterministic embedding provider for semantic-index tests."""

    model_name = "fake-embedding-provider-v3"

    _CONCEPTS = (
        ("write", ("写", "create", "generate", "implementation", "build")),
        ("latency", ("尾延迟", "latency", "延迟", "hot path", "critical path")),
        ("allocation", ("分配", "allocation", "blocking", "阻塞", "unbounded")),
        ("queue", ("队列", "queue", "data channel", "buffer", "producer consumer")),
        (
            "production",
            (
                "生产可用",
                "production",
                "production-ready",
                "production-quality",
                "polish",
                "能跑",
                "produccion",
                "本番品質",
            ),
        ),
        (
            "behavior",
            (
                "不改变行为",
                "不要改变行为",
                "preserving behavior",
                "without changing behavior",
                "sin cambiar el comportamiento",
                "changer le comportement",
                "振る舞いを変えず",
            ),
        ),
        (
            "complexity",
            (
                "意外复杂度",
                "complexity",
                "accidental complexity",
                "complejidad accidental",
                "複雑さ",
                "有点乱",
                "整理清楚",
                "maintainable",
            ),
        ),
        (
            "duplication",
            (
                "重复逻辑",
                "duplication",
                "duplicated logic",
                "repeated logic",
                "logique dupliquee",
            ),
        ),
        (
            "api",
            (
                "接口摩擦",
                "api",
                "接口调用步骤",
                "调用方负担",
                "合理默认值",
                "api ergonomics",
                "caller friction",
                "friction de l'api",
            ),
        ),
        (
            "read_only_string",
            (
                "read-only string",
                "string parameter",
                "string-like input",
                "std::string_view",
                "string_view",
            ),
        ),
        (
            "contiguous_range",
            (
                "contiguous range",
                "non-owning contiguous range",
                "non-owning range",
                "range of orders",
                "std::span",
                "span",
                "连续范围",
            ),
        ),
        (
            "ownership",
            (
                "ownership",
                "owned resource",
            ),
        ),
        (
            "takes_ownership",
            (
                "takes ownership",
                "resource ownership",
            ),
        ),
        (
            "resource",
            (
                "resource",
            ),
        ),
        (
            "grouping",
            (
                "scattered helpers",
                "scattered helpers and state",
                "cohesive component",
                "整理相关",
            ),
        ),
        (
            "hierarchy",
            (
                "调用链",
                "循环依赖",
                "层次",
                "call chain",
                "circular dependencies",
            ),
        ),
        (
            "variation",
            (
                "small variations",
                "shared control flow",
                "same control flow",
                "variation point",
            ),
        ),
        (
            "expression",
            (
                "样板代码",
                "语言原生",
                "清晰直接",
                "language-native",
                "boilerplate",
            ),
        ),
        (
            "git_commit",
            (
                "commit",
                "commits",
                "conventional commit",
                "atomic commit",
                "logical commit",
                "staged diff",
                "拆分提交",
                "changelog",
            ),
        ),
        (
            "git_commit_message",
            (
                "commit message",
                "commit title",
                "commit subject",
                "commit body",
                "提交信息",
                "提交说明",
                "提交消息",
            ),
        ),
        (
            "git_commit_code",
            (
                "提交一次代码",
                "提交代码",
                "commit this code",
                "commit code changes",
                "save completed code changes",
            ),
        ),
        (
            "git_history",
            (
                "rebase",
                "rewrite history",
                "interactive rebase",
                "squash",
                "amend",
                "force push",
                "force-with-lease",
                "reset",
                "revert",
                "改写历史",
                "变基",
                "压缩提交",
                "撤销提交",
            ),
        ),
        (
            "git_review",
            (
                "merge",
                "branch",
                "pull request",
                "pr",
                "merge conflict",
                "conflict marker",
                "merge request",
                "review branch",
                "check the diff",
                "checking the diff",
                "分支",
                "合并",
                "冲突",
                "冲突标记",
                "代码评审",
            ),
        ),
        (
            "git_branch_name",
            (
                "short branch names",
                "branch names",
                "review branch",
                "git branch",
                "pull request branch",
            ),
        ),
        (
            "git_stash_clean",
            (
                "stash",
                "git stash",
                "clean",
                "git clean",
                "untracked files",
                "ignored files",
                "dry run",
                "保存未完成修改",
                "清理未跟踪文件",
            ),
        ),
        (
            "non_code_change",
            (
                "no code changes",
                "do not change source code",
                "do not change code",
                "without changing code",
                "explain code only",
                "summarize logs without changing code",
                "rewrite documentation copy without code changes",
                "不需要改代码",
                "不要改代码",
                "不需要修改",
                "不要修改",
                "只解释",
                "先不要修改代码",
                "输出效果是正确的吗",
                "is this output correct",
            ),
        ),
        (
            "cmake_target",
            (
                "target based",
                "target-based",
                "modern cmake",
                "target_link_libraries",
                "target_include_directories",
                "target_sources",
                "usage requirements",
                "public private interface",
                "目标",
            ),
        ),
        (
            "cmake_compiler",
            (
                "compile features",
                "cxx standard",
                "compiler flags",
                "cmake_cxx_flags",
                "warnings as errors",
                "generator expressions",
                "multi config",
                "编译选项",
            ),
        ),
        (
            "cmake_sources_modules",
            (
                "file glob",
                "file_set",
                "source list",
                "generated files",
                "project option",
                "cache variable",
                "cmake module",
                "cmake function",
                "custom command",
                "源码列表",
            ),
        ),
        (
            "cmake_dependency",
            (
                "find_package",
                "fetchcontent",
                "externalproject",
                "imported target",
                "third party",
                "dependency",
                "vcpkg",
                "conan",
                "依赖",
            ),
        ),
        (
            "cmake_distribution",
            (
                "install target",
                "cmake install",
                "export targets",
                "package config",
                "cpack",
                "build interface",
                "install interface",
                "安装",
                "打包",
            ),
        ),
        (
            "cmake_repro",
            (
                "cmake preset",
                "presets",
                "cmakepresets",
                "cmakepresets.json",
                "configure preset",
                "build preset",
                "workflow preset",
                "toolchain file",
                "cross compile",
                "sysroot",
                "预设",
                "工具链",
            ),
        ),
        (
            "cmake_quality",
            (
                "add_test",
                "enable_testing",
                "discover tests",
                "sanitizer",
                "code coverage",
                "static analysis",
                "clang-tidy",
                "cppcheck",
                "测试",
            ),
        ),
        (
            "python_api",
            (
                "public api",
                "library api",
                "caller ergonomics",
                "parameters",
                "return values",
                "interface",
            ),
        ),
        (
            "python_classes",
            (
                "class hierarchy",
                "dataclass",
                "dataclasses",
                "protocol",
                "protocols",
                "inheritance",
                "composition",
            ),
        ),
        (
            "python_cli",
            (
                "cli",
                "command line",
                "argparse",
                "argv",
                "entry point",
                "console",
                "script",
            ),
        ),
        (
            "python_functions",
            (
                "function",
                "arguments",
                "defaults",
                "return values",
                "mutable default",
                "decorator",
            ),
        ),
        (
            "python_control_flow",
            (
                "loop",
                "loops",
                "comprehension",
                "generator",
                "iterator",
                "truthiness",
                "slicing",
                "dictionary",
            ),
        ),
        (
            "python_resource",
            (
                "resource",
                "cleanup",
                "context manager",
                "file handle",
                "exception",
                "broad except",
                "transaction",
            ),
        ),
        (
            "python_style",
            (
                "pep8",
                "pep 8",
                "imports",
                "import order",
                "docstring",
                "naming",
                "整理 imports",
            ),
        ),
        (
            "python_typing",
            (
                "type hints",
                "type hint",
                "typing",
                "mypy",
                "pyright",
                "typed dict",
            ),
        ),
        (
            "python_testing",
            (
                "pytest",
                "unittest",
                "python tests",
                "fixture",
                "mock",
                "测试",
            ),
        ),
        (
            "python_security",
            (
                "security",
                "subprocess",
                "shell injection",
                "secret",
                "secrets",
                "path traversal",
                "validates paths",
                "redacts",
            ),
        ),
        (
            "python_packaging",
            (
                "pyproject.toml",
                "python packaging",
                "requirements.txt",
                "wheel",
                "package manager",
            ),
        ),
        (
            "python_concurrency",
            (
                "asyncio",
                "bounded concurrency",
                "thread",
                "multiprocessing",
                "cancellation",
            ),
        ),
        (
            "python_performance",
            (
                "performance",
                "optimize",
                "benchmark",
                "benchmarks",
                "profiling",
                "cprofile",
            ),
        ),
    )

    def encode(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if any(_concept_token_matches(lowered, token) for token in tokens) else 0.0
                    for _, tokens in self._CONCEPTS
                ]
            )
        return vectors


def _concept_token_matches(text: str, token: str) -> bool:
    normalized = token.lower()
    if re.fullmatch(r"[a-z0-9_]+(?: [a-z0-9_]+)*", normalized):
        return re.search(rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])", text) is not None
    return normalized in text


class CountingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]:
        self.calls += 1
        return super().encode(texts)


class TargetedScoreEmbeddingProvider:
    """Embedding provider that gives one semantic entry a controlled score."""

    def __init__(self, *, query_marker: str, target_marker: str, score: float) -> None:
        self.query_marker = query_marker
        self.target_marker = target_marker
        self.score = score

    def encode(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]:
        vectors: list[list[float]] = []
        tail = math.sqrt(max(1.0 - (self.score * self.score), 0.0))
        for text in texts:
            lowered = text.lower()
            if self.query_marker in lowered:
                vectors.append([self.score, tail])
            elif self.target_marker in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 0.0])
        return vectors


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class AlwaysViolationVerifier:
    def supports(self, rule: dict[str, object]) -> bool:
        return True

    def verify(self, rule: dict[str, object], path: Path) -> list[Violation]:
        return [
            Violation(
                rule_id=str(rule.get("id", "")),
                severity="error",
                path=str(path),
                line=1,
                message="custom verifier violation",
            )
        ]


class PolicyRuntimeTests(unittest.TestCase):
    _REAL_EMBEDDING_PROVIDER_TESTS = {
        "test_hashing_embedding_provider_is_not_supported",
        "test_runtime_uses_openai_compatible_embedding_provider_when_configured",
        "test_python_runtime_accepts_embedding_provider_config",
        "test_runtime_requires_embedding_provider_when_no_local_model_exists",
    }

    def setUp(self) -> None:
        analyzer_service._DEFAULT_ANALYZER = None
        self.addCleanup(self._reset_default_analyzer)
        if self._testMethodName in self._REAL_EMBEDDING_PROVIDER_TESTS:
            return
        for target in (
            "ai_policy_runtime.task_analysis.analyzer.default_embedding_provider",
            "ai_policy_runtime.application.runtime.default_embedding_provider",
        ):
            patcher = patch(target, return_value=FakeEmbeddingProvider())
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _reset_default_analyzer() -> None:
        analyzer_service._DEFAULT_ANALYZER = None

    def test_task_analyzer_understands_cpp20_low_latency_queue(self) -> None:
        analysis = analyze("帮我写一个 C++20 低延迟队列")
        task = analysis.task

        self.assertGreaterEqual(analysis.confidence, 0.72)
        self.assertFalse(analysis.needs_review)
        self.assertEqual(task.domain, "cpp")
        self.assertEqual(task.task_type, "write_code")
        self.assertEqual(task.context["language"], "cpp")
        self.assertEqual(task.context["standard"], 20)
        self.assertTrue(task.context["hot_path"])
        self.assertTrue(task.context["performance_critical"])
        self.assertEqual(task.context["data_structure"], "queue")
        self.assertEqual(task.context["scenario"], "low_latency_queue")
        self.assertIn("cpp20", task.tags)
        self.assertIn("low_latency", task.tags)
        self.assertIn("code_generation", task.capabilities)

    def test_task_analyzer_extracts_matching_engine_scenario(self) -> None:
        task = analyze("为低延迟撮合引擎写一段 C++20 代码").task

        self.assertEqual(task.domain, "cpp")
        self.assertEqual(task.context["standard"], 20)
        self.assertEqual(task.context["scenario"], "matching_engine")
        self.assertIn("trading", task.tags)
        self.assertIn("systems_programming", task.tags)

    def test_cpp_refactor_does_not_infer_hot_path_without_latency_signal(self) -> None:
        task = analyze(
            "Refactor this C++20 code so it is not just working. "
            "Reduce complexity and preserve safety."
        ).task

        self.assertEqual(task.domain, "cpp")
        self.assertEqual(task.context["standard"], 20)
        self.assertNotIn("hot_path", task.context)
        self.assertNotIn("performance_critical", task.context)
        self.assertNotIn("allocation_sensitive", task.context)
        self.assertNotIn("low_latency", task.tags)

    def test_task_analyzer_extracts_cpp20_api_span_intent(self) -> None:
        task = analyze("设计一个 C++20 API，参数是连续范围，优先使用 span").task

        self.assertEqual(task.task_type, "design_api")
        self.assertEqual(task.context["standard"], 20)
        self.assertTrue(task.context["designing_api"])
        self.assertEqual(task.context["parameter_kind"], "contiguous_range")
        self.assertFalse(task.context["ownership_required"])
        self.assertIn("api_design", task.capabilities)

    def test_task_analyzer_marks_ambiguous_input_for_review(self) -> None:
        analysis = analyze("帮我处理一下这个问题")

        self.assertEqual(analysis.task.domain, "general")
        self.assertEqual(analysis.task.task_type, "unknown")
        self.assertTrue(analysis.needs_review)
        self.assertLess(analysis.confidence, 0.72)

    def test_task_analyzer_uses_embedding_semantics_for_rephrased_intent(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=FakeEmbeddingProvider(),
        )

        analysis = analyzer.analyze("写一个 C++20 数据通道，主循环里不能有分配和阻塞，尾延迟要稳")
        task = analysis.task

        self.assertEqual(task.domain, "cpp")
        self.assertEqual(task.task_type, "write_code")
        self.assertTrue(task.context["hot_path"])
        self.assertTrue(task.context["performance_critical"])
        self.assertIn("low_latency", task.tags)
        self.assertTrue(
            any(":semantic:" in item.source for item in analysis.evidence),
            [item.source for item in analysis.evidence],
        )

    def test_task_analyzer_bootstraps_generic_refinement_from_semantics(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=FakeEmbeddingProvider(),
        )

        analysis = analyzer.analyze(
            "帮我把这个模块整理到生产可用，不改变行为，"
            "减少意外复杂度，顺便把重复逻辑和接口摩擦处理掉"
        )
        task = analysis.task

        self.assertEqual(task.domain, "general")
        self.assertNotEqual(task.task_type, "unknown")
        self.assertTrue(task.context["artifact_type"] == "code")
        self.assertTrue(task.context["refinement_requested"])
        self.assertTrue(task.context["behavior_preservation_required"])
        self.assertTrue(
            set(task.context["semantic_skill_matches"]).intersection(
                {
                    "generic.code_quality.implementation_polish",
                    "generic.refactoring.duplication_extraction",
                    "generic.code_quality.api_usability",
                    "generic.code_quality.expressive_implementation",
                }
            )
        )
        self.assertTrue(analysis.activation_ready)
        self.assertTrue(
            any(":semantic:" in item.source for item in analysis.evidence),
            [item.source for item in analysis.evidence],
        )

    def test_semantic_recall_quality_eval_set(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=FakeEmbeddingProvider(),
        )
        fixture = _load_fixture("semantic_recall_eval.yaml")

        for case in fixture["cases"]:
            with self.subTest(case=case["id"]):
                analysis = analyzer.analyze(str(case["prompt"]))
                task = analysis.task

                self.assertEqual(analysis.activation_ready, case["activation_ready"])
                if "domain" in case:
                    self.assertEqual(task.domain, case["domain"])
                if "forbidden_domain" in case:
                    self.assertNotEqual(task.domain, case["forbidden_domain"])
                    self.assertNotEqual(task.context.get("language"), case["forbidden_domain"])
                if "task_type" in case:
                    self.assertEqual(task.task_type, case["task_type"])
                if "task_type_not" in case:
                    self.assertNotEqual(task.task_type, case["task_type_not"])

                for key, value in case.get("required_context", {}).items():
                    self.assertEqual(task.context.get(key), value, key)
                for key in case.get("forbidden_context", ()):
                    self.assertNotIn(key, task.context)
                for tag in case.get("required_tags", ()):
                    self.assertIn(tag, task.tags)
                for tag in case.get("forbidden_tags", ()):
                    self.assertNotIn(tag, task.tags)

                expected_any_skill = set(case.get("expected_any_skill", ()))
                if expected_any_skill:
                    self.assertTrue(
                        expected_any_skill.intersection(
                            set(task.context.get("semantic_skill_matches", ()))
                        ),
                        task.context.get("semantic_skill_matches", ()),
                    )
                if task.domain != "cpp":
                    self.assertNotIn("cpp", task.tags)

    def test_python_semantic_recall_quality_eval_set(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=FakeEmbeddingProvider(),
        )
        fixture = _load_fixture("python_semantic_recall_eval.yaml")

        for case in fixture["cases"]:
            with self.subTest(case=case["id"]):
                analysis = analyzer.analyze(str(case["prompt"]))
                task = analysis.task

                self.assertEqual(analysis.activation_ready, case["activation_ready"])
                if "domain" in case:
                    self.assertEqual(task.domain, case["domain"])
                if "task_type" in case:
                    self.assertEqual(task.task_type, case["task_type"])
                if "task_type_not" in case:
                    self.assertNotEqual(task.task_type, case["task_type_not"])
                for key, value in case.get("required_context", {}).items():
                    self.assertEqual(task.context.get(key), value, key)
                for key in case.get("forbidden_context", ()):
                    self.assertNotIn(key, task.context)
                for tag in case.get("required_tags", ()):
                    self.assertIn(tag, task.tags)
                for tag in case.get("forbidden_tags", ()):
                    self.assertNotIn(tag, task.tags)

    def test_task_analyzer_understands_git_commit_workflow(self) -> None:
        task = analyze(
            "Prepare a git commit message for the staged diff and split unrelated changes."
        ).task

        self.assertEqual(task.domain, "git")
        self.assertEqual(task.task_type, "write_commit_message")
        self.assertTrue(task.context["git_commit_message_requested"])
        self.assertIn("git_workflow", task.capabilities)
        self.assertIn("commit", task.tags)

    def test_task_analyzer_understands_chinese_commit_request(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=FakeEmbeddingProvider(),
        )
        task = analyzer.analyze(
            "提交一次代码",
            TaskSignals(project_language="python"),
        ).task

        self.assertEqual(task.domain, "git")
        self.assertEqual(task.task_type, "prepare_commit")
        self.assertIn("git_workflow", task.capabilities)

    def test_task_analyzer_understands_chinese_commit_request_with_suffix(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=FakeEmbeddingProvider(),
        )
        task = analyzer.analyze(
            "提交一次代码试试",
            TaskSignals(project_language="python"),
        ).task

        self.assertEqual(task.domain, "git")
        self.assertEqual(task.task_type, "prepare_commit")
        self.assertIn("git_workflow", task.capabilities)

    def test_short_commit_intent_uses_git_working_tree_context(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=TargetedScoreEmbeddingProvider(
                query_marker="提交一下",
                target_marker="commit now",
                score=0.48,
            ),
        )
        analysis = analyzer.analyze(
            "提交一下",
            TaskSignals(project_language="python", git_has_changes=True),
        )

        self.assertEqual(analysis.task.domain, "git")
        self.assertEqual(analysis.task.task_type, "prepare_commit")
        self.assertTrue(analysis.task.context["git_working_tree_sensitive"])
        self.assertTrue(analysis.activation_ready)

    def test_short_commit_intent_without_git_changes_stays_unknown(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=TargetedScoreEmbeddingProvider(
                query_marker="提交一下",
                target_marker="commit current changes",
                score=0.56,
            ),
        )
        analysis = analyzer.analyze(
            "提交一下",
            TaskSignals(project_language="python", git_has_changes=False),
        )

        self.assertEqual(analysis.task.domain, "python")
        self.assertEqual(analysis.task.task_type, "unknown")
        self.assertFalse(analysis.activation_ready)

    def test_long_commit_intent_uses_git_working_tree_context(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=TargetedScoreEmbeddingProvider(
                query_marker="long commit probe",
                target_marker="prepare a git commit for current changes",
                score=0.61,
            ),
        )
        analysis = analyzer.analyze(
            "long commit probe with current workspace changes",
            TaskSignals(project_language="python", git_has_changes=True),
        )

        self.assertEqual(analysis.task.domain, "git")
        self.assertEqual(analysis.task.task_type, "prepare_commit")
        self.assertTrue(analysis.task.context["git_working_tree_sensitive"])
        self.assertTrue(analysis.activation_ready)

    def test_weak_long_commit_intent_stays_unknown(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=TargetedScoreEmbeddingProvider(
                query_marker="weak long commit probe",
                target_marker="prepare a git commit for current changes",
                score=0.59,
            ),
        )
        analysis = analyzer.analyze(
            "weak long commit probe with current workspace changes",
            TaskSignals(project_language="python", git_has_changes=True),
        )

        self.assertEqual(analysis.task.domain, "python")
        self.assertEqual(analysis.task.task_type, "unknown")
        self.assertFalse(analysis.activation_ready)

    def test_project_language_signal_does_not_promote_weak_same_domain_task(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=TargetedScoreEmbeddingProvider(
                query_marker="weak same-domain probe",
                target_marker="write python code",
                score=0.59,
            ),
        )
        analysis = analyzer.analyze(
            "weak same-domain probe",
            TaskSignals(project_language="python"),
        )

        self.assertEqual(analysis.task.domain, "python")
        self.assertEqual(analysis.task.task_type, "unknown")
        self.assertFalse(analysis.activation_ready)

    def test_project_language_signal_does_not_promote_weak_cross_domain_task(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=TargetedScoreEmbeddingProvider(
                query_marker="weak cross-domain probe",
                target_marker="prepare a git branch",
                score=0.63,
            ),
        )
        analysis = analyzer.analyze(
            "weak cross-domain probe",
            TaskSignals(project_language="python"),
        )

        self.assertEqual(analysis.task.domain, "python")
        self.assertEqual(analysis.task.task_type, "unknown")
        self.assertFalse(analysis.activation_ready)

    def test_project_language_signal_still_allows_strong_git_commit_semantics(self) -> None:
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=TargetedScoreEmbeddingProvider(
                query_marker="strong git commit probe",
                target_marker="create a commit for code changes",
                score=0.7,
            ),
        )
        analysis = analyzer.analyze(
            "strong git commit probe",
            TaskSignals(project_language="python"),
        )

        self.assertEqual(analysis.task.domain, "git")
        self.assertEqual(analysis.task.task_type, "prepare_commit")
        self.assertTrue(analysis.activation_ready)

    def test_task_analyzer_allows_plain_commit_style_override(self) -> None:
        task = analyze("Write a commit message with no conventional commits.").task

        self.assertEqual(task.domain, "git")
        self.assertEqual(task.context["git_commit_style"], "imperative")
        self.assertFalse(task.context["git_conventional_commit_requested"])

    def test_task_analyzer_understands_python_best_practices(self) -> None:
        task = analyze(
            "Apply Python best practices: clean up imports, add type hints, and write pytest tests."
        ).task

        self.assertEqual(task.domain, "python")
        self.assertEqual(task.task_type, "improve_code_quality")
        self.assertEqual(task.context["language"], "python")
        self.assertTrue(task.context["python_best_practices_requested"])
        self.assertIn("python", task.tags)
        self.assertIn("best-practices", task.tags)
        self.assertIn("code_review", task.capabilities)

    def test_python_best_practices_pack_outputs_python_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Apply Python best practices: clean up imports, add type hints, and write pytest tests.",
            ("python.best_practices",),
        )
        effective = result.structured["effective_rules"]
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")
        sources = _sources(effective)

        self.assertIn("python.core.pythonic_baseline", sources)
        self.assertIn("python.style.readability_and_naming", sources)
        self.assertIn("python.typing.static_typing", sources)
        self.assertIn("python.testing.testing_practices", sources)
        self.assertIn("Do not use wildcard imports", prompt)
        self.assertIn("Prioritize type hints for public functions", prompt)
        self.assertIn("Keep tests isolated from real networks", prompt)
        self.assertNotIn("selected C++ standard", prompt)

    def test_python_professional_pack_outputs_security_and_cli_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Build a Python CLI with argparse that validates paths, avoids shell injection in subprocess calls, and redacts secrets.",
            ("python.best_practices",),
        )
        effective = result.structured["effective_rules"]
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")
        sources = _sources(effective)

        self.assertIn("python.cli.cli_applications", sources)
        self.assertIn("python.security.security_boundaries", sources)
        self.assertIn("Do not execute CLI work at import time", prompt)
        self.assertIn("Do not use eval, exec, dynamic import", prompt)
        self.assertIn("Use subprocess argument lists", prompt)
        self.assertIn("Do not log, print, serialize, or expose passwords", prompt)

    def test_python_professional_pack_outputs_packaging_async_and_performance_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Improve Python packaging in pyproject.toml and optimize asyncio performance with bounded concurrency and benchmarks.",
            ("python.best_practices",),
        )
        effective = result.structured["effective_rules"]
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")
        sources = _sources(effective)

        self.assertIn("python.packaging.project_packaging", sources)
        self.assertIn("python.concurrency.async_and_concurrency", sources)
        self.assertIn("python.performance.performance_engineering", sources)
        self.assertIn("Do not change package manager", prompt)
        self.assertIn("Do not create unbounded threads, processes, tasks", prompt)
        self.assertIn("Define the metric, measure a baseline", prompt)

    def test_git_best_practices_pack_outputs_commit_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Prepare a git commit message for the staged diff and split unrelated changes.",
            ("git.best_practices",),
        )
        effective = result.structured["effective_rules"]
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")
        sources = _sources(effective)

        self.assertIn("git.workflow.commit_hygiene", sources)
        self.assertIn("git.workflow.working_tree_safety", sources)
        self.assertIn("Make the commit message accurately describe the changes", prompt)
        self.assertIn("Keep each commit focused on one coherent reason", prompt)
        self.assertIn("Do not discard, overwrite, reset, clean", prompt)
        self.assertNotIn("selected C++ standard", prompt)

    def test_semantic_git_commit_match_activates_commit_hygiene(self) -> None:
        registry = SkillRegistry.from_dirs("skills", "packs")
        task = TaskContext(
            domain="git",
            task_type="prepare_commit",
            capabilities=("git_workflow",),
            tags=("git", "staging", "working-tree"),
            context={
                "language": "git",
                "semantic_skill_matches": ("git.workflow.commit_hygiene",),
            },
        )
        active = {skill.skill_id for skill in registry.active_skills(task)}

        self.assertIn("git.workflow.commit_hygiene", active)
        self.assertIn("git.workflow.working_tree_safety", active)

    def test_git_history_rewrite_rules_are_shared_history_safe(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "I need to squash these commits with interactive rebase and force push safely.",
            ("git.best_practices",),
        )
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")

        self.assertIn("Do not rebase, amend, reset, or force-push commits", prompt)
        self.assertIn("Use force-with-lease rather than an unconditional force push", prompt)
        self.assertIn("Use amend, squash, fixup, and interactive rebase primarily for local", prompt)

    def test_git_conflict_rules_preserve_both_sides(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Resolve this git merge conflict and prepare the branch for PR review.",
            ("git.best_practices",),
        )
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")

        self.assertIn("Resolve conflict markers by understanding both sides", prompt)
        self.assertIn("summarize the branch purpose", prompt)
        self.assertIn("Keep pull request diffs focused", prompt)

    def test_git_stash_and_clean_rules_guard_destructive_cleanup(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Stash my unfinished work, then clean untracked generated files safely.",
            ("git.best_practices",),
        )
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")
        sources = _sources(result.structured["effective_rules"])

        self.assertIn("git.workflow.stash_and_clean_safety", sources)
        self.assertIn("Run or recommend a dry-run clean before deleting", prompt)
        self.assertIn("Use stash for unfinished work", prompt)
        self.assertIn("Do not clean ignored files", prompt)

    def test_git_pr_rules_include_review_readiness(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Prepare this git branch for a pull request with a conventional PR title.",
            ("git.best_practices",),
        )
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")

        self.assertIn("Check that the branch contains only relevant commits", prompt)
        self.assertIn("Make the pull request title follow", prompt)
        self.assertTrue(
            _has_statement_containing(result.structured["effective_rules"], "Use short branch names")
        )

    def test_cmake_best_practices_pack_outputs_target_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Modernize this CMakeLists.txt to use target-based CMake with correct PUBLIC PRIVATE INTERFACE usage.",
            ("cmake.best_practices",),
        )
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")
        sources = _sources(result.structured["effective_rules"])

        self.assertIn("cmake.project.target_model", sources)
        self.assertIn("cmake.project.usage_requirements", sources)
        self.assertIn("Express include directories, compile definitions", prompt)
        self.assertIn("Use PRIVATE for implementation-only requirements", prompt)
        self.assertNotIn("selected C++ standard", prompt)

    def test_cmake_dependency_pack_outputs_imported_target_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Add a third party dependency with find_package or FetchContent and keep the build reproducible.",
            ("cmake.best_practices",),
        )
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")

        self.assertIn("Prefer dependency consumption through imported targets", prompt)
        self.assertIn("Pin FetchContent or ExternalProject dependencies", prompt)

    def test_cmake_presets_and_testing_rules_are_task_specific(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Add CMakePresets and CTest entries with sanitizer builds as opt-in quality presets.",
            ("cmake.best_practices",),
        )
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")

        self.assertIn("Keep generated build files outside the source tree", prompt)
        self.assertIn("Put stable project configure, build, test, package", prompt)
        self.assertIn("Keep sanitizers, coverage, static analysis", prompt)
        self.assertIn("Add tests through CTest-compatible entrypoints", prompt)

    def test_cmake_compiler_and_source_rules_cover_common_antipatterns(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Replace CMAKE_CXX_FLAGS and file(GLOB) with target_compile_features and explicit target_sources.",
            ("cmake.best_practices",),
        )
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")
        sources = _sources(result.structured["effective_rules"])

        self.assertIn("cmake.project.compiler_options", sources)
        self.assertIn("cmake.project.sources_options_modules", sources)
        self.assertIn("Prefer target_compile_features", prompt)
        self.assertIn("Prefer explicit target_sources entries", prompt)
        self.assertIn("Use generator expressions", prompt)

    def test_semantic_index_reuses_cached_vectors(self) -> None:
        lexicon = TaskLexicon(
            context_rules=(
                LexiconRule(
                    skill_id="cpp.performance.hot_path",
                    field="context.hot_path",
                    value=True,
                    phrases=(),
                    confidence=0.9,
                    source="test",
                    set_context={"hot_path": True},
                    semantic_texts=("tail latency must remain stable",),
                ),
            )
        )
        provider = CountingEmbeddingProvider()
        with TemporaryDirectory() as tmp:
            SemanticTaskIndex(lexicon, provider, cache_dir=tmp)
            SemanticTaskIndex(lexicon, provider, cache_dir=tmp)

        self.assertEqual(provider.calls, 1)

    def test_semantic_index_search_can_be_scoped_by_candidate_skill(self) -> None:
        lexicon = TaskLexicon(
            context_rules=(
                LexiconRule(
                    skill_id="cpp.performance.hot_path",
                    field="context.hot_path",
                    value=True,
                    phrases=(),
                    confidence=0.9,
                    source="hot",
                    semantic_texts=("tail latency must remain stable",),
                ),
                LexiconRule(
                    skill_id="python.web",
                    field="context.framework",
                    value="django",
                    phrases=(),
                    confidence=0.9,
                    source="web",
                    semantic_texts=("tail latency must remain stable",),
                ),
            )
        )
        index = SemanticTaskIndex(lexicon, FakeEmbeddingProvider(), threshold=0.1)

        matches = index.search_scoped(
            "尾延迟要稳定",
            scope=frozenset({"cpp.performance.hot_path"}),
        )

        self.assertEqual(
            [match.rule.skill_id for match in matches],
            ["cpp.performance.hot_path"],
        )

    def test_task_analysis_context_rules_use_text_match_authoring_form(self) -> None:
        lexicon = TaskLexicon.from_skills_dir("skills")

        template_rule = next(
            rule
            for rule in lexicon.context_rules
            if rule.source.endswith(":detect_template_constraints_required")
        )

        self.assertEqual(template_rule.field, "context.template_constraints_required")
        self.assertEqual(template_rule.value, True)
        self.assertIn("concept", template_rule.phrases)
        self.assertEqual(template_rule.set_context, {"template_constraints_required": True})

    def test_openai_compatible_embedding_provider_uses_batch_endpoint(self) -> None:
        provider = OpenAICompatibleEmbeddingProvider(
            OpenAICompatibleEmbeddingConfig(
                base_url="https://embedding.example.test/v1",
                model="embed-small",
                api_key="secret",
                timeout_seconds=3.0,
            )
        )
        response = FakeHttpResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )

        with patch(
            "ai_policy_runtime.task_analysis.embeddings.urlopen",
            return_value=response,
        ) as urlopen_mock:
            vectors = provider.encode(("first", "second"))

        request = urlopen_mock.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://embedding.example.test/v1/embeddings")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(payload, {"model": "embed-small", "input": ["first", "second"]})
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])

    def test_openai_compatible_embedding_config_can_be_loaded_from_env(self) -> None:
        env = {
            "AI_POLICY_EMBEDDING_PROVIDER": "openai-compatible",
            "AI_POLICY_EMBEDDING_BASE_URL": "https://gateway.example.test/v1",
            "AI_POLICY_EMBEDDING_API_KEY": "key",
            "AI_POLICY_EMBEDDING_MODEL": "embedding-model",
        }
        with patch.dict(os.environ, env, clear=True):
            provider = OpenAICompatibleEmbeddingProvider.from_env()

        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.config.base_url, "https://gateway.example.test/v1")
        self.assertEqual(provider.config.api_key, "key")
        self.assertEqual(provider.config.model, "embedding-model")

    def test_openai_compatible_embedding_provider_requires_endpoint_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {"AI_POLICY_EMBEDDING_PROVIDER": "openai-compatible"},
            clear=True,
        ):
            provider = OpenAICompatibleEmbeddingProvider.from_env()

        self.assertIsNone(provider)

    def test_hashing_embedding_provider_is_not_supported(self) -> None:
        with patch.dict(
            os.environ,
            {"AI_POLICY_EMBEDDING_PROVIDER": "hashing"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                TaskAnalyzer.from_skills_dir("skills")

    def test_env_local_provider_ignores_auto_remote_model_from_project_config(self) -> None:
        embedding = EmbeddingConfig(model="remote/model:name")
        with (
            patch.dict(
                os.environ,
                {
                    "AI_POLICY_EMBEDDING_PROVIDER": "local",
                    "AI_POLICY_EMBEDDING_MODEL": "local-model-path",
                },
                clear=True,
            ),
            patch(
                "ai_policy_runtime.task_analysis.analyzer.SentenceTransformerEmbeddingProvider"
            ) as provider,
        ):
            default_embedding_provider(".", embedding)

        provider.assert_called_once_with("local-model-path")

    def test_local_model_manager_lists_and_installs_known_model(self) -> None:
        with TemporaryDirectory() as tmp:
            manager = LocalModelManager(tmp)
            listed = manager.list()

            self.assertEqual(listed[0]["key"], "multilingual-mini")
            self.assertFalse(listed[0]["installed"])

            with patch(
                "ai_policy_runtime.services.local_models._snapshot_download"
            ) as download:
                installed = manager.install()

            download.assert_called_once()
            self.assertEqual(installed["key"], "multilingual-mini")
            self.assertTrue(installed["path"].endswith("paraphrase-multilingual-MiniLM-L12-v2"))

    def test_runtime_explain_returns_task_analysis_without_current_state(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root="."))
        result = runtime.explain("帮我写一个 C++20 低延迟队列").to_dict()

        self.assertEqual(result["task"]["domain"], "cpp")
        self.assertEqual(result["task"]["context"]["standard"], 20)
        self.assertFalse(result["needs_review"])
        self.assertTrue(result["evidence"])
        self.assertIn("project_context", result)

    def test_runtime_uses_openai_compatible_embedding_provider_when_configured(self) -> None:
        with TemporaryDirectory() as tmp:
            policy_root = Path(tmp)
            local_model = policy_root / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
            local_model.mkdir(parents=True)
            runtime = PolicyRuntime(
                RuntimeConfig.from_values(root=tmp, policy_root=policy_root)
            )
            env = {
                "AI_POLICY_EMBEDDING_PROVIDER": "openai-compatible",
                "AI_POLICY_EMBEDDING_API_KEY": "key",
                "AI_POLICY_EMBEDDING_MODEL": "text-embedding-3-small",
            }

            with patch.dict(os.environ, env, clear=True):
                provider = runtime._embedding_provider()

        self.assertIsInstance(provider, OpenAICompatibleEmbeddingProvider)

    def test_python_runtime_accepts_embedding_provider_config(self) -> None:
        runtime = PolicyRuntime(
            RuntimeConfig.from_values(
                root=".",
                policy_root=".",
                embedding_provider="openai-compatible",
                embedding_api_key="key",
                embedding_model="text-embedding-3-small",
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            provider = runtime._embedding_provider()

        self.assertIsInstance(provider, OpenAICompatibleEmbeddingProvider)
        self.assertEqual(provider.config.api_key, "key")

    def test_runtime_requires_embedding_provider_when_no_local_model_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = PolicyRuntime(RuntimeConfig.from_values(root=tmp, policy_root=tmp))

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Semantic analysis requires"):
                    runtime._embedding_provider()

    def test_project_context_reads_cmake_standard(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\n"
                "project(demo LANGUAGES CXX)\n"
                "target_compile_features(demo PRIVATE cxx_std_20)\n",
                encoding="utf-8",
            )

            analysis = ProjectContextAnalyzer(root).analyze()

        self.assertEqual(analysis.fact("domain").value, "cpp")
        self.assertEqual(analysis.context()["language"], "cpp")
        self.assertEqual(analysis.context()["standard"], 20)
        self.assertTrue(analysis.context()["selected_standard_is_known"])

    def test_project_context_prefers_compile_commands_over_cmake_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                "set(CMAKE_CXX_STANDARD 17)\n",
                encoding="utf-8",
            )
            (root / "compile_commands.json").write_text(
                json.dumps(
                    [
                        {
                            "directory": str(root),
                            "file": "main.cpp",
                            "command": "clang++ -std=c++20 -c main.cpp",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            analysis = ProjectContextAnalyzer(root).analyze()

        self.assertEqual(analysis.context()["standard"], 20)
        self.assertIn("compile_commands.json", analysis.fact("context.standard").source)

    def test_project_context_detects_generic_project_tooling(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".clang-format").write_text("BasedOnStyle: LLVM\n", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                '[project]\n'
                'requires-python = ">=3.10"\n'
                "\n"
                "[tool.ruff]\n"
                "line-length = 100\n",
                encoding="utf-8",
            )

            analysis = ProjectContextAnalyzer(root).analyze()
            context = analysis.context()

        self.assertEqual(analysis.primary_language, "python")
        self.assertTrue(context["has_clang_format"])
        self.assertTrue(context["has_ruff"])
        self.assertEqual(context["python_requires"], ">=3.10")

    def test_project_context_detects_git_working_tree_changes(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not available")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.DEVNULL)
            (root / "file.txt").write_text("change\n", encoding="utf-8")

            analysis = ProjectContextAnalyzer(root).analyze()
            context = analysis.context()

        self.assertTrue(context["git_has_changes"])
        self.assertEqual(context["git_change_count"], 1)
        self.assertTrue(context["git_has_untracked_files"])

    def test_project_context_yaml_overrides_detected_facts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".policy").mkdir()
            (root / ".policy" / "project.yaml").write_text(
                "domain: cpp\n"
                "context:\n"
                "  standard: 23\n"
                "  hot_path: true\n"
                "tags:\n"
                "  - low_latency\n",
                encoding="utf-8",
            )
            (root / "CMakeLists.txt").write_text(
                "set(CMAKE_CXX_STANDARD 17)\n",
                encoding="utf-8",
            )

            analysis = ProjectContextAnalyzer(root).analyze()

        self.assertEqual(analysis.context()["standard"], 23)
        self.assertTrue(analysis.context()["hot_path"])
        self.assertIn("low_latency", analysis.tags())

    def test_project_context_merges_missing_standard_without_overriding_prompt(self) -> None:
        task_analysis = analyze("帮我写一个 C++17 队列")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "compile_commands.json").write_text(
                json.dumps(
                    [
                        {
                            "directory": str(root),
                            "file": "main.cpp",
                            "command": "clang++ -std=c++20 -c main.cpp",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            project = ProjectContextAnalyzer(root).analyze()
            merged = merge_project_analysis(task_analysis, project)

        self.assertEqual(merged.task.context["standard"], 17)
        self.assertEqual(merged.task.domain, "cpp")

    def test_project_config_can_request_conventional_commits_for_git_tasks(self) -> None:
        task_analysis = _git_prepare_commit_analysis()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".policy").mkdir()
            (root / ".policy" / "config.json").write_text(
                json.dumps({"git": {"commitStyle": "conventional"}}),
                encoding="utf-8",
            )

            project = ProjectContextAnalyzer(root).analyze()
            merged = merge_project_analysis(task_analysis, project)

        self.assertEqual(merged.task.domain, "git")
        self.assertEqual(merged.task.context["git_commit_style"], "conventional")
        self.assertTrue(merged.task.context["git_conventional_commit_requested"])

    def test_project_config_imperative_style_does_not_request_conventional_commits(self) -> None:
        task_analysis = _git_prepare_commit_analysis()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".policy").mkdir()
            (root / ".policy" / "config.json").write_text(
                json.dumps({"git": {"commitStyle": "imperative"}}),
                encoding="utf-8",
            )

            project = ProjectContextAnalyzer(root).analyze()
            merged = merge_project_analysis(task_analysis, project)

        self.assertEqual(merged.task.context["git_commit_style"], "imperative")
        self.assertNotIn("git_conventional_commit_requested", merged.task.context)

    def test_project_context_detects_conventional_commit_tooling(self) -> None:
        task_analysis = _git_prepare_commit_analysis()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "devDependencies": {
                            "@commitlint/cli": "^19.0.0",
                            "@commitlint/config-conventional": "^19.0.0",
                        }
                    }
                ),
                encoding="utf-8",
            )

            project = ProjectContextAnalyzer(root).analyze()
            merged = merge_project_analysis(task_analysis, project)

        self.assertEqual(merged.task.context["git_commit_style"], "conventional")
        self.assertTrue(merged.task.context["git_conventional_commit_requested"])

    @unittest.skipUnless(shutil.which("git"), "git executable is required")
    def test_project_context_detects_conventional_commit_history(self) -> None:
        task_analysis = _git_prepare_commit_analysis()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(
                ["git", "init"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Policy Test"],
                cwd=root,
                check=True,
            )
            for index, message in enumerate(
                (
                    "feat: add initial file",
                    "fix(runtime): preserve hook prompt",
                    "docs: update usage notes",
                ),
                1,
            ):
                (root / "file.txt").write_text(f"{index}\n", encoding="utf-8")
                subprocess.run(["git", "add", "file.txt"], cwd=root, check=True)
                subprocess.run(
                    ["git", "commit", "-m", message],
                    cwd=root,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )

            project = ProjectContextAnalyzer(root).analyze()
            merged = merge_project_analysis(task_analysis, project)

        self.assertEqual(merged.task.context["git_commit_style"], "conventional")
        self.assertEqual(merged.task.context["git_commit_style_source"], "project_detected")
        self.assertTrue(merged.task.context["git_conventional_commit_requested"])

    def test_runtime_uses_target_project_root_separate_from_policy_root(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\n"
                "project(external LANGUAGES CXX)\n"
                "set(CMAKE_CXX_STANDARD 20)\n",
                encoding="utf-8",
            )
            runtime = PolicyRuntime(
                RuntimeConfig.from_values(root=target, policy_root=".")
            )

            result = runtime.resolve("帮我写一个低延迟队列", ("cpp.low_latency",))

            self.assertEqual(result.current, target / ".policy" / "current")
            context = result.structured["effective_rules"]["task"]["context"]
            self.assertEqual(context["domain"], "cpp")
            self.assertEqual(context["standard"], 20)
            self.assertTrue((target / ".policy" / "current" / "project-context.json").exists())

    def test_activates_dependencies_and_keeps_conditional_exception(self) -> None:
        registry = SkillRegistry(
            [
                Skill.from_mapping(
                    {
                        "skill_id": "cpp.safe",
                        "name": "C++ Safe",
                        "domains": ["cpp"],
                        "triggers": ["write_code"],
                        "capabilities": ["code_generation"],
                        "rules": {
                            "soft": [
                                {
                                    "id": "avoid_raw",
                                    "target": "raw_pointer",
                                    "action": "FORBID",
                                    "value": "raw_pointer",
                                    "description": "Avoid raw pointers.",
                                }
                            ]
                        },
                    }
                ),
                Skill.from_mapping(
                    {
                        "skill_id": "cpp.hot_path",
                        "name": "C++ Hot Path",
                        "domains": ["cpp"],
                        "triggers": ["write_code"],
                        "capabilities": ["code_generation"],
                        "tags": ["hot_path"],
                        "dependencies": ["cpp.safe"],
                        "exceptions": [
                            {
                                "when": "hot_path == true",
                                "allow": [
                                    {
                                        "id": "allow_raw_hot_path",
                                        "target": "raw_pointer",
                                        "action": "ALLOW",
                                        "value": "raw_pointer",
                                        "description": "Allow raw pointers in hot paths.",
                                    }
                                ],
                                "require": ["justification"],
                            }
                        ],
                    }
                ),
            ]
        )
        task = TaskContext(
            domain="cpp",
            task_type="write_code",
            capabilities=("code_generation",),
            tags=("hot_path",),
            context={"hot_path": True},
        )

        effective = PolicyEngine(registry).evaluate(task)

        self.assertEqual([rule.id for rule in effective.soft], ["avoid_raw"])
        self.assertEqual([rule.id for rule in effective.exceptions], ["allow_raw_hot_path"])
        self.assertEqual(effective.exceptions[0].requires, ("justification",))

    def test_hard_conflict_fails(self) -> None:
        registry = SkillRegistry(
            [
                Skill.from_mapping(
                    {
                        "skill_id": "a",
                        "name": "A",
                        "domains": ["cpp"],
                        "triggers": ["write_code"],
                        "rules": {
                            "hard": [
                                {
                                    "id": "must_x",
                                    "target": "api",
                                    "action": "REQUIRE",
                                    "value": "x",
                                }
                            ]
                        },
                    }
                ),
                Skill.from_mapping(
                    {
                        "skill_id": "b",
                        "name": "B",
                        "domains": ["cpp"],
                        "triggers": ["write_code"],
                        "rules": {
                            "hard": [
                                {
                                    "id": "forbid_x",
                                    "target": "api",
                                    "action": "FORBID",
                                    "value": "x",
                                }
                            ]
                        },
                    }
                ),
            ]
        )
        task = TaskContext(domain="cpp", task_type="write_code")

        with self.assertRaises(PolicyConflictError):
            PolicyEngine(registry).evaluate(task)

    def test_canonical_skill_dsl_shape_and_activation_condition(self) -> None:
        skill = Skill.from_mapping(
            {
                "kind": "skill",
                "api_version": "policy.skill/v1",
                "skill": {
                    "id": "cpp.dsl.canonical",
                    "name": "Canonical C++ DSL",
                    "version": "1.0.0",
                    "level": "DOMAIN",
                    "priority": 70,
                    "status": "stable",
                },
                "scope": {
                    "domains": ["cpp"],
                    "triggers": ["write_code"],
                    "capabilities": ["code_generation"],
                },
                "activation": {"when": 'language == "cpp" and standard >= 20'},
                "rules": {
                    "hard": [
                        {
                            "id": "no_ub",
                            "must_not": "undefined_behavior",
                            "target": "behavior.undefined",
                            "reason": "Undefined behavior is not acceptable.",
                        }
                    ],
                    "soft": [
                        {
                            "id": "prefer_raii",
                            "should": "raii",
                            "target": "resource_management",
                        }
                    ],
                    "preference": [
                        {
                            "id": "prefer_safety",
                            "prefer": {"higher": "safety", "lower": "performance"},
                            "target": "decision.optimization",
                        }
                    ],
                },
            }
        )
        registry = SkillRegistry([skill])
        task = TaskContext(
            domain="cpp",
            task_type="write_code",
            capabilities=("code_generation",),
            context={"language": "cpp", "standard": 20},
        )

        effective = PolicyEngine(registry).evaluate(task)

        self.assertEqual(effective.hard[0].action, RuleAction.FORBID)
        self.assertEqual(effective.soft[0].action, RuleAction.RECOMMEND)
        self.assertEqual(effective.preferences[0].value, "safety > performance")

    def test_skill_dsl_authoring_text_is_english_only(self) -> None:
        violations: list[str] = []
        for path in sorted(Path("skills").rglob("*.skill.yaml")):
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if NON_ENGLISH_DSL_PATTERN.search(line):
                    violations.append(f"{path}:{line_number}: {line.strip()}")

        self.assertEqual(
            violations,
            [],
            "Skill DSL files must stay English-only; multilingual recall belongs to embeddings.",
        )

    def test_activation_condition_filters_skill(self) -> None:
        skill = Skill.from_mapping(
            {
                "kind": "skill",
                "api_version": "policy.skill/v1",
                "skill": {
                    "id": "cpp20.only",
                    "name": "C++20 Only",
                    "version": "1.0.0",
                    "level": "DOMAIN",
                    "priority": 10,
                },
                "scope": {"domains": ["cpp"], "triggers": ["write_code"]},
                "activation": {"when": "standard >= 20"},
                "rules": {
                    "hard": [
                        {
                            "id": "requires_cpp20",
                            "must": "cpp20",
                            "target": "language_standard",
                        }
                    ],
                    "soft": [],
                    "preference": [],
                },
            }
        )
        registry = SkillRegistry([skill])

        effective = PolicyEngine(registry).evaluate(
            TaskContext(
                domain="cpp",
                task_type="write_code",
                context={"language": "cpp", "standard": 17},
            )
        )

        self.assertEqual(effective.hard, [])

    def test_general_code_task_activates_generic_code_skills(self) -> None:
        task = TaskContext(
            domain="general",
            task_type="improve_code_quality",
            capabilities=("code_review", "refactor_code"),
            tags=("code-quality", "complexity", "refactoring"),
            context={
                "artifact_type": "code",
                "refinement_requested": True,
                "behavior_preservation_required": True,
                "duplicated_logic": True,
            },
        )

        effective = PolicyEngine(SkillRegistry.from_dirs("skills", "packs")).evaluate(task)
        sources = {
            rule.source
            for rule in (
                *effective.hard,
                *effective.soft,
                *effective.preferences,
                *effective.exceptions,
            )
        }

        self.assertIn("generic.code_quality.complexity_reduction", sources)
        self.assertIn("generic.refactoring.duplication_extraction", sources)

    def test_multiline_condition_expression_is_normalized(self) -> None:
        skill = Skill.from_mapping(
            {
                "skill": {
                    "id": "cpp.multiline.condition",
                    "name": "Multiline Condition",
                    "version": "1.0.0",
                    "level": "domain",
                    "domain": "cpp",
                    "priority": 10,
                    "activation": {"when": {"language": "cpp"}},
                    "capabilities": ["code_generation"],
                },
                "rules": {
                    "soft": [
                        {
                            "id": "allocation_condition",
                            "when": (
                                'language == "cpp" and\n'
                                "(hot_path == true or performance_critical == true)"
                            ),
                            "should": "Keep allocation policy explicit.",
                            "target": "allocation",
                            "action": "recommend",
                        }
                    ]
                },
            }
        )
        task = TaskContext(
            domain="cpp",
            task_type="write_code",
            capabilities=("code_generation",),
            context={"language": "cpp", "hot_path": True},
        )

        effective = PolicyEngine(SkillRegistry([skill])).evaluate(task)

        self.assertEqual([rule.id for rule in effective.soft], ["allocation_condition"])

    def test_pack_expansion_includes_parent_and_overrides(self) -> None:
        base = Skill.from_mapping(
            {
                "skill": {
                    "id": "cpp.base",
                    "name": "Base",
                    "version": "1.0.0",
                    "level": "domain",
                    "domain": "cpp",
                    "priority": 10,
                    "activation": {"when": {"language": "cpp"}},
                    "capabilities": ["code_generation"],
                },
                "rules": {
                    "hard": [
                        {"id": "base_rule", "must": "modern_cpp", "target": "base"}
                    ]
                },
            }
        )
        hot = Skill.from_mapping(
            {
                "skill": {
                    "id": "cpp.hot",
                    "name": "Hot",
                    "version": "1.0.0",
                    "level": "domain",
                    "domain": "cpp",
                    "priority": 10,
                    "activation": {"when": {"language": "cpp"}},
                    "capabilities": ["code_generation"],
                },
                "rules": {
                    "soft": [
                        {"id": "hot_rule", "should": "avoid_alloc", "target": "alloc"}
                    ]
                },
            }
        )
        packs = PackRegistry(
            [
                SkillPack.from_mapping(
                    {
                        "pack": {"id": "cpp.safe", "name": "Safe"},
                        "includes": ["cpp.base"],
                    }
                ),
                SkillPack.from_mapping(
                    {
                        "pack": {"id": "cpp.low", "name": "Low"},
                        "extends": ["cpp.safe"],
                        "includes": ["cpp.hot"],
                        "overrides": [
                            {
                                "id": "hot_preference",
                                "when": "hot_path == true",
                                "prefer": "performance",
                                "over": "readability",
                                "target": "tradeoff",
                            }
                        ],
                    }
                ),
            ]
        )
        registry = SkillRegistry([base, hot], packs)
        task = TaskContext(
            domain="cpp",
            task_type="write_code",
            capabilities=("code_generation",),
            context={"language": "cpp", "hot_path": True},
        )

        effective = PolicyEngine(registry).evaluate(task, ("cpp.low",))

        self.assertEqual([rule.id for rule in effective.hard], ["base_rule"])
        self.assertIn("hot_rule", [rule.id for rule in effective.soft])
        self.assertIn("hot_preference", [rule.id for rule in effective.preferences])

    def test_rule_unless_filters_and_ir_metadata_is_preserved(self) -> None:
        skill = Skill.from_mapping(
            {
                "skill": {
                    "id": "cpp.metadata",
                    "name": "Metadata",
                    "version": "1.0.0",
                    "level": "domain",
                    "domain": "cpp",
                    "priority": 10,
                    "activation": {"when": {"language": "cpp"}},
                    "capabilities": ["code_generation"],
                },
                "rules": {
                    "soft": [
                        {
                            "id": "prefer_span",
                            "when": "standard >= 20",
                            "unless": "abi_boundary == true",
                            "should": "Prefer std::span.",
                            "target": "range",
                            "prefer": "std::span",
                            "over": ["pointer_and_size"],
                            "rationale": "std::span expresses bounds.",
                            "examples": ["span<const int>"],
                        }
                    ]
                },
            }
        )
        registry = SkillRegistry([skill])
        blocked = PolicyEngine(registry).evaluate(
            TaskContext(
                domain="cpp",
                task_type="write_code",
                capabilities=("code_generation",),
                context={"language": "cpp", "standard": 20, "abi_boundary": True},
            )
        )
        allowed = PolicyEngine(registry).evaluate(
            TaskContext(
                domain="cpp",
                task_type="write_code",
                capabilities=("code_generation",),
                context={"language": "cpp", "standard": 20, "abi_boundary": False},
            )
        )

        self.assertEqual(blocked.soft, [])
        self.assertEqual(allowed.soft[0].over, ("pointer_and_size",))
        self.assertIn("rationale", allowed.soft[0].to_dict())

    def test_verify_rules_reports_forbidden_text(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.cpp"
            path.write_text("int undefined_behavior = 0;\n", encoding="utf-8")

            violations = verify_rules(
                [
                    {
                        "id": "no_ub",
                        "action": "FORBID",
                        "value": "undefined_behavior",
                    }
                ],
                path,
            )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule_id, "no_ub")

    def test_file_verifier_accepts_custom_verifier_plugins(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.cpp"
            path.write_text("int main() {}\n", encoding="utf-8")
            violations = FileVerifier([AlwaysViolationVerifier()]).verify_rules(
                [{"id": "custom"}],
                path,
            )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].message, "custom verifier violation")

    def test_file_verifier_supports_regex_and_required_text(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.cpp"
            path.write_text("int * raw = nullptr;\n", encoding="utf-8")
            violations = FileVerifier().verify_rules(
                [
                    {"id": "no_raw_ptr", "action": "forbid", "pattern": r"\w+\s*\*"},
                    {"id": "require_raii", "action": "require", "value": "RAII"},
                ],
                path,
            )

        self.assertEqual({item.rule_id for item in violations}, {"no_raw_ptr", "require_raii"})

    def test_inject_preserves_manual_content_and_replaces_block(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / ".policy" / "current"
            current.mkdir(parents=True)
            (current / "effective-prompt.md").write_text("HARD:\n- A\n", encoding="utf-8")
            agents = root / "AGENTS.md"
            agents.write_text("# Manual\n\nKeep me.\n", encoding="utf-8")

            inject_current_prompt(root, "codex")
            (current / "effective-prompt.md").write_text("HARD:\n- B\n", encoding="utf-8")
            inject_current_prompt(root, "codex")

            text = agents.read_text(encoding="utf-8")

        self.assertIn("Keep me.", text)
        self.assertIn("- B", text)
        self.assertNotIn("- A", text)
        self.assertEqual(text.count(BEGIN), 1)
        self.assertEqual(text.count(END), 1)

    def test_clear_injected_prompt_removes_only_policy_block(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents = root / "AGENTS.md"
            agents.write_text(
                "# Manual\n\nKeep me.\n\n"
                f"{BEGIN}\nold rules\n{END}\n\n"
                "After.\n",
                encoding="utf-8",
            )

            clear_injected_prompt(root, "codex")
            text = agents.read_text(encoding="utf-8")

        self.assertIn("Keep me.", text)
        self.assertIn("After.", text)
        self.assertNotIn("old rules", text)
        self.assertNotIn(BEGIN, text)
        self.assertNotIn(END, text)

    def test_opencode_injects_and_clears_agents_policy_block(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / ".policy" / "current"
            current.mkdir(parents=True)
            (current / "effective-prompt.md").write_text("HARD:\n- OpenCode rule\n", encoding="utf-8")
            agents = root / "AGENTS.md"
            agents.write_text("# Manual\n\nKeep me.\n", encoding="utf-8")

            injected = inject_current_prompt(root, "opencode")
            clear_injected_prompt(root, "opencode")
            text = agents.read_text(encoding="utf-8")

        self.assertEqual(injected, agents)
        self.assertIn("Keep me.", text)
        self.assertNotIn("OpenCode rule", text)
        self.assertNotIn(BEGIN, text)
        self.assertNotIn(END, text)

    def test_cli_inject_supports_opencode_target(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / ".policy" / "current"
            current.mkdir(parents=True)
            (current / "effective-prompt.md").write_text("HARD:\n- OpenCode\n", encoding="utf-8")

            output, exit_code = CommandDispatcher().dispatch(
                argparse.Namespace(
                    command="inject",
                    root=str(root),
                    policy_root=None,
                    skills="skills",
                    packs="packs",
                    target="opencode",
                )
            )

            agents = (root / "AGENTS.md").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(output["target"], "opencode")
        self.assertIn("- OpenCode", agents)

    def test_codex_hook_reads_project_config_packs(self) -> None:
        config = {"packs": ["cpp.safe_generation", "cpp.low_latency"]}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                user_prompt_submit._configured_packs(config),
                ("cpp.safe_generation", "cpp.low_latency"),
            )

    def test_codex_hook_environment_packs_override_project_config(self) -> None:
        config = {"packs": ["cpp.safe_generation"]}
        with patch.dict(os.environ, {"AI_POLICY_PACKS": "cpp.low_latency"}, clear=True):
            self.assertEqual(
                user_prompt_submit._configured_packs(config),
                ("cpp.low_latency",),
            )

    def test_codex_hook_can_be_disabled_by_project_config(self) -> None:
        self.assertFalse(user_prompt_submit._enabled({"enabled": False}))
        self.assertFalse(user_prompt_submit._enabled({"enabled": "off"}))
        self.assertTrue(user_prompt_submit._enabled({}))

    def test_hook_defaults_to_codex_agent(self) -> None:
        self.assertTrue(user_prompt_submit._enabled_for({}, "codex"))
        self.assertFalse(user_prompt_submit._enabled_for({}, "claude"))

    def test_hook_config_filters_by_agent(self) -> None:
        config = {"enabled": True, "agents": ["claude"]}

        self.assertFalse(user_prompt_submit._enabled_for(config, "codex"))
        self.assertTrue(user_prompt_submit._enabled_for(config, "claude"))
        self.assertFalse(user_prompt_submit._enabled_for(config, "opencode"))

    def test_hook_config_supports_opencode_agent(self) -> None:
        config = {"enabled": True, "agents": ["opencode"]}

        self.assertFalse(user_prompt_submit._enabled_for(config, "codex"))
        self.assertTrue(user_prompt_submit._enabled_for(config, "opencode"))

    def test_hook_local_provider_bootstraps_semantic_dependencies(self) -> None:
        config = user_prompt_submit.ProjectHookConfig.from_mapping(
            {"embeddingProvider": "local"}
        )

        with patch.dict(os.environ, {}, clear=True), patch(
            "hooks.user_prompt_submit._bootstrap_package"
        ) as bootstrap:
            with patch(
                "builtins.__import__",
                side_effect=ModuleNotFoundError("sentence_transformers"),
            ):
                config.ensure_semantic_dependencies(Path.cwd())

        bootstrap.assert_called_once_with(semantic=True)

    def test_hook_auto_bootstraps_semantic_when_default_local_model_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_root = root / "policy"
            model = policy_root / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
            model.mkdir(parents=True)
            config = user_prompt_submit.ProjectHookConfig.from_mapping(
                {"policyRoot": str(policy_root)}
            )

            with patch.dict(os.environ, {}, clear=True), patch(
                "hooks.user_prompt_submit._bootstrap_package"
            ) as bootstrap:
                with patch(
                    "builtins.__import__",
                    side_effect=ModuleNotFoundError("sentence_transformers"),
                ):
                    config.ensure_semantic_dependencies(root)

        bootstrap.assert_called_once_with(semantic=True)

    def test_hook_auto_remote_does_not_bootstrap_semantic_dependencies(self) -> None:
        config = user_prompt_submit.ProjectHookConfig.from_mapping({})

        with patch.dict(
            os.environ,
            {"AI_POLICY_EMBEDDING_API_KEY": "key"},
            clear=True,
        ), patch("hooks.user_prompt_submit._bootstrap_package") as bootstrap:
            config.ensure_semantic_dependencies(Path.cwd())

        bootstrap.assert_not_called()

    def test_user_prompt_hook_records_turn_state_for_non_applicable_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"enabled": True, "agents": ["codex"]}),
                encoding="utf-8",
            )
            (root / "AGENTS.md").write_text(
                "# Manual\n\n"
                f"{BEGIN}\nprevious rules\n{END}\n",
                encoding="utf-8",
            )

            payload = {
                "cwd": str(root),
                "turn_id": "turn-probe",
                "session_id": "session-probe",
                "prompt": "验证 hook：policy-runtime-hook-probe-20260519",
            }
            with patch.object(user_prompt_submit, "_read_payload", return_value=payload):
                with patch.object(
                    user_prompt_submit,
                    "_resolve_effective_prompt",
                    return_value="",
                ):
                    with patch("sys.stdout", new=io.StringIO()) as stdout:
                        exit_code = user_prompt_submit.main()

            state = json.loads(
                (root / user_prompt_submit.HOOK_STATE_PATH).read_text(encoding="utf-8")
            )
            response = json.loads(stdout.getvalue())
            agents = (root / "AGENTS.md").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(state["turn_id"], "turn-probe")
        self.assertEqual(state["prompt"], "验证 hook：policy-runtime-hook-probe-20260519")
        self.assertFalse(state["effective_rules_generated"])
        self.assertIsNone(state["effective_prompt_path"])
        self.assertEqual(state["additional_context_chars"], 0)
        self.assertIsNone(state["hook_error"])
        self.assertEqual(
            response["hookSpecificOutput"],
            {"hookEventName": "UserPromptSubmit", "additionalContext": ""},
        )
        self.assertNotIn("previous rules", agents)
        self.assertNotIn(BEGIN, agents)

    def test_policy_agent_wrapper_clears_static_block_for_non_applicable_task(self) -> None:
        from ai_policy_runtime.adapters.agent import AgentWrapperOptions, PolicyAgentWrapper

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents = root / "AGENTS.md"
            agents.write_text(
                "# Manual\n\n"
                f"{BEGIN}\nstale rules\n{END}\n",
                encoding="utf-8",
            )
            wrapper = PolicyAgentWrapper(
                AgentWrapperOptions(
                    task="hello",
                    agent="codex",
                    root=root,
                    policy_root=Path.cwd(),
                    pack_ids=("cpp.safe_generation",),
                    command=("codex",),
                    execute=False,
                )
            )

            with self.assertRaises(NonApplicableTaskError):
                wrapper.run()
            text = agents.read_text(encoding="utf-8")

        self.assertNotIn("stale rules", text)
        self.assertNotIn(BEGIN, text)

    def test_user_prompt_hook_records_effective_rules_generation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"enabled": True, "agents": ["codex"]}),
                encoding="utf-8",
            )

            payload = {
                "cwd": str(root),
                "turn_id": "turn-cpp",
                "session_id": "session-cpp",
                "prompt": "帮我设计一个 C++20 低延迟队列 API",
            }
            effective_prompt = "# Effective Rules for Current Task\n"
            with patch.object(user_prompt_submit, "_read_payload", return_value=payload):
                with patch.object(
                    user_prompt_submit,
                    "_resolve_effective_prompt",
                    return_value=effective_prompt,
                ):
                    with patch("sys.stdout", new=io.StringIO()) as stdout:
                        exit_code = user_prompt_submit.main()

            state = json.loads(
                (root / user_prompt_submit.HOOK_STATE_PATH).read_text(encoding="utf-8")
            )
            response = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(state["effective_rules_generated"])
        self.assertEqual(
            state["effective_prompt_path"],
            str((root / ".policy" / "current" / "effective-prompt.md").resolve()),
        )
        self.assertEqual(state["additional_context_chars"], len(effective_prompt))
        self.assertIsNone(state["hook_error"])
        self.assertEqual(response["hookSpecificOutput"]["additionalContext"], effective_prompt)

    def test_user_prompt_hook_extracts_codex_ide_request(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"enabled": True, "agents": ["codex"]}),
                encoding="utf-8",
            )

            payload = {
                "cwd": str(root),
                "turn_id": "turn-ide",
                "session_id": "session-ide",
                "prompt": (
                    "# Context from my IDE setup:\n\n"
                    "## Active file: .policy/current/agent-hook-state.json\n\n"
                    "## Open tabs:\n"
                    "- agent-hook-state.json: .policy/current/agent-hook-state.json\n\n"
                    "## My request for Codex:\n"
                    "提交一次代码试试\n"
                ),
            }
            effective_prompt = "# Effective Rules for Current Task\n"
            with patch.object(user_prompt_submit, "_read_payload", return_value=payload):
                with patch.object(
                    user_prompt_submit,
                    "_resolve_effective_prompt",
                    return_value=effective_prompt,
                ) as resolve:
                    with patch("sys.stdout", new=io.StringIO()):
                        exit_code = user_prompt_submit.main()

            state = json.loads(
                (root / user_prompt_submit.HOOK_STATE_PATH).read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(state["prompt"], "提交一次代码试试")
        self.assertTrue(state["effective_rules_generated"])
        self.assertEqual(resolve.call_args.args[0], "提交一次代码试试")

    def test_user_prompt_hook_preserves_markdown_in_codex_request(self) -> None:
        wrapped = (
            "# Context from my IDE setup:\n\n"
            "## Active file: .policy/current/effective-prompt.md\n\n"
            "## My request for Codex:\n"
            "# Effective Rules for Current Task\n\n"
            "## Task Context\n\n"
            "- Domain: git\n\n"
            "这个输出效果是正确的吗\n"
        )

        self.assertEqual(
            user_prompt_submit._task_prompt(wrapped),
            (
                "# Effective Rules for Current Task\n\n"
                "## Task Context\n\n"
                "- Domain: git\n\n"
                "这个输出效果是正确的吗"
            ),
        )

    def test_codex_hook_config_reads_post_refinement_options(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = user_prompt_submit.ProjectHookConfig.from_mapping(
                {
                    "postRefine": "strict",
                    "postRefinePacks": ["cpp.production_refinement"],
                    "verifyTarget": "src",
                }
            )

        self.assertEqual(config.post_refine_mode, "strict")
        self.assertEqual(config.post_refine_pack_ids, ("cpp.production_refinement",))
        self.assertEqual(config.verify_target, "src")

    def test_stop_hook_allows_second_stop_to_prevent_loop(self) -> None:
        response = stop_refinement.build_stop_response({"stop_hook_active": True})

        self.assertEqual(response, {"continue": True})

    def test_stop_hook_blocks_once_for_configured_refinement(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"postRefine": "standard"}),
                encoding="utf-8",
            )
            state_path = root / user_prompt_submit.HOOK_STATE_PATH
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "turn_id": "turn-1",
                        "session_id": "session-1",
                        "prompt": "Refactor this C++20 code.",
                        "effective_rules_generated": True,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                stop_refinement,
                "build_refinement_continuation_prompt",
                return_value="Refine once.",
            ):
                response = stop_refinement.build_stop_response(
                    {
                        "cwd": str(root),
                        "stop_hook_active": False,
                        "turn_id": "turn-1",
                        "session_id": "session-1",
                    }
                )

        self.assertEqual(response, {"decision": "block", "reason": "Refine once."})

    def test_stop_hook_main_outputs_ascii_safe_json(self) -> None:
        with patch.object(stop_refinement, "_read_payload", return_value={}):
            with patch.object(
                stop_refinement,
                "build_stop_response",
                return_value={"decision": "block", "reason": "提交一次代码"},
            ):
                with patch("sys.stdout", new=io.StringIO()) as stdout:
                    exit_code = stop_refinement.main()

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("\\u63d0\\u4ea4\\u4e00\\u6b21\\u4ee3\\u7801", output)
        self.assertEqual(json.loads(output)["reason"], "提交一次代码")

    def test_stop_hook_does_not_reuse_stale_turn_state_without_matching_payload_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"postRefine": "standard"}),
                encoding="utf-8",
            )
            state_path = root / user_prompt_submit.HOOK_STATE_PATH
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "turn_id": "old-turn",
                        "session_id": "old-session",
                        "prompt": "Refactor this C++20 code.",
                        "effective_rules_generated": True,
                    }
                ),
                encoding="utf-8",
            )

            response = stop_refinement.build_stop_response(
                {"cwd": str(root), "stop_hook_active": False}
            )

        self.assertEqual(response, {"continue": True})

    def test_stop_hook_allows_turn_state_when_payload_has_no_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"postRefine": "standard"}),
                encoding="utf-8",
            )
            state_path = root / user_prompt_submit.HOOK_STATE_PATH
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "prompt": "Refactor this C++20 code.",
                        "effective_rules_generated": True,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                stop_refinement,
                "build_refinement_continuation_prompt",
                return_value="Refine turn.",
            ):
                response = stop_refinement.build_stop_response(
                    {"cwd": str(root), "stop_hook_active": False}
                )

        self.assertEqual(response, {"decision": "block", "reason": "Refine turn."})

    def test_stop_hook_skips_refinement_without_applicable_turn_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"postRefine": "standard"}),
                encoding="utf-8",
            )

            response = stop_refinement.build_stop_response(
                {"cwd": str(root), "stop_hook_active": False, "turn_id": "turn-1"}
            )

        self.assertEqual(response, {"continue": True})

    def test_stop_hook_skips_refinement_when_prompt_was_not_applicable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"postRefine": "standard"}),
                encoding="utf-8",
            )
            state_path = root / user_prompt_submit.HOOK_STATE_PATH
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "turn_id": "turn-question",
                        "session_id": "session-question",
                        "prompt": "post refine 有效果吗？",
                        "effective_rules_generated": False,
                    }
                ),
                encoding="utf-8",
            )

            response = stop_refinement.build_stop_response(
                {
                    "cwd": str(root),
                    "stop_hook_active": False,
                    "turn_id": "turn-question",
                    "session_id": "session-question",
                }
            )

        self.assertEqual(response, {"continue": True})

    def test_stop_hook_respects_agent_filter(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"agents": ["claude"], "postRefine": "standard"}),
                encoding="utf-8",
            )

            response = stop_refinement.build_stop_response(
                {"cwd": str(root), "stop_hook_active": False}
            )

        self.assertEqual(response, {"continue": True})

    def test_codex_hook_applies_openai_compatible_embedding_config(self) -> None:
        config = user_prompt_submit.ProjectHookConfig.from_mapping(
            {
                "embeddingProvider": "openai-compatible",
                "embeddingBaseUrl": "https://embedding.example.test/v1",
                "embeddingApiKey": "project-key",
                "embeddingModel": "embedding-model",
                "embeddingTimeout": "12.5",
            }
        )

        with patch.dict(os.environ, {}, clear=True):
            config.apply_environment()

            self.assertEqual(
                os.environ["AI_POLICY_EMBEDDING_PROVIDER"], "openai-compatible"
            )
            self.assertEqual(
                os.environ["AI_POLICY_EMBEDDING_BASE_URL"],
                "https://embedding.example.test/v1",
            )
            self.assertEqual(os.environ["AI_POLICY_EMBEDDING_API_KEY"], "project-key")
            self.assertEqual(os.environ["AI_POLICY_EMBEDDING_MODEL"], "embedding-model")
            self.assertEqual(os.environ["AI_POLICY_EMBEDDING_TIMEOUT"], "12.5")

    def test_codex_hook_project_embedding_config_overrides_environment(self) -> None:
        config = user_prompt_submit.ProjectHookConfig.from_mapping(
            {
                "embeddingProvider": "openai-compatible",
                "embeddingBaseUrl": "https://project.example.test/v1",
                "embeddingApiKey": "project-key",
                "embeddingModel": "project-model",
                "embeddingTimeout": "12.5",
            }
        )
        env = {
            "AI_POLICY_EMBEDDING_PROVIDER": "openai-compatible",
            "AI_POLICY_EMBEDDING_BASE_URL": "https://env.example.test/v1",
            "AI_POLICY_EMBEDDING_API_KEY": "env-key",
            "AI_POLICY_EMBEDDING_MODEL": "env-model",
            "AI_POLICY_EMBEDDING_TIMEOUT": "3",
        }

        with patch.dict(os.environ, env, clear=True):
            config.apply_environment()

            self.assertEqual(
                os.environ["AI_POLICY_EMBEDDING_PROVIDER"], "openai-compatible"
            )
            self.assertEqual(
                os.environ["AI_POLICY_EMBEDDING_BASE_URL"],
                "https://project.example.test/v1",
            )
            self.assertEqual(os.environ["AI_POLICY_EMBEDDING_API_KEY"], "project-key")
            self.assertEqual(os.environ["AI_POLICY_EMBEDDING_MODEL"], "project-model")
            self.assertEqual(os.environ["AI_POLICY_EMBEDDING_TIMEOUT"], "12.5")

    def test_codex_hook_keeps_environment_embedding_when_project_config_is_empty(
        self,
    ) -> None:
        config = user_prompt_submit.ProjectHookConfig.from_mapping({})
        env = {
            "AI_POLICY_EMBEDDING_PROVIDER": "openai-compatible",
            "AI_POLICY_EMBEDDING_BASE_URL": "https://env.example.test/v1",
            "AI_POLICY_EMBEDDING_API_KEY": "env-key",
            "AI_POLICY_EMBEDDING_MODEL": "env-model",
            "AI_POLICY_EMBEDDING_TIMEOUT": "3",
        }

        with patch.dict(os.environ, env, clear=True):
            config.apply_environment()

            for key, value in env.items():
                self.assertEqual(os.environ[key], value)

    def test_codex_hook_local_provider_clears_stale_remote_embedding_model(
        self,
    ) -> None:
        config = user_prompt_submit.ProjectHookConfig.from_mapping(
            {"embeddingProvider": "local"}
        )
        env = {
            "AI_POLICY_EMBEDDING_PROVIDER": "openai-compatible",
            "AI_POLICY_EMBEDDING_BASE_URL": "https://env.example.test/v1",
            "AI_POLICY_EMBEDDING_API_KEY": "env-key",
            "AI_POLICY_EMBEDDING_MODEL": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
            "AI_POLICY_EMBEDDING_TIMEOUT": "3",
        }

        with patch.dict(os.environ, env, clear=True):
            config.apply_environment()

            self.assertEqual(os.environ["AI_POLICY_EMBEDDING_PROVIDER"], "local")
            self.assertNotIn("AI_POLICY_EMBEDDING_BASE_URL", os.environ)
            self.assertNotIn("AI_POLICY_EMBEDDING_API_KEY", os.environ)
            self.assertNotIn("AI_POLICY_EMBEDDING_MODEL", os.environ)
            self.assertNotIn("AI_POLICY_EMBEDDING_TIMEOUT", os.environ)

    def test_codex_hook_local_provider_applies_project_model(
        self,
    ) -> None:
        config = user_prompt_submit.ProjectHookConfig.from_mapping(
            {
                "embeddingProvider": "local",
                "embeddingModel": "D:/models/paraphrase-multilingual-MiniLM-L12-v2",
            }
        )

        with patch.dict(
            os.environ,
            {"AI_POLICY_EMBEDDING_MODEL": "env-model"},
            clear=True,
        ):
            config.apply_environment()

            self.assertEqual(os.environ["AI_POLICY_EMBEDDING_PROVIDER"], "local")
            self.assertEqual(
                os.environ["AI_POLICY_EMBEDDING_MODEL"],
                "D:/models/paraphrase-multilingual-MiniLM-L12-v2",
            )

    def test_codex_hook_loads_project_config(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / ".policy"
            policy.mkdir()
            (policy / "config.json").write_text(
                json.dumps({"enabled": True, "packs": ["cpp.safe_generation"]}),
                encoding="utf-8",
            )

            self.assertEqual(
                user_prompt_submit._load_project_config(root),
                {"enabled": True, "packs": ["cpp.safe_generation"]},
            )

    def test_resolve_if_applicable_skips_non_policy_input_without_writing_current(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = PolicyRuntime(RuntimeConfig.from_values(root=tmp, policy_root="."))

            with patch(
                "ai_policy_runtime.task_analysis.embeddings.urlopen",
                side_effect=AssertionError("trivial greeting should not request embeddings"),
            ):
                result = runtime.resolve_if_applicable("hello", ("cpp.safe_generation",))

            self.assertFalse(result.applicable)
            self.assertFalse((Path(tmp) / ".policy" / "current").exists())

    def test_resolve_if_applicable_skips_status_query_without_writing_current(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = PolicyRuntime(RuntimeConfig.from_values(root=tmp, policy_root="."))

            with patch(
                "ai_policy_runtime.application.runtime.default_embedding_provider",
                return_value=FakeEmbeddingProvider(),
            ):
                result = runtime.resolve_if_applicable(
                    "请检查当前项目，并说明 AI Policy Runtime 是否通过 Claude Code plugin 启用了。",
                    ("cpp.safe_generation",),
                )

            self.assertFalse(result.applicable)
            self.assertFalse((Path(tmp) / ".policy" / "current").exists())

    def test_resolve_if_applicable_skips_semantic_status_query_without_evaluating(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = PolicyRuntime(RuntimeConfig.from_values(root=tmp, policy_root="."))

            with (
                patch(
                    "ai_policy_runtime.application.runtime.default_embedding_provider",
                    return_value=TargetedScoreEmbeddingProvider(
                        query_marker="ai policy runtime",
                        target_marker="check whether ai policy runtime is enabled",
                        score=0.7,
                    ),
                ),
                patch.object(
                    runtime,
                    "_evaluate",
                    side_effect=AssertionError("status query should not resolve rules"),
                ),
            ):
                result = runtime.resolve_if_applicable(
                    "Inspect whether AI Policy Runtime is active for this project.",
                    ("cpp.safe_generation",),
                )

            self.assertFalse(result.applicable)
            self.assertFalse((Path(tmp) / ".policy" / "current").exists())

    def test_resolve_if_applicable_skips_effective_rules_output_question(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = PolicyRuntime(RuntimeConfig.from_values(root=tmp, policy_root="."))

            result = runtime.resolve_if_applicable(
                "# Effective Rules for Current Task\n\n"
                "## Task Context\n\n"
                "- Domain: git\n\n"
                "这个输出效果是正确的吗",
                (),
            )

            self.assertFalse(result.applicable)
            self.assertFalse((Path(tmp) / ".policy" / "current").exists())

    def test_resolve_if_applicable_requires_prompt_task_before_project_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\n"
                "project(docs LANGUAGES CXX)\n"
                "set(CMAKE_CXX_STANDARD 20)\n",
                encoding="utf-8",
            )
            runtime = PolicyRuntime(RuntimeConfig.from_values(root=root, policy_root="."))

            result = runtime.resolve_if_applicable(
                "阅读一下文档中的设计思想，看看有没有破坏原有的流程，流程是否还连贯。只分析，不改",
                ("cpp.safe_generation",),
            )

            self.assertFalse(result.applicable)
            self.assertFalse(result.task_analysis["activation_ready"])
            self.assertNotIn(
                "semantic_skill_matches",
                result.task_analysis["task"]["context"],
            )
            self.assertFalse((root / ".policy" / "current").exists())

    def test_resolve_rejects_non_applicable_task_without_writing_current(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                "project(docs LANGUAGES CXX)\n"
                "set(CMAKE_CXX_STANDARD 20)\n",
                encoding="utf-8",
            )
            runtime = PolicyRuntime(RuntimeConfig.from_values(root=root, policy_root="."))

            with self.assertRaises(NonApplicableTaskError) as raised:
                runtime.resolve("阅读文档，只分析不改", ("cpp.safe_generation",))

            self.assertFalse(raised.exception.task_analysis["activation_ready"])
            self.assertFalse((root / ".policy" / "current").exists())

    def test_cli_resolve_reports_non_applicable_without_writing_current(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                "project(docs LANGUAGES CXX)\n"
                "set(CMAKE_CXX_STANDARD 20)\n",
                encoding="utf-8",
            )

            output, exit_code = CommandDispatcher().dispatch(
                argparse.Namespace(
                    command="resolve",
                    root=str(root),
                    policy_root=".",
                    skills="skills",
                    packs="packs",
                    task="阅读文档，只分析不改",
                    pack=["cpp.safe_generation"],
                    format="json",
                )
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse(output["applicable"])
            self.assertFalse((root / ".policy" / "current").exists())

    def test_pack_does_not_activate_unknown_task_from_project_context(self) -> None:
        base = Skill.from_mapping(
            {
                "skill": {
                    "id": "cpp.base",
                    "name": "Base",
                    "version": "1.0.0",
                    "level": "domain",
                    "domain": "cpp",
                    "priority": 10,
                    "activation": {
                        "when": {"language": "cpp"},
                        "triggers": ["write_code"],
                    },
                    "capabilities": ["code_generation"],
                },
                "rules": {
                    "hard": [
                        {"id": "base_rule", "must": "modern_cpp", "target": "base"}
                    ]
                },
            }
        )
        packs = PackRegistry(
            [
                SkillPack.from_mapping(
                    {
                        "pack": {"id": "cpp.safe", "name": "Safe"},
                        "includes": ["cpp.base"],
                        "overrides": [
                            {
                                "id": "pack_preference",
                                "prefer": "safety",
                                "over": "speed",
                                "target": "tradeoff",
                            }
                        ],
                    }
                )
            ]
        )
        task = TaskContext(
            domain="cpp",
            task_type="unknown",
            capabilities=(),
            context={"language": "cpp", "standard": 20},
        )

        effective = PolicyEngine(SkillRegistry([base], packs)).evaluate(task, ("cpp.safe",))

        self.assertFalse(effective.hard)
        self.assertFalse(effective.preferences)

    def test_codex_wrapper_builds_command_with_task_last(self) -> None:
        command = _build_codex_command(
            ("codex",),
            ("--approval-mode", "never"),
            "帮我写一个 C++20 低延迟队列",
        )

        self.assertEqual(
            command,
            (
                "codex",
                "--approval-mode",
                "never",
                "帮我写一个 C++20 低延迟队列",
            ),
        )

    def test_claude_wrapper_builds_command_with_task_last(self) -> None:
        command = _build_claude_command(
            ("claude",),
            ("--dangerously-skip-permissions",),
            "帮我写一个 C++20 低延迟队列",
        )

        self.assertEqual(
            command,
            (
                "claude",
                "--dangerously-skip-permissions",
                "帮我写一个 C++20 低延迟队列",
            ),
        )

    def test_opencode_wrapper_builds_command_with_task_last(self) -> None:
        command = _build_opencode_command(
            ("opencode", "run"),
            ("--model", "anthropic/claude-sonnet-4"),
            "帮我写一个 C++20 低延迟队列",
        )

        self.assertEqual(
            command,
            (
                "opencode",
                "run",
                "--model",
                "anthropic/claude-sonnet-4",
                "帮我写一个 C++20 低延迟队列",
            ),
        )

    def test_configure_claude_desktop_writes_policy_and_settings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path(tmp) / "plugin"
            (plugin_root / ".claude-plugin").mkdir(parents=True)
            (plugin_root / ".claude-plugin" / "plugin.json").write_text("{}", encoding="utf-8")
            (plugin_root / ".claude-plugin" / "marketplace.json").write_text("{}", encoding="utf-8")
            (plugin_root / "hooks").mkdir()
            (plugin_root / "hooks" / "hooks.json").write_text("{}", encoding="utf-8")

            policy_path = configure_policy(root, plugin_root)
            settings_path = configure_claude_settings(root, plugin_root, "local")

            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            settings = json.loads(settings_path.read_text(encoding="utf-8"))

        self.assertTrue(policy["enabled"])
        self.assertIn("claude", policy["agents"])
        self.assertEqual(policy["packs"], [])
        self.assertEqual(policy["policyRoot"], str(plugin_root))
        self.assertEqual(policy["git"], {"commitStyle": "auto"})
        self.assertTrue(settings["enabledPlugins"][PLUGIN_ID])
        self.assertEqual(
            settings["extraKnownMarketplaces"]["ai-policy-runtime"]["source"]["path"],
            str(plugin_root),
        )

    def test_configure_claude_desktop_updates_stale_policy_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            old_plugin = Path(tmp) / "old-plugin"
            plugin_root = Path(tmp) / "plugin"
            policy = root / ".policy"
            policy.mkdir(parents=True)
            (policy / "config.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "agents": ["claude"],
                        "policyRoot": str(old_plugin),
                    }
                ),
                encoding="utf-8",
            )

            policy_path = configure_policy(root, plugin_root)
            current = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertEqual(current["policyRoot"], str(plugin_root))

    def test_configure_claude_desktop_can_enable_post_refinement(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path(tmp) / "plugin"

            policy_path = configure_policy(
                root,
                plugin_root,
                post_refine="standard",
            )
            policy = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertEqual(policy["postRefine"], "standard")
        self.assertEqual(policy["postRefinePacks"], [DEFAULT_POST_REFINE_PACK])

    def test_configure_claude_desktop_uses_custom_post_refinement_packs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path(tmp) / "plugin"

            policy_path = configure_policy(
                root,
                plugin_root,
                post_refine="strict",
                post_refine_packs=("cpp.production_refinement", "project.refinement"),
            )
            policy = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertEqual(policy["postRefine"], "strict")
        self.assertEqual(
            policy["postRefinePacks"],
            ["cpp.production_refinement", "project.refinement"],
        )

    def test_configure_claude_desktop_can_disable_runtime_and_plugin(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path(tmp) / "plugin"

            configure_policy(root, plugin_root, enabled=True)
            policy_path = configure_policy(root, plugin_root, enabled=False)
            settings_path = configure_claude_settings(
                root,
                plugin_root,
                "local",
                enabled=False,
            )
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            settings = json.loads(settings_path.read_text(encoding="utf-8"))

        self.assertFalse(policy["enabled"])
        self.assertNotIn("claude", policy["agents"])
        self.assertFalse(settings["enabledPlugins"][PLUGIN_ID])

    def test_configure_claude_desktop_status_reports_current_features(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path(tmp) / "plugin"
            configure_policy(root, plugin_root, post_refine="standard")
            configure_claude_settings(root, plugin_root, "local")

            current = claude_desktop_status(root, plugin_root, "local")

        self.assertTrue(current["runtime_enabled"])
        self.assertTrue(current["claude_agent_enabled"])
        self.assertTrue(current["plugin_enabled"])
        self.assertTrue(current["marketplace_registered"])
        self.assertEqual(current["post_refine"], "standard")
        self.assertEqual(current["post_refine_packs"], [DEFAULT_POST_REFINE_PACK])
        self.assertEqual(current["git_commit_style"], "auto")
        self.assertTrue(current["policy_root_matches_expected"])
        self.assertTrue(current["marketplace_root_matches_expected"])

    def test_configure_claude_desktop_plugin_only_update_preserves_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path(tmp) / "plugin"
            policy_path = configure_policy(root, plugin_root, post_refine="strict")

            configure_policy(
                root,
                plugin_root,
                enabled=False,
                configure_runtime=False,
            )
            policy = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["postRefine"], "strict")

    def test_configure_claude_desktop_cli_plugin_only_does_not_create_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"

            with patch("sys.stdout", new=io.StringIO()):
                exit_code = configure_claude_desktop_main(
                    ["--root", str(root), "--enable-plugin"]
                )

            settings = json.loads(
                (root / ".claude" / "settings.local.json").read_text(encoding="utf-8")
            )
            policy_exists = (root / ".policy" / "config.json").exists()

        self.assertEqual(exit_code, 0)
        self.assertFalse(policy_exists)
        self.assertTrue(settings["enabledPlugins"][PLUGIN_ID])

    def test_configure_claude_desktop_post_refine_only_preserves_disabled_runtime(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path(tmp) / "plugin"
            policy_path = configure_policy(root, plugin_root, enabled=False, post_refine="strict")

            configure_policy(
                root,
                plugin_root,
                post_refine="off",
                configure_runtime=False,
            )
            policy = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertFalse(policy["enabled"])
        self.assertEqual(policy["postRefine"], "off")
        self.assertEqual(policy["postRefinePacks"], [])

    def test_configure_claude_desktop_cli_post_refine_only_preserves_settings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path(tmp) / "plugin"
            policy_path = configure_policy(root, plugin_root, enabled=False)
            settings_path = configure_claude_settings(root, plugin_root, "local", enabled=False)
            before_settings = settings_path.read_text(encoding="utf-8")

            with patch("sys.stdout", new=io.StringIO()):
                exit_code = configure_claude_desktop_main(
                    ["--root", str(root), "--post-refine", "standard"]
                )

            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            after_settings = settings_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertFalse(policy["enabled"])
        self.assertEqual(policy["postRefine"], "standard")
        self.assertEqual(after_settings, before_settings)

    def test_configure_claude_desktop_cli_post_refine_off_does_not_enable_new_project(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"

            with patch("sys.stdout", new=io.StringIO()):
                exit_code = configure_claude_desktop_main(
                    ["--root", str(root), "--post-refine", "off"]
                )

            policy = json.loads(
                (root / ".policy" / "config.json").read_text(encoding="utf-8")
            )
            settings_exists = (root / ".claude" / "settings.local.json").exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(policy, {"postRefine": "off", "postRefinePacks": []})
        self.assertFalse(settings_exists)

    def test_configure_claude_desktop_cli_reports_invalid_option_combinations(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"

            with patch("sys.stderr", new=io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    configure_claude_desktop_main(
                        [
                            "--root",
                            str(root),
                            "--post-refine-pack",
                            "cpp.production_refinement",
                        ]
                    )

            message = stderr.getvalue()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--post-refine-pack requires --post-refine", message)

    def test_configure_codex_enables_policy_without_claude_settings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path.cwd()

            policy_path = configure_codex_policy(root, plugin_root)
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            claude_settings_exists = (root / ".claude").exists()

        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["agents"], ["codex"])
        self.assertEqual(policy["packs"], [])
        self.assertEqual(policy["policyRoot"], str(Path.cwd()))
        self.assertEqual(policy["git"], {"commitStyle": "auto"})
        self.assertFalse(claude_settings_exists)

    def test_configure_codex_updates_stale_policy_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            old_plugin = Path(tmp) / "old-plugin"
            plugin_root = Path.cwd()
            policy = root / ".policy"
            policy.mkdir(parents=True)
            (policy / "config.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "agents": ["codex"],
                        "policyRoot": str(old_plugin),
                    }
                ),
                encoding="utf-8",
            )

            policy_path = configure_codex_policy(root, plugin_root)
            current = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertEqual(current["policyRoot"], str(plugin_root))

    def test_configure_codex_hooks_writes_project_hook_commands(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path.cwd()

            hooks_path = configure_codex_hooks(root, plugin_root)
            hooks = json.loads(hooks_path.read_text(encoding="utf-8"))

        user_prompt = hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]
        stop = hooks["hooks"]["Stop"][0]["hooks"][0]
        self.assertIn("user_prompt_submit.py", user_prompt["command"])
        self.assertIn("stop_refinement.py", stop["command"])
        self.assertNotIn("ai-policy-hook.js", user_prompt["command"])

    def test_configure_codex_hooks_preserves_existing_hooks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            hooks_dir = root / ".codex"
            hooks_dir.mkdir(parents=True)
            (hooks_dir / "hooks.json").write_text(
                json.dumps(
                    {
                        "hooks": {
                            "UserPromptSubmit": [
                                {
                                    "hooks": [
                                        {
                                            "type": "command",
                                            "command": "echo existing",
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            hooks_path = configure_codex_hooks(root, Path.cwd())
            hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
            entries = hooks["hooks"]["UserPromptSubmit"]

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["hooks"][0]["command"], "echo existing")
        self.assertIn("user_prompt_submit.py", entries[1]["hooks"][0]["command"])

    def test_configure_codex_hooks_disable_removes_only_ai_policy_hooks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            configure_codex_hooks(root, Path.cwd())
            hooks_path = root / ".codex" / "hooks.json"
            hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
            hooks["hooks"]["Stop"].insert(
                0,
                {"hooks": [{"type": "command", "command": "echo keep"}]},
            )
            hooks_path.write_text(json.dumps(hooks), encoding="utf-8")

            configure_codex_hooks(root, Path.cwd(), enabled=False)
            disabled = json.loads(hooks_path.read_text(encoding="utf-8"))

        self.assertNotIn("UserPromptSubmit", disabled["hooks"])
        self.assertEqual(disabled["hooks"]["Stop"][0]["hooks"][0]["command"], "echo keep")

    def test_configure_codex_config_enables_project_hooks_feature(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            config_path = configure_codex_config(root)

            content = config_path.read_text(encoding="utf-8")

        self.assertIn("[features]", content)
        self.assertIn("hooks = true", content)

    def test_configure_codex_config_preserves_existing_toml(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            codex = root / ".codex"
            codex.mkdir(parents=True)
            (codex / "config.toml").write_text(
                "model = \"gpt-5\"\n\n[features]\nother = true\n",
                encoding="utf-8",
            )

            config_path = configure_codex_config(root)
            content = config_path.read_text(encoding="utf-8")

        self.assertIn("model = \"gpt-5\"", content)
        self.assertIn("other = true", content)
        self.assertIn("hooks = true", content)

    def test_configure_codex_config_removes_deprecated_codex_hooks_feature(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            config = root / ".codex" / "config.toml"
            config.parent.mkdir(parents=True)
            config.write_text("[features]\ncodex_hooks = true\n", encoding="utf-8")

            config_path = configure_codex_config(root)
            content = config_path.read_text(encoding="utf-8")

        self.assertIn("hooks = true", content)
        self.assertNotIn("codex_hooks", content)

    def test_configure_codex_disable_preserves_other_agents(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            policy = root / ".policy"
            policy.mkdir(parents=True)
            (policy / "config.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "agents": ["codex", "claude"],
                        "packs": ["cpp.safe_generation"],
                    }
                ),
                encoding="utf-8",
            )

            policy_path = configure_codex_policy(root, Path.cwd(), enabled=False)
            current = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertTrue(current["enabled"])
        self.assertEqual(current["agents"], ["claude"])

    def test_configure_codex_disable_turns_runtime_off_without_other_agents(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            policy = root / ".policy"
            policy.mkdir(parents=True)
            (policy / "config.json").write_text(
                json.dumps({"enabled": True, "agents": ["codex"]}),
                encoding="utf-8",
            )

            policy_path = configure_codex_policy(root, Path.cwd(), enabled=False)
            current = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertFalse(current["enabled"])
        self.assertEqual(current["agents"], [])

    def test_configure_codex_status_reports_current_features(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path.cwd()
            configure_codex_policy(root, plugin_root)
            configure_codex_hooks(root, plugin_root)
            configure_codex_config(root)

            current = codex_status(root, plugin_root)

        self.assertTrue(current["runtime_enabled"])
        self.assertTrue(current["codex_agent_enabled"])
        self.assertTrue(current["codex_hooks_enabled"])
        self.assertTrue(current["project_hooks_present"])
        self.assertTrue(current["project_hooks_configured"])
        self.assertTrue(current["plugin_assets_present"])
        self.assertEqual(current["expected_plugin_root"], str(Path.cwd()))
        self.assertTrue(current["policy_root_matches_expected"])
        self.assertTrue(current["project_hook_runtime_roots_match_expected"])
        self.assertIn("hook_python_available", current)
        self.assertIn("hook_python_command", current)
        self.assertEqual(current["git_commit_style"], "auto")

    def test_configure_codex_status_does_not_treat_unrelated_hooks_as_configured(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            hooks_dir = root / ".codex"
            hooks_dir.mkdir(parents=True)
            (hooks_dir / "hooks.json").write_text(
                json.dumps(
                    {
                        "hooks": {
                            "UserPromptSubmit": [
                                {"hooks": [{"type": "command", "command": "echo user"}]}
                            ],
                            "Stop": [{"hooks": [{"type": "command", "command": "echo stop"}]}],
                        }
                    }
                ),
                encoding="utf-8",
            )

            current = codex_status(root, Path.cwd())

        self.assertTrue(current["project_hooks_present"])
        self.assertFalse(current["project_hooks_configured"])

    def test_configure_codex_status_reports_unavailable_hook_python(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path.cwd()
            configure_codex_policy(root, plugin_root)
            configure_codex_hooks(root, plugin_root)
            configure_codex_config(root)

            with patch.dict(os.environ, {"AI_POLICY_PYTHON": str(root / "missing-python")}, clear=False):
                current = codex_status(root, plugin_root)

        self.assertFalse(current["hook_python_available"])
        self.assertIn("missing-python", " ".join(current["hook_python_command"]))
        self.assertTrue(current["hook_python_error"])

    def test_configure_codex_cli_updates_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"

            with patch("sys.stdout", new=io.StringIO()):
                exit_code = configure_codex_main(["--root", str(root)])

            policy = json.loads(
                (root / ".policy" / "config.json").read_text(encoding="utf-8")
            )
            hooks = json.loads((root / ".codex" / "hooks.json").read_text(encoding="utf-8"))
            codex_config = (root / ".codex" / "config.toml").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(policy["agents"], ["codex"])
        self.assertIn("UserPromptSubmit", hooks["hooks"])
        self.assertIn("hooks = true", codex_config)

    def test_configure_opencode_writes_policy_and_config(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path.cwd()

            policy_path = configure_opencode_policy(root, plugin_root)
            config_path = configure_opencode_config(root)
            plugin_path = configure_opencode_plugin(root, plugin_root)
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            plugin = plugin_path.read_text(encoding="utf-8")

        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["agents"], ["opencode"])
        self.assertEqual(policy["packs"], [])
        self.assertEqual(policy["policyRoot"], str(plugin_root))
        self.assertEqual(policy["git"], {"commitStyle": "auto"})
        self.assertEqual(config["$schema"], "https://opencode.ai/config.json")
        self.assertEqual(config["instructions"], ["AGENTS.md"])
        self.assertIn("opencode-user-prompt-submit", plugin)
        self.assertIn("opencode-plugin-state.json", plugin)
        self.assertIn("opencode-post-refine-prompt.md", plugin)
        self.assertIn(str(plugin_root).replace("\\", "\\\\"), plugin)

    def test_configure_opencode_preserves_existing_instructions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            config = root / "opencode.json"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps({"instructions": ["README.md"], "model": "anthropic/claude"}),
                encoding="utf-8",
            )

            config_path = configure_opencode_config(root)
            current = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(current["instructions"], ["README.md", "AGENTS.md"])
        self.assertEqual(current["model"], "anthropic/claude")

    def test_configure_opencode_disable_preserves_other_agents(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            policy = root / ".policy"
            policy.mkdir(parents=True)
            (policy / "config.json").write_text(
                json.dumps({"enabled": True, "agents": ["opencode", "codex"]}),
                encoding="utf-8",
            )

            policy_path = configure_opencode_policy(root, Path.cwd(), enabled=False)
            current = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertTrue(current["enabled"])
        self.assertEqual(current["agents"], ["codex"])

    def test_configure_opencode_status_reports_current_features(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path.cwd()
            configure_opencode_policy(root, plugin_root)
            configure_opencode_config(root)
            configure_opencode_plugin(root, plugin_root)

            current = opencode_status(root, plugin_root)

        self.assertTrue(current["runtime_enabled"])
        self.assertTrue(current["opencode_agent_enabled"])
        self.assertTrue(current["opencode_config_present"])
        self.assertTrue(current["agents_instruction_configured"])
        self.assertTrue(current["project_plugin_present"])
        self.assertTrue(current["project_plugin_configured"])
        self.assertFalse(current["project_plugin_state_present"])
        self.assertFalse(current["project_post_refine_prompt_present"])
        self.assertTrue(current["project_plugin_runtime_root_matches_expected"])
        self.assertTrue(current["policy_root_matches_expected"])
        self.assertEqual(current["git_commit_style"], "auto")

    def test_configure_opencode_cli_updates_policy_and_config(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"

            with patch("sys.stdout", new=io.StringIO()):
                exit_code = configure_opencode_main(["--root", str(root)])

            policy = json.loads(
                (root / ".policy" / "config.json").read_text(encoding="utf-8")
            )
            config = json.loads((root / "opencode.json").read_text(encoding="utf-8"))
            plugin = root / ".opencode" / "plugins" / "ai-policy-runtime.js"
            plugin_exists = plugin.exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(policy["agents"], ["opencode"])
        self.assertEqual(config["instructions"], ["AGENTS.md"])
        self.assertTrue(plugin_exists)

    def test_clean_workspace_removes_only_ai_policy_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path.cwd()
            configure_codex_policy(root, plugin_root)
            hooks_path = configure_codex_hooks(root, plugin_root)
            codex_config = configure_codex_config(root)
            opencode_config = configure_opencode_config(root)
            opencode_plugin = configure_opencode_plugin(root, plugin_root)
            claude_settings = configure_claude_settings(root, plugin_root, "local")
            current = root / ".policy" / "current"
            current.mkdir(parents=True)
            (current / "agent-hook-state.json").write_text("{}", encoding="utf-8")
            opencode_state = current / "opencode-plugin-state.json"
            opencode_state.write_text("{}", encoding="utf-8")
            opencode_prompt = current / "opencode-post-refine-prompt.md"
            opencode_prompt.write_text("Refine once.\n", encoding="utf-8")

            hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
            hooks["hooks"]["Stop"].append(
                {"hooks": [{"type": "command", "command": "echo keep"}]}
            )
            hooks_path.write_text(json.dumps(hooks), encoding="utf-8")

            result = clean_workspace(root)
            cleaned_hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
            cleaned_codex_config = codex_config.read_text(encoding="utf-8")
            cleaned_opencode = json.loads(opencode_config.read_text(encoding="utf-8"))
            cleaned_claude = json.loads(claude_settings.read_text(encoding="utf-8"))

        self.assertIn(str(root / ".policy" / "config.json"), result["removed"])
        self.assertIn(str(root / ".policy" / "current"), result["removed"])
        self.assertIn(str(opencode_state), result["removed"])
        self.assertIn(str(opencode_prompt), result["removed"])
        self.assertIn(str(opencode_plugin), result["removed"])
        self.assertNotIn("UserPromptSubmit", cleaned_hooks["hooks"])
        self.assertEqual(
            cleaned_hooks["hooks"]["Stop"][0]["hooks"][0]["command"],
            "echo keep",
        )
        self.assertIn("hooks = true", cleaned_codex_config)
        self.assertNotIn("instructions", cleaned_opencode)
        self.assertNotIn(PLUGIN_ID, cleaned_claude.get("enabledPlugins", {}))
        self.assertNotIn("ai-policy-runtime", cleaned_claude.get("extraKnownMarketplaces", {}))

    def test_clean_workspace_disables_codex_hooks_when_no_hooks_remain(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            plugin_root = Path.cwd()
            configure_codex_hooks(root, plugin_root)
            codex_config = configure_codex_config(root)

            clean_workspace(root)
            cleaned_codex_config = codex_config.read_text(encoding="utf-8")

        self.assertIn("hooks = false", cleaned_codex_config)

    def test_cli_cleanup_removes_project_config_and_current_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            config = root / ".policy" / "config.json"
            current = root / ".policy" / "current"
            current.mkdir(parents=True)
            config.write_text('{"enabled": true}\n', encoding="utf-8")
            (current / "trace.json").write_text("{}", encoding="utf-8")

            output, exit_code = CommandDispatcher().dispatch(
                argparse.Namespace(
                    command="cleanup",
                    root=str(root),
                    keep_current=False,
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertIsInstance(output, dict)
        self.assertFalse(config.exists())
        self.assertFalse(current.exists())

    def test_npm_package_exposes_ai_policy_commands(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(package["bin"]["ai-policy"], "bin/ai-policy.js")
        self.assertNotIn("ai-policy-runtime", package["bin"])
        self.assertIn(".codex-plugin/*.json", package["files"])
        self.assertIn(".claude-plugin/*.json", package["files"])
        self.assertIn("hooks/*.json", package["files"])
        self.assertIn("hooks/*.js", package["files"])
        self.assertIn("hooks/*.py", package["files"])
        self.assertIn("docs/reference/**/*.yaml", package["files"])
        self.assertIn("skills/**/*.yaml", package["files"])
        self.assertIn("packs/*.yaml", package["files"])

    def test_plugin_metadata_matches_package_version(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))
        claude_plugin = json.loads(
            (Path(".claude-plugin") / "plugin.json").read_text(encoding="utf-8")
        )
        codex_plugin = json.loads(
            (Path(".codex-plugin") / "plugin.json").read_text(encoding="utf-8")
        )
        marketplace = json.loads(
            (Path(".claude-plugin") / "marketplace.json").read_text(encoding="utf-8")
        )
        marketplace_plugin = marketplace["plugins"][0]

        self.assertEqual(claude_plugin["version"], package["version"])
        self.assertEqual(codex_plugin["version"], package["version"])
        self.assertEqual(marketplace_plugin["version"], package["version"])
        self.assertEqual(claude_plugin["description"], marketplace_plugin["description"])

    def test_vscode_embedding_provider_does_not_offer_disabled_mode(self) -> None:
        package = json.loads(
            (Path("vscode-extension") / "package.json").read_text(encoding="utf-8")
        )
        enum = package["contributes"]["configuration"]["properties"][
            "aiPolicy.embeddingProvider"
        ]["enum"]

        self.assertEqual(enum, ["", "openai-compatible", "local"])

    def test_npm_pack_does_not_include_python_bytecode(self) -> None:
        npm = shutil.which("npm")
        if npm is None:
            self.skipTest("npm is not available")

        with TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [npm, "pack", "--dry-run"],
                check=True,
                capture_output=True,
                text=True,
                env=_npm_test_env(npm_config_cache=str(Path(tmp) / "npm-cache")),
            )
        output = completed.stdout + completed.stderr

        self.assertNotIn("__pycache__", output)
        self.assertNotRegex(output, r"\.pyc\b")

    def test_claude_hooks_use_packaged_node_wrapper(self) -> None:
        hooks = json.loads((Path("hooks") / "hooks.json").read_text(encoding="utf-8"))

        user_prompt = hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]
        stop = hooks["hooks"]["Stop"][0]["hooks"][0]

        self.assertEqual(user_prompt["command"], "node")
        self.assertEqual(
            user_prompt["args"],
            ["${CLAUDE_PLUGIN_ROOT}/bin/ai-policy-hook.js", "claude-user-prompt-submit"],
        )
        self.assertEqual(user_prompt["timeout"], 120)
        self.assertEqual(stop["command"], "node")
        self.assertEqual(
            stop["args"],
            ["${CLAUDE_PLUGIN_ROOT}/bin/ai-policy-hook.js", "claude-stop-refinement"],
        )
        self.assertEqual(stop["timeout"], 120)

    def test_ai_policy_status_command_uses_installed_package_root(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available")

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            env = {
                **os.environ,
                "AI_POLICY_PYTHON": sys.executable,
            }
            completed = subprocess.run(
                [
                    "node",
                    "bin/ai-policy.js",
                    "status",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            current = json.loads(completed.stdout)

        self.assertEqual(current["expected_plugin_root"], str(_node_package_root()))
        self.assertFalse(current["runtime_enabled"])
        self.assertFalse((root / ".policy" / "config.json").exists())

    def test_ai_policy_configure_codex_command_updates_policy_and_hooks(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available")

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            env = {
                **os.environ,
                "AI_POLICY_PYTHON": sys.executable,
            }
            subprocess.run(
                [
                    "node",
                    "bin/ai-policy.js",
                    "configure",
                    "codex",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            policy = json.loads(
                (root / ".policy" / "config.json").read_text(encoding="utf-8")
            )
            hooks = json.loads((root / ".codex" / "hooks.json").read_text(encoding="utf-8"))
            codex_config = (root / ".codex" / "config.toml").read_text(encoding="utf-8")
            claude_settings_exists = (root / ".claude").exists()

        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["agents"], ["codex"])
        self.assertIn("UserPromptSubmit", hooks["hooks"])
        self.assertIn("hooks = true", codex_config)
        self.assertFalse(claude_settings_exists)

    def test_ai_policy_configure_opencode_command_updates_policy_and_config(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available")

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            env = {
                **os.environ,
                "AI_POLICY_PYTHON": sys.executable,
            }
            subprocess.run(
                [
                    "node",
                    "bin/ai-policy.js",
                    "configure",
                    "opencode",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            policy = json.loads(
                (root / ".policy" / "config.json").read_text(encoding="utf-8")
            )
            config = json.loads((root / "opencode.json").read_text(encoding="utf-8"))

        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["agents"], ["opencode"])
        self.assertEqual(config["instructions"], ["AGENTS.md"])

    def test_ai_policy_embedding_configure_writes_project_config(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available")

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            env = {
                **os.environ,
                "AI_POLICY_PYTHON": sys.executable,
            }
            completed = subprocess.run(
                [
                    "node",
                    "bin/ai-policy.js",
                    "embedding",
                    "configure",
                    "--root",
                    str(root),
                    "--provider",
                    "openai-compatible",
                    "--base-url",
                    "https://embedding.example.test/v1",
                    "--api-key",
                    "project-key",
                    "--model",
                    "embedding-model",
                    "--timeout",
                    "12.5",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            output = json.loads(completed.stdout)
            policy = json.loads(
                (root / ".policy" / "config.json").read_text(encoding="utf-8")
            )

        self.assertEqual(output["embedding"]["provider"], "openai-compatible")
        self.assertEqual(policy["embeddingProvider"], "openai-compatible")
        self.assertEqual(policy["embeddingBaseUrl"], "https://embedding.example.test/v1")
        self.assertEqual(policy["embeddingApiKey"], "project-key")
        self.assertEqual(policy["embeddingModel"], "embedding-model")
        self.assertEqual(policy["embeddingTimeout"], "12.5")

    def test_ai_policy_disable_clears_claude_runtime_state(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available")

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            env = {
                **os.environ,
                "AI_POLICY_PYTHON": sys.executable,
            }
            subprocess.run(
                [
                    "node",
                    "bin/ai-policy.js",
                    "configure",
                    "claude",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            completed = subprocess.run(
                [
                    "node",
                    "bin/ai-policy.js",
                    "disable",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            policy = json.loads(
                (root / ".policy" / "config.json").read_text(encoding="utf-8")
            )

        self.assertEqual(completed.returncode, 0)
        self.assertFalse(policy["enabled"])
        self.assertNotIn("claude", policy["agents"])

    def test_runtime_from_args_reads_project_policy_and_embedding_config(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            policy_root = Path(tmp) / "policy-assets"
            config = root / ".policy" / "config.json"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps(
                    {
                        "policyRoot": str(policy_root),
                        "embeddingProvider": "openai_compatible",
                        "embeddingBaseUrl": "https://embedding.example.test/v1",
                        "embeddingApiKey": "project-key",
                        "embeddingModel": "embedding-model",
                        "embeddingTimeout": "12.5",
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                root=str(root),
                policy_root=None,
                skills="skills",
                packs="packs",
            )

            runtime = _runtime_from_args(args)

        self.assertEqual(runtime.config.policy_root, policy_root)
        self.assertIsNotNone(runtime.config.embedding)
        assert runtime.config.embedding is not None
        self.assertEqual(runtime.config.embedding.provider, "openai-compatible")
        self.assertEqual(
            runtime.config.embedding.base_url,
            "https://embedding.example.test/v1",
        )
        self.assertEqual(runtime.config.embedding.api_key, "project-key")
        self.assertEqual(runtime.config.embedding.model, "embedding-model")
        self.assertEqual(runtime.config.embedding.timeout_seconds, 12.5)

    def test_runtime_from_args_resolves_project_local_model_from_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            config = root / ".policy" / "config.json"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps(
                    {
                        "embeddingProvider": "LOCAL",
                        "embeddingModel": "models/custom-model",
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                root=str(root),
                policy_root=None,
                skills="skills",
                packs="packs",
            )

            runtime = _runtime_from_args(args)

        self.assertIsNotNone(runtime.config.embedding)
        assert runtime.config.embedding is not None
        self.assertEqual(runtime.config.embedding.provider, "local")
        self.assertEqual(runtime.config.embedding.model, str(root / "models" / "custom-model"))

    def test_cli_embedding_configure_local_can_install_default_model(self) -> None:
        args = argparse.Namespace(
            command="embedding",
            action="configure",
            root="",
            provider="local",
            base_url=None,
            api_key=None,
            model=None,
            timeout=None,
            policy_root=None,
            install=True,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            args.root = str(root)
            with patch(
                "ai_policy_runtime.services.local_models._snapshot_download"
            ) as download:
                output, exit_code = CommandDispatcher().dispatch(args)

            policy = json.loads(
                (root / ".policy" / "config.json").read_text(encoding="utf-8")
            )

        download.assert_called_once()
        self.assertEqual(exit_code, 0)
        self.assertEqual(output["embedding"]["provider"], "local")
        self.assertEqual(policy["embeddingProvider"], "local")
        self.assertTrue(policy["embeddingModel"].endswith("paraphrase-multilingual-MiniLM-L12-v2"))
        self.assertEqual(output["installed_model"]["key"], "multilingual-mini")

    def test_cli_embedding_status_reports_local_model_installation_state(self) -> None:
        args = argparse.Namespace(
            command="embedding",
            action="status",
            root="",
            provider=None,
            base_url=None,
            api_key=None,
            model=None,
            timeout=None,
            policy_root=None,
            install=False,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            model = root / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
            model.mkdir(parents=True)
            args.root = str(root)

            output, exit_code = CommandDispatcher().dispatch(args)

        self.assertEqual(exit_code, 0)
        self.assertTrue(output["embedding"]["local_models"][0]["installed"])

    def test_embedding_health_forced_local_ignores_remote_environment(self) -> None:
        args = argparse.Namespace(
            command="embedding",
            action="status",
            root="",
            provider=None,
            base_url=None,
            api_key=None,
            model=None,
            timeout=None,
            policy_root=None,
            install=False,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            config = root / ".policy" / "config.json"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps({"embeddingProvider": "local"}),
                encoding="utf-8",
            )
            args.root = str(root)
            with patch.dict(
                os.environ,
                {
                    "AI_POLICY_EMBEDDING_BASE_URL": "https://embedding.example.test/v1",
                    "AI_POLICY_EMBEDDING_API_KEY": "key",
                },
                clear=True,
            ):
                output, exit_code = CommandDispatcher().dispatch(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(output["embedding"]["provider"], "local")
        self.assertFalse(output["embedding"]["ok"])
        self.assertTrue(output["embedding"]["remote_configured"])

    def test_embedding_health_auto_accepts_remote_environment(self) -> None:
        args = argparse.Namespace(
            command="embedding",
            action="status",
            root="",
            provider=None,
            base_url=None,
            api_key=None,
            model=None,
            timeout=None,
            policy_root=None,
            install=False,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            args.root = str(root)
            with patch.dict(
                os.environ,
                {"AI_POLICY_EMBEDDING_API_KEY": "key"},
                clear=True,
            ):
                output, exit_code = CommandDispatcher().dispatch(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(output["embedding"]["provider"], "auto")
        self.assertTrue(output["embedding"]["ok"])
        self.assertTrue(output["embedding"]["remote_configured"])

    def test_cli_embedding_test_runs_provider_probe(self) -> None:
        args = argparse.Namespace(
            command="embedding",
            action="test",
            root="",
            provider=None,
            base_url=None,
            api_key=None,
            model=None,
            timeout=None,
            policy_root=None,
            install=False,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            args.root = str(root)
            with patch.dict(
                os.environ,
                {"AI_POLICY_EMBEDDING_API_KEY": "key"},
                clear=True,
            ), patch(
                "ai_policy_runtime.services.embedding_health.default_embedding_provider",
                return_value=FakeEmbeddingProvider(),
            ) as provider:
                output, exit_code = CommandDispatcher().dispatch(args)

        self.assertEqual(exit_code, 0)
        provider.assert_called_once()
        self.assertTrue(output["embedding"]["probe_ok"])
        self.assertGreater(output["embedding"]["vector_dimensions"], 0)

    def test_cli_embedding_test_reports_probe_failure(self) -> None:
        args = argparse.Namespace(
            command="embedding",
            action="test",
            root="",
            provider=None,
            base_url=None,
            api_key=None,
            model=None,
            timeout=None,
            policy_root=None,
            install=False,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            config = root / ".policy" / "config.json"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps(
                    {
                        "embeddingProvider": "local",
                        "embeddingModel": "missing-model",
                    }
                ),
                encoding="utf-8",
            )
            args.root = str(root)

            output, exit_code = CommandDispatcher().dispatch(args)

        self.assertEqual(exit_code, 1)
        self.assertFalse(output["embedding"]["probe_ok"])
        self.assertIn("ai-policy embedding configure", output["embedding"]["probe_error"])

    def test_embedding_status_resolves_relative_local_model_from_project_root(self) -> None:
        args = argparse.Namespace(
            command="embedding",
            action="status",
            root="",
            provider=None,
            base_url=None,
            api_key=None,
            model=None,
            timeout=None,
            policy_root=None,
            install=False,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            config = root / ".policy" / "config.json"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps(
                    {
                        "embeddingProvider": "local",
                        "embeddingModel": "models/custom-model",
                    }
                ),
                encoding="utf-8",
            )
            args.root = str(root)
            with patch(
                "ai_policy_runtime.services.embedding_health.check_sentence_transformer_model",
                return_value={"usable": True, "error": None, "next_step": None},
            ) as check:
                output, exit_code = CommandDispatcher().dispatch(args)

        self.assertEqual(exit_code, 0)
        check.assert_called_once_with(root / "models" / "custom-model")
        self.assertTrue(output["embedding"]["ok"])

    def test_forced_local_embedding_prefers_explicit_model_over_default_install(self) -> None:
        with TemporaryDirectory() as tmp:
            policy_root = Path(tmp) / "policy"
            default_model = policy_root / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
            default_model.mkdir(parents=True)
            provider = object()

            with patch.dict(
                os.environ,
                {
                    "AI_POLICY_EMBEDDING_PROVIDER": "local",
                    "AI_POLICY_EMBEDDING_MODEL": "models/custom-model",
                },
                clear=True,
            ), patch(
                "ai_policy_runtime.task_analysis.analyzer.SentenceTransformerEmbeddingProvider",
                return_value=provider,
            ) as sentence_transformer:
                selected = default_embedding_provider(policy_root)

        self.assertIs(selected, provider)
        sentence_transformer.assert_called_once_with("models/custom-model")

    def test_ai_policy_status_codex_command_is_read_only(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available")

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            env = {
                **os.environ,
                "AI_POLICY_PYTHON": sys.executable,
            }
            completed = subprocess.run(
                [
                    "node",
                    "bin/ai-policy.js",
                    "status",
                    "--agent",
                    "codex",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            current = json.loads(completed.stdout)
            policy_exists = (root / ".policy" / "config.json").exists()

        self.assertFalse(current["runtime_enabled"])
        self.assertFalse(current["codex_agent_enabled"])
        self.assertTrue(current["plugin_assets_present"])
        self.assertFalse(policy_exists)

    def test_ai_policy_doctor_reports_runtime_health(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available")

        env = {
            **os.environ,
            "AI_POLICY_PYTHON": sys.executable,
            "AI_POLICY_EMBEDDING_PROVIDER": "openai-compatible",
            "AI_POLICY_EMBEDDING_API_KEY": "test-key",
            "AI_POLICY_EMBEDDING_MODEL": "test-embedding-model",
        }
        completed = subprocess.run(
            ["node", "bin/ai-policy.js", "doctor"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        current = json.loads(completed.stdout)

        self.assertTrue(current["ok"])
        self.assertEqual(current["packageRoot"], str(_node_package_root()))
        self.assertTrue(current["usingExplicitPython"])
        self.assertEqual(
            Path(current["venvPython"]).parents[1],
            Path(current["stateDir"]) / "venv",
        )
        self.assertTrue(current["checks"]["claudePlugin"])
        self.assertTrue(current["checks"]["skills"])

    def test_npm_tarball_install_exposes_ai_policy_command(self) -> None:
        npm = shutil.which("npm")
        if npm is None:
            self.skipTest("npm is not available")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            npm_env = {
                **_npm_test_env(
                    AI_POLICY_PYTHON=sys.executable,
                    npm_config_cache=str(root / "npm-cache"),
                )
            }
            pack = subprocess.run(
                [npm, "pack", "--silent"],
                check=True,
                capture_output=True,
                text=True,
                env=npm_env,
            )
            tarball_name = pack.stdout.strip().splitlines()[-1]
            tarball = Path.cwd() / tarball_name
            self.addCleanup(lambda: tarball.exists() and tarball.unlink())
            prefix = root / "prefix"
            project = root / "project"
            subprocess.run(
                [npm, "install", "--prefix", str(prefix), "-g", str(tarball)],
                check=True,
                capture_output=True,
                text=True,
                env=npm_env,
            )
            command = (
                prefix / "ai-policy.cmd"
                if sys.platform.startswith("win")
                else prefix / "bin" / "ai-policy"
            )
            completed = subprocess.run(
                [str(command), "status", "--root", str(project)],
                check=True,
                capture_output=True,
                text=True,
                env=npm_env,
            )
            current = json.loads(completed.stdout)
            plugin_root_exists = Path(current["expected_plugin_root"]).exists()
            policy_exists = (project / ".policy" / "config.json").exists()

        self.assertTrue(plugin_root_exists)
        self.assertFalse(current["runtime_enabled"])
        self.assertFalse(policy_exists)

    def test_post_refinement_task_preserves_scope_and_behavior(self) -> None:
        task = build_post_refinement_task("Refactor the matching engine API.", "standard")

        self.assertIn("Preserve observable behavior", task)
        self.assertIn("remove accidental complexity", task)
        self.assertIn("Do not broaden scope", task)
        self.assertIn("Refactor the matching engine API.", task)

    def test_post_refinement_pack_merge_preserves_order_and_deduplicates(self) -> None:
        pack_ids = merge_pack_ids(
            ("cpp.low_latency", "cpp.production_refinement"),
            ("cpp.production_refinement", "cpp.safe_generation"),
        )

        self.assertEqual(
            pack_ids,
            ("cpp.low_latency", "cpp.production_refinement", "cpp.safe_generation"),
        )

    def test_effective_rules_renderer_matches_output_spec(self) -> None:
        skill = Skill.from_mapping(
            {
                "skill": {
                    "id": "cpp.render",
                    "name": "Render",
                    "version": "1.0.0",
                    "level": "domain",
                    "domain": "cpp",
                    "priority": 10,
                    "activation": {"when": {"language": "cpp"}},
                    "capabilities": ["code_generation"],
                },
                "rules": {
                    "hard": [
                        {
                            "id": "no_ub",
                            "must_not": "undefined_behavior",
                            "target": "undefined_behavior",
                            "reason": "Avoid undefined behavior.",
                        }
                    ],
                    "preference": [
                        {
                            "id": "prefer_safety",
                            "prefer": "safety",
                            "over": "performance",
                            "target": "decision_priority",
                        }
                    ],
                },
            }
        )
        task = TaskContext(
            domain="cpp",
            task_type="write_code",
            capabilities=("code_generation",),
            context={"language": "cpp", "standard": 20},
        )
        rules = PolicyEngine(SkillRegistry([skill])).evaluate(task)
        rendered = EffectiveRulesRenderer().to_mapping(
            task=task,
            task_id="task_test",
            summary="Render test",
            rules=rules,
            trace={"active_skills": ["cpp.render"]},
        )
        effective = rendered["effective_rules"]

        self.assertEqual(effective["schema_version"], 1)
        self.assertIn("task", effective)
        self.assertIn("preference", effective)
        self.assertEqual(effective["hard"][0]["statement"], "Avoid undefined behavior.")
        self.assertEqual(effective["hard"][0]["source"]["skill"], "cpp.render")
        self.assertEqual(effective["preference"][0]["prefer"], "safety")

    def test_effective_prompt_keeps_bullets_on_separate_lines(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Refactor this C++20 code so it is not just working. "
            "Reduce complexity and preserve safety.",
            ("cpp.production_refinement",),
        )
        prompt = (result.current / "effective-prompt.md").read_text(encoding="utf-8")

        self.assertNotIn(".- ", prompt)
        self.assertNotIn("Semantic Skill Matches", prompt)
        self.assertIn("Preserve the existing observable behavior", prompt)
        self.assertIn("Avoid undefined behavior.", prompt)
        self.assertIn("Group related state, helper functions, and behavior", prompt)
        self.assertIn("Verify behavior preservation.", prompt)
        self.assertIn(
            "Verify no new ownership, lifetime, resource, bounds, or "
            "undefined-behavior risks were introduced.",
            prompt,
        )
        self.assertIn(
            "Verify the refactoring reduced accidental complexity without "
            "introducing over-abstraction.",
            prompt,
        )
        self.assertNotIn("Verify: Do not use unchecked bounds access", prompt)
        self.assertLessEqual(_section_bullet_count(prompt, "Verification Requirements"), 5)
        self.assertLessEqual(_section_bullet_count(prompt, "HARD Rules"), 8)
        self.assertLessEqual(_section_bullet_count(prompt, "SOFT Rules"), 12)

        detailed_checks = [
            item["statement"]
            for item in result.structured["effective_rules"]["verification"]["required"]
        ]
        self.assertTrue(
            any("Do not use unchecked bounds access" in item for item in detailed_checks)
        )
        self.assertTrue(
            any("Preserve the existing observable behavior" in item for item in detailed_checks)
        )

    def test_generic_refinement_prompt_omits_cpp_standard_verification(self) -> None:
        task = TaskContext(
            domain="general",
            task_type="improve_code_quality",
            capabilities=("code_review", "refactor_code"),
            tags=("code-quality", "complexity", "refactoring"),
            context={
                "artifact_type": "code",
                "refinement_requested": True,
                "behavior_preservation_required": True,
                "duplicated_logic": True,
            },
        )
        rules = PolicyEngine(SkillRegistry.from_dirs("skills", "packs")).evaluate(task)
        rendered = EffectiveRulesRenderer().to_mapping(
            task=task,
            task_id="task_test",
            summary="Generic refinement",
            rules=rules,
            trace={"active_skills": []},
        )
        prompt = EffectiveRulesRenderer().to_prompt(rendered)

        self.assertIn("Preserve the existing observable behavior", prompt)
        self.assertIn("Extract duplicated logic", prompt)
        self.assertIn("Prefer shared responsibility abstraction", prompt)
        self.assertIn("Prefer finished component", prompt)
        self.assertNotIn("selected C++ standard", prompt)
        self.assertNotIn("Prefer clear call chain", prompt)
        self.assertLessEqual(_section_bullet_count(prompt, "SOFT Rules"), 9)
        self.assertLessEqual(_section_bullet_count(prompt, "Preferences"), 4)

    def test_prompt_quality_eval_set(self) -> None:
        fixture = _load_fixture("prompt_quality_eval.yaml")
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=FakeEmbeddingProvider(),
        )
        registry = SkillRegistry.from_dirs("skills", "packs")
        renderer = EffectiveRulesRenderer()

        for case in fixture["cases"]:
            with self.subTest(case=case["id"]):
                analysis = analyzer.analyze(str(case["prompt"]))
                effective = PolicyEngine(registry).evaluate(analysis.task)
                if case.get("applicable", True) is False:
                    self.assertFalse(analysis.activation_ready)
                    self.assertFalse(_has_policy_content_for_test(effective))
                    continue
                self.assertTrue(analysis.activation_ready, analysis.to_dict())
                self.assertTrue(_has_policy_content_for_test(effective))
                rendered = renderer.to_mapping(
                    task=analysis.task,
                    task_id=str(case["id"]),
                    summary=str(case["prompt"]),
                    rules=effective,
                    trace={"active_skills": []},
                )
                prompt = renderer.to_prompt(rendered)

                for text in case.get("include", ()):
                    self.assertIn(text, prompt)
                for text in case.get("exclude", ()):
                    self.assertNotIn(text, prompt)
                if "max_hard_rules" in case:
                    self.assertLessEqual(
                        _section_bullet_count(prompt, "HARD Rules"),
                        case["max_hard_rules"],
                    )
                if "max_soft_rules" in case:
                    self.assertLessEqual(
                        _section_bullet_count(prompt, "SOFT Rules"),
                        case["max_soft_rules"],
                    )
                if "max_preferences" in case:
                    self.assertLessEqual(
                        _section_bullet_count(prompt, "Preferences"),
                        case["max_preferences"],
                    )
                if "max_verification" in case:
                    self.assertLessEqual(
                        _section_bullet_count(prompt, "Verification Requirements"),
                        case["max_verification"],
                    )

    def test_python_prompt_quality_eval_set(self) -> None:
        fixture = _load_fixture("python_prompt_quality_eval.yaml")
        analyzer = TaskAnalyzer.from_skills_dir(
            "skills",
            embeddings=FakeEmbeddingProvider(),
        )
        registry = SkillRegistry.from_dirs("skills", "packs")
        renderer = EffectiveRulesRenderer()

        for case in fixture["cases"]:
            with self.subTest(case=case["id"]):
                analysis = analyzer.analyze(str(case["prompt"]))
                effective = PolicyEngine(registry).evaluate(
                    analysis.task,
                    ("python.best_practices",),
                )
                if case.get("applicable", True) is False:
                    self.assertFalse(analysis.activation_ready)
                    self.assertFalse(_has_policy_content_for_test(effective))
                    continue
                self.assertTrue(analysis.activation_ready, analysis.to_dict())
                self.assertTrue(_has_policy_content_for_test(effective))
                rendered = renderer.to_mapping(
                    task=analysis.task,
                    task_id=str(case["id"]),
                    summary=str(case["prompt"]),
                    rules=effective,
                    trace={"active_skills": [], "packs": ["python.best_practices"]},
                )
                prompt = renderer.to_prompt(rendered)

                for text in case.get("include", ()):
                    self.assertIn(text, prompt)
                for text in case.get("exclude", ()):
                    self.assertNotIn(text, prompt)
                if "max_hard_rules" in case:
                    self.assertLessEqual(
                        _section_bullet_count(prompt, "HARD Rules"),
                        case["max_hard_rules"],
                    )
                if "max_soft_rules" in case:
                    self.assertLessEqual(
                        _section_bullet_count(prompt, "SOFT Rules"),
                        case["max_soft_rules"],
                    )
                if "max_preferences" in case:
                    self.assertLessEqual(
                        _section_bullet_count(prompt, "Preferences"),
                        case["max_preferences"],
                    )
                if "max_verification" in case:
                    self.assertLessEqual(
                        _section_bullet_count(prompt, "Verification Requirements"),
                        case["max_verification"],
                    )

    def test_effective_rules_schema_validator_reports_missing_field(self) -> None:
        diagnostics = validate_effective_rules_mapping(
            {"effective_rules": {"schema_version": 1}},
            "inline",
        )

        self.assertTrue(diagnostics)

    def test_cpp17_string_view_fixture_resolves_version_safe_rules(self) -> None:
        fixture = _load_fixture("cpp17_string_view_task.yaml")
        effective = _resolve_fixture(fixture)
        statements = _statements(effective)
        sources = _sources(effective)

        self.assertIn("cpp.standard.cpp17.best_practices", sources)
        self.assertIn("cpp.standard.standard_availability", sources)
        self.assertTrue(_has_statement_containing(effective, "std::string_view"))
        self.assertTrue(_has_statement_containing(effective, "C++20-only facilities"))
        self.assertFalse(_has_statement_containing(effective, "std::span"))
        self.assertFalse(_has_statement_containing(effective, "C++20 concepts"))
        self.assertFalse(_has_statement_containing(effective, "std::jthread"))

    def test_cpp20_span_fixture_resolves_contiguous_range_rules(self) -> None:
        fixture = _load_fixture("cpp20_span_task.yaml")
        effective = _resolve_fixture(fixture)
        statements = _statements(effective)
        sources = _sources(effective)

        self.assertIn("cpp.standard.cpp17.best_practices", sources)
        self.assertIn("cpp.standard.cpp20.best_practices", sources)
        self.assertIn("cpp.standard.standard_availability", sources)
        self.assertTrue(_has_statement_containing(effective, "std::span"))
        self.assertTrue(_has_statement_containing(effective, "unavailable in the selected C++ standard"))
        self.assertFalse(_has_statement_containing(effective, "std::string_view"))

    def test_cpp20_low_latency_fixture_keeps_safety_above_performance(self) -> None:
        fixture = _load_fixture("cpp20_low_latency_task.yaml")
        effective = _resolve_fixture(fixture)
        statements = _statements(effective)
        sources = _sources(effective)

        self.assertIn("cpp.safety.undefined_behavior", sources)
        self.assertIn("cpp.performance.hot_path", sources)
        self.assertIn("cpp.performance.allocation_control", sources)
        self.assertTrue(_has_statement_containing(effective, "std::span"))
        self.assertIn("safety > performance", statements)
        self.assertIn("performance > readability", statements)

    def test_cpp_api_design_fixture_resolves_interface_rules(self) -> None:
        fixture = _load_fixture("cpp_api_design_task.yaml")
        effective = _resolve_fixture(fixture)
        sources = _sources(effective)

        self.assertIn("cpp.api_design.interface_intent", sources)
        self.assertIn("cpp.api_design.parameter_passing", sources)
        self.assertIn("cpp.api_design.ownership_in_interfaces", sources)

    def test_cpp_review_lifetime_fixture_resolves_review_safety_rules(self) -> None:
        fixture = _load_fixture("cpp_review_lifetime_task.yaml")
        effective = _resolve_fixture(fixture)
        sources = _sources(effective)

        self.assertIn("cpp.safety.ownership_and_lifetime", sources)
        self.assertIn("cpp.resource_management.raii", sources)
        self.assertIn("cpp.safety.undefined_behavior", sources)
        self.assertTrue(effective["verification"]["required"])

    def test_cpp20_template_constraints_prefer_concepts(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Write a C++20 generic template API with explicit template constraints.",
            ("cpp.modernization",),
        )
        effective = result.structured["effective_rules"]
        statements = _statements(effective)

        self.assertIn(
            "Prefer C++20 concepts and requires-clauses over SFINAE or std::enable_if "
            "when template constraints are part of the public interface.",
            statements,
        )
        self.assertIn(
            "Avoid exposing unconstrained template interfaces when the valid argument set "
            "has meaningful semantic requirements.",
            statements,
        )
        self.assertIn(
            "Structure generic constraints so invalid arguments fail with actionable "
            "diagnostics near the template interface.",
            statements,
        )
        self.assertIn("named_semantic_concept > repeated_ad_hoc_requires_expression", statements)

    def test_cpp17_template_constraints_use_readable_sfinae_fallback(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Write a C++17 generic template API with explicit template constraints.",
            ("cpp.modernization",),
        )
        statements = _statements(result.structured["effective_rules"])

        self.assertIn(
            "Use readable SFINAE, std::enable_if, or type traits when template "
            "constraints are required in pre-C++20 code.",
            statements,
        )
        self.assertIn(
            "Avoid exposing unconstrained template interfaces when the valid argument set "
            "has meaningful semantic requirements.",
            statements,
        )
        self.assertIn(
            "Structure generic constraints so invalid arguments fail with actionable "
            "diagnostics near the template interface.",
            statements,
        )
        self.assertNotIn(
            "Prefer C++20 concepts and requires-clauses over SFINAE or std::enable_if "
            "when template constraints are part of the public interface.",
            statements,
        )

    def test_cpp_production_refinement_extracts_template_for_type_variation(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Refactor these C++20 functions. They have shared C++ logic with small "
            "variations and similar C++ functions differ only by type, so extract "
            "a template function if it reduces duplication.",
            ("cpp.production_refinement",),
        )
        statements = _statements(result.structured["effective_rules"])

        self.assertIn(
            "Extract a template function, template class, constrained overload, or "
            "policy parameter when similar C++ implementations share most control or "
            "data flow and differ only by a small number of type or policy decisions.",
            statements,
        )
        self.assertIn(
            "Avoid introducing a template abstraction when the similarity is incidental, "
            "variation points are unclear, or the resulting API and diagnostics become "
            "harder to understand than the specialized implementations.",
            statements,
        )

    def test_resolve_cli_can_output_effective_prompt(self) -> None:
        from argparse import Namespace

        output, exit_code = CommandDispatcher().dispatch(
            Namespace(
                command="resolve",
                root=".",
                policy_root=".",
                skills="skills",
                packs="packs",
                task="Write a C++17 function that accepts a read-only string parameter.",
                pack=[],
                format="prompt",
            )
        )

        self.assertEqual(exit_code, 0)
        self.assertIsInstance(output, str)
        self.assertIn("# Effective Rules for Current Task", output)
        self.assertIn("Prefer std::string_view", output)
        self.assertNotIn('"effective_rules"', output)

    def test_resolve_cli_defaults_to_effective_prompt(self) -> None:
        from argparse import Namespace

        output, exit_code = CommandDispatcher().dispatch(
            Namespace(
                command="resolve",
                root=".",
                policy_root=".",
                skills="skills",
                packs="packs",
                task="Write a C++17 function that accepts a read-only string parameter.",
                pack=[],
                format="prompt",
            )
        )

        self.assertEqual(exit_code, 0)
        self.assertIsInstance(output, str)
        self.assertIn("Prefer available standard facility over unavailable or unapproved facility.", output)
        self.assertNotIn("available_standard_facility > unavailable_or_unapproved_facility", output)

    def test_generic_production_refinement_pack_outputs_refinement_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Refactor this code so it is not just working. Reduce complexity, "
            "group scattered logic, and make the API easier to use.",
            ("generic.production_refinement",),
        )
        effective = result.structured["effective_rules"]
        statements = _statements(effective)

        self.assertIn(
            "Preserve the existing observable behavior while reducing complexity "
            "unless the task explicitly asks for a behavior change.",
            statements,
        )
        self.assertIn(
            "Remove accidental complexity that does not contribute to correctness, "
            "extensibility, performance, or clarity.",
            statements,
        )
        self.assertTrue(_has_statement_containing(effective, "Group related variables"))
        self.assertTrue(_has_statement_containing(effective, "Minimize the number of steps"))
        self.assertFalse(any(item.startswith("Introduce abstractions") for item in statements))

    def test_cpp_production_refinement_pack_combines_generic_and_cpp_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Refactor this C++20 code so it is not just working. Reduce complexity "
            "and preserve safety.",
            ("cpp.production_refinement",),
        )
        effective = result.structured["effective_rules"]
        sources = _sources(effective)

        self.assertIn("generic.code_quality.complexity_reduction", sources)
        self.assertIn("cpp.safety.undefined_behavior", sources)
        self.assertTrue(_has_statement_containing(effective, "observable behavior"))
        self.assertTrue(_has_statement_containing(effective, "undefined behavior"))

    def test_python_production_refinement_pack_combines_generic_and_python_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Refactor this Python package so it is production-ready. Reduce complexity, "
            "preserve behavior, keep typing clear, and verify pytest coverage.",
            ("python.production_refinement",),
        )
        effective = result.structured["effective_rules"]
        sources = _sources(effective)

        self.assertIn("generic.code_quality.complexity_reduction", sources)
        self.assertIn("python.core.pythonic_baseline", sources)
        self.assertIn("python.typing.static_typing", sources)
        self.assertIn("python.testing.testing_practices", sources)
        self.assertTrue(_has_statement_containing(effective, "observable behavior"))
        self.assertTrue(_has_statement_containing(effective, "type hints"))

    def test_cmake_production_refinement_pack_combines_generic_and_cmake_rules(self) -> None:
        runtime = PolicyRuntime(RuntimeConfig.from_values(root=".", policy_root="."))
        result = runtime.resolve(
            "Refactor this CMakeLists.txt project to be production-ready. Reduce complexity, "
            "use target-based CMake, find_package imported targets, CMakePresets, "
            "and preserve behavior.",
            ("cmake.production_refinement",),
        )
        effective = result.structured["effective_rules"]
        sources = _sources(effective)

        self.assertIn("generic.code_quality.complexity_reduction", sources)
        self.assertIn("cmake.project.target_model", sources)
        self.assertIn("cmake.dependencies.package_management", sources)
        self.assertIn("cmake.reproducibility.presets_toolchains", sources)
        self.assertTrue(_has_statement_containing(effective, "observable behavior"))
        self.assertTrue(_has_statement_containing(effective, "target"))


class MultipleSkillPathTests(unittest.TestCase):
    """Tests for the extra skills/packs directory feature."""

    def _write_skill(self, root: Path, skill_id: str, *, name: str | None = None) -> Path:
        slug = skill_id.replace(".", "_")
        path = root / f"{slug}.skill.yaml"
        body = (
            "kind: skill\n"
            "api_version: policy.skill/v1\n"
            "skill:\n"
            f"  id: {skill_id}\n"
            f"  name: {name or skill_id}\n"
            "  version: 1.0.0\n"
            "  status: stable\n"
            "  level: domain\n"
            "  domain: extras_test\n"
            "  priority: 50\n"
            "  activation:\n"
            "    when: \"true\"\n"
            "rules: {}\n"
        )
        path.write_text(body, encoding="utf-8")
        return path

    def _write_pack(self, root: Path, pack_id: str, includes: tuple[str, ...]) -> Path:
        slug = pack_id.replace(".", "_")
        path = root / f"{slug}.pack.yaml"
        body = (
            "kind: pack\n"
            "api_version: policy.skill/v1\n"
            "pack:\n"
            f"  id: {pack_id}\n"
            f"  name: {pack_id}\n"
            "  version: 1.0.0\n"
            "includes:\n"
            + "".join(f"  - {item}\n" for item in includes)
            + "excludes: []\n"
            + "overrides: []\n"
        )
        path.write_text(body, encoding="utf-8")
        return path

    def test_default_config_is_backward_compatible(self) -> None:
        """No extras → behavior identical to single-path constructor."""

        config = RuntimeConfig.from_values(root=".")
        self.assertEqual(config.extra_skills_dirs, ())
        self.assertEqual(config.extra_packs_dirs, ())
        self.assertEqual(config.on_duplicate, "error")
        paths = config.paths
        self.assertEqual(paths.extra_skills, ())
        self.assertEqual(paths.extra_packs, ())
        self.assertEqual(paths.all_skills, (paths.skills,))
        self.assertEqual(paths.all_packs, (paths.packs,))

    def test_runtime_paths_resolves_extras_relative_to_policy_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = RuntimeConfig.from_values(
                root=root,
                extra_skills_dirs=("custom/skills", str(root / "abs_skills")),
                extra_packs_dirs=("custom/packs",),
            )
            paths = config.paths
            self.assertEqual(paths.extra_skills[0], root / "custom" / "skills")
            self.assertEqual(paths.extra_skills[1], root / "abs_skills")
            self.assertEqual(paths.extra_packs[0], root / "custom" / "packs")
            self.assertEqual(paths.all_skills[0], paths.skills)

    def test_invalid_on_duplicate_raises(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeConfig.from_values(root=".", on_duplicate="bogus")

    def test_from_dirs_multi_merges_skills(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            extra = root / "extra"
            primary.mkdir()
            extra.mkdir()
            self._write_skill(primary, "extras_test.alpha")
            self._write_skill(extra, "extras_test.bravo")
            registry = SkillRegistry.from_dirs_multi(
                (primary, extra), (), on_duplicate="error"
            )
            ids = {skill.skill_id for skill in registry.all()}
            self.assertEqual(ids, {"extras_test.alpha", "extras_test.bravo"})

    def test_from_dirs_multi_merges_packs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills = root / "skills"
            primary_packs = root / "packs"
            extra_packs = root / "extra_packs"
            skills.mkdir()
            primary_packs.mkdir()
            extra_packs.mkdir()
            self._write_skill(skills, "extras_test.alpha")
            self._write_pack(primary_packs, "primary.pack", ("extras_test.alpha",))
            self._write_pack(extra_packs, "extra.pack", ("extras_test.alpha",))
            registry = SkillRegistry.from_dirs_multi(
                (skills,), (primary_packs, extra_packs), on_duplicate="error"
            )
            self.assertEqual({"primary.pack", "extra.pack"}, set(registry.packs._packs))

    def test_duplicate_skill_id_default_errors(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            extra = root / "extra"
            primary.mkdir()
            extra.mkdir()
            self._write_skill(primary, "extras_test.alpha", name="primary version")
            self._write_skill(extra, "extras_test.alpha", name="extra version")
            with self.assertRaises(ValueError):
                SkillRegistry.from_dirs_multi(
                    (primary, extra), (), on_duplicate="error"
                )

    def test_duplicate_skill_id_first_wins(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            extra = root / "extra"
            primary.mkdir()
            extra.mkdir()
            self._write_skill(primary, "extras_test.alpha", name="primary version")
            self._write_skill(extra, "extras_test.alpha", name="extra version")
            registry = SkillRegistry.from_dirs_multi(
                (primary, extra), (), on_duplicate="first_wins"
            )
            self.assertEqual(registry.get("extras_test.alpha").name, "primary version")

    def test_duplicate_skill_id_last_wins(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            extra = root / "extra"
            primary.mkdir()
            extra.mkdir()
            self._write_skill(primary, "extras_test.alpha", name="primary version")
            self._write_skill(extra, "extras_test.alpha", name="extra version")
            registry = SkillRegistry.from_dirs_multi(
                (primary, extra), (), on_duplicate="last_wins"
            )
            self.assertEqual(registry.get("extras_test.alpha").name, "extra version")

    def test_duplicate_pack_id_obeys_on_duplicate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills = root / "skills"
            primary_packs = root / "packs"
            extra_packs = root / "extra_packs"
            skills.mkdir()
            primary_packs.mkdir()
            extra_packs.mkdir()
            self._write_skill(skills, "extras_test.alpha")
            self._write_pack(primary_packs, "shared.pack", ("extras_test.alpha",))
            self._write_pack(extra_packs, "shared.pack", ("extras_test.alpha",))
            with self.assertRaises(ValueError):
                SkillRegistry.from_dirs_multi(
                    (skills,), (primary_packs, extra_packs), on_duplicate="error"
                )

    def test_cli_args_populate_extras(self) -> None:
        from ai_policy_runtime.interfaces.cli import _runtime_from_args

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "skills").mkdir()
            (root / "packs").mkdir()
            args = argparse.Namespace(
                root=str(root),
                policy_root=None,
                skills="skills",
                packs="packs",
                extra_skills=["custom/skills"],
                extra_packs=["custom/packs"],
                on_duplicate="last_wins",
            )
            runtime = _runtime_from_args(args)
            self.assertEqual(runtime.config.extra_skills_dirs, ("custom/skills",))
            self.assertEqual(runtime.config.extra_packs_dirs, ("custom/packs",))
            self.assertEqual(runtime.config.on_duplicate, "last_wins")

    def test_cli_reads_extras_from_project_config(self) -> None:
        from ai_policy_runtime.interfaces.cli import _runtime_from_args

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "skills").mkdir()
            (root / "packs").mkdir()
            (root / ".policy").mkdir()
            (root / ".policy" / "config.json").write_text(
                json.dumps(
                    {
                        "extraSkillsDirs": ["from_config/skills"],
                        "extraPacksDirs": ["from_config/packs"],
                        "onDuplicate": "first_wins",
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                root=str(root),
                policy_root=None,
                skills="skills",
                packs="packs",
                extra_skills=[],
                extra_packs=[],
                on_duplicate=None,
            )
            runtime = _runtime_from_args(args)
            self.assertEqual(
                runtime.config.extra_skills_dirs, ("from_config/skills",)
            )
            self.assertEqual(
                runtime.config.extra_packs_dirs, ("from_config/packs",)
            )
            self.assertEqual(runtime.config.on_duplicate, "first_wins")


if __name__ == "__main__":
    unittest.main()
