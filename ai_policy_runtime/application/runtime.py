from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_policy_runtime.domain.config import RuntimeConfig
from ai_policy_runtime.domain.diagnostics import Diagnostic
from ai_policy_runtime.domain.rule import EffectiveRules
from ai_policy_runtime.services.engine import PolicyEngine
from ai_policy_runtime.services.injector import inject_current_prompt
from ai_policy_runtime.services.project_context import (
    ProjectAnalysis,
    ProjectContextAnalyzer,
    merge_project_analysis,
)
from ai_policy_runtime.services.repair import (
    RepairInstruction,
    RepairPlanner,
    RepairPlanWriter,
)
from ai_policy_runtime.services.registry import SkillRegistry
from ai_policy_runtime.services.state import write_current_state
from ai_policy_runtime.services.validator import validate_repository
from ai_policy_runtime.services.verification import (
    Violation,
    verify_current_state,
    write_violations,
)
from ai_policy_runtime.task_analysis import (
    EmbeddingProvider,
    TaskAnalyzer,
    TaskSignals,
)
from ai_policy_runtime.task_analysis.analyzer import default_embedding_provider
from ai_policy_runtime.task_analysis.embeddings import cosine_similarity


class NonApplicableTaskError(RuntimeError):
    """Raised when a task does not activate any task-scoped policy."""

    def __init__(self, task_analysis: dict[str, Any]) -> None:
        super().__init__("Task did not produce applicable Effective Rules.")
        self.task_analysis = task_analysis

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": False,
            "result": None,
            "task_analysis": self.task_analysis,
        }


@dataclass(frozen=True)
class ResolveResult:
    """Result of resolving a task into Effective Rules."""

    current: Path
    effective_rules: EffectiveRules
    structured: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"current": str(self.current), **self.structured}


@dataclass(frozen=True)
class ResolveApplicabilityResult:
    """Result of attempting to resolve policy only when task rules apply."""

    applicable: bool
    resolve_result: ResolveResult | None
    task_analysis: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "result": self.resolve_result.to_dict() if self.resolve_result else None,
            "task_analysis": self.task_analysis,
        }


@dataclass(frozen=True)
class ExplainResult:
    """Result of task analysis without resolving Effective Rules."""

    structured: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.structured


@dataclass(frozen=True)
class RunResult:
    """Result of the MVP workflow: resolve -> inject -> optional verify."""

    current: Path
    injected: Path
    agent: str
    verified: bool
    violations: tuple[Violation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": str(self.current),
            "injected": str(self.injected),
            "agent": self.agent,
            "verified": self.verified,
            "violations": [item.to_dict() for item in self.violations],
        }


class PolicyRuntime:
    """High-level service API used by CLI, tests, and future adapters."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()

    def validate(self) -> list[Diagnostic]:
        paths = self.config.paths
        return validate_repository(paths.all_skills, paths.all_packs)

    def resolve(self, task_text: str, pack_ids: tuple[str, ...] = ()) -> ResolveResult:
        project, analysis, effective_rules, contributing_skills = self._evaluate(
            task_text,
            pack_ids,
        )
        if not analysis.activation_ready or not _has_policy_content(effective_rules):
            raise NonApplicableTaskError(analysis.to_dict())
        return self._write_resolved_state(
            task_text,
            analysis,
            project,
            effective_rules=effective_rules,
            pack_ids=pack_ids,
            contributing_skills=contributing_skills,
        )

    def resolve_if_applicable(
        self,
        task_text: str,
        pack_ids: tuple[str, ...] = (),
    ) -> ResolveApplicabilityResult:
        if (
            _is_runtime_status_query(task_text)
            or _is_trivial_non_task_query(task_text)
            or self._is_semantic_runtime_status_query(task_text)
        ):
            return ResolveApplicabilityResult(
                applicable=False,
                resolve_result=None,
                task_analysis=_non_applicable_task_analysis(),
            )
        project, analysis, effective_rules, contributing_skills = self._evaluate(
            task_text,
            pack_ids,
        )
        if not analysis.activation_ready or not _has_policy_content(effective_rules):
            return ResolveApplicabilityResult(
                applicable=False,
                resolve_result=None,
                task_analysis=analysis.to_dict(),
            )
        resolved = self._write_resolved_state(
            task_text,
            analysis,
            project,
            effective_rules=effective_rules,
            pack_ids=pack_ids,
            contributing_skills=contributing_skills,
        )
        return ResolveApplicabilityResult(
            applicable=True,
            resolve_result=resolved,
            task_analysis=analysis.to_dict(),
        )

    def explain(self, task_text: str) -> ExplainResult:
        """Analyze a task and return evidence without writing runtime state."""

        project = self._analyze_project()
        analysis = self._analyze(task_text, project)
        structured = analysis.to_dict()
        structured["project_context"] = project.to_dict()
        return ExplainResult(structured)

    def inject(self, target: str = "codex") -> Path:
        return inject_current_prompt(self.config.paths.root, target)

    def verify(self, target: str | Path = ".") -> tuple[Violation, ...]:
        violations = tuple(verify_current_state(self.config.paths.root, target))
        write_violations(self.config.paths.root, list(violations))
        return violations

    def repair_plan(self) -> tuple[RepairInstruction, ...]:
        """Generate repair instructions from current verification violations."""

        violations_path = self.config.paths.current / "violations.json"
        if not violations_path.exists():
            raise FileNotFoundError(f"Violations not found: {violations_path}")
        import json

        data = json.loads(violations_path.read_text(encoding="utf-8"))
        violations = [
            Violation(
                rule_id=str(item.get("rule_id", "")),
                severity=str(item.get("severity", "error")),
                path=str(item.get("path", "")),
                line=int(item.get("line", 0)),
                message=str(item.get("message", "")),
            )
            for item in data.get("violations", ())
        ]
        instructions = tuple(RepairPlanner().plan(violations))
        RepairPlanWriter(self.config.paths.root).write(list(instructions))
        return instructions

    def run(
        self,
        task_text: str,
        *,
        pack_ids: tuple[str, ...] = (),
        agent: str = "custom",
        verify_target: str | Path | None = None,
    ) -> RunResult:
        resolved = self.resolve(task_text, pack_ids)
        injected = self.inject(agent)
        violations = self.verify(verify_target) if verify_target else ()
        return RunResult(
            current=resolved.current,
            injected=injected,
            agent=agent,
            verified=verify_target is not None,
            violations=tuple(violations),
        )

    def _analyze(self, task_text: str, project: ProjectAnalysis):
        paths = self.config.paths
        supported_domains = self._supported_domains()
        project_language = (
            project.primary_language
            if project.primary_language in supported_domains
            else None
        )
        embeddings = self._embedding_provider()
        analysis = TaskAnalyzer.from_skills_dirs(
            paths.all_skills,
            embeddings=embeddings,
            semantic=True,
            cache_dir=paths.root / ".policy" / "cache" / "semantic-index",
        ).analyze(
            task_text,
            TaskSignals(
                project_language=project_language,
                git_has_changes=_project_git_has_changes(project),
            ),
        )
        return merge_project_analysis(
            analysis,
            project,
            supported_domains=supported_domains,
        )

    def _analyze_project(self) -> ProjectAnalysis:
        return ProjectContextAnalyzer(self.config.paths.root).analyze()

    def _evaluate(
        self,
        task_text: str,
        pack_ids: tuple[str, ...],
    ):
        paths = self.config.paths
        project = self._analyze_project()
        analysis = self._analyze(task_text, project)
        registry = SkillRegistry.from_dirs_multi(
            paths.all_skills,
            paths.all_packs,
            on_duplicate=self.config.on_duplicate,
        )
        effective_rules = PolicyEngine(registry).evaluate(analysis.task, pack_ids)
        contributing_skills = _contributing_skill_ids(effective_rules)
        return project, analysis, effective_rules, contributing_skills

    def _write_resolved_state(
        self,
        task_text: str,
        analysis,
        project: ProjectAnalysis,
        *,
        effective_rules: EffectiveRules,
        pack_ids: tuple[str, ...],
        contributing_skills: list[str],
    ) -> ResolveResult:
        current, structured = write_current_state(
            root=self.config.paths.root,
            task=analysis.task,
            effective_rules=effective_rules,
            trace={
                "task": task_text,
                "packs": list(pack_ids),
                "project_context": project.to_dict(),
                "task_analysis": analysis.to_dict(),
                "active_skills": contributing_skills,
                "conflict_count": len(effective_rules.conflicts),
            },
            project_context=project.to_dict(),
        )
        return ResolveResult(
            current=current,
            effective_rules=effective_rules,
            structured=structured,
        )

    def _supported_domains(self) -> set[str]:
        try:
            registry = SkillRegistry.from_dirs_multi(
                self.config.paths.all_skills,
                self.config.paths.all_packs,
                on_duplicate=self.config.on_duplicate,
            )
        except Exception:
            return set()
        return {domain for skill in registry.all() for domain in skill.domains}

    def _embedding_provider(self) -> EmbeddingProvider | None:
        return default_embedding_provider(
            self.config.policy_root or self.config.paths.root,
            self.config.embedding,
        )

    def _is_semantic_runtime_status_query(self, task_text: str) -> bool:
        if not _mentions_runtime_subject(task_text):
            return False
        try:
            vectors = self._embedding_provider().encode((task_text, *_RUNTIME_STATUS_INTENTS))
        except RuntimeError:
            return False
        if len(vectors) < 2:
            return False
        query = vectors[0]
        return max(cosine_similarity(query, vector) for vector in vectors[1:]) >= 0.62


def _contributing_skill_ids(effective_rules: EffectiveRules) -> list[str]:
    sources = {
        rule.source
        for group in (
            effective_rules.hard,
            effective_rules.soft,
            effective_rules.preferences,
            effective_rules.exceptions,
        )
        for rule in group
        if rule.source
    }
    return sorted(sources)


def _project_git_has_changes(project: ProjectAnalysis) -> bool:
    fact = project.fact("context.git_has_changes")
    return bool(fact and fact.value is True and fact.confidence >= 0.7)


def _has_policy_content(effective_rules: EffectiveRules) -> bool:
    return any(
        (
            effective_rules.hard,
            effective_rules.soft,
            effective_rules.preferences,
            effective_rules.exceptions,
        )
    )


def _is_runtime_status_query(task_text: str) -> bool:
    text = " ".join(task_text.lower().split())
    status_terms = (
        "status",
        "enabled",
        "enable",
        "configured",
        "configuration",
    )
    return _mentions_runtime_subject(task_text) and any(
        term in text for term in status_terms
    )


def _mentions_runtime_subject(task_text: str) -> bool:
    text = " ".join(task_text.lower().split())
    runtime_terms = (
        "ai policy runtime",
        "policy runtime",
        "effective rules",
        "claude code plugin",
        "codex plugin",
        "plugin",
    )
    return any(term in text for term in runtime_terms)


_RUNTIME_STATUS_INTENTS = (
    "check whether AI Policy Runtime is enabled",
    "show AI Policy Runtime status",
    "verify the plugin configuration",
    "inspect the current project runtime setup",
    "confirm whether the Claude Code plugin is configured correctly",
    "explain whether the Effective Rules output is correct",
)


def _is_trivial_non_task_query(task_text: str) -> bool:
    text = " ".join(task_text.lower().split())
    return text in {
        "hello",
        "hi",
        "hey",
    }


def _non_applicable_task_analysis() -> dict[str, Any]:
    return {
        "task": {
            "domain": "general",
            "task_type": "unknown",
            "capabilities": [],
            "tags": [],
            "context": {},
        },
        "confidence": 0.2,
        "needs_review": True,
        "activation_ready": False,
        "evidence": [],
    }
