from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from ai_policy_runtime.domain.pack import PackRegistry, SkillPack
from ai_policy_runtime.domain.rule import Rule
from ai_policy_runtime.domain.skill import Skill
from ai_policy_runtime.domain.task import TaskContext
from ai_policy_runtime.infrastructure.conditions import ConditionError, evaluate_condition
from ai_policy_runtime.infrastructure.loader import load_packs_from_dir, load_skills_from_dir


VALID_ON_DUPLICATE = ("error", "first_wins", "last_wins")


GENERIC_ACTIVATION_TAGS = frozenset(
    {
        "cpp",
        "cmake",
        "git",
        "generation",
        "review",
        "core-guidelines",
        "standard-library",
    }
)

GENERIC_CORE_SKILLS = frozenset(
    {
        "generic.code_quality.complexity_reduction",
        "generic.code_quality.implementation_polish",
    }
)

GENERIC_CONTEXT_GATES = {
    "generic.code_quality.api_usability": ("user_facing_api",),
    "generic.architecture.hierarchy_cleanup": ("unclear_hierarchy",),
    "generic.refactoring.component_grouping": ("scattered_related_logic",),
    "generic.refactoring.duplication_extraction": ("duplicated_logic",),
    "generic.code_quality.expressive_implementation": ("expressive_implementation",),
    "generic.refactoring.parameterized_abstraction": ("similar_logic_with_small_variation",),
}

CMAKE_CONTEXT_GATES = {
    "cmake.project.target_model": ("cmake_target_model_requested",),
    "cmake.project.usage_requirements": ("cmake_usage_requirements_requested",),
    "cmake.project.compiler_options": ("cmake_compiler_options_requested",),
    "cmake.project.sources_options_modules": (
        "cmake_source_management_requested",
        "cmake_generated_files_requested",
        "cmake_options_modules_requested",
    ),
    "cmake.dependencies.package_management": ("cmake_dependency_requested",),
    "cmake.distribution.install_export_package": ("cmake_distribution_requested",),
    "cmake.reproducibility.presets_toolchains": (
        "cmake_presets_requested",
        "cmake_toolchain_requested",
    ),
    "cmake.quality.testing_and_analysis": (
        "cmake_testing_requested",
        "cmake_quality_requested",
    ),
}


class SkillRegistry:
    """In-memory index for Skill activation and Pack expansion."""

    def __init__(
        self, skills: Iterable[Skill] = (), packs: PackRegistry | None = None
    ) -> None:
        self._skills: dict[str, Skill] = {}
        self.packs = packs or PackRegistry()
        for skill in skills:
            self.register(skill)

    @classmethod
    def from_dir(cls, path: str | Path) -> "SkillRegistry":
        return cls(load_skills_from_dir(path))

    @classmethod
    def from_dirs(cls, skills_path: str | Path, packs_path: str | Path) -> "SkillRegistry":
        return cls(
            load_skills_from_dir(skills_path),
            PackRegistry(load_packs_from_dir(packs_path)),
        )

    @classmethod
    def from_dirs_multi(
        cls,
        skills_paths: Sequence[str | Path],
        packs_paths: Sequence[str | Path],
        *,
        on_duplicate: str = "error",
    ) -> "SkillRegistry":
        """Build a registry from multiple skill/pack directories.

        Earlier paths win in ``first_wins`` mode; later paths win in
        ``last_wins`` mode; ``error`` raises on duplicate skill_id or pack_id.
        """

        if on_duplicate not in VALID_ON_DUPLICATE:
            raise ValueError(
                f"on_duplicate must be one of {VALID_ON_DUPLICATE}, got: {on_duplicate!r}"
            )
        skills = _merge_skills(
            (load_skills_from_dir(path) for path in skills_paths),
            on_duplicate=on_duplicate,
        )
        packs = _merge_packs(
            (load_packs_from_dir(path) for path in packs_paths),
            on_duplicate=on_duplicate,
        )
        return cls(skills, PackRegistry(packs))

    def register(self, skill: Skill) -> None:
        if skill.skill_id in self._skills:
            raise ValueError(f"Duplicate skill_id: {skill.skill_id}")
        self._skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> Skill:
        return self._skills[skill_id]

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def activate(
        self, task: TaskContext, pack_ids: Iterable[str] = ()
    ) -> tuple[list[Skill], tuple[Rule, ...]]:
        activation = ActivationSet(self._skills)
        activation.add(skill for skill in self._skills.values() if self._matches(skill, task))
        pack_overrides = (
            self._apply_packs(activation, pack_ids)
            if _pack_activation_allowed(task)
            else []
        )
        activation.include_dependencies()
        return activation.filtered(), tuple(pack_overrides)

    def active_skills(self, task: TaskContext, pack_ids: Iterable[str] = ()) -> list[Skill]:
        skills, _ = self.activate(task, pack_ids)
        return skills

    def _resolve_skill_pattern(self, pattern: str) -> list[str]:
        if pattern.endswith(".*"):
            prefix = pattern[:-1]
            return [skill_id for skill_id in self._skills if skill_id.startswith(prefix)]
        return [pattern]

    def _apply_packs(
        self, activation: "ActivationSet", pack_ids: Iterable[str]
    ) -> list[Rule]:
        overrides: list[Rule] = []
        for pack_id in pack_ids:
            includes, pack_overrides = self.packs.expand(pack_id)
            overrides.extend(pack_overrides)
            for pattern in includes:
                activation.add_by_id(self._resolve_skill_pattern(pattern))
        return overrides

    def _matches(self, skill: Skill, task: TaskContext) -> bool:
        if skill.status in {"deprecated", "removed"}:
            return False
        if skill.domains and not _domain_matches(skill.domains, task):
            return False
        if not _generic_code_skill_allowed(skill, task):
            return False
        context_gate_match = _cmake_context_gate_matches(skill, task)
        if skill.triggers and task.task_type not in skill.triggers and not context_gate_match:
            return False
        if (
            skill.capabilities
            and not set(skill.capabilities).intersection(task.capabilities)
            and not context_gate_match
        ):
            return False
        if not _tags_match(skill.tags, task.tags) and not _semantic_skill_matches(skill, task):
            return False
        if not self._context_matches(skill, task):
            return False
        try:
            return evaluate_condition(skill.activation_when, task.condition_context())
        except ConditionError:
            return False

    def _context_matches(self, skill: Skill, task: TaskContext) -> bool:
        for key, expected in skill.context.items():
            if key not in task.context:
                return False
            if not _matches_expected(task.context[key], expected):
                return False
        return True


def _matches_expected(actual: object, expected: object) -> bool:
    if isinstance(expected, str) and expected.startswith(">="):
        try:
            return float(actual) >= float(expected[2:])
        except (TypeError, ValueError):
            return False
    return actual == expected


def _domain_matches(skill_domains: tuple[str, ...], task: TaskContext) -> bool:
    if task.domain in skill_domains:
        return True
    if "generic_code" not in skill_domains:
        return False
    if task.domain != "general":
        return False
    return task.context.get("artifact_type") == "code" or task.context.get("language") == "generic_code"


def _generic_code_skill_allowed(skill: Skill, task: TaskContext) -> bool:
    if "generic_code" not in skill.domains:
        return True
    if task.context.get("artifact_type") != "code":
        return True
    if skill.skill_id in GENERIC_CORE_SKILLS:
        return True
    return any(task.context.get(key) is True for key in GENERIC_CONTEXT_GATES.get(skill.skill_id, ()))


def _cmake_context_gate_matches(skill: Skill, task: TaskContext) -> bool:
    if "cmake" not in skill.domains:
        return False
    if task.domain != "cmake" and task.context.get("language") != "cmake":
        return False
    return any(task.context.get(key) is True for key in CMAKE_CONTEXT_GATES.get(skill.skill_id, ()))


def _tags_match(skill_tags: tuple[str, ...], task_tags: tuple[str, ...]) -> bool:
    meaningful_skill_tags = set(skill_tags) - GENERIC_ACTIVATION_TAGS
    if not meaningful_skill_tags:
        return True
    return bool(meaningful_skill_tags.intersection(task_tags))


def _semantic_skill_matches(skill: Skill, task: TaskContext) -> bool:
    matches = task.context.get("semantic_skill_matches", ())
    if isinstance(matches, str):
        return matches == skill.skill_id
    if isinstance(matches, (list, tuple, set, frozenset)):
        return skill.skill_id in matches
    return False


def _pack_activation_allowed(task: TaskContext) -> bool:
    """Packs refine an identified task; they must not create one from context alone."""

    if task.task_type == "unknown":
        return False
    if task.domain != "general":
        return True
    return task.context.get("artifact_type") == "code" or bool(task.context.get("language"))


def _merge_skills(
    groups: Iterable[Iterable[Skill]], *, on_duplicate: str
) -> list[Skill]:
    merged: dict[str, Skill] = {}
    for group in groups:
        for skill in group:
            if skill.skill_id not in merged:
                merged[skill.skill_id] = skill
                continue
            if on_duplicate == "error":
                raise ValueError(f"Duplicate skill_id: {skill.skill_id}")
            if on_duplicate == "last_wins":
                merged[skill.skill_id] = skill
    return list(merged.values())


def _merge_packs(
    groups: Iterable[Iterable[SkillPack]], *, on_duplicate: str
) -> list[SkillPack]:
    merged: dict[str, SkillPack] = {}
    for group in groups:
        for pack in group:
            if pack.pack_id not in merged:
                merged[pack.pack_id] = pack
                continue
            if on_duplicate == "error":
                raise ValueError(f"Duplicate pack_id: {pack.pack_id}")
            if on_duplicate == "last_wins":
                merged[pack.pack_id] = pack
    return list(merged.values())


class ActivationSet:
    """Mutable active-skill set with dependency and compatibility handling."""

    def __init__(self, available: dict[str, Skill]) -> None:
        self._available = available
        self._active: dict[str, Skill] = {}

    def add(self, skills: Iterable[Skill]) -> None:
        for skill in skills:
            self._active[skill.skill_id] = skill

    def add_by_id(self, skill_ids: Iterable[str]) -> None:
        self.add(self._available[skill_id] for skill_id in skill_ids if skill_id in self._available)

    def include_dependencies(self) -> None:
        changed = True
        while changed:
            changed = False
            for skill in list(self._active.values()):
                for dependency in skill.dependencies:
                    if dependency not in self._active and dependency in self._available:
                        self._active[dependency] = self._available[dependency]
                        changed = True

    def filtered(self) -> list[Skill]:
        active_ids = set(self._active)
        return [
            skill
            for skill in self._active.values()
            if skill.status not in {"deprecated", "removed"}
            and not any(item in active_ids for item in skill.incompatibilities)
        ]
