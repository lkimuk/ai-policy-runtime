from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai_policy_runtime.domain.diagnostics import Diagnostic
from ai_policy_runtime.infrastructure.conditions import ConditionError, evaluate_condition
from ai_policy_runtime.infrastructure.loader import (
    load_mapping,
    load_packs_from_dir,
    load_skills_from_dir,
)
from ai_policy_runtime.services.schema_validation import JsonSchemaValidator


VALID_LEVELS = {"platform", "domain", "project", "task", "user"}
VALID_STATUS = {"experimental", "stable", "deprecated"}
GROUP_KEYWORDS = {
    "hard": {"must", "must_not"},
    "soft": {"should", "should_not", "allow"},
    "preference": {"prefer"},
}


class DslValidator:
    """Validate Skill DSL and Pack files and emit stable diagnostics."""

    def __init__(
        self,
        *,
        skills: "SkillDslValidator | None" = None,
        packs: "PackDslValidator | None" = None,
    ) -> None:
        self._skills = skills or SkillDslValidator()
        self._packs = packs or PackDslValidator()

    def validate_repository(
        self,
        skills_dir: str | Path | Sequence[str | Path],
        packs_dir: str | Path | Sequence[str | Path],
    ) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        skills_paths = _normalize_paths(skills_dir)
        packs_paths = _normalize_paths(packs_dir)
        skill_ids: set[str] = set()

        for skills_path in skills_paths:
            for path in sorted(skills_path.rglob("*.skill.yaml")):
                data = load_mapping(path)
                diagnostics.extend(self._skills.validate(data, str(path)))
                skill = data.get("skill", {})
                if skill.get("id"):
                    skill_ids.add(str(skill["id"]))

        diagnostics.extend(self._validate_dependencies(skills_paths, skill_ids))
        for packs_path in packs_paths:
            diagnostics.extend(self._packs.validate_files(packs_path))
        diagnostics.extend(self._validate_packs(packs_paths, skill_ids))
        return diagnostics

    def _validate_dependencies(
        self, skills_paths: Sequence[Path], skill_ids: set[str]
    ) -> list[Diagnostic]:
        try:
            return [
                Diagnostic("E005", f"Unresolved dependency: {dependency}", skill.skill_id)
                for skills_path in skills_paths
                for skill in load_skills_from_dir(skills_path)
                for dependency in skill.dependencies
                if dependency not in skill_ids
            ]
        except Exception as exc:
            return [Diagnostic("E000", f"Skill loading failed: {exc}")]

    def _validate_packs(
        self, packs_paths: Sequence[Path], skill_ids: set[str]
    ) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        try:
            packs = [
                pack for packs_path in packs_paths for pack in load_packs_from_dir(packs_path)
            ]
        except Exception as exc:
            return [Diagnostic("E000", f"Pack loading failed: {exc}")]

        pack_ids = {pack.pack_id for pack in packs}
        for pack in packs:
            diagnostics.extend(
                Diagnostic("E005", f"Unresolved parent pack: {parent}", pack.pack_id)
                for parent in pack.extends
                if parent not in pack_ids
            )
            diagnostics.extend(_validate_pack_skill_refs(pack, skill_ids))
        return diagnostics


class EffectiveRulesValidator:
    """Validate standardized effective-rules output."""

    def __init__(self, schemas: JsonSchemaValidator | None = None) -> None:
        self._schemas = schemas or JsonSchemaValidator()

    def validate(self, data: dict[str, Any], path: str = "") -> list[Diagnostic]:
        diagnostics = self._schemas.validate("effective-rules", data, path)
        effective = data.get("effective_rules")
        if not isinstance(effective, dict):
            return diagnostics or [Diagnostic("E001", "Missing top-level effective_rules", path)]
        for field in ("schema_version", "task", "hard", "soft", "preference"):
            if field not in effective:
                diagnostics.append(
                    Diagnostic("E001", f"Missing effective_rules field: {field}", path)
                )
        task = effective.get("task", {})
        if not isinstance(task, dict) or "context" not in task:
            diagnostics.append(
                Diagnostic("E001", "Missing effective_rules.task.context", path)
            )
        for group in ("hard", "soft", "preference"):
            for rule in effective.get(group, ()) or ():
                diagnostics.extend(_validate_effective_rule(rule, group, path))
        return diagnostics


class SkillDslValidator:
    """Validate one Skill DSL document."""

    def __init__(self, schemas: JsonSchemaValidator | None = None) -> None:
        self._schemas = schemas or JsonSchemaValidator()

    def validate(self, data: dict[str, Any], path: str = "") -> list[Diagnostic]:
        return [*self._schemas.validate("skill", data, path), *validate_skill_mapping(data, path)]


class PackDslValidator:
    """Validate Pack DSL documents and references."""

    def __init__(self, schemas: JsonSchemaValidator | None = None) -> None:
        self._schemas = schemas or JsonSchemaValidator()

    def validate(self, data: dict[str, Any], path: str = "") -> list[Diagnostic]:
        return [*self._schemas.validate("pack", data, path), *validate_pack_mapping(data, path)]

    def validate_files(self, packs_path: Path) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for path in sorted(packs_path.rglob("*.pack.yaml")):
            diagnostics.extend(self.validate(load_mapping(path), str(path)))
        return diagnostics


_DEFAULT_VALIDATOR = DslValidator()
_EFFECTIVE_RULES_VALIDATOR = EffectiveRulesValidator()


def validate_repository(
    skills_dir: str | Path | Sequence[str | Path],
    packs_dir: str | Path | Sequence[str | Path],
) -> list[Diagnostic]:
    return _DEFAULT_VALIDATOR.validate_repository(skills_dir, packs_dir)


def _normalize_paths(
    value: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def validate_effective_rules_mapping(data: dict[str, Any], path: str = "") -> list[Diagnostic]:
    return _EFFECTIVE_RULES_VALIDATOR.validate(data, path)


def validate_effective_rules_file(path: str | Path) -> list[Diagnostic]:
    return validate_effective_rules_mapping(load_mapping(path), str(path))


def validate_skill_mapping(data: dict[str, Any], path: str = "") -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    skill = data.get("skill")
    if not isinstance(skill, dict):
        return [Diagnostic("E001", "Missing required top-level section: skill", path)]

    for field in ("id", "name", "version", "level", "domain", "priority", "activation"):
        if field not in skill:
            diagnostics.append(Diagnostic("E001", f"Missing required skill field: {field}", path))

    level = str(skill.get("level", "")).lower()
    if level and level not in VALID_LEVELS:
        diagnostics.append(Diagnostic("E002", f"Invalid skill level: {skill.get('level')}", path))

    status = str(skill.get("status", "stable")).lower()
    if status not in VALID_STATUS:
        diagnostics.append(Diagnostic("E002", f"Invalid skill status: {skill.get('status')}", path))

    activation = skill.get("activation", {})
    if isinstance(activation, dict):
        _validate_condition(activation.get("when"), path, diagnostics)

    rules = data.get("rules", {})
    if not isinstance(rules, dict):
        diagnostics.append(Diagnostic("E001", "Missing or invalid rules section", path))
        return diagnostics

    for group in ("hard", "soft", "preference"):
        if group not in rules:
            continue
        if not isinstance(rules[group], list):
            diagnostics.append(Diagnostic("E001", f"Rules group must be a list: {group}", path))
            continue
        for item in rules[group]:
            diagnostics.extend(_validate_rule_object(item, group, path))

    for exception in data.get("exceptions", ()) or ():
        if not isinstance(exception, dict):
            diagnostics.append(Diagnostic("E001", "Exception must be an object", path))
            continue
        if "id" not in exception or "when" not in exception:
            diagnostics.append(Diagnostic("E001", "Exception requires id and when", path))
        _validate_condition(exception.get("when"), path, diagnostics)

    return diagnostics


def validate_pack_mapping(data: dict[str, Any], path: str = "") -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    pack = data.get("pack")
    if not isinstance(pack, dict):
        return [Diagnostic("E001", "Missing required top-level section: pack", path)]
    if "id" not in pack:
        diagnostics.append(Diagnostic("E001", "Missing required pack field: id", path))
    if "includes" not in data:
        diagnostics.append(Diagnostic("E001", "Missing required pack field: includes", path))
    return diagnostics


def _validate_effective_rule(rule: Any, group: str, path: str) -> list[Diagnostic]:
    if not isinstance(rule, dict):
        return [Diagnostic("E001", f"Effective {group} rule must be an object", path)]
    diagnostics = [
        Diagnostic("E001", f"Effective {group} rule missing {field}: {rule.get('id')}", path)
        for field in ("id", "target", "action", "statement", "source")
        if field not in rule
    ]
    source = rule.get("source")
    if not isinstance(source, dict) or "skill" not in source or "rule" not in source:
        diagnostics.append(Diagnostic("E001", f"Effective {group} rule has invalid source: {rule.get('id')}", path))
    return diagnostics


def _validate_pack_skill_refs(pack: Any, skill_ids: set[str]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for pattern in (*pack.includes, *pack.excludes):
        if pattern.endswith(".*"):
            prefix = pattern[:-1]
            if not any(skill_id.startswith(prefix) for skill_id in skill_ids):
                diagnostics.append(
                    Diagnostic(
                        "E010",
                        f"Pack wildcard matched no skills: {pattern}",
                        pack.pack_id,
                    )
                )
        elif pattern not in skill_ids:
            diagnostics.append(
                Diagnostic("E005", f"Unresolved pack skill reference: {pattern}", pack.pack_id)
            )
    return diagnostics


def _validate_pack_files(packs_path: Path) -> list[Diagnostic]:
    return PackDslValidator().validate_files(packs_path)


def _validate_rule_object(item: Any, group: str, path: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if not isinstance(item, dict):
        diagnostics.append(Diagnostic("E001", f"Rule in {group} must be an object", path))
        return diagnostics
    if "id" not in item:
        diagnostics.append(Diagnostic("E001", f"Rule in {group} missing id", path))
    keywords = GROUP_KEYWORDS[group]
    used = keywords.intersection(item)
    if len(used) != 1:
        diagnostics.append(
            Diagnostic("E003", f"Rule must use exactly one primary keyword: {item.get('id')}", path)
        )
    elif not used.issubset(GROUP_KEYWORDS[group]):
        diagnostics.append(
            Diagnostic(
                "E003",
                f"Invalid keyword {next(iter(used))} for rules.{group}: {item.get('id')}",
                path,
            )
        )
    if "target" not in item:
        diagnostics.append(Diagnostic("E001", f"Rule missing target: {item.get('id')}", path))
    _validate_condition(item.get("when"), path, diagnostics)
    _validate_condition(item.get("unless"), path, diagnostics)
    return diagnostics


def _validate_condition(
    condition: Any, path: str, diagnostics: list[Diagnostic]
) -> None:
    if condition is None or isinstance(condition, dict):
        return
    if isinstance(condition, str):
        try:
            evaluate_condition(condition, _DUMMY_CONTEXT)
        except ConditionError as exc:
            if "Unknown condition identifier" not in str(exc):
                diagnostics.append(Diagnostic("E004", f"Invalid condition: {condition}", path))
    else:
        diagnostics.append(Diagnostic("E004", f"Invalid condition type: {type(condition).__name__}", path))


_DUMMY_CONTEXT = {
    "language": "cpp",
    "standard": 20,
    "hot_path": True,
    "parameter_kind": "read_only_string",
    "ownership_required": False,
    "abi_boundary": False,
    "c_api_boundary": False,
    "designing_api": True,
    "selected_standard_is_known": True,
    "operation": "string_prefix_or_suffix_check",
    "trigger": "write_code",
}
