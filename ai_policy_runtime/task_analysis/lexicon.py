from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai_policy_runtime.infrastructure.loader import PolicyLoader


@dataclass(frozen=True)
class LexiconRule:
    """Data-driven text match rule loaded from Skill metadata."""

    skill_id: str
    field: str
    value: Any
    phrases: tuple[str, ...]
    confidence: float
    source: str
    set_context: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    semantic_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class TriggerProfile:
    """Capabilities associated with a task trigger."""

    trigger: str
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillProfile:
    """Skill-level metadata used to gate semantic recall."""

    skill_id: str
    domain: str | None = None
    triggers: tuple[str, ...] = ()
    standard_min: int | None = None

    def matches(self, gate: "TaskGate") -> bool:
        """Return whether this skill may participate in semantic recall."""

        if gate.domain and self.domain and gate.domain != self.domain:
            return False
        if gate.task_type and self.triggers and gate.task_type not in self.triggers:
            return False
        if gate.standard is not None and self.standard_min is not None:
            return gate.standard >= self.standard_min
        return True


@dataclass(frozen=True)
class TaskLexicon:
    """Runtime task-analysis lexicon assembled from installed Skills."""

    skill_profiles: tuple[SkillProfile, ...] = ()
    skill_rules: tuple[LexiconRule, ...] = ()
    domain_rules: tuple[LexiconRule, ...] = ()
    trigger_rules: tuple[LexiconRule, ...] = ()
    context_rules: tuple[LexiconRule, ...] = ()
    trigger_profiles: tuple[TriggerProfile, ...] = ()

    @classmethod
    def from_skills_dir(cls, path: str | Path) -> "TaskLexicon":
        return cls.from_skills_dirs((path,))

    @classmethod
    def from_skills_dirs(cls, paths: Sequence[str | Path]) -> "TaskLexicon":
        loader = PolicyLoader()
        skill_profiles: list[SkillProfile] = []
        skill_rules: list[LexiconRule] = []
        domain_rules: list[LexiconRule] = []
        trigger_rules: list[LexiconRule] = []
        context_rules: list[LexiconRule] = []
        trigger_capabilities: dict[str, set[str]] = {}

        for file_path in _iter_skill_files_multi(paths):
            document = SkillAnalysisDocument(loader.load_mapping(file_path), file_path)
            skill_profiles.append(document.skill_profile())
            if skill_rule := document.skill_rule():
                skill_rules.append(skill_rule)
            if domain_rule := document.domain_rule():
                domain_rules.append(domain_rule)
            for trigger, values in document.trigger_capabilities().items():
                if values:
                    trigger_capabilities.setdefault(trigger, set()).update(values)
            trigger_rules.extend(document.trigger_rules())
            context_rules.extend(document.context_rules())

        return cls(
            skill_profiles=tuple(skill_profiles),
            skill_rules=tuple(skill_rules),
            domain_rules=tuple(domain_rules),
            trigger_rules=tuple(trigger_rules),
            context_rules=tuple(context_rules),
            trigger_profiles=tuple(
                TriggerProfile(trigger=trigger, capabilities=tuple(sorted(values)))
                for trigger, values in sorted(trigger_capabilities.items())
            ),
        )

    def capabilities_for(self, trigger: str) -> tuple[str, ...]:
        values = {
            capability
            for profile in self.trigger_profiles
            if profile.trigger == trigger
            for capability in profile.capabilities
        }
        return tuple(sorted(values))

    def domain_for_skill(self, skill_id: str) -> str | None:
        """Return the declared domain for a skill id, if one exists."""

        for profile in self.skill_profiles:
            if profile.skill_id == skill_id:
                return profile.domain
        return None

    def semantic_scope(self, gate: "TaskGate") -> frozenset[str]:
        """Return skill ids eligible for second-stage semantic recall."""

        return frozenset(
            profile.skill_id
            for profile in self.skill_profiles
            if profile.matches(gate)
        )

    def generic_semantic_scope(self) -> frozenset[str]:
        """Return language-independent skills eligible for no-domain bootstrapping."""

        return frozenset(
            profile.skill_id
            for profile in self.skill_profiles
            if profile.domain in {None, "generic_code"}
        )


class SkillAnalysisDocument:
    """Task-analysis view over a raw Skill DSL mapping."""

    def __init__(self, data: dict[str, Any], path: Path) -> None:
        self._data = data
        self._path = path
        self._meta = data.get("skill", {})
        self._analysis = data.get("task_analysis", self._meta.get("task_analysis", {}))
        self.skill_id = str(self._meta.get("id", data.get("skill_id", path.stem)))

    def skill_profile(self) -> SkillProfile:
        """Return gate metadata for this Skill."""

        activation = self._activation()
        domain = self._meta.get("domain") or _first(self._data.get("domains"))
        return SkillProfile(
            skill_id=self.skill_id,
            domain=str(domain) if domain else None,
            triggers=tuple(_strings(activation.get("triggers", ()))),
            standard_min=_standard_min(activation.get("when")),
        )

    def skill_rule(self) -> LexiconRule | None:
        """Return a skill-level semantic recall rule derived from Skill metadata."""

        texts = (
            str(self._meta.get("name", "")),
            str(self._meta.get("description", self._data.get("description", ""))),
            *_strings(self._meta.get("tags", ())),
        )
        semantic_texts = _normalize_phrases(texts)
        if not semantic_texts:
            return None
        return LexiconRule(
            skill_id=self.skill_id,
            field="skill",
            value=self.skill_id,
            phrases=(),
            confidence=float(self._analysis.get("skill_confidence", 0.68)),
            source=f"skill:{self.skill_id}:description",
            semantic_texts=semantic_texts,
        )

    def domain_rule(self) -> LexiconRule | None:
        """Return the domain rule declared by this Skill, if any."""

        domain = self._meta.get("domain") or _first(self._data.get("domains"))
        if not domain:
            return None
        phrases = _strings(self._analysis.get("domain_aliases", ())) or (str(domain),)
        return LexiconRule(
            skill_id=self.skill_id,
            field="domain",
            value=str(domain),
            phrases=_normalize_phrases(phrases),
            confidence=float(self._analysis.get("domain_confidence", 0.9)),
            source=f"skill:{self.skill_id}:domain",
            semantic_texts=_normalize_phrases(
                _strings(self._analysis.get("domain_semantics", ()))
            ),
        )

    def trigger_rules(self) -> tuple[LexiconRule, ...]:
        """Return task-trigger rules declared by this Skill."""

        aliases_by_trigger = dict(self._analysis.get("trigger_aliases", {}))
        semantics = dict(self._analysis.get("trigger_semantics", {}))
        triggers = sorted({*aliases_by_trigger, *semantics})
        return tuple(
            LexiconRule(
                skill_id=self.skill_id,
                field="task_type",
                value=str(trigger),
                phrases=_normalize_phrases(_strings(aliases_by_trigger.get(trigger, ()))),
                confidence=float(self._analysis.get("trigger_confidence", 0.82)),
                source=f"skill:{self.skill_id}:trigger:{trigger}",
                semantic_texts=_normalize_phrases(
                    _strings(semantics.get(trigger, ()))
                ),
            )
            for trigger in triggers
        )

    def context_rules(self) -> tuple[LexiconRule, ...]:
        """Return context-setting rules declared by this Skill."""

        return tuple(
            self._context_rule(item)
            for item in self._analysis.get("context_rules", ())
            if isinstance(item, dict)
        )

    def trigger_capabilities(self) -> dict[str, set[str]]:
        """Return trigger-specific capability declarations."""

        return {
            str(trigger): set(_strings(values))
            for trigger, values in dict(
                self._analysis.get("trigger_capabilities", {})
            ).items()
        }

    def _context_rule(self, item: dict[str, Any]) -> LexiconRule:
        set_context = dict(item.get("set", {}))
        field, value = _context_evidence_field_value(set_context)
        rule_id = item.get("id")
        source_suffix = f"context:{rule_id}" if rule_id else "context"
        return LexiconRule(
            skill_id=self.skill_id,
            field=field,
            value=value,
            phrases=_normalize_phrases(_strings(item.get("when_text_matches", ()))),
            confidence=float(item.get("confidence", 0.8)),
            source=f"skill:{self.skill_id}:{source_suffix}",
            set_context=set_context,
            tags=tuple(_strings(item.get("tags", ()))),
            semantic_texts=_normalize_phrases(_strings(item.get("semantic_match", ()))),
        )

    def _activation(self) -> dict[str, Any]:
        activation = self._data.get("activation", self._meta.get("activation", {}))
        return activation if isinstance(activation, dict) else {}


@dataclass(frozen=True)
class TaskGate:
    """First-stage hard facts that constrain semantic recall."""

    domain: str | None = None
    task_type: str | None = None
    standard: int | None = None


def _standard_min(value: Any) -> int | None:
    if isinstance(value, dict):
        raw = value.get("standard")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str) and raw.startswith(">="):
            try:
                return int(raw[2:])
            except ValueError:
                return None
    return None


def _iter_skill_files(path: str | Path) -> Iterable[Path]:
    root = Path(path)
    if not root.exists():
        return ()
    return (
        item
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.name.endswith((".skill.yaml", ".skill.yml", ".skill.json"))
    )


def _iter_skill_files_multi(paths: Sequence[str | Path]) -> Iterable[Path]:
    for path in paths:
        yield from _iter_skill_files(path)


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return None


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _normalize_phrases(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(" ".join(value.lower().strip().split()) for value in values if value)


def _context_evidence_field_value(set_context: dict[str, Any]) -> tuple[str, Any]:
    generic_context_keys = {
        "artifact_type",
        "refinement_requested",
        "behavior_preservation_required",
    }
    for key, value in set_context.items():
        if value is not True and key not in generic_context_keys:
            return f"context.{key}", value
    for key, value in set_context.items():
        if key not in generic_context_keys:
            return f"context.{key}", value
    for key, value in set_context.items():
        if value is not True:
            return f"context.{key}", value
    if set_context:
        key, value = next(iter(set_context.items()))
        return f"context.{key}", value
    return "context", True
