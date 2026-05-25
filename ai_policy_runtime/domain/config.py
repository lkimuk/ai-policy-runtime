from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


VALID_ON_DUPLICATE = ("error", "first_wins", "last_wins")


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved filesystem paths used by the policy runtime."""

    root: Path
    skills: Path
    packs: Path
    current: Path
    extra_skills: tuple[Path, ...] = ()
    extra_packs: tuple[Path, ...] = ()

    @property
    def all_skills(self) -> tuple[Path, ...]:
        return (self.skills, *self.extra_skills)

    @property
    def all_packs(self) -> tuple[Path, ...]:
        return (self.packs, *self.extra_packs)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding provider configuration for embedded Python runtime usage."""

    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    """User-facing runtime configuration with conservative defaults."""

    root: Path = Path(".")
    policy_root: Path | None = None
    skills_dir: str = "skills"
    packs_dir: str = "packs"
    extra_skills_dirs: tuple[str, ...] = ()
    extra_packs_dirs: tuple[str, ...] = ()
    on_duplicate: str = "error"
    embedding: EmbeddingConfig | None = field(default=None)

    def __post_init__(self) -> None:
        if self.on_duplicate not in VALID_ON_DUPLICATE:
            raise ValueError(
                f"on_duplicate must be one of {VALID_ON_DUPLICATE}, got: {self.on_duplicate!r}"
            )

    @property
    def paths(self) -> RuntimePaths:
        root = self.root
        policy_root = self.policy_root or root
        return RuntimePaths(
            root=root,
            skills=_resolve_policy_path(policy_root, self.skills_dir),
            packs=_resolve_policy_path(policy_root, self.packs_dir),
            current=root / ".policy" / "current",
            extra_skills=tuple(
                _resolve_policy_path(policy_root, value) for value in self.extra_skills_dirs
            ),
            extra_packs=tuple(
                _resolve_policy_path(policy_root, value) for value in self.extra_packs_dirs
            ),
        )

    @classmethod
    def from_values(
        cls,
        *,
        root: str | Path = ".",
        policy_root: str | Path | None = None,
        skills_dir: str = "skills",
        packs_dir: str = "packs",
        extra_skills_dirs: Sequence[str | Path] = (),
        extra_packs_dirs: Sequence[str | Path] = (),
        on_duplicate: str = "error",
        embedding_provider: str | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
        embedding_timeout_seconds: float | None = None,
    ) -> "RuntimeConfig":
        embedding = None
        if any(
            value is not None
            for value in (
                embedding_provider,
                embedding_base_url,
                embedding_api_key,
                embedding_model,
                embedding_timeout_seconds,
            )
        ):
            embedding = EmbeddingConfig(
                provider=embedding_provider,
                base_url=embedding_base_url,
                api_key=embedding_api_key,
                model=embedding_model,
                timeout_seconds=embedding_timeout_seconds,
            )
        return cls(
            root=Path(root),
            policy_root=Path(policy_root) if policy_root is not None else None,
            skills_dir=skills_dir,
            packs_dir=packs_dir,
            extra_skills_dirs=tuple(str(item) for item in extra_skills_dirs),
            extra_packs_dirs=tuple(str(item) for item in extra_packs_dirs),
            on_duplicate=on_duplicate,
            embedding=embedding,
        )


def _resolve_policy_path(policy_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return policy_root / path
