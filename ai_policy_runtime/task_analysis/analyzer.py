from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from ai_policy_runtime.domain.config import EmbeddingConfig

from .deterministic_extractor import DeterministicTaskExtractor
from .embeddings import (
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from .lexicon import TaskLexicon
from .schema import TaskAnalysis, TaskSignals
from .semantic_index import SemanticTaskIndex


class TaskAnalyzer:
    """Task analyzer using exact facts plus embedding semantic recall."""

    def __init__(
        self,
        *,
        deterministic: DeterministicTaskExtractor | None = None,
    ) -> None:
        self._deterministic = deterministic or build_extractor("skills")

    @classmethod
    def from_skills_dir(
        cls,
        path: str | Path,
        *,
        embeddings: EmbeddingProvider | None = None,
        semantic: bool = True,
        cache_dir: str | Path | None = None,
    ) -> "TaskAnalyzer":
        """Build an analyzer whose deterministic rules come from Skill metadata."""

        return cls.from_skills_dirs(
            (path,),
            embeddings=embeddings,
            semantic=semantic,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_skills_dirs(
        cls,
        paths: Sequence[str | Path],
        *,
        embeddings: EmbeddingProvider | None = None,
        semantic: bool = True,
        cache_dir: str | Path | None = None,
    ) -> "TaskAnalyzer":
        """Build an analyzer from multiple skill directories."""

        return cls(
            deterministic=build_extractor(
                paths,
                embeddings,
                semantic=semantic,
                cache_dir=cache_dir,
            )
        )

    def analyze(self, text: str, signals: TaskSignals | None = None) -> TaskAnalysis:
        """Analyze user input into a structured task context."""

        return self._deterministic.extract(text, signals)


def build_extractor(
    skills_dir: str | Path | Sequence[str | Path],
    embeddings: EmbeddingProvider | None = None,
    *,
    semantic: bool = True,
    cache_dir: str | Path | None = None,
) -> DeterministicTaskExtractor:
    """Create the default task extractor with sensible runtime defaults."""

    paths: Sequence[str | Path]
    if isinstance(skills_dir, (str, Path)):
        paths = (skills_dir,)
    else:
        paths = tuple(skills_dir)
    lexicon = TaskLexicon.from_skills_dirs(paths)
    if not semantic:
        return DeterministicTaskExtractor(lexicon)
    provider = embeddings or default_embedding_provider()
    return DeterministicTaskExtractor(
        lexicon,
        SemanticTaskIndex(lexicon, provider, cache_dir=cache_dir),
    )


def default_embedding_provider(
    local_model_root: str | Path = ".",
    embedding: EmbeddingConfig | None = None,
) -> EmbeddingProvider:
    """Return the best configured embedding provider.

    Selection order defaults toward OpenAI-compatible embeddings when remote
    configuration is present, while preserving offline options:

    1. Explicit provider choices from AI_POLICY_EMBEDDING_PROVIDER.
    2. OpenAI-compatible /v1/embeddings when configured.
    3. Local bundled sentence-transformers model when available.

    If no semantic provider is available, analysis fails instead of silently
    downgrading to deterministic matching.
    """

    provider = _configured_provider_name(embedding)
    if provider == "openai-compatible":
        remote = _openai_compatible_provider(embedding)
        if remote is None:
            raise RuntimeError(
                "AI_POLICY_EMBEDDING_PROVIDER requests OpenAI-compatible embeddings, "
                "but no endpoint configuration was found."
            )
        return remote

    local_model = Path(local_model_root) / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
    if provider == "local":
        configured_model = (
            embedding.model
            if embedding and embedding.provider and embedding.model
            else os.environ.get("AI_POLICY_EMBEDDING_MODEL")
        )
        if configured_model:
            return SentenceTransformerEmbeddingProvider(configured_model)
        if local_model.exists():
            return SentenceTransformerEmbeddingProvider(str(local_model))
        local_provider = optional_sentence_transformer_provider()
        if local_provider is not None:
            return local_provider
        raise RuntimeError(
            "AI_POLICY_EMBEDDING_PROVIDER requests local embeddings, but no "
            "local sentence-transformers model could be loaded. Run: "
            "ai-policy embedding configure --provider local --install, or pass "
            "--model <existing sentence-transformers model path>."
        )
    if provider:
        raise RuntimeError(f"Unsupported AI_POLICY_EMBEDDING_PROVIDER: {provider}")
    if remote := _openai_compatible_provider(embedding):
        return remote
    if local_model.exists():
        try:
            return SentenceTransformerEmbeddingProvider(str(local_model))
        except RuntimeError:
            pass
    raise RuntimeError(
        "Semantic analysis requires an embedding provider. Configure an "
        "OpenAI-compatible endpoint or install the local sentence-transformers model. "
        "Run: ai-policy embedding configure --provider local --install"
    )


def optional_sentence_transformer_provider() -> EmbeddingProvider | None:
    """Return a configured transformer provider, or None when unavailable."""

    try:
        return SentenceTransformerEmbeddingProvider()
    except RuntimeError:
        return None


def _openai_compatible_provider(
    embedding: EmbeddingConfig | None = None,
) -> OpenAICompatibleEmbeddingProvider | None:
    if embedding is None:
        return OpenAICompatibleEmbeddingProvider.from_env()
    env_config = OpenAICompatibleEmbeddingProvider.from_env()
    if not any(
        value is not None
        for value in (
            embedding.provider,
            embedding.base_url,
            embedding.api_key,
            embedding.model,
            embedding.timeout_seconds,
        )
    ):
        return env_config

    from .embeddings import OpenAICompatibleEmbeddingConfig

    explicit = OpenAICompatibleEmbeddingConfig.from_values(
        base_url=embedding.base_url or (env_config.config.base_url if env_config else None),
        model=embedding.model or (env_config.config.model if env_config else None),
        api_key=embedding.api_key or (env_config.config.api_key if env_config else None),
        timeout_seconds=embedding.timeout_seconds
        or (env_config.config.timeout_seconds if env_config else None),
    )
    return OpenAICompatibleEmbeddingProvider(explicit) if explicit else None


def _configured_provider_name(embedding: EmbeddingConfig | None = None) -> str:
    if embedding and embedding.provider:
        return embedding.provider.strip().lower().replace("_", "-")
    return (
        os.environ.get("AI_POLICY_EMBEDDING_PROVIDER", "")
        .strip()
        .lower()
        .replace("_", "-")
    )
