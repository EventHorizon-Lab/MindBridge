"""Validated local policy for one open memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from mindbridge.exceptions import ValidationError
from mindbridge.infrastructure.local.store import RECALL_MAX_ROWS
from mindbridge.kernel.contracts import index_recipe_for, validated_index_quantization
from mindbridge.kernel.validation import (
    positive_int,
    positive_seconds,
    strict_bool,
    unit_interval,
    validated_decay_half_life,
    validated_retrieval_mode,
)
from mindbridge.types import IndexQuantization, RetentionPolicy, RetrievalMode


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated local policy for one open memory, in the units the planes consume."""

    index_quantization: IndexQuantization
    index_recipe: str
    retrieval_mode: RetrievalMode
    speaker_similarity: float
    speaker_margin: float
    face_similarity: float
    face_margin: float
    identity_link_min_assets: int
    index_speech: bool
    independent_evidence: bool
    minimum_relevance: float
    ambiguity_margin: float
    evidence_budget: int | None
    recall_planning: bool
    recall_set_budget: int
    recall_set_max_rows: int
    recall_rounds: int
    evidence_expansion: bool
    evidence_expansion_max_rows: int
    decay_half_life: timedelta | None
    reinforce_on_answer: bool
    memory_budget_records: int | None
    query_failure_window: timedelta
    query_failure_history: int
    retention: RetentionPolicy


def _expansion_max_rows(value: object) -> int:
    """Bound the expansion read the way the store bounds every other recall read.

    The value goes straight to `related_memories(max_rows=...)`, which refuses anything above
    `RECALL_MAX_ROWS`. Unbounded here, a caller who set it high built a memory that opened fine
    and then raised out of the store on every `ask`; policy is validated once, at wiring.
    """
    rows = positive_int(value, "evidence_expansion_max_rows")
    if rows > RECALL_MAX_ROWS:
        raise ValidationError(f"evidence_expansion_max_rows must be at most {RECALL_MAX_ROWS}")
    return rows


def resolve_settings(
    *,
    index_speech: bool,
    independent_evidence: bool,
    index_quantization: IndexQuantization,
    retrieval_mode: RetrievalMode,
    minimum_relevance: float,
    ambiguity_margin: float,
    evidence_budget_chars: int | None,
    recall_planning: bool,
    recall_set_budget_chars: int,
    recall_set_max_rows: int,
    recall_rounds: int,
    evidence_expansion: bool,
    evidence_expansion_max_rows: int,
    decay_half_life_days: float | None,
    reinforce_on_answer: bool,
    speaker_similarity: float,
    speaker_margin: float,
    face_similarity: float,
    face_margin: float,
    identity_link_min_assets: int,
    memory_budget_records: int | None,
    query_failure_window_seconds: float,
    query_failure_history: int,
    retention: RetentionPolicy,
) -> Settings:
    """Validate every `MemoryConfig` field once, at wiring, and fail before storage opens."""
    quantization = validated_index_quantization(index_quantization)
    settings = Settings(
        index_quantization=quantization,
        index_recipe=index_recipe_for(quantization),
        retrieval_mode=validated_retrieval_mode(retrieval_mode),
        speaker_similarity=unit_interval(speaker_similarity, "speaker_similarity"),
        speaker_margin=unit_interval(speaker_margin, "speaker_margin"),
        face_similarity=unit_interval(face_similarity, "face_similarity"),
        face_margin=unit_interval(face_margin, "face_margin"),
        identity_link_min_assets=positive_int(identity_link_min_assets, "identity_link_min_assets"),
        index_speech=strict_bool(index_speech, "index_speech"),
        independent_evidence=strict_bool(independent_evidence, "independent_evidence"),
        minimum_relevance=unit_interval(minimum_relevance, "minimum_relevance"),
        ambiguity_margin=unit_interval(ambiguity_margin, "ambiguity_margin"),
        evidence_budget=None
        if evidence_budget_chars is None
        else positive_int(evidence_budget_chars, "evidence_budget_chars"),
        recall_planning=strict_bool(recall_planning, "recall_planning"),
        recall_set_budget=positive_int(recall_set_budget_chars, "recall_set_budget_chars"),
        recall_set_max_rows=positive_int(recall_set_max_rows, "recall_set_max_rows"),
        recall_rounds=positive_int(recall_rounds, "recall_rounds"),
        evidence_expansion=strict_bool(evidence_expansion, "evidence_expansion"),
        evidence_expansion_max_rows=_expansion_max_rows(evidence_expansion_max_rows),
        decay_half_life=validated_decay_half_life(decay_half_life_days),
        reinforce_on_answer=strict_bool(reinforce_on_answer, "reinforce_on_answer"),
        memory_budget_records=(
            None
            if memory_budget_records is None
            else positive_int(memory_budget_records, "memory_budget_records")
        ),
        query_failure_window=positive_seconds(
            query_failure_window_seconds, "query_failure_window_seconds"
        ),
        query_failure_history=positive_int(query_failure_history, "query_failure_history"),
        retention=retention,
    )
    if not isinstance(settings.retention, RetentionPolicy):
        raise ValidationError("retention must be a RetentionPolicy value")
    return settings
