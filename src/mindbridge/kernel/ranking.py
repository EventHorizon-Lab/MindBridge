"""Pure ranking signals over index candidates and hydrated records.

Every function is a deterministic function of the hits it is given, the query terms, and the
reference clock. The retrieval plane decides what to hydrate and what to keep; this module
decides only how strongly a surviving candidate answers the query and what its trace says.
"""

from __future__ import annotations

import builtins
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import lru_cache
from typing import cast

from mindbridge.infrastructure.local.store import IndexCandidate, StoredMemory
from mindbridge.infrastructure.local.zvec_index import IndexHit
from mindbridge.kernel.temporal import overlaps_temporal_range
from mindbridge.types import (
    RetrievalCandidateTrace,
    RetrievalMode,
    RetrievalRejection,
    RetrievalTrace,
    SearchHit,
)

RANK_FLOOR = 0.3


RANK_CEILING = 1.5


_DECAY_REINFORCEMENT_LIMIT = 20


_CONFIRMATION_WEIGHT = 0.05


# Covering every distinctive query term is evidence in a way that placing well in the full-text
# index is not, so a complete match ranks as near-certain and a memory quoting the whole question
# cannot be buried by an unrelated dense neighbour. Nothing short of complete coverage enters the
# ranking score through this branch. A general floor under every full-text hit used to, at 0.24,
# but that compared an index-side quantity against a cosine, and how often it decided anything
# turned entirely on where the configured embedder puts its cosines: replaying 841 dev queries,
# the share of dense candidates it could outrank was 0.92% on LoCoMo-Refined, 8.48% on
# Mem-Gallery and 50.26% on MemLens. Deleting it left R@1, R@5, R@10, R@20, R@100 and MRR
# unchanged to four decimals on all three and the top ten identical on 99.87% of queries.
# The threshold leaves slack for summation order; the coverage ratio is otherwise exactly one.
# Across both replayed corpora two candidates in more than twenty thousand reached full coverage,
# so this rescues the exact-phrase case without moving the measurement at all. The value sits
# between the two behaviours the search tests pin: after the coverage lift it must clear a
# mediocre dense neighbour and must still lose to strong semantic evidence, because a memory that
# merely echoes the question back is a complete term match and is not the answer.
LEXICAL_FULL_COVERAGE = 0.999


LEXICAL_FULL_COVERAGE_RELEVANCE = 0.75


# Weight on the IDF-weighted query-term coverage, which unlike the rank proxy is a normalised
# score, so it is the term that actually combines the two routes. It is applied as a lift toward
# one across the remaining headroom rather than as an addition, because a clamped sum turns every
# strong candidate into exactly 1.0 and loses the ordering among them.
MAX_LEXICAL_RERANK_BONUS = 0.3


_NEGATION_WEIGHT = 6.0


_LEXICAL_NOISE_TERMS = frozenset(
    {
        "a",
        "an",
        "answer",
        "are",
        "based",
        "did",
        "do",
        "does",
        "how",
        "is",
        "memory",
        "memories",
        "only",
        "please",
        "question",
        "return",
        "the",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
)


_LEXICAL_TERM = re.compile(r"\w+")


# Runs `\w+` cannot split into words, because they are written without spaces and no segmenter is
# available here. What follows is adjacent-character bigram splitting, which is script-agnostic,
# so no Japanese text is handed to a Chinese segmenter by being listed here.
#
# This is deliberately narrower than `_ngram_text` in the Zvec index, which strips only Latin
# words and bigrams whatever is left. The two answer different questions. The index asks what a
# query could possibly match; this one asks which runs `\w+` fails to split at all, and
# bigramming a script it already splits would only dilute the word terms the coverage ratio is
# tuned on. Korean is excluded for that reason: it is space-delimited, so `\w+` yields eojeol.
#
# ponytail: an enumerated list of unspaced scripts, and a script missing from it silently gets
# one useless whole-run term -- which is how Thai went unretrievable. Inverting it to enumerate
# the space-delimited scripts is worse, not better: that list is far longer, and omitting an
# entry there breaks a script that works today. Add a range when a corpus needs it.
_UNSEGMENTED_RUN = re.compile(
    "["
    "\u0e00-\u0e7f"  # Thai
    "\u0e80-\u0eff"  # Lao
    "\u0f00-\u0fff"  # Tibetan
    "\u1000-\u109f"  # Myanmar
    "\u1780-\u17ff"  # Khmer
    "\u3040-\u30ff"  # Hiragana and Katakana
    "\u31f0-\u31ff"  # Katakana Phonetic Extensions
    "\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\uff66-\uff9f"  # Halfwidth Katakana
    "\U00020000-\U0003ffff"  # CJK Unified Ideographs Extension B and later
    "]+"
)


# What the index's ngram tokenizer treats as a token boundary: it is configured with
# `token_chars` of letter and digit, so everything else separates rather than disappears.
# Thai and Lao write their vowels and tones as combining marks, which are neither, so a bigram
# must not be formed across one -- measured against the index, "ยว" matches and the pair that
# spans the mark before it does not.
_TOKEN_BREAK = re.compile(r"[\W_]")


# The Chinese counterpart of `_LEXICAL_NOISE_TERMS`: particles, copulas, prepositions,
# conjunctions, pronouns, and interrogatives. Stripping them matters more than it does in
# English, because every one of them otherwise carries IDF weight into the coverage ratio that
# performs cross-route fusion, and a question is mostly these characters. Negation characters
# are excluded on purpose; `_NEGATION_TERMS` weights those up rather than away.
_CJK_NOISE_CHARACTERS = frozenset(
    "的了着地得之吗呢吧啊呀嘛么们"
    "是在就都也还把被给对从跟和与"
    "或及而但因所以什谁哪怎何我你"
    "他她它这那其请"
)


_NEGATION_TERMS = frozenset(
    {"no", "not", "never", "neither", "nor", "without", "不", "没", "未", "无", "非", "别"}
)


_NEGATED_CONTRACTION = re.compile(r"n['\u2019]t\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class IndexCandidates:
    dense: tuple[IndexHit, ...]
    lexical: tuple[IndexHit, ...]
    exhausted: bool


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    hits: tuple[SearchHit, ...]
    trace: RetrievalTrace | None = None
    matched_dense_index_ids: tuple[tuple[str, str], ...] = ()


def merge_lexical_routes(routes: Sequence[Sequence[IndexHit]]) -> tuple[IndexHit, ...]:
    if not routes:
        return ()
    if len(routes) == 1:
        return tuple(routes[0])
    return tuple(sorted(merge_index_hits(*routes), key=lambda hit: -hit.relevance))


def merge_index_hits(*groups: Sequence[IndexHit]) -> tuple[IndexHit, ...]:
    merged: dict[str, IndexHit] = {}
    for group in groups:
        for hit in group:
            current = merged.get(hit.id)
            if current is None:
                merged[hit.id] = hit
            else:
                merged[hit.id] = IndexHit(
                    id=hit.id,
                    relevance=max(current.relevance, hit.relevance),
                    confidence=max(cast(float, current.confidence), cast(float, hit.confidence)),
                    lexical_match=current.lexical_match or hit.lexical_match,
                )
    return tuple(merged.values())


def search_outcome(
    hits: Sequence[SearchHit],
    candidates: Sequence[RetrievalCandidateTrace] | None,
    *,
    candidate_limit: int,
    exhaustive: bool,
    ambiguous: bool = False,
    matched_dense_index_ids: Mapping[str, str] | None = None,
) -> SearchOutcome:
    trace = (
        None
        if candidates is None
        else RetrievalTrace(
            candidates=tuple(candidates),
            candidate_limit=candidate_limit,
            exhaustive=exhaustive,
            ambiguous=ambiguous,
        )
    )
    return SearchOutcome(
        hits=tuple(hits),
        trace=trace,
        matched_dense_index_ids=tuple((matched_dense_index_ids or {}).items()),
    )


def parent_index_ids(documents: Sequence[IndexCandidate]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, builtins.list[str]] = {}
    for document in documents:
        grouped.setdefault(document.memory_id, []).append(document.embedding_id)
    return {memory_id: tuple(index_ids) for memory_id, index_ids in grouped.items()}


def _early_candidate_trace(
    memory_id: str,
    index_ids: tuple[str, ...],
    dense_relevance: Mapping[str, float],
    dense_confidence: Mapping[str, float],
    lexical_index_relevance: Mapping[str, float],
    lexical_matches: set[str],
    rejected_by: RetrievalRejection,
    *,
    retrieval_mode: RetrievalMode,
) -> RetrievalCandidateTrace:
    lexical_match = memory_id in lexical_matches
    lexical_gate_scale = (
        1.0 if retrieval_mode is RetrievalMode.LEXICAL else LEXICAL_FULL_COVERAGE_RELEVANCE
    )
    return RetrievalCandidateTrace(
        memory_id=memory_id,
        index_ids=index_ids,
        dense_relevance=dense_relevance.get(memory_id, 0.0),
        dense_confidence=dense_confidence.get(memory_id, 0.0),
        # Hybrid needs term coverage from hydrated content, so its lexical ranking contribution
        # stays unknown here. Lexical mode ranks directly by native FTS relevance, which the
        # index already supplied.
        lexical_relevance=(
            lexical_index_relevance.get(memory_id, 0.0)
            if retrieval_mode is RetrievalMode.LEXICAL and lexical_match
            else None
        ),
        lexical_match=lexical_match,
        # An upper bound on what the gate would have scored: time and observation confidence were
        # not evaluated for this rejected candidate.
        gate_relevance=max(
            dense_relevance.get(memory_id, 0.0),
            lexical_gate_scale * lexical_index_relevance.get(memory_id, 0.0),
        ),
        rejected_by=rejected_by,
    )


def extend_hydration_traces(
    target: builtins.list[RetrievalCandidateTrace] | None,
    candidates: IndexCandidates,
    index_ids: Sequence[str],
    hydrated_documents: Sequence[IndexCandidate],
    accepted_documents: Sequence[IndexCandidate],
    index_ids_by_memory: Mapping[str, tuple[str, ...]],
    *,
    retrieval_mode: RetrievalMode,
) -> None:
    if target is None:
        return
    (
        dense_relevance,
        dense_confidence,
        lexical_index_relevance,
        lexical_matches,
        _matched_dense_index_ids,
    ) = parent_index_signals(candidates, hydrated_documents)
    dense_by_id = {hit.id: hit for hit in candidates.dense}
    lexical_by_id = {hit.id: hit for hit in candidates.lexical}
    lexical_gate_scale = (
        1.0 if retrieval_mode is RetrievalMode.LEXICAL else LEXICAL_FULL_COVERAGE_RELEVANCE
    )
    hydrated_ids = {document.embedding_id for document in hydrated_documents}
    for index_id in index_ids:
        if index_id in hydrated_ids:
            continue
        dense_hit = dense_by_id.get(index_id)
        lexical_hit = lexical_by_id.get(index_id)
        dense_hit_confidence = None if dense_hit is None else cast(float, dense_hit.confidence)
        target.append(
            RetrievalCandidateTrace(
                memory_id=None,
                index_ids=(index_id,),
                dense_relevance=None if dense_hit is None else dense_hit.relevance,
                dense_confidence=dense_hit_confidence,
                lexical_relevance=(
                    None
                    if lexical_hit is None or retrieval_mode is not RetrievalMode.LEXICAL
                    else lexical_hit.relevance
                ),
                lexical_match=lexical_hit is not None,
                gate_relevance=max(
                    0.0 if dense_hit is None else dense_hit.relevance,
                    0.0 if lexical_hit is None else lexical_gate_scale * lexical_hit.relevance,
                ),
                rejected_by=RetrievalRejection.STALE_INDEX,
            )
        )
    accepted_parent_ids = {document.memory_id for document in accepted_documents}
    for memory_id, parent_index_ids in index_ids_by_memory.items():
        if memory_id in accepted_parent_ids:
            continue
        target.append(
            _early_candidate_trace(
                memory_id,
                parent_index_ids,
                dense_relevance,
                dense_confidence,
                lexical_index_relevance,
                lexical_matches,
                RetrievalRejection.OCCURRENCE_RANGE,
                retrieval_mode=retrieval_mode,
            )
        )


def extend_missing_memory_traces(
    target: builtins.list[RetrievalCandidateTrace] | None,
    parent_ids: Sequence[str],
    memories: Sequence[StoredMemory],
    index_ids_by_memory: Mapping[str, tuple[str, ...]],
    dense_relevance: Mapping[str, float],
    dense_confidence: Mapping[str, float],
    lexical_index_relevance: Mapping[str, float],
    lexical_matches: set[str],
    *,
    retrieval_mode: RetrievalMode,
) -> None:
    if target is None:
        return
    hydrated_ids = {memory.memory_id for memory in memories}
    for memory_id in parent_ids:
        if memory_id not in hydrated_ids:
            target.append(
                _early_candidate_trace(
                    memory_id,
                    index_ids_by_memory[memory_id],
                    dense_relevance,
                    dense_confidence,
                    lexical_index_relevance,
                    lexical_matches,
                    RetrievalRejection.MISSING_MEMORY,
                    retrieval_mode=retrieval_mode,
                )
            )


def extend_memory_type_traces(
    target: builtins.list[RetrievalCandidateTrace] | None,
    memories: Sequence[StoredMemory],
    kept: frozenset[str],
    index_ids_by_memory: Mapping[str, tuple[str, ...]],
    dense_relevance: Mapping[str, float],
    dense_confidence: Mapping[str, float],
    lexical_index_relevance: Mapping[str, float],
    lexical_matches: set[str],
    *,
    retrieval_mode: RetrievalMode,
) -> None:
    if target is None:
        return
    for memory in memories:
        if memory.memory_type not in kept:
            target.append(
                _early_candidate_trace(
                    memory.memory_id,
                    index_ids_by_memory[memory.memory_id],
                    dense_relevance,
                    dense_confidence,
                    lexical_index_relevance,
                    lexical_matches,
                    RetrievalRejection.MEMORY_TYPE,
                    retrieval_mode=retrieval_mode,
                )
            )


def record_ranked_trace(
    target: dict[str, RetrievalCandidateTrace] | None,
    memory_id: str,
    index_ids_by_memory: Mapping[str, tuple[str, ...]],
    dense_relevance: Mapping[str, float],
    dense_confidence: Mapping[str, float],
    *,
    lexical_relevance: float,
    lexical_rerank_bonus: float,
    lexical_match: bool,
    gate_relevance: float,
    base_relevance: float,
    reinforcement_factor: float,
    temporal_factor: float | None,
    retention_factor: float | None,
    final_score: float,
) -> None:
    if target is None:
        return
    target[memory_id] = RetrievalCandidateTrace(
        memory_id=memory_id,
        index_ids=index_ids_by_memory[memory_id],
        dense_relevance=dense_relevance.get(memory_id, 0.0),
        dense_confidence=dense_confidence.get(memory_id, 0.0),
        lexical_relevance=lexical_relevance,
        lexical_rerank_bonus=lexical_rerank_bonus,
        lexical_match=lexical_match,
        gate_relevance=gate_relevance,
        base_relevance=base_relevance,
        reinforcement_factor=reinforcement_factor,
        temporal_factor=temporal_factor,
        retention_factor=retention_factor,
        final_score=final_score,
    )


def qualified_candidates(
    ranked: Sequence[tuple[StoredMemory, float, float, bool]],
    *,
    minimum_relevance: float,
    trace_candidates: builtins.list[RetrievalCandidateTrace] | None,
    ranked_traces: Mapping[str, RetrievalCandidateTrace] | None,
) -> builtins.list[tuple[StoredMemory, float, float, bool]]:
    qualified = []
    for item in ranked:
        if item[2] >= minimum_relevance:
            qualified.append(item)
        elif trace_candidates is not None and ranked_traces is not None:
            trace_candidates.append(
                replace(
                    ranked_traces[item[0].memory_id],
                    rejected_by=RetrievalRejection.MINIMUM_RELEVANCE,
                )
            )
    return qualified


def extend_ranked_traces(
    target: builtins.list[RetrievalCandidateTrace] | None,
    ranked_traces: Mapping[str, RetrievalCandidateTrace] | None,
    ranked: Sequence[tuple[StoredMemory, float, float, bool]],
    *,
    limit: int,
    ambiguous: bool,
) -> None:
    if target is None or ranked_traces is None:
        return
    for rank, item in enumerate(ranked, start=1):
        rejected_by = None
        if ambiguous:
            rejected_by = RetrievalRejection.AMBIGUITY if rank <= 2 else RetrievalRejection.LIMIT
        elif rank > limit:
            rejected_by = RetrievalRejection.LIMIT
        target.append(
            replace(
                ranked_traces[item[0].memory_id],
                rank=rank,
                rejected_by=rejected_by,
            )
        )


def parent_index_signals(
    candidates: IndexCandidates,
    documents: Sequence[IndexCandidate],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], set[str], dict[str, str]]:
    dense_by_id = {hit.id: hit for hit in candidates.dense}
    lexical_by_id = {hit.id: hit for hit in candidates.lexical}
    dense_relevance: dict[str, float] = {}
    dense_confidence: dict[str, float] = {}
    lexical_relevance: dict[str, float] = {}
    lexical_matches: set[str] = set()
    winning_dense: dict[str, tuple[float, str]] = {}
    for document in documents:
        memory_id = document.memory_id
        embedding_id = document.embedding_id
        dense_hit = dense_by_id.get(embedding_id)
        if dense_hit is not None:
            dense_relevance[memory_id] = max(
                dense_relevance.get(memory_id, 0.0), dense_hit.relevance
            )
            dense_confidence[memory_id] = max(
                dense_confidence.get(memory_id, 0.0), cast(float, dense_hit.confidence)
            )
            current = winning_dense.get(memory_id)
            candidate = (dense_hit.relevance, embedding_id)
            if (
                current is None
                or candidate[0] > current[0]
                or (candidate[0] == current[0] and candidate[1] < current[1])
            ):
                winning_dense[memory_id] = candidate
        lexical_hit = lexical_by_id.get(embedding_id)
        if lexical_hit is not None:
            lexical_matches.add(memory_id)
            lexical_relevance[memory_id] = max(
                lexical_relevance.get(memory_id, 0.0), lexical_hit.relevance
            )
    return (
        dense_relevance,
        dense_confidence,
        lexical_relevance,
        lexical_matches,
        {memory_id: value[1] for memory_id, value in winning_dense.items()},
    )


def retrieval_is_ambiguous(
    ranked: Sequence[tuple[StoredMemory, float, float, bool]],
    *,
    margin: float,
    temporal_range: tuple[datetime, datetime] | None,
) -> bool:
    if margin == 0.0 or len(ranked) < 2:
        return False
    first, second = ranked[:2]
    if first[3]:
        return False
    difference = first[1] - second[1] if temporal_range is not None else first[2] - second[2]
    return difference < margin


def ranked_relevance(
    memory: StoredMemory,
    relevance: float,
    *,
    reference_at: datetime,
    temporal_range: tuple[datetime, datetime] | None,
    decay_half_life: timedelta | None,
) -> float:
    return ranking_signals(
        memory,
        relevance,
        reference_at=reference_at,
        temporal_range=temporal_range,
        decay_half_life=decay_half_life,
    )[0]


def ranking_signals(
    memory: StoredMemory,
    relevance: float,
    *,
    reference_at: datetime,
    temporal_range: tuple[datetime, datetime] | None,
    decay_half_life: timedelta | None,
) -> tuple[float, float, float | None, float | None]:
    ranking_reference = temporal_range[1] if temporal_range is not None else reference_at
    confirmed_at = memory.last_accessed_at
    confirmations = (
        min(memory.access_count, _DECAY_REINFORCEMENT_LIMIT)
        if confirmed_at is not None and confirmed_at <= ranking_reference
        else 0
    )
    reinforcement_factor = 1.0 + _CONFIRMATION_WEIGHT * math.log2(1.0 + confirmations)
    score = bounded_scale(
        relevance,
        reinforcement_factor,
    )
    temporal_factor = None
    if temporal_range is not None:
        temporal_factor = temporal_relevance(
            memory.occurred_at,
            memory.occurred_end,
            temporal_range,
        )
        score = bounded_scale(
            score,
            temporal_factor,
        )
    retention_factor = None
    if decay_half_life is not None:
        decay_reference = ranking_reference
        accessed_at = memory.last_accessed_at
        anchor = memory.occurred_end or memory.occurred_at or memory.updated_at
        access_count = 0
        if accessed_at is not None and accessed_at <= decay_reference:
            anchor = accessed_at
            access_count = confirmations
        age = max(0.0, (decay_reference - anchor).total_seconds())
        strength = 1.0 + math.log2(1.0 + min(access_count, _DECAY_REINFORCEMENT_LIMIT))
        retention = 2.0 ** (-age / (decay_half_life.total_seconds() * strength))
        retention_factor = RANK_FLOOR + (RANK_CEILING - RANK_FLOOR) * retention
        score = bounded_scale(
            score,
            retention_factor,
        )
    return score, reinforcement_factor, temporal_factor, retention_factor


def bounded_scale(score: float, factor: float) -> float:
    if factor <= 1.0:
        return score * factor
    return score + (1.0 - score) * (factor - 1.0)


def lexical_relevance(
    query: str,
    memories: Sequence[StoredMemory],
) -> dict[str, float]:
    query_terms = lexical_query_terms(query)
    if not query_terms or not memories:
        return {}
    documents = {memory.memory_id: _lexical_terms(memory.content) for memory in memories}
    frequencies = Counter(term for terms in documents.values() for term in query_terms & terms)
    count = len(documents)
    weights = {
        term: (math.log((count + 1.0) / (frequencies[term] + 1.0)) + 1.0)
        * (_NEGATION_WEIGHT if term in _NEGATION_TERMS else 1.0)
        for term in query_terms
    }
    total = sum(weights.values())
    return {
        memory_id: sum(weights[term] for term in query_terms & terms) / total
        for memory_id, terms in documents.items()
    }


# Every search re-derives the terms of ~100 hydrated candidates whose content has not changed
# since the last search that ranked them; the terms are a pure function of the text. The cache is
# process-wide and holds the term sets, which run to ~15x the text for unspaced CJK, so only texts
# under this length are cached: 1024 entries x 2 000 chars bounds it near 32 MiB.
_LEXICAL_TERMS_CACHE_CHARS = 2_000


def _lexical_terms(value: str) -> frozenset[str]:
    if len(value) > _LEXICAL_TERMS_CACHE_CHARS:
        return _lexical_terms_uncached(value)
    return _lexical_terms_cached(value)


@lru_cache(maxsize=1024)
def _lexical_terms_cached(value: str) -> frozenset[str]:
    return _lexical_terms_uncached(value)


def _lexical_terms_uncached(value: str) -> frozenset[str]:
    """Split text into words, plus adjacent-character bigrams for runs `\\w+` cannot split.

    `\\w+` matches an entire unspaced run as one token. That token is by construction the rarest
    term in any corpus, so it took the largest IDF weight into the coverage ratio and could never
    match anything, which put `LEXICAL_FULL_COVERAGE` — the one term that actually fuses the
    dense and full-text routes — permanently out of reach for every multi-character Chinese
    query. Runs are therefore removed before word splitting rather than left alongside their
    parts, and re-emitted as bigrams: the dependency-free stand-in for a segmenter, and the
    reason a query needs an adjacent pair to match rather than one character that happens to
    appear somewhere.

    Bigrams only, and not the single characters as well. Emitting both made the term count a
    property of the language rather than of the question -- eleven terms for `爱丽丝面包店`
    against three for the `Alice bakery Tuesday` it translates -- which moved
    `LEXICAL_FULL_COVERAGE` out of reach for Chinese at a length it stayed reachable for
    English. A lone single-character run therefore yields no term at all, and `search` drops the
    lexical route for such a query. The stemmed full-text field would in fact match it, because
    the `standard` tokenizer splits Han per character, but a single character is too weak a term
    to rank on: it would return `lexical_match` with a coverage of zero, contributing candidates
    and no ranking signal, and the dense route still answers the query.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold()
    terms = set(_LEXICAL_TERM.findall(_UNSEGMENTED_RUN.sub(" ", normalized)))
    for run in _UNSEGMENTED_RUN.findall(normalized):
        for piece in _TOKEN_BREAK.split(run):
            terms.update(piece[index : index + 2] for index in range(len(piece) - 1))
    if _NEGATED_CONTRACTION.search(normalized) is not None:
        terms.add("not")
    return frozenset(terms)


def lexical_query_terms(value: str) -> frozenset[str]:
    """Query terms with the scaffolding dropped, so coverage measures the distinctive ones.

    A bigram is scaffolding only when both of its characters are, so `什么` goes and `的饮`
    stays; otherwise a question's function words would keep full coverage unreachable through
    the bigrams they appear in.
    """
    return frozenset(
        term
        for term in _lexical_terms(value)
        if term not in _LEXICAL_NOISE_TERMS
        and not all(character in _CJK_NOISE_CHARACTERS for character in term)
    )


def temporal_relevance(
    occurred_at: datetime | None,
    occurred_end: datetime | None,
    temporal_range: tuple[datetime, datetime],
) -> float:
    """Boost a memory that overlaps the asked time range; leave every other memory alone.

    The factor used to decay from the ceiling to `RANK_FLOOR` with distance from the range, and to
    hand the floor to a memory with no event time at all. Replayed on the validation libraries
    that penalty never recovered a gold memory and only reordered: making the factor additive --
    ceiling on overlap, neutral otherwise -- won or tied on every paired question of 810 across
    LoCoMo and ATM-Bench (R@5 +2.6 pp on ATM's temporal slice, one loss), left the questions
    without a temporal phrase bit-identical, and needs one constant fewer. A memory that misses
    the window is still ranked on how well it matches; it is no longer pushed under the memories
    that merely happened at the right time.
    """
    if occurred_at is not None and overlaps_temporal_range(
        occurred_at, occurred_end, temporal_range
    ):
        return RANK_CEILING
    return 1.0
