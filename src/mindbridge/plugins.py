"""Explicit capability composition for one local memory instance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from mindbridge.exceptions import ValidationError
from mindbridge.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    FaceBackend,
    FormationBackend,
    GenerationBackend,
    SpeechBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
)
from mindbridge.types import IndexQuantization

_StrictBool = Annotated[bool, Field(strict=True)]
_UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1)]
_PositiveFloat = Annotated[float, Field(strict=True, gt=0)]
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryPlugins:
    """Already-constructed capability adapters owned and closed by ``Memory``."""

    embedder: EmbeddingBackend
    answerer: GenerationBackend | None = None
    transcriber: SpeechBackend | TranscriptionBackend | None = None
    vision_describer: VisionDescriptionBackend | None = None
    face_analyzer: FaceBackend | None = None
    former: FormationBackend | None = None
    consolidator: ConsolidationBackend | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.embedder, EmbeddingBackend):
            raise ValidationError("plugins.embedder must implement EmbeddingBackend")
        if self.answerer is not None and not isinstance(self.answerer, GenerationBackend):
            raise ValidationError("plugins.answerer must implement GenerationBackend")
        if self.transcriber is not None and not isinstance(
            self.transcriber,
            (SpeechBackend, TranscriptionBackend),
        ):
            raise ValidationError(
                "plugins.transcriber must implement SpeechBackend or TranscriptionBackend"
            )
        if self.vision_describer is not None and not isinstance(
            self.vision_describer,
            VisionDescriptionBackend,
        ):
            raise ValidationError(
                "plugins.vision_describer must implement VisionDescriptionBackend"
            )
        if self.face_analyzer is not None and not isinstance(self.face_analyzer, FaceBackend):
            raise ValidationError("plugins.face_analyzer must implement FaceBackend")
        if self.former is not None and not isinstance(self.former, FormationBackend):
            raise ValidationError("plugins.former must implement FormationBackend")
        if self.consolidator is not None and not isinstance(
            self.consolidator,
            ConsolidationBackend,
        ):
            raise ValidationError("plugins.consolidator must implement ConsolidationBackend")


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryConfig:
    """Value-only local policy kept separate from runtime capability objects."""

    # On by default because a configured speech backend has already produced its analysis by the
    # time `add` reaches the index: reusing that text costs no extra model call and no extra token.
    # Without it a video memory's lexical index document can be as short as 29 characters. It is a
    # no-op unless `plugins.transcriber` is a `SpeechBackend`; a plain `TranscriptionBackend`
    # reaches the index through the separate transcript-derivation path either way.
    index_speech: _StrictBool = True
    index_quantization: IndexQuantization = IndexQuantization.NONE
    # A floor on evidence relevance: the cosine the dense route reports, or the demoted full-text
    # contribution when only the lexical route matched, adjusted by temporal proximity when the
    # query asked about a time and by the observation's own confidence. Relevance is floored at 0,
    # so `0` admits everything.
    #
    # It is NOT a floor on `SearchHit.score`. Reinforcement and `decay_half_life_days` retention
    # shape the score and the order but stay outside the gate, because the query never mentioned
    # them -- so an admitted hit can report a score below this value. That asymmetry is load
    # bearing: both factors are bounded below by `_RANK_FLOOR`, so with retention inside the gate
    # a perfectly relevant memory decayed to 0.30 and to 0.09 once a dated question's window also
    # missed it, under this default, and "prefer recent" silently became "hide old".
    #
    # `0.10` is not a round number and must not be tidied into one: the floor used to gate a
    # rescaled `(1 + cosine) / 2` confidence, where the old 0.55 default admitted cosine 0.10, so
    # 0.10 reproduces the previously shipped effective floor exactly and fixing the scale does not
    # also silently tighten the policy. Raising it is a separate, separately measurable decision.
    minimum_relevance: _UnitInterval = 0.10
    ambiguity_margin: _UnitInterval = 0.01
    # `ask` grounds on `limit` memories and then keeps admitting lower-ranked ones while the
    # evidence stays inside this character budget. Retrieval scores separate the right memory
    # from the rest only weakly, so the answering model does the final selection and benefits
    # from seeing more candidates. `None` grounds on exactly `limit`.
    #
    # This raises a floor; it is not a ceiling. The `limit` hits are kept unconditionally, so the
    # prompt is bounded by `limit` times the size of a memory and setting this can only make it
    # larger. On LoCoMo-Refined it turns a 20-memory window into 56.3 and the answer prompt from
    # 2 856 tokens to 7 648 for a score change of 0.003, inside that task's 0.029 noise. Nor is
    # `limit` a token control once this is set: at limit 3 and limit 20 the same budget refills
    # the window to the same 56.3 memories, Jaccard 0.986. Bound a prompt by lowering `limit`
    # with this left at `None`.
    evidence_budget_chars: _PositiveInt | None = None
    decay_half_life_days: _PositiveFloat | None = None
    # `ask` counts the evidence it cited, which is what keeps the reinforcement factor in
    # `_ranking_signals` from being pinned at 1.0 for a caller that never calls `reinforce`.
    # Measurement needs it off: reinforcing during a run makes one question's retrieval depend
    # on which earlier questions already answered and, under concurrency, on the order their
    # updates committed, so an evaluation stops being reproducible from its seed.
    reinforce_on_answer: _StrictBool = True
    # UNPROVENANCED AND KNOWN-WRONG IN THE SAFE DIRECTION. Kept only because every alternative
    # available without a measurement is wrong in the unsafe direction. Do not tune this by
    # intuition; the calibration plan is in `docs/configuration.md`.
    #
    # What it costs when wrong. Too high: every encounter fails to match and mints a new person, so
    # voice identity silently stops working — 1667 voice segments produced 1743 identities in a real
    # run, and 0.78 is above CAM++'s same-speaker cosine, which is that failure. Too low: two people
    # merge into one identity, which hands one person's memories to another. Fragmentation costs
    # recall; a false merge is a correctness and privacy failure, so this errs high deliberately.
    #
    # Why upstream's published number does not rescue it. The pinned CAM++ recipe
    # (`iic/speech_campplus_sv_zh-cn_16k-common`, revision `a045b2af`, see `models/funasr.py`)
    # ships `configuration.json` with `"model": {"yesOrno_thr": 0.31}`, and ModelScope's own
    # `speaker_verification_light_pipeline` reads that field as its decision threshold — raw
    # `torch.nn.CosineSimilarity`, `yes` iff `score >= thr` — the same quantity
    # `_accepted_identity` compares. But 0.31 is calibrated for ONE pair of embeddings, and this
    # matcher accepts on a `max` over up to 20 stored exemplars. Max-over-N is monotone in N, so a
    # pair threshold is a strict LOWER bound on the correct max-over-N threshold, never the value:
    # measured on this matcher, the max-over-20 score for mutually orthogonal random 192-d
    # impostors already reaches 0.28 (mean 0.14, up from mean 0.00 at N=1), and real same-language
    # same-channel impostors sit above orthogonal. A crafted cross-person vector scoring 0.45
    # against one bank is falsely merged at 0.31 and refused at 0.78.
    #
    # `speaker_margin` does NOT guard this, and no margin value would. The margin branch is
    # `len(ranked) > 1`, so with a single enrolled identity it never runs at all; with several, the
    # `max` inflates only the winner, so the gap to the runner-up stays wide and the margin passes.
    # The instrument that would work is a threshold calibrated for max-over-N, or an accept decision
    # taken on a single exemplar while the bank continues to rank. Both live in the store.
    speaker_similarity: _UnitInterval = 0.78
    # Keep non-zero: it is the tie-break between two candidate identities, which is a real
    # protection even though it is not a protection against the max-over-N inflation above.
    speaker_margin: _UnitInterval = 0.05
    # Upstream SFace's own `_threshold_cosine`, adopted verbatim — this is what provenance looks
    # like, and what `speaker_similarity` above still lacks. Same max-over-N caveat applies.
    face_similarity: _UnitInterval = 0.363
    face_margin: _UnitInterval = 0.05
    identity_link_min_assets: _PositiveInt = 2
    # The PRESSURE trigger's whole definition: how many active records this instance is meant to
    # hold. `None` -- the default -- means the host has declared no budget, and no amount of
    # growth is evidence that consolidating or forgetting anything is useful, so the trigger
    # derives nothing. A count rather than bytes on disk because that is the quantity the
    # deliberation acts on (records to consolidate or forget) and the one a host can reason about
    # without knowing how media is stored.
    memory_budget_records: _PositiveInt | None = None
    # How far back `consolidation_candidates` looks for repeated recall failures. Two or more
    # near-equal queries that returned nothing inside this window are one QUERY_FAILURE
    # candidate; one failure is not a signal, and a failure from last month is not this week's
    # gap. One hour matches an interactive session.
    query_failure_window_seconds: _PositiveFloat = 3600.0
    # How many recall failures the store keeps at all. The table is a bounded signal buffer, not
    # a query log: the oldest rows falling out is the retention policy.
    query_failure_history: _PositiveInt = 512


# Clearer name for new code; keep the original public value intact for compatibility.
MemorySettings = MemoryConfig
