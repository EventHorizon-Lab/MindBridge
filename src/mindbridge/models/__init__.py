"""Frozen model adapters used by MindBridge."""

from mindbridge.models.jina import (
    DEFAULT_JINA_OMNI_DIMENSION,
    DEFAULT_JINA_OMNI_MODEL_ID,
    DEFAULT_JINA_OMNI_REVISION,
    DEFAULT_JINA_RETRIEVAL_SPACE,
    DEFAULT_JINA_TEXT_MODEL_ID,
    DEFAULT_JINA_TEXT_REVISION,
    JinaModality,
    JinaOmniEmbedder,
)
from mindbridge.models.openai_claim_consolidation import (
    CONSOLIDATE_CLAIMS_PROMPT_VERSION,
    OpenAIOmniClaimConsolidator,
)
from mindbridge.models.openai_consolidation import (
    CONSOLIDATE_EPISODES_PROMPT_VERSION,
    OpenAIOmniEpisodeConsolidator,
)
from mindbridge.models.openai_embeddings import OpenAIJinaEmbedder, OpenAIJinaTextEmbedder
from mindbridge.models.openai_omni import (
    ANSWER_FROM_EVIDENCE_PROMPT_VERSION,
    DEFAULT_OMNI_MODEL_ID,
    OpenAIOmniAnswerer,
    normalize_openai_base_url,
)
from mindbridge.models.openai_perception import (
    PERCEIVE_EVENTS_PROMPT_VERSION,
    OpenAIOmniEventPerceiver,
)
from mindbridge.models.openai_summary_consolidation import (
    CONSOLIDATE_SUMMARIES_PROMPT_VERSION,
    OpenAIOmniSummaryConsolidator,
)

__all__ = [
    "ANSWER_FROM_EVIDENCE_PROMPT_VERSION",
    "CONSOLIDATE_CLAIMS_PROMPT_VERSION",
    "CONSOLIDATE_EPISODES_PROMPT_VERSION",
    "CONSOLIDATE_SUMMARIES_PROMPT_VERSION",
    "DEFAULT_JINA_OMNI_DIMENSION",
    "DEFAULT_JINA_OMNI_MODEL_ID",
    "DEFAULT_JINA_OMNI_REVISION",
    "DEFAULT_JINA_RETRIEVAL_SPACE",
    "DEFAULT_JINA_TEXT_MODEL_ID",
    "DEFAULT_JINA_TEXT_REVISION",
    "DEFAULT_OMNI_MODEL_ID",
    "PERCEIVE_EVENTS_PROMPT_VERSION",
    "JinaModality",
    "JinaOmniEmbedder",
    "OpenAIJinaEmbedder",
    "OpenAIJinaTextEmbedder",
    "OpenAIOmniAnswerer",
    "OpenAIOmniClaimConsolidator",
    "OpenAIOmniEpisodeConsolidator",
    "OpenAIOmniEventPerceiver",
    "OpenAIOmniSummaryConsolidator",
    "normalize_openai_base_url",
]
