# Evidence selection and provider input boundaries — 2026-10-27

## Decision

R04 is accurate for `ask`: `evidence_budget_chars` is a soft expansion budget. It is not a
provider input limit. Preserve that retrieval contract and repair the boundary accounting,
error classification, and guidance. This review used checkout `b6b5ad28` plus the changes
described here; concurrent formation, ranking, and local-index work is outside this review.

The responsibilities differ:

- The answering plane selects sufficient evidence. Its mandatory ranked window includes the
  existing modality floor, then it admits supplementary evidence while the budget allows.
- The model adapter assembles system instructions, question, serialized evidence, provenance,
  labels, and native media. Its payload safety bounds use bytes and media counts.
- The configured provider applies its tokenizer, media processing, and context window,
  including reserved output/reasoning capacity. MindBridge has no universal token preflight.
- `compile_context` independently enforces a hard character budget on its rendered bundle.
  Its media text-equivalent charges still do not constitute a provider token guarantee.

Turning the existing expansion setting into a hard cap would change retrieval policy and lose
evidence even for a caller that requested a minimum ranked window. It would also leave system
instructions and model-specific media costs unbounded in token units. A count of records cannot
fix that either: one record may exceed a small model's entire input capacity.

The [baseline loss decomposition](2026-09-12-baseline-loss-decomposition.md) records why the
window was widened, including gains, additional input cost, and corpora not remeasured. It is
historical evidence for retaining the policy, not a fresh measurement of this checkout or proof
that wider context always improves quality. No new retrieval-quality result is claimed here.

## Current external practice

Sources checked on 2026-10-27:

- OpenAI's [token counting guide](https://developers.openai.com/api/docs/guides/token-counting)
  recommends counting the actual structured input, including framing and media. Its endpoint
  accepts Responses input. This project emits Chat Completions and supports arbitrary compatible
  servers, so counting a translated Responses request would not establish an exact count for
  the actual request. Adding a generic tokenizer or dividing characters by four would not
  establish that guarantee either.
- Anthropic's [token counting guide](https://platform.claude.com/docs/en/build-with-claude/token-counting)
  includes system prompts, tools, images, and PDFs, with endpoint-specific restrictions, and
  describes counting as an estimate. A counter's capability must be explicit for the chosen
  provider and request format.
- The [vLLM model configuration](https://docs.vllm.ai/en/latest/api/vllm/config/model/)
  defines `max_model_len` over prompt and output. Server capacity is independent of the kernel's
  evidence packing heuristic.
- The recent [EPGM preprint](https://www.alphaxiv.org/pdf/2609.25913) frames retrieval as evidence
  completion under a token budget: “No single fragment provides complete support within
  the retrieval budget.” Its retrieval
  objective supports preserving complete supporting evidence; its experiments do not validate
  MindBridge's policy or supply a provider-independent tokenizer.

## Changes and compatibility

The 4 MiB answer-message text bound now includes the system prompt and each emitted attachment
label, as well as the serialized user text. It is checked before reading native media. The
on-demand media diagnostic also charges its system prompt. Existing inline-media byte and video
count limits remain independent. These checks bound message text, not HTTP JSON framing,
provider extensions supplied through `extra_body`, or provider tokens.
Both local text guards retain `payload_too_large` and now identify stage `generate`.

Provider `context_length_exceeded` now takes precedence over generic bad-request and media-error
handling. Compatible-server context recognition remains available when no semantic code is
present, including vLLM's numeric HTTP-status code. An explicit different semantic code cannot
be overridden by incidental message words. A final answer request is not
retried with evidence removed after context overflow. The separate, recorded embedding fallback
from an oversized video to ordered stills remains covered.

Compatibility change: a recognized provider context overflow now reports
`ModelError(reason="context_length_exceeded")`, `retryable=false`, and HTTP 413, previously
`request_rejected`/502. No signature, dependency, schema, prompt wording, or default evidence
selection changed. REST and Python references document the new reason.

## Validation scope

Acceptance fixtures cover UTF-8 message bytes, exact-limit admission and one-byte-over rejection,
text and image answers, system-prompt rejection before media reads or HTTP, the optional media
diagnostic, coded and no-code context errors, ordinary bad requests, media-error precedence,
streaming parity, and the permanent REST recovery contract. The existing public SDK fixture
retains three 100-character records with `limit=3` and `evidence_budget_chars=10`; the independent
context compiler tests continue to exercise its hard rendered-character ceiling.

No real model service was called and no model revision was evaluated. The transport fixtures
exercise request construction and failure handling; they establish no new answer-quality claim.
The task changes neither retrieval selection nor answer/diagnostic prompt wording.
