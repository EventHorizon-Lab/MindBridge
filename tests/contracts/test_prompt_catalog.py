"""Prompt provenance contract tests."""

import hashlib

from mindbridge.benchmarks.prompts import BENCHMARK_PROMPTS
from mindbridge.prompts import ALL_PROMPTS, PromptSpec

_EXPECTED_FINGERPRINTS = {
    "active_speaker_v3": "9eba83fb24bb4c2a0eb9f69f76cb949b70571b54df8659364e68e1a96cfc023f",
    "aml_extract_facts_v1": "55d9ba0c29cc04247730e6b756ada69e2a20f46f15af58d0be1bcace97cd2d3d",
    "answer_from_evidence_v12": (
        "6a939ed81473a45bfc28267b5501849fe2e5e8acee7c023d5c97bed27913ab70"
    ),
    "consolidate_claims_v4": "4c075d9b5fe18a737bf93f3b4a99c3a84f3401c705482e4f166b89a116794693",
    "consolidate_episodes_v3": "40963f8c2a177f2172672bbbb4be74539a5560a62f44cbffcfa171918859f931",
    "consolidate_summaries_v4": (
        "05e2c2e1c06f73ad1fb177efa719f5b1a6bdffbeebbb2f9a52f404150b1bbd09"
    ),
    "perceive_events_v11": "4b37da84a5ecddb560f8a16facbead56e526f220899453cfc67c55a32b6c6c3c",
    "resolve_entities_v1": "413e59c736ac9478ff9595c03e7282459154c2f192f9cc5413793f601541c993",
    "segment_speech_v1": "819e6429099f4d4bc852d1db482dc65c5e19d2f41f6fc1bbe576a32bc6562850",
    "select_occurrences_v2": "e5062faf64a439dd5232c132bde032ac677e5712ed7594997124f957cf8c0aa3",
}
_EXPECTED_BENCHMARK_FINGERPRINTS = {
    "egomem_reason_query_v1": ("be1b4861320908f7575c5b72151bd715f9ac5f3498dfe0de2d8f8dc9faac0bf6"),
    "egotempo_query_v1": "27327bbac7f294f1d6ed16e675bdd8e416ee2c5bca0911967ce68a5c4f60cce8",
    "memlens_query_v1": "0b9b44d0b6c65d9131fb947155bb5a57587bae5d2881504c4dea62b0127a6f2f",
    "video_mme_query_v1": "28b657a26654f73b0d692bcb17de7d160dbb02aab8ad89b2b889edcdc86c4626",
    "video_mme_v2_query_v1": ("96c676324eb0a2e4b02531f2ea0303980c6033a5b4cb442f17fa1fbab72820c1"),
}


def test_prompt_catalog_has_stable_versioned_content() -> None:
    """Changing prompt text without reviewing its version must fail this contract."""
    _require_versioned_provenance(ALL_PROMPTS)
    assert _fingerprints(ALL_PROMPTS) == _EXPECTED_FINGERPRINTS


def test_benchmark_prompts_stay_out_of_the_production_catalog() -> None:
    """Official dataset wordings are evaluation inputs, not part of the product contract."""
    _require_versioned_provenance(BENCHMARK_PROMPTS)
    assert _fingerprints(BENCHMARK_PROMPTS) == _EXPECTED_BENCHMARK_FINGERPRINTS
    assert all(prompt.used_by.startswith("mindbridge.benchmarks.") for prompt in BENCHMARK_PROMPTS)
    assert not {prompt.name for prompt in ALL_PROMPTS} & {
        prompt.name for prompt in BENCHMARK_PROMPTS
    }
    assert all(not prompt.used_by.startswith("mindbridge.benchmarks.") for prompt in ALL_PROMPTS)


def _require_versioned_provenance(prompts: tuple[PromptSpec, ...]) -> None:
    assert len({prompt.name for prompt in prompts}) == len(prompts)
    assert all(prompt.version.startswith(f"{prompt.name}_v") for prompt in prompts)
    assert all(prompt.purpose and prompt.used_by for prompt in prompts)


def _fingerprints(prompts: tuple[PromptSpec, ...]) -> dict[str, str]:
    return {
        prompt.version: hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        for prompt in prompts
    }
