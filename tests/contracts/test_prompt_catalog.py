"""Prompt provenance contract tests."""

import hashlib

from mindbridge.benchmarks.prompts import BENCHMARK_PROMPTS
from mindbridge.prompts import ALL_PROMPTS, PromptSpec

_EXPECTED_FINGERPRINTS = {
    "active_speaker_v2": "0e6f8763910e580cbc3a9323dde717119eb5fd481e532af3f19c5a6cfb61b1cb",
    "aml_extract_facts_v1": "55d9ba0c29cc04247730e6b756ada69e2a20f46f15af58d0be1bcace97cd2d3d",
    "answer_from_evidence_v10": (
        "5b1fa751f8b9cdc87584a94048b0dcf514948db1c7151edc14e3542a3f29d84b"
    ),
    "consolidate_claims_v2": ("65e3de18448f2879ef8bc83c19290a46cf66d64052230242ae96851a5ecc93a2"),
    "consolidate_episodes_v2": ("f2455d3e72319d68f94a1f4f7f4fab3845f3a9e34f138823013f346228d13c22"),
    "consolidate_summaries_v3": (
        "29fd2da85ecaea2deddbee31a28595feb33ce0171fc0711347a9e9548ad74926"
    ),
    "perceive_events_v9": ("1d130f8eef164d988efdfa4870b3350280faf9d720d27da1f370d89239318792"),
    "segment_speech_v1": ("819e6429099f4d4bc852d1db482dc65c5e19d2f41f6fc1bbe576a32bc6562850"),
    "select_occurrences_v2": ("e5062faf64a439dd5232c132bde032ac677e5712ed7594997124f957cf8c0aa3"),
}
_EXPECTED_BENCHMARK_FINGERPRINTS = {
    "egomem_reason_query_v1": ("be1b4861320908f7575c5b72151bd715f9ac5f3498dfe0de2d8f8dc9faac0bf6"),
    "egotempo_query_v1": "27327bbac7f294f1d6ed16e675bdd8e416ee2c5bca0911967ce68a5c4f60cce8",
    "memlens_query_v1": "0b9b44d0b6c65d9131fb947155bb5a57587bae5d2881504c4dea62b0127a6f2f",
    "video_mme_query_v1": "28b657a26654f73b0d692bcb17de7d160dbb02aab8ad89b2b889edcdc86c4626",
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
