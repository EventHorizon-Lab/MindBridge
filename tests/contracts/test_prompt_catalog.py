"""Prompt provenance contract tests."""

import hashlib

from mindbridge.benchmarks.prompts import BENCHMARK_PROMPTS
from mindbridge.prompts import ALL_PROMPTS, PromptSpec

_EXPECTED_FINGERPRINTS = {
    "active_speaker_v3": "9eba83fb24bb4c2a0eb9f69f76cb949b70571b54df8659364e68e1a96cfc023f",
    "aml_extract_facts_v1": "55d9ba0c29cc04247730e6b756ada69e2a20f46f15af58d0be1bcace97cd2d3d",
    "answer_from_evidence_v13": (
        "3ad311c35132f97182a8d50363d9c7ab177ee93d7c4d3606a2945ed3e5a89c7c"
    ),
    "consolidate_claims_v4": "4c075d9b5fe18a737bf93f3b4a99c3a84f3401c705482e4f166b89a116794693",
    "consolidate_episodes_v3": "40963f8c2a177f2172672bbbb4be74539a5560a62f44cbffcfa171918859f931",
    "consolidate_summaries_v4": (
        "05e2c2e1c06f73ad1fb177efa719f5b1a6bdffbeebbb2f9a52f404150b1bbd09"
    ),
    "perceive_events_v12": "51919991687649817ed1466dbb9be42980ab4d9099093f19607bc0ce42ab98c2",
    "resolve_entities_v1": "413e59c736ac9478ff9595c03e7282459154c2f192f9cc5413793f601541c993",
    "segment_speech_v1": "819e6429099f4d4bc852d1db482dc65c5e19d2f41f6fc1bbe576a32bc6562850",
    "select_occurrences_v2": "e5062faf64a439dd5232c132bde032ac677e5712ed7594997124f957cf8c0aa3",
}
_EXPECTED_BENCHMARK_FINGERPRINTS = {
    "atm_bench_query_v1": "db075ed927d575e3fcfd468e481f91f79e1b05e8a83af61d312b686d249a240f",
    "atm_bench_number_format_v1": (
        "eb8993f2f4cdb105e3a865b631670d993016c9e5fefaa53fe0c0317c0e1e26fc"
    ),
    "atm_bench_list_recall_format_v1": (
        "93e503847d7e359446907a401446269a9e0245c5fe2de80efbf3dd3e7b441b01"
    ),
    "atm_bench_open_end_format_v1": (
        "31f6be646fde418dea229240c5b7883c08112d3cb782255f18b75ce7a204604c"
    ),
    "egomem_reason_query_v1": ("be1b4861320908f7575c5b72151bd715f9ac5f3498dfe0de2d8f8dc9faac0bf6"),
    "egotempo_query_v1": "27327bbac7f294f1d6ed16e675bdd8e416ee2c5bca0911967ce68a5c4f60cce8",
    "memlens_query_v1": "0b9b44d0b6c65d9131fb947155bb5a57587bae5d2881504c4dea62b0127a6f2f",
    "mem_gallery_conflict_v1": "fa5b1f1961f7a0bd3575371f5d1100197644bd62307d9fc927458a7368c6eab3",
    "mem_gallery_query_v1": "bb9386c15cfd4b49d5007e65cc4bb88b34067d83e8f7ee37f94ee8688bf310c0",
    "mem_gallery_refusal_v1": "da6381ac3fdccb029389e96f28dd7c37c416a2fdccc80645470311216d66d6da",
    "mem_gallery_search_v1": "340f461de7fba952c39a9fc4f82b36edb60bc17ec7cd076307f868941bb70f52",
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
