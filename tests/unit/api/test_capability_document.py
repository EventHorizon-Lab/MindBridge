"""Each surface's capability document must promise only what that surface can serve.

`MemoryCapabilities.operations` is derived from the declared backends, but a derivation is not a
routing table. The memory control plane is SDK-only by design (`docs/context-os.md`), so
`consolidate` has no `/v1` route and none of the fifteen MCP tools, and the embodied operations
reach a route or a tool only when the host opts in. An agent that read `consolidate` in the MCP
greeting would go looking for a tool that is not there, so every surface narrows the list to what
it serves -- except `mindbridge doctor`, which reports for a CLI that does have the command.
"""

import json
from typing import Any, cast

from fastapi.testclient import TestClient

from mindbridge import cli
from mindbridge.api import app as rest
from mindbridge.api import mcp as mcp_adapter
from mindbridge.types import MemoryCapabilities, Modality

# Every optional backend declared at once, so each operation is derivable and the publishing
# surface is the only thing that can be the reason one is missing.
EVERYTHING = MemoryCapabilities(
    embedding=frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO}),
    embedding_model="jina-v5-omni",
    embedding_space="space_1",
    embedding_dimension=8,
    generation=frozenset({Modality.TEXT}),
    transcription=frozenset({Modality.AUDIO}),
    vision=frozenset({Modality.IMAGE}),
    face=frozenset({Modality.IMAGE}),
    formation=frozenset({Modality.TEXT}),
    generation_model="qwen3-omni",
    transcription_space="funasr-nano:cam++",
    vision_model="qwen3-omni",
    face_model="insightface",
    formation_model="qwen3-omni",
    consolidation_model="qwen3-omni",
    speaker_recognition=True,
)
_EMBODIED = frozenset({"speech", "faces"})


class _Declared:
    """Adapter construction reads the capability declaration and nothing else."""

    @property
    def capabilities(self) -> MemoryCapabilities:
        return EVERYTHING


class _Embedder:
    """The smallest thing `declared_capabilities` accepts, so doctor needs no store."""

    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "jina-v5-omni"
    embedding_space = "space_1"
    embedding_dimension = 8


class _Consolidator:
    consolidation_model = "qwen3-omni"
    consolidation_recipe = "qwen3-omni:consolidate"


def _healthz(*, embodied_operations: bool) -> dict[str, Any]:
    app = rest.create_app(memory=cast(Any, _Declared()), embodied_operations=embodied_operations)
    with TestClient(app) as client:
        body = client.get("/healthz").json()
    return cast(dict[str, Any], body["capabilities"])


def _greeting(*, embodied_operations: bool, write_operations: bool = True) -> dict[str, Any]:
    server = mcp_adapter.build_mcp_server(
        cast(Any, _Declared()),
        embodied_operations=embodied_operations,
        write_operations=write_operations,
    )
    instructions = server.instructions
    assert instructions is not None
    return cast(dict[str, Any], json.loads(instructions[instructions.index("{") :]))


def test_no_network_surface_advertises_the_consolidation_it_cannot_serve() -> None:
    """The control plane has no route and no tool, so neither may name it however it is declared."""
    assert "consolidate" in EVERYTHING.operations

    assert "consolidate" not in _healthz(embodied_operations=True)["operations"]
    assert "consolidate" not in _greeting(embodied_operations=True)["operations"]


def test_only_the_operation_list_is_narrowed() -> None:
    """The declared backends are the same everywhere; only what a surface routes to differs."""
    published = _healthz(embodied_operations=True)
    declared = EVERYTHING.document()

    assert published["operations"] != declared["operations"]
    assert {name: value for name, value in published.items() if name != "operations"} == {
        name: value for name, value in declared.items() if name != "operations"
    }


def test_a_withheld_embodied_group_is_withheld_from_the_document_too() -> None:
    """With the switch off the routes and tools are never registered, so nothing serves them."""
    assert EVERYTHING.operations >= _EMBODIED

    for published in (_healthz(embodied_operations=False), _greeting(embodied_operations=False)):
        assert not _EMBODIED & frozenset(published["operations"])
    for published in (_healthz(embodied_operations=True), _greeting(embodied_operations=True)):
        assert frozenset(published["operations"]) >= _EMBODIED


def test_a_read_only_mcp_server_does_not_advertise_the_write_path_operations() -> None:
    """Formation and captioning run on the write path only, so nothing reaches them read-only.

    `add_memory` is the only tool that forms -- `settle` and `capture` have none -- and vision
    captions are derived where a frame is stored, so a read-only server that still named either
    promised an agent a capability it had no way to invoke. `transcribe` is different: a read
    tool transcribes the audio a question arrives as. REST keeps all of them, because
    `createMemory` is always registered there.
    """
    write_path = frozenset({"formation", "describe_vision"})
    assert EVERYTHING.operations >= write_path

    read_only = frozenset(_greeting(embodied_operations=True, write_operations=False)["operations"])
    assert not write_path & read_only
    assert "transcribe" in read_only
    assert frozenset(_greeting(embodied_operations=True)["operations"]) >= write_path
    assert frozenset(_healthz(embodied_operations=True)["operations"]) >= write_path


def test_the_two_network_surfaces_narrow_one_document_identically() -> None:
    """REST and MCP mirror the same switches, so their two filters must not drift apart."""
    for embodied in (False, True):
        assert (
            _healthz(embodied_operations=embodied)["operations"]
            == _greeting(embodied_operations=embodied)["operations"]
        )


def test_doctor_still_publishes_the_whole_derivation() -> None:
    """`mindbridge consolidate` exists, so the CLI's document is the derivation, unnarrowed."""
    probed = cli._doctor_capabilities({"embedder": _Embedder(), "consolidator": _Consolidator()})

    assert probed is not None
    assert "consolidate" in cast(list[str], probed["operations"])
