"""AML chunk extraction tests."""

from datetime import datetime, timezone

import pytest

from mindbridge.api.aml_contracts import AmlMessage
from mindbridge.application.aml_extraction import extract_memories
from mindbridge.core import MemoryType, ModelOutputError, ModelReference
from mindbridge.models import GenerateRequest, GenerateResult

_NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


class _StubGenerator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.requests.append(request)
        return GenerateResult(
            text=self.text,
            model_reference=ModelReference(model_id="qwen3.8-max", revision="test"),
        )


@pytest.mark.asyncio
async def test_extract_memories_maps_types_and_message_timestamps() -> None:
    generator = _StubGenerator(
        '{"memories": [{"summary": "Rob moved to Sweden.", "type": "episodic"},'
        ' {"summary": "Rob prefers tea.", "type": "semantic"}]}'
    )
    messages = (
        AmlMessage(role="user", content="Rob moved to Sweden.", timestamp=1_704_067_200_000),
    )

    memories = await extract_memories(generator, messages, now=_NOW)

    assert [memory.summary for memory in memories] == [
        "Rob moved to Sweden.",
        "Rob prefers tea.",
    ]
    assert memories[0].memory_type is MemoryType.EPISODIC
    assert memories[1].memory_type is MemoryType.SEMANTIC
    assert memories[0].occurred_at == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert generator.requests[0].json_mode is True


@pytest.mark.asyncio
async def test_extract_memories_falls_back_to_now_without_timestamps() -> None:
    generator = _StubGenerator('{"memories": [{"summary": "Rob likes tea.", "type": "semantic"}]}')
    messages = (AmlMessage(role="user", content="I like tea."),)

    memories = await extract_memories(generator, messages, now=_NOW)

    assert memories[0].occurred_at == _NOW


@pytest.mark.asyncio
async def test_extract_memories_accepts_an_empty_chunk() -> None:
    generator = _StubGenerator('{"memories": []}')
    messages = (AmlMessage(role="user", content="ok"),)

    assert await extract_memories(generator, messages, now=_NOW) == ()


@pytest.mark.asyncio
async def test_extract_memories_rejects_unparseable_output() -> None:
    generator = _StubGenerator("not json")
    messages = (AmlMessage(role="user", content="hello"),)

    with pytest.raises(ModelOutputError):
        await extract_memories(generator, messages, now=_NOW)
