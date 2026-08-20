"""AML chunk extraction tests."""

from datetime import datetime, timezone

import pytest

from mindbridge.api.aml_contracts import AmlMessage
from mindbridge.application.aml_extraction import extract_memories
from mindbridge.core import MemoryType, ModelOutputError, ModelReference
from mindbridge.models import GenerateRequest, GenerateResult, TextPart

_NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


class _StubGenerator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.requests.append(request)
        return GenerateResult(
            text=self.text,
            model_reference=ModelReference(model_id="qwen3.8-max"),
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

    outcome = await extract_memories(generator, messages, now=_NOW)

    assert [memory.summary for memory in outcome.memories] == [
        "Rob moved to Sweden.",
        "Rob prefers tea.",
    ]
    assert outcome.memories[0].memory_type is MemoryType.EPISODIC
    assert outcome.memories[1].memory_type is MemoryType.SEMANTIC
    assert outcome.memories[0].occurred_at == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert outcome.skipped == 0
    assert generator.requests[0].json_mode is True


@pytest.mark.asyncio
async def test_extract_memories_grounds_occurred_at_and_renders_the_timestamp() -> None:
    """Blocking 3 (final review, 2026-08-17): the extractor never saw a
    message's timestamp -- `_render_chunk` rendered only `role: content`. This
    both left `occurred_at` unable to be grounded from the actual
    conversation (a loss `min(timestamps)` alone can't fix if the model never
    sees an absolute anchor for relative expressions like "last week") and
    made the rendered chunk untestable for carrying that anchor at all.
    """
    generator = _StubGenerator(
        '{"memories": [{"summary": "Rob moved to Sweden.", "type": "episodic"}]}'
    )
    messages = (
        AmlMessage(role="user", content="I moved last week.", timestamp=1_704_067_200_000),
        AmlMessage(role="assistant", content="Nice!", timestamp=1_704_153_600_000),
    )

    outcome = await extract_memories(generator, messages, now=_NOW)

    # occurred_at is grounded from the earliest message timestamp, not `now`.
    assert outcome.memories[0].occurred_at == datetime(2024, 1, 1, tzinfo=timezone.utc)

    # The rendered chunk carries a readable anchor for each message, not just
    # role/content, so the extractor can ground relative expressions.
    [request] = generator.requests
    [part] = request.input.parts
    assert isinstance(part, TextPart)
    assert "2024-01-01" in part.text
    assert "2024-01-02" in part.text


@pytest.mark.asyncio
async def test_extract_memories_falls_back_to_now_without_timestamps() -> None:
    generator = _StubGenerator('{"memories": [{"summary": "Rob likes tea.", "type": "semantic"}]}')
    messages = (AmlMessage(role="user", content="I like tea."),)

    outcome = await extract_memories(generator, messages, now=_NOW)

    assert outcome.memories[0].occurred_at == _NOW


@pytest.mark.asyncio
async def test_extract_memories_accepts_an_empty_chunk() -> None:
    generator = _StubGenerator('{"memories": []}')
    messages = (AmlMessage(role="user", content="ok"),)

    outcome = await extract_memories(generator, messages, now=_NOW)

    assert outcome.memories == ()
    assert outcome.skipped == 0


@pytest.mark.asyncio
async def test_extract_memories_rejects_unparseable_output() -> None:
    generator = _StubGenerator("not json")
    messages = (AmlMessage(role="user", content="hello"),)

    with pytest.raises(ModelOutputError):
        await extract_memories(generator, messages, now=_NOW)


@pytest.mark.asyncio
async def test_extract_memories_skips_a_non_dict_item_without_losing_siblings() -> None:
    generator = _StubGenerator(
        '{"memories": [{"summary": "Rob moved to Sweden.", "type": "episodic"},'
        ' "not an object",'
        ' {"summary": "Rob prefers tea.", "type": "semantic"}]}'
    )
    messages = (AmlMessage(role="user", content="hello"),)

    outcome = await extract_memories(generator, messages, now=_NOW)

    assert [memory.summary for memory in outcome.memories] == [
        "Rob moved to Sweden.",
        "Rob prefers tea.",
    ]
    assert outcome.skipped == 1


@pytest.mark.asyncio
async def test_extract_memories_skips_an_unrecognized_type() -> None:
    generator = _StubGenerator('{"memories": [{"summary": "Rob likes dogs.", "type": "opinion"}]}')
    messages = (AmlMessage(role="user", content="hello"),)

    outcome = await extract_memories(generator, messages, now=_NOW)

    assert outcome.memories == ()
    assert outcome.skipped == 1


@pytest.mark.asyncio
async def test_extract_memories_skips_a_blank_summary() -> None:
    generator = _StubGenerator('{"memories": [{"summary": "   ", "type": "semantic"}]}')
    messages = (AmlMessage(role="user", content="hello"),)

    outcome = await extract_memories(generator, messages, now=_NOW)

    assert outcome.memories == ()
    assert outcome.skipped == 1


@pytest.mark.asyncio
async def test_extract_memories_rejects_a_payload_without_memories_key() -> None:
    generator = _StubGenerator('{"notes": []}')
    messages = (AmlMessage(role="user", content="hello"),)

    with pytest.raises(ModelOutputError):
        await extract_memories(generator, messages, now=_NOW)
