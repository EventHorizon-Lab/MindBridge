"""AML case and chunking tests."""

from mindbridge.benchmarks.aml.cases import (
    MAX_MESSAGES_PER_CHUNK,
    chunk_messages,
)


def _message(content: str) -> dict[str, object]:
    return {"role": "user", "content": content}


def test_chunk_splits_at_twenty_messages() -> None:
    chunks = chunk_messages([_message("hi") for _ in range(45)])

    assert [len(chunk) for chunk in chunks] == [20, 20, 5]
    assert sum(len(chunk) for chunk in chunks) == 45


def test_chunk_splits_before_exceeding_the_word_budget() -> None:
    chunks = chunk_messages([_message("word " * 1_500) for _ in range(3)])

    assert all(len(chunk) <= MAX_MESSAGES_PER_CHUNK for chunk in chunks)
    assert [len(chunk) for chunk in chunks] == [1, 1, 1]


def test_chunk_keeps_a_single_oversized_message_whole() -> None:
    chunks = chunk_messages([_message("word " * 5_000)])

    assert len(chunks) == 1
    assert len(chunks[0]) == 1


def test_chunk_of_nothing_is_empty() -> None:
    assert chunk_messages([]) == ()
