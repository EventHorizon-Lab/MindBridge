from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from mindbridge import (
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    ObservationContext,
    RetrievalScope,
    ValidationError,
)


class DrinkFormer:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "drink-test"
    formation_space = "drink-test:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        results: list[tuple[FormationProposal, ...]] = []
        for item in inputs:
            drink = "coffee" if "coffee" in item.content.text else "tea"
            results.append(
                (
                    FormationProposal(
                        kind=MemoryKind.STATE,
                        content=f"The user's preferred drink is {drink}",
                        subject="user",
                        predicate="preferred_drink",
                        value=drink,
                        confidence=0.95,
                        valid_from=item.context.valid_from,
                        valid_until=item.context.valid_until,
                    ),
                )
            )
        return tuple(results)

    def close(self) -> None:
        pass


def _state_values(memory: Memory, scope: RetrievalScope) -> set[str | None]:
    return {
        hit.context.value
        for hit in memory.search("preferred drink", limit=10, scope=scope)
        if hit.context is not None and hit.context.kind is MemoryKind.STATE
    }


def test_state_corrections_preserve_valid_and_transaction_time(tmp_path: Path) -> None:
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    jan_middle = datetime(2026, 1, 15, tzinfo=timezone.utc)
    feb_middle = datetime(2026, 2, 15, tzinfo=timezone.utc)

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=DrinkFormer(),
        minimum_relevance=0,
    ) as memory:
        tea_source = memory.add(
            "I prefer tea",
            context=ObservationContext(
                basis=EvidenceBasis.USER_STATEMENT,
                valid_from=january,
            ),
        )
        tea_state = next(
            hit
            for hit in memory.search("preferred drink", limit=10)
            if hit.context is not None and hit.context.kind is MemoryKind.STATE
        )
        known_before_correction = datetime.now(timezone.utc)
        memory.add(
            "Correction: I prefer coffee",
            context=ObservationContext(
                basis=EvidenceBasis.USER_STATEMENT,
                valid_from=february,
            ),
        )
        known_after_correction = datetime.now(timezone.utc)

        assert _state_values(
            memory,
            RetrievalScope(valid_at=jan_middle, known_at=known_after_correction),
        ) == {"tea"}
        assert _state_values(
            memory,
            RetrievalScope(valid_at=feb_middle, known_at=known_after_correction),
        ) == {"coffee"}
        assert _state_values(
            memory,
            RetrievalScope(valid_at=feb_middle, known_at=known_before_correction),
        ) == {"tea"}
        assert memory.get(tea_source.id).id == tea_source.id
        retired = memory.get(tea_state.id).context
        assert retired is not None
        assert retired.evidence_ids == (tea_source.id,)


def test_expired_derived_memory_is_soft_forgotten_but_source_remains(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=DrinkFormer(),
        minimum_relevance=0,
    ) as memory:
        source = memory.add(
            "I prefer tea",
            context=ObservationContext(
                basis=EvidenceBasis.USER_STATEMENT,
                valid_from=start,
                valid_until=end,
            ),
        )
        values = _state_values(
            memory,
            RetrievalScope(valid_at=datetime(2026, 1, 3, tzinfo=timezone.utc)),
        )

        assert values == set()
        assert memory.get(source.id).id == source.id


def test_state_can_return_to_an_old_value_without_losing_backfilled_time(
    tmp_path: Path,
) -> None:
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    april = datetime(2026, 4, 1, tzinfo=timezone.utc)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=DrinkFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("I prefer tea", context=ObservationContext(valid_from=january))
        memory.add("I prefer coffee", context=ObservationContext(valid_from=february))
        memory.add("I prefer tea again", context=ObservationContext(valid_from=march))
        known = datetime.now(timezone.utc)

        assert _state_values(
            memory,
            RetrievalScope(valid_at=january, known_at=known),
        ) == {"tea"}
        assert _state_values(
            memory,
            RetrievalScope(valid_at=february, known_at=known),
        ) == {"coffee"}
        assert _state_values(
            memory,
            RetrievalScope(valid_at=april, known_at=known),
        ) == {"tea"}


def test_later_backfill_wins_its_full_valid_interval_without_a_gap(tmp_path: Path) -> None:
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    april = datetime(2026, 4, 1, tzinfo=timezone.utc)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=DrinkFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add(
            "I prefer tea",
            context=ObservationContext(valid_from=february, valid_until=march),
        )
        memory.add(
            "Correction: I preferred coffee",
            context=ObservationContext(valid_from=january, valid_until=april),
        )
        known = datetime.now(timezone.utc)

        assert _state_values(
            memory,
            RetrievalScope(
                valid_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
                known_at=known,
            ),
        ) == {"coffee"}


def test_known_at_never_leaks_a_later_unformed_record(tmp_path: Path) -> None:
    known_before_add = datetime.now(timezone.utc)
    with Memory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0) as memory:
        source = memory.add("future transaction witness")

        assert (
            memory.search(
                "future transaction witness",
                limit=10,
                scope=RetrievalScope(known_at=known_before_add),
            )
            == ()
        )
        assert memory.get(source.id) == source


def test_valid_at_keeps_records_without_typed_validity(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0) as memory:
        observation = memory.add("undated raw observation")

        # An observation declares no validity interval, so no `valid_at` can refute it. Dropping
        # it would turn every historical question into a silent empty answer.
        hits = memory.search(
            "undated raw observation",
            limit=10,
            scope=RetrievalScope(valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        )

        assert [hit.id for hit in hits] == [observation.id]


def test_get_keeps_provenance_for_a_fully_retired_state(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=DrinkFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("I prefer tea", context=ObservationContext(valid_from=start))
        tea = next(
            hit
            for hit in memory.search("preferred drink", limit=10)
            if hit.context is not None and hit.context.value == "tea"
        )
        memory.add("I prefer coffee", context=ObservationContext(valid_from=start))

        retired = memory.get(tea.id).context
        assert retired is not None
        assert retired.retired_at is not None
        assert retired.lineage_id is not None
        assert retired.evidence_ids


def test_deleting_latest_state_source_restores_prior_valid_tail(tmp_path: Path) -> None:
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=DrinkFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("I prefer tea", context=ObservationContext(valid_from=january))
        coffee = memory.add(
            "Correction: I prefer coffee",
            context=ObservationContext(valid_from=february),
        )
        assert _state_values(memory, RetrievalScope(valid_at=march)) == {"coffee"}

        assert memory.delete(coffee.id) is True

        assert _state_values(memory, RetrievalScope(valid_at=march)) == {"tea"}


def test_deleting_latest_derived_state_restores_prior_valid_tail(tmp_path: Path) -> None:
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=DrinkFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("I prefer tea", context=ObservationContext(valid_from=january))
        coffee_source = memory.add(
            "I prefer coffee",
            context=ObservationContext(valid_from=february),
        )
        coffee_state = next(
            hit
            for hit in memory.search("preferred drink", limit=10)
            if hit.context is not None and hit.context.value == "coffee"
        )

        assert memory.delete(coffee_state.id) is True

        assert _state_values(memory, RetrievalScope(valid_at=march)) == {"tea"}
        assert memory.get(coffee_source.id) == coffee_source


def test_deleting_middle_state_source_replays_the_remaining_lineage(tmp_path: Path) -> None:
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=DrinkFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("I prefer tea", context=ObservationContext(valid_from=january))
        coffee = memory.add(
            "I prefer coffee",
            context=ObservationContext(valid_from=february),
        )
        memory.add("I prefer tea again", context=ObservationContext(valid_from=march))

        assert memory.delete(coffee.id) is True

        assert _state_values(
            memory,
            RetrievalScope(valid_at=datetime(2026, 2, 15, tzinfo=timezone.utc)),
        ) == {"tea"}
        assert _state_values(
            memory,
            RetrievalScope(valid_at=datetime(2026, 3, 15, tzinfo=timezone.utc)),
        ) == {"tea"}


def test_one_source_can_form_the_same_state_in_disjoint_intervals(tmp_path: Path) -> None:
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)

    class EvolutionFormer(DrinkFormer):
        def form(
            self,
            inputs: Sequence[FormationInput],
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                (
                    FormationProposal(
                        kind=MemoryKind.STATE,
                        content="The user's preferred drink is tea",
                        subject="user",
                        predicate="preferred_drink",
                        value="tea",
                        valid_from=january,
                        valid_until=february,
                    ),
                    FormationProposal(
                        kind=MemoryKind.STATE,
                        content="The user's preferred drink is coffee",
                        subject="user",
                        predicate="preferred_drink",
                        value="coffee",
                        valid_from=february,
                        valid_until=march,
                    ),
                    FormationProposal(
                        kind=MemoryKind.STATE,
                        content="The user's preferred drink is tea",
                        subject="user",
                        predicate="preferred_drink",
                        value="tea",
                        valid_from=march,
                    ),
                )
                for _value in inputs
            )

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=EvolutionFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("My drink preference changed twice")

        assert _state_values(
            memory,
            RetrievalScope(valid_at=datetime(2026, 1, 15, tzinfo=timezone.utc)),
        ) == {"tea"}
        assert _state_values(
            memory,
            RetrievalScope(valid_at=datetime(2026, 2, 15, tzinfo=timezone.utc)),
        ) == {"coffee"}
        assert _state_values(
            memory,
            RetrievalScope(valid_at=datetime(2026, 3, 15, tzinfo=timezone.utc)),
        ) == {"tea"}


def test_expired_observation_validity_soft_forgets_without_a_former(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=2)
    with Memory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0) as memory:
        expired = memory.add(
            "the guest wifi password is swordfish",
            context=ObservationContext(valid_from=start, valid_until=now - timedelta(days=1)),
        )
        live = memory.add(
            "the front door code is 1234",
            context=ObservationContext(valid_from=start),
        )

        # Default retrieval drops the expired record; identity reads keep it.
        assert [hit.id for hit in memory.search("password code", limit=10)] == [live.id]
        assert memory.get(expired.id).content == "the guest wifi password is swordfish"
        assert {record.id for record in memory.list(limit=10).items} == {expired.id, live.id}

        # A scope inside the expired window still reaches it.
        inside = start + timedelta(hours=1)
        assert expired.id in {
            hit.id
            for hit in memory.search(
                "password code", limit=10, scope=RetrievalScope(valid_at=inside)
            )
        }


def test_valid_until_requires_valid_from() -> None:
    try:
        ObservationContext(valid_until=datetime.now(timezone.utc))
    except ValidationError as error:
        assert "valid_until must be later than valid_from" in str(error)
    else:  # pragma: no cover - the contract is that this raises
        raise AssertionError("valid_until alone must be rejected")
