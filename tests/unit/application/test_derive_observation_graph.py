"""What one validated perception becomes: the text a reader sees, and who a person is."""

from datetime import datetime, timedelta, timezone

from mindbridge.application.derive_observation_graph import (
    DerivedObservationGraph,
    derive_observation_graph,
)
from mindbridge.application.perception import (
    EventPerception,
    PerceivedClaim,
    PerceivedCount,
    PerceivedEvent,
)
from mindbridge.core import (
    AnonymousIdentityObservation,
    ClaimType,
    DeviceId,
    Event,
    EventId,
    EvidenceId,
    EvidenceSpan,
    IdentityKind,
    IdentityScope,
    MediaObjectId,
    ModelReference,
    Observation,
    ObservationId,
    RelationType,
    SensorKind,
    TenantId,
)

NOW = datetime(2026, 8, 25, 9, 0, tzinfo=timezone.utc)
TENANT = TenantId("tenant_01")
MODEL = ModelReference(model_id="qwen/qwen3.5-omni")
EVIDENCE = EvidenceId("evidence_01")
FIRST_OBSERVATION = ObservationId("observation_01")
PROMPT_VERSION = "perceive_events_v11"


def _perceived_claim(exact_count: PerceivedCount | None) -> PerceivedClaim:
    """The MM-Lifelong worked example: the store held a caption, the question asked how many."""
    return PerceivedClaim(
        claim_type=ClaimType.FACT,
        statement="Small monsters attack the player character in the clearing.",
        confidence=0.8,
        evidence_ids=(EVIDENCE,),
        valid_from_ms=0,
        valid_to_ms=12_000,
        exact_count=exact_count,
    )


def _graph_of(
    claim: PerceivedClaim,
    observation: Observation | None = None,
) -> DerivedObservationGraph:
    observation = observation or _observation()
    perceived = PerceivedEvent(
        start_ms=0,
        end_ms=12_000,
        description="The player character fights through a forest clearing.",
        salience=0.6,
        evidence_ids=(EVIDENCE,),
        claims=(claim,),
    )
    event = Event(
        event_id=EventId("event_01"),
        tenant_id=TENANT,
        observation_ids=(observation.observation_id,),
        evidence_ids=(EVIDENCE,),
        occurred_at=observation.occurred_at,
        ended_at=observation.occurred_at + timedelta(milliseconds=12_000),
        description=perceived.description,
        salience=perceived.salience,
        created_at=NOW,
        model_reference=MODEL,
        prompt_version=PROMPT_VERSION,
    )
    return derive_observation_graph(
        observation,
        EventPerception(
            events=(perceived,),
            model_reference=MODEL,
            prompt_version=PROMPT_VERSION,
        ),
        (event,),
        (_evidence_span(observation),),
        NOW,
    )


def test_a_counted_claim_carries_its_number_in_the_text_a_reader_gets() -> None:
    """The typed count is only worth having if it reaches the one field recall hands the model.

    The answer pipeline is given `MemoryRecord.summary` and nothing else about a claim, so a
    count stored beside the statement would be invisible to every question that needs it.
    """
    graph = _graph_of(_perceived_claim(PerceivedCount(subject="small monsters", value=3)))

    statement = graph.claims[0].statement
    assert statement.endswith("(exactly 3 small monsters)")
    assert [memory.summary for memory in graph.memories if memory.summary == statement]


def test_an_uncounted_claim_is_left_exactly_as_it_was_written() -> None:
    """Nothing is appended when there is no count, so an absent one costs the text nothing."""
    graph = _graph_of(_perceived_claim(None))

    assert (
        graph.claims[0].statement == "Small monsters attack the player character in the clearing."
    )


def test_two_claims_differing_only_in_their_count_are_two_records() -> None:
    """The count is content, so it has to reach the identity the retry-stable ID is derived from.

    Hashing the unrendered statement would file "exactly 3" and "exactly 5" under one claim ID,
    and the strict writer would then reject the second as a conflicting identity for the first.
    """
    three = _graph_of(_perceived_claim(PerceivedCount(subject="small monsters", value=3)))
    five = _graph_of(_perceived_claim(PerceivedCount(subject="small monsters", value=5)))

    assert three.claims[0].claim_id != five.claims[0].claim_id


def test_a_device_identity_is_one_person_across_observations() -> None:
    """A device-scoped pseudonym is an edge match against an enrolled gallery, so it is durable.

    This is the key recall's co-mention expansion walks to reach the other clips a person is in,
    and it must not change when the same face is seen again in the next observation.
    """
    first = _graph_of(_perceived_claim(None), _observation(ObservationId("observation_01")))
    second = _graph_of(_perceived_claim(None), _observation(ObservationId("observation_02")))

    assert _person_ids(first) == _person_ids(second)


def test_an_observation_scoped_identity_is_not_reused_by_the_next_observation() -> None:
    """The scope says the pseudonym is only safe inside its own observation; the graph must obey.

    A within-clip diarization label is picked per clip, so the same string in two observations is
    two strangers. Keyed verbatim, every caller-supplied `speaker_0` became one tenant-wide
    person and the MENTIONS edges then joined their events -- a merge asserted from nothing.
    """
    first = _graph_of(
        _perceived_claim(None),
        _observation(ObservationId("observation_01"), scope=IdentityScope.OBSERVATION),
    )
    second = _graph_of(
        _perceived_claim(None),
        _observation(ObservationId("observation_02"), scope=IdentityScope.OBSERVATION),
    )

    assert _person_ids(first) and _person_ids(second)
    assert _person_ids(first) != _person_ids(second)


def _person_ids(graph: DerivedObservationGraph) -> set[str]:
    """The identity-backed people one observation contributed, named by their entity IDs."""
    entities = {entity.entity_id for entity in graph.entities if entity.canonical_name is None}
    mentioned = {
        relation.target_id
        for relation in graph.relations
        if relation.relation_type is RelationType.MENTIONS
    }
    # Asserting over a collection that could legitimately be empty: an entity nothing mentions is
    # not reachable, so the intersection is what the test is actually about.
    return {str(entity_id) for entity_id in entities & mentioned}


def _observation(
    observation_id: ObservationId = FIRST_OBSERVATION,
    *,
    scope: IdentityScope = IdentityScope.DEVICE,
) -> Observation:
    return Observation(
        observation_id=observation_id,
        tenant_id=TENANT,
        device_id=DeviceId("device_01"),
        boot_id="boot_01",
        sequence=1,
        sensor=SensorKind.CAMERA,
        media_object_ids=(MediaObjectId("media_01"),),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=12),
        observed_at=NOW,
        clock_offset_ms=0,
        identity_observations=(
            AnonymousIdentityObservation(
                identity_id="speaker_0",
                kind=IdentityKind.VOICE,
                start_ms=0,
                end_ms=8_000,
                confidence=0.9,
                model_reference=ModelReference(model_id="funasr/campplus"),
                scope=scope,
            ),
        ),
    )


def _evidence_span(observation: Observation) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=EVIDENCE,
        tenant_id=TENANT,
        observation_id=observation.observation_id,
        media_object_id=MediaObjectId("media_01"),
        start_ms=0,
        end_ms=12_000,
        created_at=NOW,
    )
