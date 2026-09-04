"""The three data-subject rights: consent, export, and retention.

Every test drives the public SDK. What they assert is that a person's own statement restrains the
kernel, that an export answers with everything held about one subject and nothing about anybody
else, and that retention deletes exactly what a declared policy names.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    AssetRef,
    Blob,
    ConsentClaim,
    ConsentState,
    ContextBudget,
    ContextUnknownKind,
    EvidenceBasis,
    FaceAnalysis,
    FaceEmbedding,
    FormationInput,
    FormationProposal,
    IdentityNotFoundError,
    Memory,
    MemoryIntent,
    MemoryKind,
    MemoryNotFoundError,
    MemoryOperation,
    MemoryRecord,
    MemoryTrigger,
    MemoryType,
    Modality,
    NamedActor,
    RetentionPolicy,
    RetentionReport,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
    ValidationError,
    cli,
)

OCCURRED = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)


def _people(name: str | None) -> tuple[str, ...]:
    """Return which people one asset contains, from the leading characters of its name.

    `a-1.mp4` is one person, `b-1.mp4` is another, and `ab-1.wav` is a clip they share.
    """
    if name is not None and name.startswith("ab"):
        return ("a", "b")
    return ("b",) if name is not None and name.startswith("b") else ("a",)


def _person(name: str | None) -> str:
    return _people(name)[0]


def _drift(name: str | None) -> float:
    """Return a small per-asset offset, so two clips of one person enroll two exemplars.

    Keyed on the digit in the asset's name (`a-1.mp4`), not on its content hash, so the test
    controls exactly which observations are distinguishable.
    """
    digit = 0 if name is None or len(name) < 3 or not name[2].isdigit() else int(name[2])
    return 0.02 * (digit + 1)


class PersonSpeech:
    """One voice per person, drifting slightly per asset so a second exemplar is enrollable."""

    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "person-speech"
    transcription_space = "person-speech:test"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        analyses = []
        for asset in assets:
            drift = _drift(asset.name)
            heard = _people(asset.name)
            speakers = []
            turns = []
            for index, person in enumerate(heard):
                base = (1.0, 0.0) if person == "a" else (0.0, 1.0)
                values = (base[0] + drift, base[1] + drift)
                speakers.append(SpeakerEmbedding(str(index), values))
                turns.append(
                    SpeechTurn(index * 900, (index + 1) * 900, "I live next door", str(index))
                )
            analyses.append(SpeechAnalysis(turns=tuple(turns), speakers=tuple(speakers)))
        return tuple(analyses)

    def close(self) -> None:
        pass


class PersonFace:
    """One face per person, drifting slightly per asset so a second exemplar is enrollable."""

    face_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    face_model = "person-face"
    face_space = "person-face:2:test"
    face_analysis_space = "person-face-analysis:test"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        analyses = []
        for asset in assets:
            base = (0.0, 1.0) if _person(asset.name) == "a" else (1.0, 0.0)
            drift = _drift(asset.name)
            values = (base[0] + drift, base[1] + drift)
            analyses.append(
                FaceAnalysis((FaceEmbedding("face-0", values, (0.1, 0.1, 0.4, 0.5), None),))
            )
        return tuple(analyses)

    def close(self) -> None:
        pass


class ScriptedConsolidator:
    """Proposes one fixed batch, so a refused intent can be observed through the public report."""

    consolidation_model = "consolidator-test"
    consolidation_recipe = "consolidator-test:v1"

    def __init__(self, *operations: MemoryOperation) -> None:
        self.operations = tuple(operations)

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        return self.operations

    def close(self) -> None:
        pass


def _memory(tmp_path: Path, **kwargs: object) -> Memory:
    return Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        transcriber=PersonSpeech(),
        face_analyzer=PersonFace(),
        identity_link_min_assets=1,
        minimum_relevance=0,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------------------------
# Consent


def test_consent_is_a_reversible_assertion_in_the_operation_log(tmp_path: Path) -> None:
    """Recording consent logs it, supersedes the previous statement, and rolls back to it."""
    with _memory(tmp_path) as memory:
        clip = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        identity_id = memory.faces(clip.id)[0].identity_id

        assert memory.consent(identity_id) is None

        granted = memory.record_consent(identity_id, ConsentState.GRANTED, note="said yes")
        assert granted is not None
        assert granted.operation.intent is MemoryIntent.CONSENT
        assert granted.operation.consent == ConsentClaim(
            identity_id=identity_id,
            state=ConsentState.GRANTED,
            note="said yes",
        )
        assert memory.consent(identity_id) is ConsentState.GRANTED
        # Restating what already stands changes nothing and logs nothing.
        assert memory.record_consent(identity_id, ConsentState.GRANTED, note="said yes") is None

        withdrawn = memory.record_consent(identity_id, ConsentState.WITHDRAWN)
        assert withdrawn is not None
        assert memory.consent(identity_id) is ConsentState.WITHDRAWN

        logged = [record for record in memory.operations() if record.operation.consent is not None]
        assert [record.operation.consent.state for record in logged] == [  # type: ignore[union-attr]
            ConsentState.WITHDRAWN,
            ConsentState.GRANTED,
        ]

        # Reversing the withdrawal restores exactly the statement it superseded.
        assert memory.rollback(withdrawn.operation_id) is True
        assert memory.consent(identity_id) is ConsentState.GRANTED
        assert memory.rollback(granted.operation_id) is True
        assert memory.consent(identity_id) is None

        # Naming is a separate lineage, so neither statement disturbed the other.
        memory.register_identity(identity_id, "Ann")
        assert memory.record_consent(identity_id, ConsentState.WITHHELD) is not None
        profile = memory.identity(identity_id)
        assert profile is not None and profile.name == "Ann"
        assert memory.consent(identity_id) is ConsentState.WITHHELD


def test_consent_refuses_an_unknown_identity_and_an_invalid_state(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        with pytest.raises(IdentityNotFoundError):
            memory.record_consent("identity_missing", ConsentState.GRANTED)
        with pytest.raises(ValidationError):
            memory.record_consent("identity_1", "granted")  # type: ignore[arg-type]


def test_no_model_may_state_or_retire_consent(tmp_path: Path) -> None:
    """A `CONSENT` proposal is refused as unauthorized, and a standing one cannot be retired.

    Consent is the one claim whose whole value is that a person made it. A backend that could
    propose one could manufacture permission; a backend that could forget one could withdraw a
    refusal by tidying it away.
    """
    with _memory(tmp_path) as memory:
        clip = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        identity_id = memory.faces(clip.id)[0].identity_id
        recorded = memory.record_consent(identity_id, ConsentState.WITHDRAWN)
        assert recorded is not None
        (assertion_id,) = recorded.created_ids

    proposed = MemoryOperation(
        intent=MemoryIntent.CONSENT,
        consent=ConsentClaim(identity_id=identity_id, state=ConsentState.GRANTED),
    )
    with _memory(tmp_path, consolidator=ScriptedConsolidator(proposed)) as memory:
        assert [reason for _operation, reason in memory.consolidate().rejected] == ["unauthorized"]
        # The same refusal on the public replay path, which is not a shortcut around it either.
        with pytest.raises(ValidationError) as refused:
            memory.apply(proposed)
        assert refused.value.reason == "unauthorized"
        assert memory.consent(identity_id) is ConsentState.WITHDRAWN

    retire = MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(assertion_id,))
    with _memory(tmp_path, consolidator=ScriptedConsolidator(retire)) as memory:
        with pytest.raises(ValidationError) as refused:
            memory.apply(retire)
        assert refused.value.reason == "consent_assertion"
        assert memory.consent(identity_id) is ConsentState.WITHDRAWN


# ---------------------------------------------------------------------------------------------
# What withheld consent restrains


def _enrolled(memory: Memory, identity_id: str) -> tuple[int, int]:
    """Return how many face and voice exemplars one identity's template holds.

    Read by erasing the person, which is the only public call that counts them -- and which is
    exactly the request consent is not: a template survives a refusal and does not survive this.
    """
    erasure = memory.forget_identity(identity_id)
    return erasure.face_exemplars, erasure.voice_exemplars


def test_withdrawn_consent_stops_enrolment_but_not_recognition(tmp_path: Path) -> None:
    """A second observation adds no exemplar, yet still resolves to the same person."""
    with _memory(tmp_path / "consented") as memory:
        first = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        identity_id = memory.faces(first.id)[0].identity_id
        second = memory.add(Blob(b"a-2 returns", "video/mp4", "a-2.mp4"))
        assert memory.faces(second.id)[0].identity_id == identity_id
        assert _enrolled(memory, identity_id) == (2, 2)

    with _memory(tmp_path / "withdrawn") as memory:
        first = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        identity_id = memory.faces(first.id)[0].identity_id
        assert memory.record_consent(identity_id, ConsentState.WITHDRAWN) is not None

        second = memory.add(Blob(b"a-2 returns", "video/mp4", "a-2.mp4"))
        # Recognition still answers, from the exemplars already held.
        assert memory.faces(second.id)[0].identity_id == identity_id
        assert memory.speech(second.id)[0].speaker_id == identity_id
        # Nothing was learned from it.
        assert _enrolled(memory, identity_id) == (1, 1)


def test_withheld_consent_stops_the_cross_modal_merge(tmp_path: Path) -> None:
    """A voice and a face that would corroborate stay two identities while a refusal stands."""
    with _memory(tmp_path / "merged") as memory:
        voice = memory.add(Blob(b"a-1 speaking", "audio/wav", "a-1.wav"))
        face = memory.add(Blob(b"a-2 seen", "image/png", "a-2.png"))
        voice_id = memory.speech(voice.id)[0].speaker_id
        face_id = memory.faces(face.id)[0].identity_id
        assert voice_id is not None and voice_id != face_id

        both = memory.add(Blob(b"a-3 seen and heard", "video/mp4", "a-3.mp4"))
        memory.faces(both.id)
        merged = [
            record
            for record in memory.operations()
            if record.operation.intent is MemoryIntent.MERGE
        ]
        assert len(merged) == 1

    with _memory(tmp_path / "refused") as memory:
        voice = memory.add(Blob(b"a-1 speaking", "audio/wav", "a-1.wav"))
        face = memory.add(Blob(b"a-2 seen", "image/png", "a-2.png"))
        voice_id = memory.speech(voice.id)[0].speaker_id
        face_id = memory.faces(face.id)[0].identity_id
        assert voice_id is not None
        assert memory.record_consent(face_id, ConsentState.WITHHELD) is not None

        both = memory.add(Blob(b"a-3 seen and heard", "video/mp4", "a-3.mp4"))
        memory.faces(both.id)
        assert not [
            record
            for record in memory.operations()
            if record.operation.intent is MemoryIntent.MERGE
        ]
        assert memory.identity(voice_id) is not None
        assert memory.identity(face_id) is not None
        assert memory.speech(both.id)[0].speaker_id == voice_id
        assert memory.faces(both.id)[0].identity_id == face_id


def test_a_withheld_person_leaves_a_compiled_bundle_with_an_unknown(tmp_path: Path) -> None:
    """The named actor is dropped, the omission is reported, and the evidence stays."""
    with _memory(tmp_path) as memory:
        clip = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        identity_id = memory.faces(clip.id)[0].identity_id
        memory.register_identity(identity_id, "Ann", relationship="neighbour")
        episode = memory.add("Ann fixed the gate on Tuesday", occurred_at=OCCURRED)

        budget = ContextBudget(max_items=10)
        bundle = memory.compile("who is Ann", budget=budget)
        named = [
            hit
            for hit in bundle.actors
            if isinstance(hit, MemoryRecord) or getattr(hit, "content", "").startswith("Ann")
        ]
        assert named, "the naming assertion should reach the actors section"
        assert ContextUnknownKind.CONSENT_WITHHELD not in {
            unknown.kind for unknown in bundle.unknowns
        }

        assert memory.record_consent(identity_id, ConsentState.WITHDRAWN) is not None
        bundle = memory.compile("who is Ann", budget=budget)
        assert not [
            hit for hit in bundle.actors if getattr(hit, "content", "").startswith("Ann is a")
        ]
        withheld = [
            unknown
            for unknown in bundle.unknowns
            if unknown.kind is ContextUnknownKind.CONSENT_WITHHELD
        ]
        assert len(withheld) == 1
        assert "consent" in withheld[0].detail
        # Their memories are untouched: consent governs being a recognized person, not the event.
        assert episode.id in {hit.id for hit in bundle.hits}


def test_a_withheld_person_is_not_named_by_a_clip_alone(tmp_path: Path) -> None:
    """Consent is read from every identity edge that can name somebody, not only ENTITY hits.

    The restrained set used to be derived from the bound `ENTITY` hits retrieval returned, so a
    bundle that reached only a person's clip -- their naming assertion outside the budget's own
    type bound -- found nothing bound, consulted no consent at all, and then resolved the clip's
    own face edge into exactly the `NamedActor` that `WITHDRAWN` consent promises to omit.
    """
    with _memory(tmp_path) as memory:
        clip = memory.add(
            Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"),
            memory_type=MemoryType.EPISODIC,
            occurred_at=OCCURRED,
        )
        identity_id = memory.faces(clip.id)[0].identity_id
        memory.register_identity(identity_id, "Ann", relationship="neighbour")
        # Episodic evidence only, so the naming assertion -- a semantic `ENTITY` -- is never
        # retrieved, and the clip's face edge is the only thing that can name her.
        budget = ContextBudget(max_items=10, memory_types=frozenset({MemoryType.EPISODIC}))

        bundle = memory.compile("who arrives", budget=budget)
        assert clip.id in {hit.id for hit in bundle.hits}
        assert [actor.name for actor in bundle.actors if isinstance(actor, NamedActor)] == ["Ann"]

        assert memory.record_consent(identity_id, ConsentState.WITHDRAWN) is not None
        bundle = memory.compile("who arrives", budget=budget)

        # Named nowhere, and not quietly downgraded to a provisional actor either.
        assert bundle.actors == ()
        withheld = [
            unknown
            for unknown in bundle.unknowns
            if unknown.kind is ContextUnknownKind.CONSENT_WITHHELD
        ]
        assert len(withheld) == 1
        # The clip itself stays: consent governs recognition, not whether she arrived.
        assert clip.id in {hit.id for hit in bundle.hits}


# ---------------------------------------------------------------------------------------------
# Export


def test_an_export_carries_the_whole_subject_and_nobody_else(tmp_path: Path) -> None:
    """Naming assertion, episodes including a forgotten one, and every log row that moved them."""
    with _memory(tmp_path) as memory:
        first = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        subject_id = memory.faces(first.id)[0].identity_id
        memory.register_identity(subject_id, "Ann", relationship="neighbour")
        second = memory.add(Blob(b"a-2 returns", "video/mp4", "a-2.mp4"))
        assert memory.faces(second.id)[0].identity_id == subject_id
        forgotten = memory.forget((second.id,))
        assert forgotten is not None

        other = memory.add(Blob(b"b-1 arrives", "video/mp4", "b-1.mp4"))
        other_id = memory.faces(other.id)[0].identity_id
        assert other_id != subject_id

        bundle = memory.export(identity_id=subject_id)
        exported = {record.id for record in bundle.records}
        naming = [
            record.id
            for record in bundle.records
            if record.context is not None and record.context.kind is MemoryKind.ENTITY
        ]

        assert bundle.identity_id == subject_id
        assert [profile.name for profile in bundle.identities] == ["Ann"]
        assert {first.id, second.id} <= exported
        assert len(naming) == 1
        # A forgotten record is still held, so an export still answers with it.
        assert next(record for record in bundle.records if record.id == second.id).forgotten_at
        assert other.id not in exported

        intents = [record.operation.intent for record in bundle.operations]
        assert MemoryIntent.MERGE in intents
        assert MemoryIntent.IDENTIFY in intents
        assert forgotten.operation_id in {record.operation_id for record in bundle.operations}
        # The other person's own merge belongs to them, not to this subject.
        moved = {
            change.identity_id
            for record in bundle.operations
            if (change := record.operation.identity) is not None
        }
        assert other_id not in moved

        # Media travels by reference: identity and digest, never bytes.
        assets = [asset for record in bundle.records for asset in record.assets]
        assert assets and all(asset.sha256 and asset.size_bytes for asset in assets)

        # Naming a record set instead answers about those records alone.
        narrow = memory.export(memory_ids=(first.id,))
        assert [record.id for record in narrow.records] == [first.id]
        assert narrow.identity_id is None and narrow.identities == ()

        with pytest.raises(ValidationError):
            memory.export()
        with pytest.raises(ValidationError):
            memory.export(identity_id=subject_id, memory_ids=(first.id,))
        with pytest.raises(IdentityNotFoundError):
            memory.export(identity_id="identity_missing")


def test_a_clip_two_people_share_is_exported_to_both_of_them(tmp_path: Path) -> None:
    """Pinned, not accidental: a shared record is evidence for every subject in it.

    One recording containing two people is held about both of them, so withholding it from
    either subject's export would answer their access request with less than is held. The
    consequence is that each export carries the other person's observations embedded in that
    record, which `export()`'s contract states rather than silently trims.
    """
    with _memory(tmp_path) as memory:
        alone = memory.add(Blob(b"a-1 speaking", "audio/wav", "a-1.wav"))
        first_id = memory.speech(alone.id)[0].speaker_id
        other = memory.add(Blob(b"b-1 speaking", "audio/wav", "b-1.wav"))
        second_id = memory.speech(other.id)[0].speaker_id
        assert first_id is not None and second_id is not None and first_id != second_id

        shared = memory.add(Blob(b"ab-1 both speaking", "audio/wav", "ab-1.wav"))
        heard = {segment.speaker_id for segment in memory.speech(shared.id)}
        assert heard == {first_id, second_id}

        for subject, neighbour in ((first_id, second_id), (second_id, first_id)):
            bundle = memory.export(identity_id=subject)
            exported = {record.id for record in bundle.records}
            assert shared.id in exported, "a shared recording is held about both subjects"
            assert bundle.identity_id == subject
            # And only the shared one is: the other subject's solo recording stays theirs.
            assert exported == {shared.id, alone.id if subject == first_id else other.id}
            assert neighbour not in {profile.identity_id for profile in bundle.identities}


# ---------------------------------------------------------------------------------------------
# Retention

# Small enough that anything already written is past it, so a test does not have to wait.
_AGED = 1e-9


def test_a_dry_run_deletes_nothing_and_a_real_run_deletes_only_what_aged_out(
    tmp_path: Path,
) -> None:
    policy = RetentionPolicy(media_days=_AGED, forgotten_days=3650.0)
    with _memory(tmp_path, retention=policy) as memory:
        clip = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        (asset,) = clip.assets
        note = memory.add("the gate sticks in the rain", occurred_at=OCCURRED)
        assert memory.forget((note.id,)) is not None

        planned = memory.apply_retention(dry_run=True)
        assert planned.dry_run is True
        assert planned.media_memory_ids == (clip.id,)
        assert planned.asset_ids == (asset.id,)
        # Forgotten far more recently than the policy allows, so it is not a candidate.
        assert planned.forgotten_memory_ids == ()
        assert planned.deleted == 1
        assert memory.get(clip.id).id == clip.id
        assert memory.get(note.id).forgotten_at is not None

        applied = memory.apply_retention()
        assert applied.dry_run is False
        assert applied.media_memory_ids == (clip.id,)
        assert applied.asset_ids == (asset.id,)
        assert applied.forgotten_memory_ids == ()

        with pytest.raises(MemoryNotFoundError):
            memory.get(clip.id)
        assert memory.get(note.id).forgotten_at is not None

        # Nothing is left to age out, so a second pass is a no-op rather than a repeat.
        assert memory.apply_retention().deleted == 0


def test_retention_finalizes_cognitive_forgetting_and_abandons_failed_captures(
    tmp_path: Path,
) -> None:
    policy = RetentionPolicy(forgotten_days=_AGED, capture_failure_days=_AGED)
    with _memory(tmp_path, retention=policy) as memory:
        kept = memory.add("the ladder is in the garage", occurred_at=OCCURRED)
        note = memory.add("the gate sticks in the rain", occurred_at=OCCURRED)
        assert memory.forget((note.id,)) is not None
        queued = memory.capture("a captured observation")

        # Never attempted, so nothing has failed and the promise to retry stands.
        assert memory.apply_retention(dry_run=True).capture_memory_ids == ()

        report = memory.apply_retention()
        assert report.forgotten_memory_ids == (note.id,)
        assert report.media_memory_ids == ()
        with pytest.raises(MemoryNotFoundError):
            memory.get(note.id)
        assert memory.get(kept.id).id == kept.id
        assert [row.memory_id for row in memory.pending_captures()] == [queued.id]


def test_an_undeclared_policy_deletes_nothing(tmp_path: Path) -> None:
    """`None` is "no policy", not "keep for zero days"; nothing ages out because a clock ticked."""
    with _memory(tmp_path) as memory:
        clip = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        note = memory.add("the gate sticks in the rain", occurred_at=OCCURRED)
        assert memory.forget((note.id,)) is not None

        report = memory.apply_retention()
        assert report == RetentionReport(dry_run=False)
        assert memory.get(clip.id).id == clip.id
        assert memory.get(note.id).forgotten_at is not None


# ---------------------------------------------------------------------------------------------
# The CLI translation of the three rights


def test_the_cli_translates_consent_export_and_retention(tmp_path: Path) -> None:
    """The CLI owns decoding and encoding only; every decision above is the SDK's."""
    parser = cli._parser()
    with _memory(tmp_path, retention=RetentionPolicy(forgotten_days=_AGED)) as memory:
        clip = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        identity_id = memory.faces(clip.id)[0].identity_id

        stated = cli._LOCAL["record-consent"](
            memory,
            parser.parse_args(
                ["record-consent", identity_id, "withdrawn", "--note", "at the door"]
            ),
        )
        operation = cast(dict[str, object], stated["operation"])
        assert operation["intent"] == "consent"
        assert operation["consent"] == {
            "identity_id": identity_id,
            "state": "withdrawn",
            "note": "at the door",
        }
        assert cli._LOCAL["consent"](memory, parser.parse_args(["consent", identity_id])) == {
            "consent": "withdrawn"
        }

        exported = cli._LOCAL["export"](
            memory, parser.parse_args(["export", "--identity-id", identity_id])
        )
        assert exported["identity_id"] == identity_id
        assert clip.id in {
            cast(dict[str, object], record)["id"]
            for record in cast(list[object], exported["records"])
        }
        # Serializable as it stands: an export is a document a subject is handed.
        assert json.loads(json.dumps(exported))["identity_id"] == identity_id

        note = memory.add("the gate sticks in the rain", occurred_at=OCCURRED)
        assert memory.forget((note.id,)) is not None
        planned = cli._LOCAL["apply-retention"](
            memory, parser.parse_args(["apply-retention", "--dry-run"])
        )
        assert planned == {
            "dry_run": True,
            "media_memory_ids": [],
            "forgotten_memory_ids": [note.id],
            "asset_ids": [],
            "capture_memory_ids": [],
            "deleted": 1,
        }
        assert memory.get(note.id).forgotten_at is not None

        applied = cli._LOCAL["apply-retention"](memory, parser.parse_args(["apply-retention"]))
        assert applied["forgotten_memory_ids"] == [note.id]
        with pytest.raises(MemoryNotFoundError):
            memory.get(note.id)


class ScriptedFormer:
    """Returns one fixed proposal for every observation, whatever it observed."""

    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "former-test"
    formation_space = "former-test:v1"

    def __init__(self, *proposals: FormationProposal) -> None:
        self.proposals = tuple(proposals)

    def form(
        self,
        inputs: Sequence[FormationInput],
    ) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(self.proposals for _value in inputs)

    def close(self) -> None:
        pass


def test_formation_cannot_manufacture_consent_through_an_ordinary_proposal(
    tmp_path: Path,
) -> None:
    """The predicate is reserved: a model's STATE claim about it binds to nobody.

    `_apply_memory_operation` refuses a proposed CONSENT operation, but ordinary formation
    never passes through it -- a `FormationBackend` returning a bound-looking STATE proposal on
    the `add()` path would otherwise have written a row every consent read believes.
    """
    with _memory(tmp_path) as memory:
        clip = memory.add(Blob(b"a-1 arrives", "video/mp4", "a-1.mp4"))
        identity_id = memory.faces(clip.id)[0].identity_id
        memory.register_identity(identity_id, "Alice")
        assert memory.record_consent(identity_id, ConsentState.GRANTED) is not None

    forged = FormationProposal(
        kind=MemoryKind.STATE,
        content="Alice withdrew her consent",
        # The basis a model may claim, and the one `record_consent` writes: neither the state
        # nor the paperwork can be told apart by looking, which is why the predicate is what
        # is reserved.
        basis=EvidenceBasis.USER_STATEMENT,
        subject="Alice",
        predicate="consent",
        value="withdrawn",
    )
    with _memory(tmp_path, former=ScriptedFormer(forged)) as memory:
        logged_before = len(memory.operations())
        memory.add("Alice came by again", occurred_at=OCCURRED)

        assert memory.consent(identity_id) is ConsentState.GRANTED
        assert len(memory.operations()) == logged_before
        # The claim is kept, as an ordinary STATE bound to nobody: a model's inference is
        # evidence of what the model thought, not of what the person said.
        formed = [
            record
            for record in memory.list(limit=100).items
            if record.context is not None and record.context.predicate == "consent"
        ]
        assert [record.context.identity_id for record in formed if record.context] == [
            None,
            identity_id,
        ]

    # The consolidation path binds through the same guard, so replaying a hand-written
    # operation carrying the same proposal cannot reach the identity either.
    with _memory(tmp_path, consolidator=ScriptedConsolidator()) as memory:
        source = memory.add("Alice said something else", occurred_at=OCCURRED)
        applied = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=forged,
            )
        )
        (derived,) = applied.created_ids
        context = memory.get(derived).context
        assert context is not None and context.identity_id is None
        assert memory.consent(identity_id) is ConsentState.GRANTED
