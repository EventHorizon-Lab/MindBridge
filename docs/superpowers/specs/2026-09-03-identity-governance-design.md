# Context OS round 2: identity as governed knowledge

Date: 2026-09-03. Base: the master merge on `claude/mindbridge-context-os-upgrade-0c41c9`
(schema v11). This round makes naming a person an auditable, reversible, evidence-bearing
operation, and binds the biometric identity registry to the semantic subject space.

The acceptance scenario is the specification. See [the household-companion scenario](2026-09-03-household-companion-scenario.md): a stranger
arrives, introduces themselves, is misheard, is confused with a courier, and finally asks to be
erased. Every API below exists to make one step of that scenario correct.

## Three defects this round fixes

Verified against source at `4846264`:

1. **Naming has no governance.** `register_identity` is `UPDATE identities SET name = ?` plus a
   destructive rewrite of `memory_records.content`. No version, no actor, no evidence, no undo.
   The MCP tool carries no authority wording, so any caller can rename anyone. Meanwhile
   `memory_semantics` next door has full bitemporal versioning, independent-evidence grouping,
   noisy-OR confidence, and computed visibility.
2. **Two disjoint identity spaces.** No column, constraint, or query joins an `identity_id` to
   `memory_semantics.subject`. The only bridge is asset-mediated and lands on `memory_records`,
   never on a semantic row. So the recognized person and the subject of typed claims about that
   person are two unrelated things that happen to share a string.
3. **`unlink_identity` does not refresh the speech index.** Reversing a merge restores the
   identity but leaves the other person's name burned into stored content. Found while designing
   this round; it is a present-tense bug, not a consequence of the redesign.

## Design principle

Do not build a second versioning system for identities. The semantic plane already has one.
Make the name a semantic assertion and make `identities.name` a projection of it, recomputed the
way `_refresh_evidence_projection` already recomputes confidence and visibility.

One mechanism then fixes all three defects: naming gains versions, evidence, and rollback;
the assertion carries the identity binding that closes the gap to the subject space; and the
projection refresh is the hook that finally repaints the index on unlink.

## Schema v12

| Change | Purpose |
| --- | --- |
| `memory_semantics.identity_id TEXT REFERENCES identities(identity_id) ON DELETE SET NULL` plus an index on `(identity_id, memory_id)` | Bind a typed claim to the recognized person it is about. `SET NULL` matches the documented erasure promise that forgetting a person is not forgetting the evening. |

No new table. A naming assertion is an ordinary memory record with `kind = ENTITY` and
`identity_id` set, so it inherits versions, evidence, confidence, visibility, and rollback.

## A. Naming becomes an assertion

`MemoryKind.ENTITY` with a non-null `identity_id` is a naming assertion. Its `subject` is the
canonical person key, its content is the human-readable claim, and its basis decides visibility:

| Basis | Who | Visible |
| --- | --- | --- |
| `USER_STATEMENT` | The household owner through `register_identity`, or the person naming themselves | Immediately |
| `MODEL_INFERENCE` | The agent through the control plane | Only with two independent evidence groups |

That rule is not new. It is exactly the existing `TRAIT` visibility policy in
`_semantic_visibility`, reused rather than reimplemented.

**`identities.name` becomes a projection.** It is recomputed from the current visible naming
assertion of that identity whenever assertions, evidence, or merges change, and every recompute
runs the existing `_refresh_speaker_memories` so the indexed text always matches the projection.
`register_identity` and `register_speaker` keep their signatures and now write through this path.

**Behaviour change to flag:** naming a person now creates a searchable memory record. This is
intended. It is what makes the neighbour's name retrievable knowledge rather than a label on a
row, and it is why the scenario's step 4 can work at all. It is still a visible change to
`list()` and `search()` results for any host that names people.

**`unlink_identity` fix, landing first as its own commit.** Unlink recomputes the projection for
both identities and refreshes the index, so a reversed merge stops claiming the wrong name.

## B. `IDENTIFY`, the fifth intent

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityClaim:
    identity_id: str
    name: str
    relationship: str | None = None


class MemoryIntent(str, Enum):
    ...
    IDENTIFY = "identify"
```

`MemoryOperation` gains `claim: IdentityClaim | None`, required for `IDENTIFY` and rejected for
every other intent. The kernel, not the model, builds the `ENTITY` proposal from the claim, so
the backend's job stays small: name the identity, cite the evidence.

Kernel validation for `IDENTIFY`:

- the identity exists and is not erased;
- every cited evidence id is in the shown set and exists;
- at least one cited evidence memory actually involves that identity, through
  `speech_segments` or `face_observations`, so a name cannot be attached to a person from
  evidence that never contained them;
- the name passes the existing `_identity_name` rules.

Effects: commit the ENTITY assertion with its evidence, recompute the projection, refresh the
index, and log the operation. Two `IDENTIFY` operations on one identity share a lineage keyed on
the identity, so the second supersedes the first through the existing bitemporal path, and
`rollback()` carries the previous version back and repaints the index. That is scenario step 5.

Authority stays where round 1 put it. `IDENTIFY` is a proposal an agent may make. Physical
erasure remains `forget_identity` under host authority and is not an intent.

## C. Binding claims to people

When formation or consolidation proposes a typed memory whose subject resolves to a known
identity, the kernel stamps `memory_semantics.identity_id`. Resolution is deterministic: the
subject matches the canonical subject of a visible naming assertion after the existing NFKC
casefold normalization. No model decides this.

`_formation_lineage_id` keys on `identity_id` when present, so claims about one person converge
even when the model writes the name differently across turns. This is what makes one person one
subject.

**On unlink,** every derived memory bound to the merged-away identity is re-evaluated. If all of
its evidence assets moved back with the restored identity, it is rebound. Otherwise its
`identity_id` is cleared, so the claim survives but stops being attributed to the wrong person.
Claims are never silently left pointing at someone they were never about. That is scenario
step 6.

## D. Provisional identities

No new state. An identity is confirmed when it has a visible naming assertion; otherwise it is
provisional. `IdentityProfile` gains `confirmed: bool` and `evidence_ids: tuple[str, ...]`, both
derived. `ContextBundle.actors` includes provisional identities, labelled, so an agent can say
that an unrecognized person is present rather than omitting them. That is scenario steps 2
and 3.

## Work split

| Member | Owns | Order |
| --- | --- | --- |
| E, core | The unlink index fix as commit one; schema v12; naming as assertion; the projection and its refresh; rerouting `register_identity`, `register_speaker`, `unlink_identity`, `forget_identity` | first |
| F, surfaces | `IdentityClaim`, `IDENTIFY`, the consolidator adapter and its prompt, subject binding and lineage keying, `IdentityProfile` fields, `compile()` actors, MCP authority wording, docs | after E lands |

## Out of scope

Coordinate transforms, a blocklist for erased people, cross-device identity, and any change to
how a face and a voice are bound. A model never proposes a biometric merge; corroboration stays
deterministic kernel policy, because the measured alternative collapsed four people into one.
