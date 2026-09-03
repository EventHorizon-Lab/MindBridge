# Round 2 acceptance scenario: the household companion

The round is done when this scenario runs end to end against the public SDK with fake backends,
and every assertion below holds. The scenario is the specification; the API exists to serve it.

## Cast

- Wearer or robot, off camera.
- Mei, a household member, already registered and corroborated.
- A stranger who turns out to be Li, the neighbour from next door.
- A delivery courier who appears once and is never named.

## Sequence and what must hold

1. **Stranger arrives.** Robot captures two clips containing the stranger's face and voice.
   The kernel merges the face and the voice into one identity deterministically, without any
   model proposing the binding. Nothing in the control plane is involved.
   *Assert:* one identity, `faces()` and `speech()` agree on its id.

2. **The identity is not yet a person.** Before anyone names them, the compiled context for
   "who is in the room" must be able to say that an unnamed, unconfirmed person is present.
   *Assert:* the bundle's actors section names the identity as provisional, not as a stranger
   silently missing from context.

3. **Self-introduction.** The stranger says "I am Li, I live next door." The agent proposes an
   identify operation citing that observation as evidence.
   *Assert:* the operation is logged with its evidence, the identity carries the name, and
   `operations()` shows it. The agent never wrote storage directly.

4. **The name reaches recall.** A later question about Li retrieves the earlier observations
   captured while Li was still unnamed.
   *Assert:* retrieval finds them, and the semantic subject for Li is the same subject that
   typed memories about Li use. One person, one subject.

5. **Mishearing is reversible.** The agent misheard: the neighbour is Li Hua, not Li. A second
   identify operation supersedes the first, and rolling back the second restores the first.
   *Assert:* both directions work, the audit trail shows both, and no indexed text is left
   claiming the retracted name.

6. **Wrong merge.** The courier's voice was briefly bound to Li's face. Unlinking splits them.
   *Assert:* claims formed while they were one identity are retracted or re-attributed, not
   silently left attached to the wrong person. This is the defect this round exists to fix.

7. **Right to be forgotten.** Li asks to be erased. The household owner, not the agent,
   executes physical erasure.
   *Assert:* templates, aliases, indexed name, and derived claims are gone; the operation log
   records that erasure happened without resurrecting what was erased.

## Boundaries this round must not cross

- A model never binds a face to a voice. Corroboration stays deterministic kernel policy.
- Physical erasure stays host authority. It is not a proposal intent.
- An unconfirmed identity must be distinguishable from a confirmed one at the point of use,
  because what an agent may say in front of whom depends on it.

## Open design questions to settle in the spec

1. Does the identity registry gain a semantic subject, or does a person become an ENTITY memory
   carrying the identity id in its context? The second keeps one evidence model.
2. What happens to name text already burned into indexed content on rename, unlink, and erase?
   If it is not recomputed, rollback is a lie.
3. Does a provisional identity need a stored state, or is it derivable from evidence count?
   Prefer derivable, so there is no new state to keep consistent.
