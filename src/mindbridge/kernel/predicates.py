"""Pure, conservative predicate resolution and stable scoped claim keys.

Identifiers and their meanings are immutable. Registry revisions may add reviewed aliases;
they must not recycle identifiers or treat similarity as equivalence.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass

from mindbridge.types import EvidenceBasis, MemoryKind

PREDICATE_REGISTRY_VERSION = "mindbridge-predicates-v1"


@dataclass(frozen=True, slots=True)
class PredicateDefinition:
    identifier: str
    aliases: frozenset[str]
    kinds: frozenset[str]
    value_type: str = "label"


_CLAIM_KINDS = frozenset({"state", "trait", "relation"})
PREDICATES = (
    PredicateDefinition(
        "preferred_drink",
        frozenset({"preferred_drink", "favorite_beverage", "favourite_beverage", "喜欢的饮料"}),
        _CLAIM_KINDS,
    ),
    PredicateDefinition(
        "home_city", frozenset({"home_city", "city_of_residence", "居住城市"}), _CLAIM_KINDS
    ),
    PredicateDefinition(
        "current_location", frozenset({"current_location", "当前位置"}), _CLAIM_KINDS
    ),
)
_ALIASES = {alias: definition for definition in PREDICATES for alias in definition.aliases}


@dataclass(frozen=True, slots=True)
class PredicateResolution:
    key: str | None
    match: str
    cardinality: str
    value_type: str | None = None


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    return unicodedata.normalize("NFKC", value).casefold().strip() or None


def resolve_predicate(
    predicate: str | None, *, kind: str, basis: str, value: str | None
) -> PredicateResolution:
    literal = normalize_label(predicate)
    cardinality = (
        "one"
        if kind == MemoryKind.STATE.value
        or (kind == MemoryKind.TRAIT.value and basis == EvidenceBasis.USER_STATEMENT.value)
        else "many"
    )
    definition = _ALIASES.get(literal or "")
    if definition is None:
        return PredicateResolution(literal, "literal", cardinality)
    # This vocabulary accepts scalar-shaped labels, rejecting JSON containers and common list
    # delimiters. This syntax guard cannot establish the semantic type of arbitrary prose.
    # Cardinality comes from the explicit kind/basis contract, never from predicate wording.
    label = normalize_label(value)
    if (
        kind not in definition.kinds
        or label is None
        or len(label) > 255
        or not label.isprintable()
        or any(character in label for character in ",;、[]{}\n/|")
    ):
        return PredicateResolution(literal, "ineligible", cardinality)
    return PredicateResolution(
        definition.identifier,
        "canonical" if literal == definition.identifier else "alias",
        cardinality,
        definition.value_type,
    )


def claim_lineage_id(
    *,
    kind: str,
    predicate: str | None,
    subject: str | None,
    identity_id: str | None = None,
    frame_id: str | None = None,
    anchor: str | None = None,
    resolution: PredicateResolution,
) -> str:
    """Hash the resolved or literal predicate within its identity and spatial scope."""
    payload: dict[str, object] = {
        "kind": kind,
        "predicate": normalize_label(predicate),
        "frame_id": frame_id,
        "anchor": anchor,
        **(
            {"subject": normalize_label(subject)}
            if identity_id is None
            else {"subject": None, "identity_id": identity_id}
        ),
    }
    version = "v1"
    if resolution.value_type is not None:
        version = "v2"
        del payload["predicate"]
        payload.update(
            predicate_id=resolution.key,
            # TRAIT's existing contract lets a host assertion override accumulating model
            # inferences in this same slot. Splitting by basis would bypass that protection.
            cardinality=("assertion-dependent" if kind == "trait" else resolution.cardinality),
            value_type=resolution.value_type,
        )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"mindbridge-lineage-{version}:{encoded}".encode()).hexdigest()
