"""Named recipes over the model backends this package bundles.

A recipe is a name for a construction the caller could equally write by hand. It is not a plugin
system and not a registry: the table is closed, every entry pins its model identity to a constant
that already lives in this package, and each function returns the constructed object so the caller
owns it, can inspect it, and can throw it away. A backend this package does not bundle is
constructed by the application and passed to ``Memory`` directly.

``load=True`` additionally exercises the backend's loader before returning, which is how
``mindbridge doctor`` turns a missing runtime into one line instead of a silent ingestion failure.
How deep that probe reaches depends on what each backend publishes; ``probe()`` reports which.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Literal, cast

from mindbridge.exceptions import ValidationError
from mindbridge.models.base import (
    EmbeddingBackend,
    GenerationBackend,
    SpeechBackend,
    TranscriptionBackend,
)
from mindbridge.models.funasr import (
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_RECIPE,
    FunASRTranscriber,
)
from mindbridge.models.jina import (
    DEFAULT_JINA_DIMENSION,
    DEFAULT_JINA_MODEL_ID,
    DEFAULT_JINA_REVISION,
    JinaOmniEmbedder,
)
from mindbridge.models.openai_sdk import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    OpenAIModels,
)

if TYPE_CHECKING:
    from openai import OpenAI

Slot = Literal["embedder", "answerer", "transcriber"]

JINA_OMNI = "jina-omni"
FUNASR = "funasr"
OPENAI = "openai"

# Which slots each recipe may fill. Only `openai` accepts an `openai:<model>` suffix, because it
# is the only bundled backend whose model is an argument rather than a pinned local revision.
_SLOTS: Mapping[str, frozenset[str]] = {
    FUNASR: frozenset({"transcriber"}),
    JINA_OMNI: frozenset({"embedder"}),
    OPENAI: frozenset({"embedder", "answerer", "transcriber"}),
}
_PARAMETERIZED = frozenset({OPENAI})
_PROBES: Mapping[str, str] = {JINA_OMNI: "weights", FUNASR: "import", OPENAI: "client"}
_OPENAI_MODELS: Mapping[str, str] = {
    "embedder": DEFAULT_EMBEDDING_MODEL,
    "answerer": DEFAULT_GENERATION_MODEL,
    "transcriber": DEFAULT_TRANSCRIPTION_MODEL,
}
# MindBridge never reads a credential. The official SDK performs its own documented lookup, so the
# source of the key is reportable while the value never enters this process's output.
OPENAI_CREDENTIAL = "OPENAI_API_KEY (openai SDK default)"


def names() -> tuple[str, ...]:
    """Every recipe name. ``openai`` also accepts an ``openai:<model>`` suffix."""
    return tuple(sorted(_SLOTS))


def slots(name: str) -> tuple[str, ...]:
    """The composition slots one recipe may fill."""
    family, _model = _split(name)
    return tuple(sorted(_SLOTS[family]))


def require_slot(name: str, slot: str) -> None:
    """Raise unless one recipe can fill the named composition slot."""
    family, _model = _split(name)
    if slot not in _SLOTS[family]:
        accepted = ", ".join(f"--{value}" for value in slots(name))
        raise ValidationError(f"recipe {name} cannot fill --{slot}; it fills: {accepted}")


def probe(name: str) -> str:
    """What ``load=True`` exercises: pinned ``weights``, a runtime ``import``, or a ``client``."""
    family, _model = _split(name)
    return _PROBES[family]


def describe(name: str) -> dict[str, object]:
    """Static identity of one recipe. Resolves nothing and constructs nothing."""
    family, model = _split(name)
    if family == JINA_OMNI:
        return {
            "recipe": name,
            "class": _qualified(JinaOmniEmbedder),
            "slots": list(slots(name)),
            "models": {"embedder": DEFAULT_JINA_MODEL_ID},
            "revision": DEFAULT_JINA_REVISION,
            "embedding_dimension": DEFAULT_JINA_DIMENSION,
            # Disclosed where the recipe is chosen, not only in a README footer. The licence
            # covers the pinned weights, not MindBridge.
            "license": "CC BY-NC 4.0",
            "extra": "local",
        }
    if family == FUNASR:
        return {
            "recipe": name,
            "class": _qualified(FunASRTranscriber),
            "slots": list(slots(name)),
            "models": {"transcriber": DEFAULT_FUNASR_MODEL_ID},
            "revision": DEFAULT_FUNASR_RECIPE.model_revision,
            "extra": "local",
        }
    return {
        "recipe": name,
        "class": _qualified(OpenAIModels),
        "slots": list(slots(name)),
        "models": {
            slot: default if model is None else model for slot, default in _OPENAI_MODELS.items()
        },
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "credential": OPENAI_CREDENTIAL,
        "extra": "openai",
    }


def embedder(name: str, *, load: bool = False) -> EmbeddingBackend:
    """Return the embedding backend one recipe names; the caller owns and closes it."""
    return cast(EmbeddingBackend, _build(name, slot="embedder", load=load))


def answerer(name: str, *, load: bool = False) -> GenerationBackend:
    """Return the generation backend one recipe names; the caller owns and closes it."""
    return cast(GenerationBackend, _build(name, slot="answerer", load=load))


def transcriber(name: str, *, load: bool = False) -> SpeechBackend | TranscriptionBackend:
    """Return the speech backend one recipe names; the caller owns and closes it."""
    return cast(
        "SpeechBackend | TranscriptionBackend",
        _build(name, slot="transcriber", load=load),
    )


def _build(name: str, *, slot: Slot, load: bool) -> object:
    family, model = _split(name)
    require_slot(name, slot)
    if family == JINA_OMNI:
        return JinaOmniEmbedder.load() if load else JinaOmniEmbedder()
    if family == FUNASR:
        if load:
            # FunASR publishes no loader, so the deepest honest probe is the deferred import an
            # under-declared dependency breaks. `probe()` reports that, so nothing overstates it.
            import_module("funasr")
        return FunASRTranscriber(DEFAULT_FUNASR_RECIPE)
    # Constructing the SDK client is the load: it is where the official SDK resolves its own
    # credentials and fails when none are present.
    selected = _OPENAI_MODELS[slot] if model is None else model
    client = _openai_client()
    if slot == "embedder":
        return OpenAIModels(client, embedding_model=selected)
    if slot == "answerer":
        return OpenAIModels(client, generation_model=selected)
    return OpenAIModels(client, transcription_model=selected)


def _openai_client() -> OpenAI:
    return cast("OpenAI", import_module("openai").OpenAI())


def _split(name: str) -> tuple[str, str | None]:
    family, separator, model = name.partition(":")
    if family not in _SLOTS or (separator and family not in _PARAMETERIZED):
        raise ValidationError(f"unknown recipe: {name}; known recipes: {', '.join(names())}")
    if separator and not model.strip():
        raise ValidationError(f"recipe {family} requires a model identifier after the colon")
    return family, model.strip() or None


def _qualified(value: type) -> str:
    return f"{value.__module__}.{value.__qualname__}"
