"""Lazy model plugin discovery through Python entry points."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import cast

from mindbridge.application.capabilities import Embedder, Generator
from mindbridge.configuration import (
    MissingConfigurationError,
    copy_plugin_configuration,
    validate_plugin_name,
)

PluginConfig = Mapping[str, object]
_Factory = Callable[[PluginConfig], object]


def validate_generator_configuration(name: str, config: PluginConfig) -> None:
    """Validate a bundled generator schema without constructing its client."""
    if name != "openai":
        return
    from mindbridge.models.openai import _GeneratorConfig

    if not config.get("api_key"):
        raise MissingConfigurationError("MINDBRIDGE_GENERATOR_API_KEY")
    _GeneratorConfig.model_validate(config)


def validate_embedder_configuration(name: str, config: PluginConfig) -> None:
    """Validate a bundled embedder schema without loading a model or client."""
    if name == "openai":
        from mindbridge.models.defaults import require_distinct_embedding_space
        from mindbridge.models.openai import _EmbedderConfig as _OpenAIEmbedderConfig

        if not config.get("api_key"):
            raise MissingConfigurationError("MINDBRIDGE_EMBEDDER_API_KEY")
        openai_config = _OpenAIEmbedderConfig.model_validate(config)
        require_distinct_embedding_space(openai_config.model_id, openai_config.space_id)
    elif name in {"jina", "sentence-transformers"}:
        from mindbridge.models.defaults import require_distinct_embedding_space
        from mindbridge.models.jina import _EmbedderConfig as _SentenceTransformersConfig

        local_config = _SentenceTransformersConfig.model_validate(config)
        require_distinct_embedding_space(
            local_config.model_id,
            local_config.space_id,
            model_revision=local_config.model_revision,
        )


def load_generator(name: str, config: PluginConfig) -> Generator:
    """Load only the selected generation plugin."""
    return cast(Generator, _load("mindbridge.generators", name, config, Generator))


def load_embedder(name: str, config: PluginConfig) -> Embedder:
    """Load only the selected embedding plugin."""
    embedder = cast(Embedder, _load("mindbridge.embedders", name, config, Embedder))
    # Reading the member is the check. Since 3.12 `isinstance` resolves protocol members
    # statically, so an explicit subclass that inherited `Embedder.space_reference`'s raising
    # body satisfies the structural check in `_load` on every version this package supports
    # -- before 3.12 it raised from there instead. Doing it here keeps the rejection at load
    # time on all of them, rather than at whichever call site reads the space first.
    _ = embedder.space_reference
    return embedder


async def close_model(model: object) -> None:
    """Release a loaded plugin that chose to own a client or device."""
    close = getattr(model, "close", None)
    if close is not None:
        await close()


def _load(
    group: str,
    name: str,
    config: PluginConfig,
    capability: object,
) -> object:
    validate_plugin_name(name)
    matches = tuple(point for point in _entry_points(group) if point.name == name)
    if not matches:
        raise LookupError(f"{group} plugin {name!r} is not installed")
    if len(matches) != 1:
        raise LookupError(f"{group} plugin {name!r} is installed more than once")
    validated_config = copy_plugin_configuration(config, "plugin configuration")
    loaded = matches[0].load()
    if not callable(loaded):
        raise TypeError(f"{group} plugin {name!r} entry point must be callable")
    factory = cast(_Factory, loaded)
    plugin = factory(validated_config)
    capability_type = cast(type[object], capability)
    if not isinstance(plugin, capability_type):
        raise TypeError(f"{group} plugin {name!r} does not implement {capability_type.__name__}")
    return plugin


def _entry_points(group: str) -> tuple[EntryPoint, ...]:
    return tuple(entry_points(group=group))
