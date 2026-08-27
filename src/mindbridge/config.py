"""Minimal configuration for local storage and OpenAI-compatible model endpoints."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from mindbridge.exceptions import ValidationError
from mindbridge.models.base import ModelCapabilities
from mindbridge.types import Modality

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_GENERATION_MODEL = "gpt-5-mini"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_EMBEDDING_DIMENSION = 1_536
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_CAPABILITIES = ModelCapabilities(
    embedding=frozenset({Modality.TEXT}),
    generation=frozenset({Modality.TEXT}),
    transcription=frozenset({Modality.AUDIO}),
)


@dataclass(frozen=True, slots=True)
class Config:
    """Settings for three independently placeable OpenAI-compatible operations."""

    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_OPENAI_BASE_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_space: str | None = None
    generation_model: str = DEFAULT_GENERATION_MODEL
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    transcription_space: str | None = None
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    media_transport: Literal["data", "file"] = "data"
    allowed_url_hosts: frozenset[str] = frozenset()
    capabilities: ModelCapabilities = DEFAULT_CAPABILITIES
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_base_url: str | None = None
    generation_api_key: str | None = field(default=None, repr=False)
    generation_base_url: str | None = None
    transcription_api_key: str | None = field(default=None, repr=False)
    transcription_base_url: str | None = None

    def __post_init__(self) -> None:
        api_key = _key(self.api_key, "api_key")
        base_url = _base_url(self.base_url)
        embedding_model = _text(self.embedding_model, "embedding_model")
        generation_model = _text(self.generation_model, "generation_model")
        transcription_model = _text(self.transcription_model, "transcription_model")
        if isinstance(self.embedding_dimension, bool) or not isinstance(
            self.embedding_dimension, int
        ):
            raise ValidationError("embedding_dimension must be a positive integer")
        if self.embedding_dimension <= 0:
            raise ValidationError("embedding_dimension must be a positive integer")
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, int | float
        ):
            raise ValidationError("timeout_seconds must be a positive number")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValidationError("timeout_seconds must be a positive number")
        if self.media_transport not in {"data", "file"}:
            raise ValidationError("media_transport must be 'data' or 'file'")
        allowed_url_hosts = _hosts(self.allowed_url_hosts)
        if not isinstance(self.capabilities, ModelCapabilities):
            raise ValidationError("capabilities must be a ModelCapabilities value")

        embedding_base_url = (
            base_url if self.embedding_base_url is None else _base_url(self.embedding_base_url)
        )
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "embedding_model", embedding_model)
        # The endpoint is deliberately absent from this recipe: it is a documented default and
        # an existing data directory has it persisted, so folding it in here would lock every
        # such directory shut. `Memory` records the endpoint as its own store-metadata value.
        object.__setattr__(
            self,
            "embedding_space",
            (
                _text(self.embedding_space, "embedding_space")
                if self.embedding_space is not None
                else f"{embedding_model}:{self.embedding_dimension}:l2-v1"
            ),
        )
        object.__setattr__(self, "generation_model", generation_model)
        object.__setattr__(self, "transcription_model", transcription_model)
        transcription_base_url = (
            base_url
            if self.transcription_base_url is None
            else _base_url(self.transcription_base_url)
        )
        object.__setattr__(
            self,
            "transcription_space",
            (
                _text(self.transcription_space, "transcription_space")
                if self.transcription_space is not None
                else f"{transcription_model}:asr-v1"
            ),
        )
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "allowed_url_hosts", allowed_url_hosts)
        object.__setattr__(
            self,
            "embedding_api_key",
            _key(self.embedding_api_key, "embedding_api_key") or api_key,
        )
        object.__setattr__(self, "embedding_base_url", embedding_base_url)
        object.__setattr__(
            self,
            "generation_api_key",
            _key(self.generation_api_key, "generation_api_key") or api_key,
        )
        object.__setattr__(
            self,
            "generation_base_url",
            base_url if self.generation_base_url is None else _base_url(self.generation_base_url),
        )
        object.__setattr__(
            self,
            "transcription_api_key",
            _key(self.transcription_api_key, "transcription_api_key") or api_key,
        )
        object.__setattr__(self, "transcription_base_url", transcription_base_url)

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Config:
        """Read explicit model settings without requiring keys for storage-only use."""
        source = os.environ if environ is None else environ
        return cls(
            api_key=_setting(source, "OPENAI_API_KEY"),
            base_url=_setting(source, "OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
            embedding_api_key=_setting(source, "MINDBRIDGE_EMBEDDING_API_KEY"),
            embedding_base_url=_setting(source, "MINDBRIDGE_EMBEDDING_BASE_URL"),
            generation_api_key=_setting(source, "MINDBRIDGE_GENERATION_API_KEY"),
            generation_base_url=_setting(source, "MINDBRIDGE_GENERATION_BASE_URL"),
            transcription_api_key=_setting(source, "MINDBRIDGE_TRANSCRIPTION_API_KEY"),
            transcription_base_url=_setting(source, "MINDBRIDGE_TRANSCRIPTION_BASE_URL"),
            embedding_model=_setting(source, "MINDBRIDGE_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL,
            embedding_space=_setting(source, "MINDBRIDGE_EMBEDDING_SPACE"),
            generation_model=_setting(source, "MINDBRIDGE_GENERATION_MODEL")
            or DEFAULT_GENERATION_MODEL,
            transcription_model=_setting(source, "MINDBRIDGE_TRANSCRIPTION_MODEL")
            or DEFAULT_TRANSCRIPTION_MODEL,
            transcription_space=_setting(source, "MINDBRIDGE_TRANSCRIPTION_SPACE"),
            embedding_dimension=_integer(
                source.get("MINDBRIDGE_EMBEDDING_DIMENSION"),
                DEFAULT_EMBEDDING_DIMENSION,
                "MINDBRIDGE_EMBEDDING_DIMENSION",
            ),
            timeout_seconds=_number(
                source.get("MINDBRIDGE_TIMEOUT_SECONDS"),
                DEFAULT_TIMEOUT_SECONDS,
                "MINDBRIDGE_TIMEOUT_SECONDS",
            ),
            media_transport=_transport(source.get("MINDBRIDGE_MEDIA_TRANSPORT", "data")),
            allowed_url_hosts=_hosts_from_environment(source.get("MINDBRIDGE_ALLOWED_URL_HOSTS")),
            capabilities=ModelCapabilities(
                embedding=_modalities(
                    source.get("MINDBRIDGE_EMBEDDING_MODALITIES"),
                    DEFAULT_CAPABILITIES.embedding,
                    "MINDBRIDGE_EMBEDDING_MODALITIES",
                ),
                generation=_modalities(
                    source.get("MINDBRIDGE_GENERATION_MODALITIES"),
                    DEFAULT_CAPABILITIES.generation,
                    "MINDBRIDGE_GENERATION_MODALITIES",
                ),
                transcription=_modalities(
                    source.get("MINDBRIDGE_TRANSCRIPTION_MODALITIES"),
                    DEFAULT_CAPABILITIES.transcription,
                    "MINDBRIDGE_TRANSCRIPTION_MODALITIES",
                ),
            ),
        )


def _setting(source: Mapping[str, str], name: str) -> str | None:
    """Read one environment value, treating a blank string as absent."""
    value = source.get(name)
    return None if _unset(value) else value


def _unset(value: str | None) -> bool:
    """Report whether an environment value is absent.

    A blank string is how a shell, `docker run -e` and a compose file clear a variable, so it
    must mean "use the default" rather than "an empty setting".
    """
    return value is None or not value.strip()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be non-empty text")
    return value.strip()


def _key(value: object | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be text")
    return value.strip() or None


def _base_url(value: object) -> str:
    text = _text(value, "base_url")
    try:
        parsed = urlsplit(text)
        _ = parsed.port
    except ValueError:
        raise ValidationError(
            "base_url must be an HTTP(S) URL without credentials or a query"
        ) from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in text)
    ):
        raise ValidationError("base_url must be an HTTP(S) URL without credentials or a query")
    parts = [part for part in parsed.path.split("/") if part]
    while parts and parts[-1].casefold() == "v1":
        parts.pop()
    parts.append("v1")
    return urlunsplit((parsed.scheme, parsed.netloc, "/" + "/".join(parts), "", ""))


def _integer(value: str | None, default: int, name: str) -> int:
    if value is None or _unset(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a positive integer") from None


def _number(value: str | None, default: float, name: str) -> float:
    if value is None or _unset(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a positive number") from None


def _transport(value: str) -> Literal["data", "file"]:
    if _unset(value):
        return "data"
    if value not in {"data", "file"}:
        raise ValidationError("MINDBRIDGE_MEDIA_TRANSPORT must be 'data' or 'file'")
    return "data" if value == "data" else "file"


def _hosts(values: object) -> frozenset[str]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise ValidationError("allowed_url_hosts must be a collection of hostnames")
    supplied = tuple(values)
    if any(not isinstance(value, str) for value in supplied):
        raise ValidationError("allowed_url_hosts must contain only hostnames")
    try:
        hosts = frozenset(
            value.strip().rstrip(".").encode("idna").decode("ascii").casefold()
            for value in supplied
        )
    except UnicodeError:
        raise ValidationError("allowed_url_hosts contains an invalid hostname") from None
    if any(
        not host or "://" in host or any(character in host for character in "/:?#@")
        for host in hosts
    ):
        raise ValidationError("allowed_url_hosts must contain hostnames without URLs or ports")
    return hosts


def _hosts_from_environment(value: str | None) -> frozenset[str]:
    return (
        frozenset()
        if value is None
        else frozenset(item.strip() for item in value.split(",") if item.strip())
    )


def _modalities(
    value: str | None,
    default: frozenset[Modality],
    name: str,
) -> frozenset[Modality]:
    if value is None or _unset(value):
        return default
    try:
        parsed = {Modality(item.strip().lower()) for item in value.split(",") if item.strip()}
    except ValueError:
        raise ValidationError(f"{name} contains an invalid modality") from None
    if Modality.OMNI in parsed:
        parsed.remove(Modality.OMNI)
        parsed.update({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})
    return frozenset(parsed)
