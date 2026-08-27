"""Small contracts shared only by benchmark dataset adapters."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_048),
]
Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


class ContractModel(BaseModel):
    """Strictly shaped, immutable benchmark value."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MediaKind(str, Enum):
    """Media kinds present in benchmark releases."""

    IMAGE = "image"
    VIDEO = "video"
