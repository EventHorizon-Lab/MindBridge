"""Read-only, physically isolated compilation replays for paired evaluations.

The helper deliberately receives an already-open public ``AsyncMemory``.  Configuration and
model construction stay with the evaluation driver; this module only clones a closed store and
calls the public ``Memory.compile`` surface with a frozen question clock.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from mindbridge import AsyncMemory, ContextBudget, RetrievalScope
from mindbridge.types import ContentInput


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    """The complete immutable input to one compiled-context replay."""

    question: ContentInput
    reference_at: datetime
    budget: ContextBudget
    scope: RetrievalScope | None = None


@dataclass(frozen=True, slots=True)
class ReplayContext:
    """Exact text and ordered media identity delivered to a downstream generator."""

    rendered: str
    media: tuple[tuple[str, str, str | None, str | None], ...]


@dataclass(frozen=True, slots=True)
class GeneratorPayload:
    """Exact final provider request bytes captured after prompt and media preparation."""

    body: bytes

    @property
    def sha256(self) -> str:
        return sha256(self.body).hexdigest()


def same_generator_input(left: GeneratorPayload, right: GeneratorPayload) -> bool:
    """Say whether an answer may be reused as an exact paired control.

    The provider-ready body includes the question, system prompt, model controls, and the actual
    prepared media payload. Deliberately avoid normalisation: any byte or ordering difference
    requires a new generation and score.
    """
    return left.body == right.body


def clone_closed_store(source: Path, destination: Path) -> Path:
    """Copy a closed store into an unused physical directory for one replay arm.

    A replay never shares the original ``data_dir`` and never overwrites a destination.  The
    caller must close the source store before cloning it, which keeps SQLite and derived-index
    files internally consistent.
    """
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"closed store directory does not exist: {source}")
    if source == destination:
        raise ValueError("paired replay destination must differ from its source store")
    if destination.is_relative_to(source):
        raise ValueError("paired replay destination must not be nested under its source store")
    if destination.exists():
        raise FileExistsError(f"paired replay destination already exists: {destination}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    return destination


async def replay_compile(memory: AsyncMemory, request: ReplayRequest) -> ReplayContext:
    """Compile through the public SDK and retain the unnormalised generator input."""
    bundle = await memory.compile(
        request.question,
        budget=request.budget,
        reference_at=request.reference_at,
        scope=request.scope,
    )
    media = tuple(
        (hit.id, asset.id, asset.sha256, asset.media_type)
        for hit in bundle.hits
        for asset in hit.assets
    )
    return ReplayContext(rendered=bundle.render(), media=media)
