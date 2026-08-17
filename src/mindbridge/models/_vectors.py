"""The shared vector contract every embedding adapter enforces at its boundary."""

from __future__ import annotations

import math

from mindbridge.core import ModelOutputError


def validate_embedding_vector(values: tuple[float, ...], dimension: int) -> None:
    """Reject a malformed or unnormalized vector before it reaches a versioned index.

    The tolerance absorbs F16 serving jitter only. `Embedding` repeats the normalization
    invariant as a domain rule; an adapter additionally owns the configured dimension,
    because only it knows what the deployment asked the provider to return.
    """
    if len(values) != dimension or not all(math.isfinite(value) for value in values):
        raise ModelOutputError("embedding vector has invalid dimension or values")
    norm = math.hypot(*values)
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-6):
        raise ModelOutputError("embedding vector is not L2-normalized")
