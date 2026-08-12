"""Reproducible component smoke for the production Jina Omni adapter."""

from __future__ import annotations

import argparse
import asyncio
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version

from pydantic import AwareDatetime

from mindbridge.contracts import ContractModel
from mindbridge.models.jina import JinaOmniEmbedder

QUERY = "Where did the robot put the red screwdriver?"
RELEVANT_DOCUMENT = "The robot put the red screwdriver beside the blue toolbox."
UNRELATED_DOCUMENT = "The weather outside is sunny and warm."


class JinaSmokeResult(ContractModel):
    """Versioned output needed to reproduce an embedding smoke run."""

    created_at: AwareDatetime
    model_id: str
    revision: str
    device: str
    dimension: int
    query_norm: float
    relevant_similarity: float
    unrelated_similarity: float
    python_version: str
    torch_version: str
    transformers_version: str
    sentence_transformers_version: str
    passed: bool


async def run_jina_smoke(*, revision: str, device: str) -> JinaSmokeResult:
    """Load the pinned model and compare one relevant and irrelevant document."""
    embedder = JinaOmniEmbedder.load(revision=revision, device=device)
    query = (await embedder.encode_queries((QUERY,)))[0]
    relevant, unrelated = await embedder.encode_documents((RELEVANT_DOCUMENT, UNRELATED_DOCUMENT))
    query_norm = math.sqrt(sum(value * value for value in query))
    relevant_similarity = _dot(query, relevant)
    unrelated_similarity = _dot(query, unrelated)
    return JinaSmokeResult(
        created_at=datetime.now(timezone.utc),
        model_id=embedder.model_reference.model_id,
        revision=embedder.model_reference.revision,
        device=device,
        dimension=len(query),
        query_norm=query_norm,
        relevant_similarity=relevant_similarity,
        unrelated_similarity=unrelated_similarity,
        python_version=platform.python_version(),
        torch_version=version("torch"),
        transformers_version=version("transformers"),
        sentence_transformers_version=version("sentence-transformers"),
        passed=(
            len(query) == embedder.dimension
            and math.isclose(query_norm, 1.0, rel_tol=1e-4, abs_tol=1e-6)
            and relevant_similarity > unrelated_similarity
        ),
    )


def main() -> None:
    """Run the smoke and emit a machine-readable manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    result = asyncio.run(run_jina_smoke(revision=arguments.revision, device=arguments.device))
    print(result.model_dump_json(indent=2))
    if not result.passed:
        raise SystemExit(1)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


if __name__ == "__main__":
    main()
