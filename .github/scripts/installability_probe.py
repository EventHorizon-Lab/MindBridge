"""Prove one installed optional surface works, not merely that it imports.

Every heavy dependency in this package sits behind ``import_module`` inside a function so the
base import stays cold. That discipline hides missing declarations: importing
``mindbridge.models.funasr`` succeeds without touching one ``funasr`` symbol, so an import-only
CI probe cannot tell an installable extra from a broken one. Each leg below therefore imports the
supported modules and then runs the loader those modules defer to.

Usage: ``python installability_probe.py <leg>`` inside a venv holding only that leg's extra.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path


class _StubEmbedder:
    """Smallest object satisfying ``EmbeddingBackend``.

    Every leg needs an embedding backend and no leg may pull a model in to get one, so this is
    also a standing check that the protocol stays small enough to implement by hand.
    """

    embedding_model = "installability-probe"
    embedding_space = "installability-probe:2:v1"
    embedding_dimension = 2

    def __init__(self) -> None:
        from mindbridge.types import Modality

        self.embedding_capabilities = frozenset({Modality.TEXT})

    def embed(
        self,
        inputs: Sequence[object],
        task: object = None,
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in inputs)

    def close(self) -> None:
        return None


def _memory_flow(tracer: object = None) -> None:
    """Write and read one record end to end: SQLite, the outbox, and the Zvec loader."""
    from mindbridge import Memory

    with (
        tempfile.TemporaryDirectory() as directory,
        Memory(Path(directory) / "store", embedder=_StubEmbedder(), tracer=tracer) as memory,
    ):
        record = memory.add("the installability probe wrote this record")
        assert memory.get(record.id) is not None
        assert memory.search("installability probe", limit=1)
        assert memory.list(limit=1).items
        memory.reindex()
        assert memory.search("installability probe", limit=1)
        assert memory.delete(record.id)


def _base() -> None:
    _memory_flow()
    _product_cli()


def _product_cli() -> None:
    """Run the `mindbridge` console script's entry point on the base dependency set.

    The script is declared whether or not its module imports, and every composition path except
    `--url` needs an extra, so the cheapest honest end-to-end check is a resolution that touches
    argparse, `mindbridge.recipes`, and the JSON writer without constructing a backend.
    """
    from mindbridge.cli import main

    assert main(["--url", "http://127.0.0.1:1", "-q", "--explain", "doctor"]) == 0


def _benchmarks() -> None:
    """Reach the download client and the Parquet reader the catalog tasks load lazily."""
    from mindbridge.benchmarks.cli import main

    assert main(["--help"]) == 0
    importlib.import_module("httpx")
    importlib.import_module("pyarrow.parquet")
    importlib.import_module("nltk")


def _observability() -> None:
    """Export real operation spans through the SDK this extra exists to supply."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from mindbridge._telemetry import TRACER_NAME

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _memory_flow(provider.get_tracer(TRACER_NAME))
    provider.shutdown()
    assert {"mindbridge.add", "mindbridge.search"} <= {
        span.name for span in exporter.get_finished_spans()
    }


def _openai() -> None:
    """Build the adapter over a real SDK client without issuing a request."""
    from openai import OpenAI

    from mindbridge import OpenAIModels

    client = OpenAI(api_key="installability-probe", base_url="http://127.0.0.1:1/v1")
    models = OpenAIModels(client)
    assert models.embedding_dimension > 0
    models.close()


def _face() -> None:
    """Load the OpenCV APIs used by the face backend without downloading model weights."""
    cv = importlib.import_module("cv2")
    assert all(hasattr(cv, name) for name in ("FaceDetectorYN", "FaceRecognizerSF", "VideoCapture"))


def _local() -> None:
    """Load every third-party module the local backends resolve at first use.

    ``import funasr.utils.load_utils`` is not enough: since 1.4.7 FunASR guards ``torchaudio``
    behind ``funasr.utils.torchaudio_compat`` and falls back for decode and resampling, so that
    import succeeds with torchaudio absent. The one operation with no fallback is
    ``torchaudio.functional.forced_align``, which Fun-ASR-Nano -- the default recipe -- calls for
    its native timestamps, so assert that symbol instead.
    """
    import cairosvg
    import torchaudio.functional

    from mindbridge.models.jina import _require_local_extra

    for name in ("funasr", "librosa", "sentence_transformers", "soundfile", "torch"):
        importlib.import_module(name)
    _require_local_extra()
    assert hasattr(torchaudio.functional, "forced_align")
    # The pinned Jina revision converts image/svg+xml through cairosvg, and memory.py maps .svg.
    assert cairosvg.svg2png(
        bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>'
    )


def _server() -> None:
    """Build the real ASGI application over a real memory instance."""
    from mindbridge import Memory
    from mindbridge.api import create_app

    with (
        tempfile.TemporaryDirectory() as directory,
        Memory(Path(directory) / "store", embedder=_StubEmbedder()) as memory,
    ):
        app = create_app(memory=memory)
    # Generating the schema resolves every route's request and response model. Route objects are
    # not the assertion: a floating FastAPI can nest the /v1 router where `.path` does not reach.
    assert set(app.openapi()["paths"]) == {
        "/healthz",
        "/v1/answers",
        "/v1/memories",
        "/v1/memories/batch",
        "/v1/memories/search",
        "/v1/memories/{memory_id}",
    }


def _mcp() -> None:
    """Build the real MCP server and resolve the five supported tool schemas."""
    import asyncio

    from mindbridge import Memory
    from mindbridge.api.mcp import build_mcp_server

    with (
        tempfile.TemporaryDirectory() as directory,
        Memory(Path(directory) / "store", embedder=_StubEmbedder()) as memory,
    ):
        server = build_mcp_server(memory)
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "add_memory",
        "ask_memory",
        "delete_memory",
        "get_memory",
        "search_memories",
    }


MODULES: dict[str, tuple[str, ...]] = {
    "base": (
        "mindbridge",
        "mindbridge.benchmarks",
        "mindbridge.cli",
        "mindbridge.benchmarks.isolation",
        "mindbridge.benchmarks.local_index_benchmark",
        "mindbridge.exceptions",
        "mindbridge.infrastructure.local",
        "mindbridge.memory",
        "mindbridge.models.base",
        "mindbridge.models.openai_sdk",
        "mindbridge.recipes",
        "mindbridge.types",
    ),
    "benchmarks": (
        "mindbridge.benchmarks.cli",
        "mindbridge.benchmarks.download",
        "mindbridge.benchmarks.eval",
        "mindbridge.benchmarks.official_scorers",
        "mindbridge.benchmarks.prepare_media",
        "mindbridge.benchmarks.task_catalog",
    ),
    "observability": ("mindbridge", "mindbridge._telemetry"),
    "openai": ("mindbridge", "mindbridge.models.openai_sdk"),
    "face": ("mindbridge", "mindbridge.models.opencv_face"),
    "local": (
        "mindbridge.models.funasr",
        "mindbridge.models.jina",
        "mindbridge.models.sentence_transformers",
    ),
    "server": ("mindbridge.api", "mindbridge.api.app"),
    "mcp": ("mindbridge.api.mcp",),
}

LOADERS: dict[str, Callable[[], None]] = {
    "base": _base,
    "benchmarks": _benchmarks,
    "observability": _observability,
    "openai": _openai,
    "face": _face,
    "local": _local,
    "server": _server,
    "mcp": _mcp,
}


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1 or argv[0] not in LOADERS:
        print(f"usage: installability_probe.py {{{','.join(LOADERS)}}}", file=sys.stderr)
        return 2
    leg = argv[0]
    modules = MODULES[leg]
    for module in modules:
        importlib.import_module(module)
    LOADERS[leg]()
    print(f"{leg}: imported {len(modules)} modules and ran the loader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
