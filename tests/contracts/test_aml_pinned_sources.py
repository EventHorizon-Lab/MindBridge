"""Vendored AML source contract: `benchmarks/aml/PINNED.md` must be true.

Cheap 7 (final review, 2026-08-17): `PINNED.md`'s entire purpose is being a
claim a reviewer can check -- "these vendored files match this sha256, as of
this revision" -- but nothing in the suite ever checked it. A vendored file
edited (accidentally or otherwise) without updating the pin would go
undetected indefinitely; this test makes that drift a CI failure instead.

`PINNED.md` now also records one deliberate local delta, and a hash alone
cannot protect it: re-pinning a fresh upstream copy matches its own hash
while dropping the delta. So the delta gets its own check here, keyed off
what the code does rather than off what it hashes to.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from mindbridge.file_integrity import sha256_file

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PINNED_MD = _REPO_ROOT / "benchmarks" / "aml" / "PINNED.md"
_PIN_LINE = re.compile(r"^([0-9a-f]{64})\s+(\S+)$")


def _pinned_entries() -> list[tuple[str, str]]:
    entries = [
        (match.group(1), match.group(2))
        for line in _PINNED_MD.read_text(encoding="utf-8").splitlines()
        if (match := _PIN_LINE.match(line))
    ]
    assert entries, f"no sha256 pin lines found in {_PINNED_MD}"
    return entries


def test_pinned_md_lists_at_least_the_six_vendored_pipelines() -> None:
    paths = {path for _, path in _pinned_entries()}
    for benchmark in ("beam", "clbench", "locomo-refined", "longmemeval-s", "scriptmem"):
        assert f"benchmarks/aml/pipelines/{benchmark}/pipeline.py" in paths
    assert "benchmarks/aml/pipelines/personamem/pipeline_v1.py" in paths
    assert "benchmarks/aml/pipelines/personamem/pipeline_v2.py" in paths


def test_every_pinned_file_matches_its_recorded_sha256() -> None:
    for expected_sha256, relative_path in _pinned_entries():
        vendored_path = _REPO_ROOT / relative_path
        assert vendored_path.is_file(), f"{relative_path} is pinned but does not exist"
        actual_sha256 = sha256_file(vendored_path)
        assert actual_sha256 == expected_sha256, (
            f"{relative_path} does not match its pin in {_PINNED_MD.relative_to(_REPO_ROOT)} "
            f"(expected {expected_sha256}, got {actual_sha256}) -- vendored AML files carry only "
            "the one delta that file records; re-vendor, re-apply it, and re-pin instead"
        )


def _pinned_pipelines() -> list[Path]:
    paths = [_REPO_ROOT / path for _, path in _pinned_entries() if path.endswith("pipeline_v1.py")]
    paths += [
        _REPO_ROOT / path
        for _, path in _pinned_entries()
        if path.endswith(("pipeline.py", "pipeline_v2.py"))
    ]
    assert len(paths) == 7, f"expected 7 vendored pipelines, found {len(paths)}"
    return paths


def _jsonl_readers_splitting_on_unicode_boundaries(source: str) -> list[str]:
    """Name every function that parses JSON out of `splitlines()` pieces.

    `splitlines()` breaks on U+2028, U+2029 and U+0085 as well as on
    newlines. None is a JSON line delimiter and a JSON string may carry all
    three raw, so a reader built on it cuts records in half mid-string. It
    breaks on U+000B, U+000C and U+001C-U+001E too, but RFC 8259 forbids
    those unescaped inside a string, so they cost no valid record either
    way. Matching on the pair of calls rather
    than on the function's name keeps this true if a re-vendor renames
    `read_jsonl`, and keeps it quiet if one legitimately splits prose.
    """
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        called = {
            child.func.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        if "splitlines" in called and "loads" in called:
            offenders.append(node.name)
    return offenders


def test_no_vendored_pipeline_reads_jsonl_with_splitlines() -> None:
    """The one deliberate local delta from upstream, asserted so a re-vendor keeps it.

    These files write their own intermediate answers files with
    `ensure_ascii=False` and read them straight back, so a raw separator
    reaches them from upstream's own output whatever this repo writes -- and
    `docs/benchmarking.md` also runs them over this repo's shards. Upstream
    read both with `splitlines()`, which made scoring CL-Bench raise
    `JSONDecodeError: Unterminated string`. Re-pinning a fresh upstream copy
    silently reinstates that, which is exactly the drift `PINNED.md` exists to
    prevent -- so the delta gets a check of its own, not just a recorded hash.
    """
    offenders = {
        path.relative_to(_REPO_ROOT).as_posix(): names
        for path in _pinned_pipelines()
        if (
            names := _jsonl_readers_splitting_on_unicode_boundaries(
                path.read_text(encoding="utf-8")
            )
        )
    }
    assert not offenders, (
        f"vendored JSONL readers still split on Unicode line boundaries: {offenders} -- "
        "re-apply the `split(chr(10))` delta recorded in benchmarks/aml/PINNED.md"
    )
