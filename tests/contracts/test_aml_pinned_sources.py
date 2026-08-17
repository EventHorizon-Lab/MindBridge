"""Vendored AML source contract: `benchmarks/aml/PINNED.md` must be true.

Cheap 7 (final review, 2026-08-17): `PINNED.md`'s entire purpose is being a
claim a reviewer can check -- "these vendored files match this sha256, as of
this revision" -- but nothing in the suite ever checked it. A vendored file
edited (accidentally or otherwise) without updating the pin would go
undetected indefinitely; this test makes that drift a CI failure instead.
"""

from __future__ import annotations

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
            f"(expected {expected_sha256}, got {actual_sha256}) -- vendored AML files must never "
            "be edited; re-vendor and re-pin instead"
        )
