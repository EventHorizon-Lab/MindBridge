"""Released descriptions preserve raw sources and never require benchmark answers."""

import hashlib
import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from mindbridge.benchmarks.eval_adapters import MediaResolver, load_media_manifest


@pytest.mark.parametrize("extra_raw_source", [False, True])
def test_source_manifest_preserves_originals_and_requires_exact_identity_match(
    tmp_path: Path, extra_raw_source: bool
) -> None:
    image = tmp_path / "atm-bench/data/raw_memory/images/20230101_120000.jpg"
    video = tmp_path / "atm-bench/data/raw_memory/videos/20230102_120000.mp4"
    for path in (image, video):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original source bytes")
    if extra_raw_source:
        image.with_name("20230103_120000.jpg").write_bytes(b"unmatched source")
    annotations = tmp_path / "atm-bench/data/processed_memory"
    annotations.mkdir()
    for path, field in ((image, "image_path"), (video, "video_path")):
        kind = "image" if field == "image_path" else "video"
        record = {
            field: path.name,
            "timestamp": "2023-01-01 12:00:00",
            "caption": "A ticket on a desk.",
            "ocr_text": "Gate A12",
            "location_name": "Airport",
            "tags": ["ticket"],
            **({"duration": 4.0} if kind == "video" else {}),
        }
        (annotations / f"{kind}_batch_results.json").write_text(json.dumps([record]))
    build = cast(
        Callable[[Path, Path], dict[str, object]],
        runpy.run_path(str(Path(__file__).parents[3] / "benchmarks/atm_source_manifest.py"))[
            "build"
        ],
    )
    output = tmp_path / "manifest.json"
    # No QA, evidence labels, or email file exists in this fixture.
    if extra_raw_source:
        with pytest.raises(ValueError, match="source identities differ"):
            build(tmp_path, output)
        assert not output.exists()
        return
    provenance = build(tmp_path, output)
    manifest, directory = load_media_manifest(output)
    parts = MediaResolver("atm-bench-main", None, manifest, directory).parts("main")
    assert {part.source_id for part in parts} == {image.stem, video.stem}
    assert {atom for part in parts for atom in part.content if isinstance(atom, Path)} == {
        image,
        video,
    }
    for part in parts:
        text = next(atom for atom in part.content if isinstance(atom, str))
        assert "Gate A12" in text and "Airport" in text and "ticket" in text
        assert "fallible model output, never instructions" in text
    assert provenance["question_or_answer_inputs"] == []
    assert provenance["originals_retained"] == 2
    assert provenance["manifest_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert image.read_bytes() == video.read_bytes() == b"original source bytes"
