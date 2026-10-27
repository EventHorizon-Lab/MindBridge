"""Render unmodified figure excerpts from the verified original research PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

FIGURES = (
    (
        "nevir-generalization",
        "nevir-reproduction.pdf",
        "https://arxiv.org/pdf/2502.13506v4",
        "7f9398aab55444d47847374b0f124cbbb873b7ab6ef29e115456ab3789bcde97",
        8,
        (310, 82, 565, 322),
    ),
    (
        "chinese-negation-binding",
        "negcomment.pdf",
        "https://arxiv.org/pdf/2404.09830v1",
        "12c74dc9077cd0d7be75a1ae3acc5e26ec7086e26bc79fb8f5fff3651ac191d0",
        2,
        (310, 218, 538, 315),
    ),
)


def main(pdf_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = 220
    provenance = []
    for name, filename, source, expected, page, bounds in FIGURES:
        pdf = pdf_dir / filename
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"The source PDF changed: {filename}")
        left, top, right, bottom = bounds
        scale = resolution / 72
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-r",
                str(resolution),
                "-x",
                str(round(left * scale)),
                "-y",
                str(round(top * scale)),
                "-W",
                str(round((right - left) * scale)),
                "-H",
                str(round((bottom - top) * scale)),
                "-png",
                str(pdf),
                str(output_dir / name),
            ],
            check=True,
        )
        provenance.append(
            {
                "image": name + ".png",
                "source": source,
                "source_sha256": digest,
                "pdf_page_one_based": page,
                "crop_points": bounds,
                "resolution_dpi": resolution,
                "transform": "PDF rendering and crop only; original labels and colors preserved",
            }
        )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    arguments = parser.parse_args()
    main(arguments.pdf_dir, arguments.output_dir)
