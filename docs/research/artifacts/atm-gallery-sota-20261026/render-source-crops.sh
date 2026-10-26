#!/usr/bin/env bash
# Reproduce unmodified table crops from the cited arXiv source PDFs.
set -euo pipefail
output_dir="$(cd "$(dirname "$0")" && pwd)"
scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT
curl --fail --location --silent --show-error https://arxiv.org/pdf/2608.01543v1 -o "$scratch_dir/vmem.pdf"
curl --fail --location --silent --show-error https://arxiv.org/pdf/2601.03515v1 -o "$scratch_dir/gallery.pdf"
pdftoppm -f 6 -l 6 -r 260 -singlefile -png "$scratch_dir/vmem.pdf" "$scratch_dir/vmem"
pdftoppm -f 8 -l 8 -r 260 -singlefile -png "$scratch_dir/gallery.pdf" "$scratch_dir/gallery"
python3 - "$scratch_dir" "$output_dir" <<'PY'
import sys
from pathlib import Path

from PIL import Image

scratch, output = map(Path, sys.argv[1:])
# Coordinates are twice the visually inspected 130-dpi page coordinates.
for source, target, box in (
    ("vmem", "vmem-table3.png", (1150, 1066, 2020, 1296)),
    ("gallery", "gallery-table4.png", (1196, 310, 1790, 802)),
):
    with Image.open(scratch / f"{source}.png") as page:
        page.crop(box).save(output / target)
PY
