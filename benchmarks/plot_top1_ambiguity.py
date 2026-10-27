# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=2,<3"]
# ///
"""Render measured operating points; run in an isolated uv script environment."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    style = runpy.run_path(str(arguments.style))
    style["use_style"]()
    result = json.loads(arguments.results.read_text(encoding="utf-8"))
    figure, axis = style["figure"](width=style["TEXT"], ratio=0.68)
    for policy, label, color, marker in (
        ("route_exemption", "Route exemption", "red", "s"),
        ("coverage_exemption", "Coverage + discriminative term", "orange", "^"),
        ("no_exemption", "No exemption (shipped)", "blue", "o"),
    ):
        points = sorted(
            {
                (point["coverage"] * 100, point["risk"] * 100)
                for point in result["points"]
                if point["policy"] == policy and point["risk"] is not None
            }
        )
        x, y = zip(*points, strict=True)
        color_value = style["PALETTE"][color]
        axis.scatter(x, y, color=color_value, marker=marker, s=17, alpha=0.6, label=label)
        # Connect attainable points only as a visual guide; no interpolated risk claims.
        axis.plot(x, y, color=color_value, linewidth=0.7, alpha=0.5)
        default = next(
            point
            for point in result["points"]
            if point["policy"] == policy and point["margin"] == 0.01
        )
        axis.scatter(
            default["coverage"] * 100,
            default["risk"] * 100,
            color=color_value,
            marker=marker,
            s=55,
            edgecolors="black",
            linewidths=0.7,
        )
    axis.set_xlabel("Singleton coverage (%) — higher is better")
    axis.set_ylabel("Unsafe retrievals / accepted (%) — lower is better")
    axis.set_xlim(0, 100)
    axis.set_ylim(-2, 55)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left", frameon=False)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    style["save"](figure, str(arguments.output))


if __name__ == "__main__":
    main()
