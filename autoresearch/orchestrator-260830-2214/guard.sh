#!/usr/bin/env bash
set -euo pipefail

uv lock --check --default-index https://pypi.org/simple
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy
uv run --frozen pytest -W error
git diff --check
