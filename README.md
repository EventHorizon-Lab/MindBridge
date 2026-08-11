# MindBridge

MindBridge is an Agentic Native Embodied Memory System: Memory-as-a-Service for machines that can see and hear.

## Documentation

- [Technical implementation architecture](docs/technical-architecture.md)

## Development

MindBridge supports Python 3.10 and 3.11. Python 3.10 is kept as the compatibility floor for Jetson deployments.

Install the project and development tools with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
```

Run the required local quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
git diff --check
```
