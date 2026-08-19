# Pull request

## What changed

<!-- What this does, and why. Link the issue if there is one. -->

## Why

<!-- The problem being solved. If it was a bug, say what the root cause was. -->

## Validation

<!-- Commands you actually ran and what happened. "Tests pass" is not validation. -->

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -W error
```

- [ ] Quality gates pass
- [ ] Re-ran the gates after merging the base branch
- [ ] Integration suite ran against a real PostgreSQL (`MINDBRIDGE_REQUIRE_INTEGRATION=1`), or
      this change cannot affect recall, consolidation, or deletion
- [ ] New behaviour has a test that fails when the behaviour regresses — verified by breaking it
- [ ] Markdown lint and link check pass, if documentation changed

## Operator impact

- [ ] New or changed environment variables — documented in `docs/configuration.md`
- [ ] New migration — number re-checked after merging base
- [ ] New dependency — justified below
- [ ] Public contract changed (`mindbridge.contracts`, REST, MCP, CLI, entry points)
- [ ] None of the above

<!-- If any box above is checked, describe what an operator has to do. -->

## Follow-up

<!-- Anything deliberately left out of scope. -->
