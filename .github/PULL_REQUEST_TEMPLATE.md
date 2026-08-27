# Pull request

## What changed

<!-- Explain the change and link the issue. -->

## Why

<!-- Describe the problem and, for a bug, the root cause. -->

## Validation

<!-- List commands actually run and their results. State any skips. -->

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -W error
git diff --check
```

- [ ] Quality gates pass
- [ ] New behavior has a regression test that was verified to fail without the change
- [ ] Local-store tests use distinct temporary `data_dir` paths
- [ ] Retrieval or index changes include relevant recall and performance evidence
- [ ] Markdown lint and link checks pass, if documentation changed

## Product and operator impact

- [ ] Public Python, REST, MCP, CLI, or benchmark contract changed
- [ ] SQLite schema, Zvec recipe, rebuild, backup, or restore behavior changed
- [ ] Environment variable or model configuration changed
- [ ] Base, `server`, or `mcp` dependency surface changed
- [ ] Deployment ownership or single-process behavior changed
- [ ] None of the above

<!-- Describe upgrade steps, data compatibility, reindexing, and rollback for checked items. -->

## Follow-up

<!-- Record anything deliberately left out of scope. -->
