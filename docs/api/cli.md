# Command-line reference

## Current status

The current release provides one console command: `mindbridge-bench`.

```bash
mindbridge-bench --help
mindbridge-bench --version
```

Supported benchmark subcommands are:

```bash
mindbridge-bench eval --tasks list
mindbridge-bench locomo-refined --help
mindbridge-bench local-index --help
```

The MindBridge product CLI required by the architecture is not implemented yet. This is a current
capability gap, not an intentional limit to benchmarking. The product CLI must expose MindBridge's
own memory operations and remain aligned with the SDK and MCP.

The complete CLI surface has two command families:

- Product commands dispatch to the shared `Memory` execution plane.
- Benchmark commands exercise the public SDK and remain leaf tooling, not an alternate product API.

`mindbridge-bench` is only the currently shipped family.

Agents that need a supported memory interface today should use the typed [MCP tools](mcp.md) or the
OpenAPI-documented [REST API](rest.md). `mindbridge-bench` is not a product memory interface.

Run REST with the application's ASGI server:

```bash
uvicorn my_application:app --workers 1
```

Run MCP from application code:

```python
with memory:
    build_mcp_server(memory).run("stdio")
```

Call `memory.reindex()` and `memory.optimize()` from the owning process. A separate maintenance
process cannot open the same physical directory while the application owns it.

## Required product CLI contract

The product CLI is a first-class developer and agent surface over the shared `Memory` execution
plane. Its capability inventory is derived from the SDK; it does not define another memory API.
When implemented, it must:

- Cover the SDK's product operations, including memory CRUD, search, grounded answers, speech, and
  maintenance. Any transport-driven omission must be explicit and justified.
- Preserve SDK and MCP vocabulary, defaults, IDs, cursors, result fields, side effects, and error
  semantics.
- Run non-interactively, accept generated input through stdin, and never require a terminal prompt.
- Provide stable JSON input/output, write only data to stdout, and send diagnostics to stderr.
- Use documented exit codes and the same stable error codes as REST and MCP.
- Bound or paginate every collection and return opaque cursors unchanged.
- Preserve idempotency and clearly identify destructive operations.
- Report the selected model/runtime identity without exposing credentials.

Provider setup remains explicit and belongs to the same application composition used by the SDK and
MCP. A shorter CLI must not create a second configuration system.

For a standalone invocation, that composition constructs one `Memory`, dispatches the requested SDK
operation, and closes it. If another process already owns the directory, the CLI must address that
owner through a supported transport; it must not open a second `Memory`. This preserves both one
execution plane and the one-directory, one-owner rule.
