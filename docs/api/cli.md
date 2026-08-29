# Command-line reference

MindBridge provides one console command: `mindbridge-bench`.

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

There is no generic product, REST, or MCP CLI. Such a command would have to guess a provider,
credentials, retry policy, and model capabilities. Host applications construct their provider SDK
clients and `Memory` explicitly.

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
