# Model adapter guidelines

Follow the root [repository guidelines](../../../AGENTS.md). Keep clients in this package thin:
the kernel owns memory behavior and validates the meaning of proposed changes.

- Build requests from the configured model's supported parameters. Do not add sampling controls
  to reasoning models that reject them, silently discard explicit settings, or change models to
  make a request succeed. Test emitted requests against realistic provider constraints.
- Keep compatible-server protocols explicit. Strict JSON Schema is preferred when the endpoint
  supports it; JSON mode still requires local structural and semantic validation. Never silently
  downgrade a configured strict schema after a provider rejection.
- Treat a provider refusal separately from malformed output and truncation. Buffered and streamed
  calls must preserve the same failure classification, usage accounting, and resource cleanup.
  A refusal must not enter the malformed-response retry path.
- Classify errors from SDK types and documented provider codes. Distinguish temporary overload
  and rate limits from exhausted credits, spend limits, and invalid configuration; respect the
  client's retry budget rather than adding an unbounded retry loop.
- Record provider request IDs and returned model identity on model traces without recording
  credentials, prompts, memory content, response bodies, or refusal text.
- Include output protocol and effective generation controls in derived-content recipe identities.
  Prompt or model changes need quality-evaluation evidence, including insufficient evidence and
  adversarial stored content, in addition to transport tests.
