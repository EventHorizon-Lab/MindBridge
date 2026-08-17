"""Replay AML cases through the deployed Add/Search contract.

Talks to `/aml/add` and `/aml/search` over HTTP only -- nothing here may
import `mindbridge.api`, so this module stays usable against a real deployed
server, not just an in-process one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import httpx

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion, chunk_messages

EmitFn = Callable[[AmlQuestion, Sequence[dict[str, object]]], dict[str, object]]

# `Identifier` (mindbridge.contracts) caps every wire id at 255 characters.
# Measured against the real vendored corpora, the longest `case.user_id` any
# loader produces is PersonaMem v1's ~83-character
# `personamem-v1:{64-char sha256 shared_context_id}:{end_index}`, so the
# `eval:{run_id}:{benchmark}:...` prefix this driver adds has well over 150
# characters of headroom for any reasonable run_id/benchmark. `run_id` is
# still caller-supplied and itself allowed up to 255 characters, so an
# adversarial run_id can still overflow the combined id. Raise loudly on
# that rather than truncate: a truncated id silently merges two retrieval
# scopes, which is exactly the failure this harness exists to catch.
_MAX_IDENTIFIER_LENGTH = 255

# Preserves the previous hardcoded behaviour for callers (tests, mainly) that
# don't pass `timeout` explicitly. `cli.py` threads `--request-timeout-seconds`
# through instead of relying on this default for a real run.
_DEFAULT_TIMEOUT_SECONDS = 600.0


def _require_identifier_length(value: str, label: str) -> None:
    if len(value) > _MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"{label} is {len(value)} characters, over the "
            f"{_MAX_IDENTIFIER_LENGTH}-character Identifier limit: {value!r}"
        )


def row_id(case: AmlCase, question: AmlQuestion) -> str:
    """The id every row this driver writes for `question` carries under `"id"`.

    `run_case` builds a row as `{"id": f"{case.user_id}#{question.question_id}",
    ...}` and then applies `row.update(question.payload)` -- so whenever a
    loader's payload carries its own `"id"` (every loader but `locomo.py`),
    that value wins over the driver's own format. This is the single place
    that resolves which one applies, so a caller (the CLI's resume check,
    Blocking 2 in the 2026-08-17 final review) can predict a row's id without
    duplicating -- and risking drifting from -- `run_case`'s own logic.
    """
    payload_id = question.payload.get("id")
    return str(payload_id) if payload_id is not None else f"{case.user_id}#{question.question_id}"


def eval_user_id(run_id: str, benchmark: str, case_user_id: str) -> str:
    """Build the eval-scoped AML `user_id` this driver sends over the wire.

    Exposed so the CLI (Task 13) can recompute the exact same id -- e.g. to
    report a run's `user_id` -> `tenant_id` mapping in its manifest -- without
    duplicating this format string and risking drift.
    """
    return f"eval:{run_id}:{benchmark}:{case_user_id}"


async def run_case(
    client: httpx.AsyncClient,
    case: AmlCase,
    *,
    run_id: str,
    benchmark: str,
    top_k: int,
    emit: EmitFn,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    """Add every chunk in order, then emit one scored-pipeline row per question."""
    user_id = eval_user_id(run_id, benchmark, case.user_id)
    session_id = user_id
    _require_identifier_length(user_id, "eval user_id")

    for index, chunk in enumerate(chunk_messages(case.messages)):
        request_id = f"{user_id}:chunk-{index}"
        _require_identifier_length(request_id, "eval request_id")
        payload = {
            "request_id": request_id,
            "messages": list(chunk),
            "user_id": user_id,
            "session_id": session_id,
        }
        response = await client.post("/aml/add", json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        if (
            body.get("request_id") != request_id
            or body.get("user_id") != user_id
            or body.get("session_id") != session_id
        ):
            raise ValueError(f"add did not echo its identifiers for {request_id}")

    rows: list[dict[str, object]] = []
    for question in case.questions:
        response = await client.post(
            "/aml/search",
            json={"query": question.question, "user_id": user_id, "top_k": top_k},
            timeout=timeout,
        )
        response.raise_for_status()
        retrieved = response.json().get("data", [])
        row: dict[str, object] = {"id": row_id(case, question), "question": question.question}
        row.update(question.payload)
        row.update(emit(question, retrieved))
        rows.append(row)
    return rows


def _joined(retrieved: Sequence[dict[str, object]]) -> str:
    lines = []
    for item in retrieved:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        created_at = str(item.get("created_at") or "").strip()
        lines.append(f"- [{created_at}] {content}" if created_at else f"- {content}")
    return "\n".join(lines)


def emit_retrieved_context(
    _question: AmlQuestion,
    retrieved: Sequence[dict[str, object]],
) -> dict[str, object]:
    """LoCoMo, LongMemEval, and BEAM all read `retrieved_context`."""
    return {"retrieved_context": _joined(retrieved)}


def emit_chat_history(
    _question: AmlQuestion,
    retrieved: Sequence[dict[str, object]],
) -> dict[str, object]:
    """PersonaMem v2 reads `chat_history` where v1 reads `context_messages`.

    v2's pipeline does fall back to `context_messages`, but relying on that
    fallback would work by accident today and break silently if upstream
    reorders the chain, so this gets its own emitter rather than reusing
    `emit_context_messages`.
    """
    return {"chat_history": _message_list(retrieved)}


def emit_selected(
    _question: AmlQuestion,
    retrieved: Sequence[dict[str, object]],
) -> dict[str, object]:
    """CL-Bench reads `retrieval.selected` with `text` and `created_at` keys."""
    return {
        "retrieval": {
            "selected": [
                {"text": item.get("content"), "created_at": item.get("created_at")}
                for item in retrieved
            ]
        }
    }


def emit_context_messages(
    _question: AmlQuestion,
    retrieved: Sequence[dict[str, object]],
) -> dict[str, object]:
    """PersonaMem v1 reads an already-sliced `context_messages` list."""
    return {"context_messages": _message_list(retrieved)}


def _message_list(retrieved: Sequence[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(item.get("content") or "")}
        for item in retrieved
        if str(item.get("content") or "").strip()
    ]
