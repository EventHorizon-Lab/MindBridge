"""Checks for the shared client lifecycle, unit scheduling, and prepared-media index."""

import ast
import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from mindbridge.benchmarks import cli_common
from mindbridge.benchmarks.cli_common import (
    CoreArguments,
    connected_memory,
    core_arguments,
    core_parser,
    index_prepared,
    limit_units,
    report,
    report_unit,
    run_units,
    select_by_id,
)
from mindbridge.sdk import MindBridge


@dataclass(frozen=True, slots=True)
class _Prepared:
    unit_id: str


@dataclass(frozen=True, slots=True)
class _Indexed:
    index: int


async def test_connected_memory_closes_the_client_when_the_run_raises() -> None:
    """A failed benchmark must not leak the connection pool it opened."""
    opened: list[MindBridge] = []
    async with connected_memory(_arguments()) as memory:
        opened.append(memory)
        assert not memory._client.is_closed
        with pytest.raises(RuntimeError):
            async with connected_memory(_arguments()) as inner:
                inner_client = inner._client
                raise RuntimeError("benchmark failed mid-run")
        assert inner_client.is_closed
    assert opened[0]._client.is_closed


async def test_connected_memory_reads_the_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded invocation never carries the credential the run used."""
    monkeypatch.setenv("MINDBRIDGE_API_KEY", "runtime-secret-000000000000000000")
    async with connected_memory(_arguments()) as memory:
        assert memory._client.headers["authorization"] == "Bearer runtime-secret-000000000000000000"


async def test_connected_memory_allows_an_unauthenticated_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINDBRIDGE_API_KEY", raising=False)
    async with connected_memory(_arguments()) as memory:
        assert "authorization" not in memory._client.headers


def test_select_by_id_returns_the_whole_release_when_nothing_is_requested() -> None:
    """Empty means "no subset", which is how every runner spells a full official split."""
    items = (_Prepared("a"), _Prepared("b"))

    assert select_by_id(items, (), key=lambda i: i.unit_id, label="units") == items


def test_select_by_id_keeps_release_order_not_request_order() -> None:
    """A prediction artifact has to line up with the annotation, whatever order was asked for."""
    items = (_Prepared("first"), _Prepared("second"), _Prepared("third"))

    selected = select_by_id(items, ("third", "first"), key=lambda i: i.unit_id, label="units")

    assert selected == (items[0], items[2])


def test_select_by_id_rejects_a_duplicated_request() -> None:
    with pytest.raises(ValueError, match=r"^EgoTempo question IDs must not contain duplicates$"):
        select_by_id(
            (_Prepared("q1"),),
            ("q1", "q1"),
            key=lambda i: i.unit_id,
            label="EgoTempo question IDs",
        )


def test_select_by_id_names_the_benchmark_unit_in_its_refusal() -> None:
    """The label is the part an operator reads, so each benchmark keeps its own wording."""
    with pytest.raises(ValueError, match=r"^unknown LoCoMo-Refined sample IDs: nope$"):
        select_by_id(
            (_Prepared("conv-01"),),
            ("nope",),
            key=lambda i: i.unit_id,
            label="LoCoMo-Refined sample IDs",
        )


def test_select_by_id_reports_integer_units_in_numeric_order() -> None:
    """String ordering would print "10, 100, 9" and read as corrupt annotation data."""
    with pytest.raises(ValueError, match=r"^unknown MM-Lifelong question indices: 9, 10, 100$"):
        select_by_id(
            (_Indexed(1),),
            (9, 10, 100),
            key=lambda item: item.index,
            label="MM-Lifelong question indices",
        )


def test_limit_keeps_the_first_units_of_a_release() -> None:
    """`--limit` is what makes a smoke run of any of these cheap enough to iterate on."""
    items = (_Prepared("first"), _Prepared("second"), _Prepared("third"))

    selected = select_by_id(items, (), key=lambda i: i.unit_id, label="units", limit=2)

    assert selected == items[:2]


def test_limit_composes_with_an_explicit_selection_rather_than_competing_with_it() -> None:
    """Truncating after selection is what lets `--limit 1 --sample-id X --sample-id Y` mean X."""
    items = (_Prepared("first"), _Prepared("second"), _Prepared("third"))

    selected = select_by_id(
        items, ("third", "second"), key=lambda i: i.unit_id, label="units", limit=1
    )

    assert selected == (items[1],)


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_that_would_select_nothing_is_refused(limit: int) -> None:
    """An empty selection reaches the manifest as a run of nothing rather than as a mistake."""
    items = (_Prepared("first"),)

    with pytest.raises(ValueError, match="--limit must be a positive count of units"):
        select_by_id(items, (), key=lambda i: i.unit_id, label="units", limit=limit)


def test_limit_units_truncates_a_set_a_benchmark_filtered_after_selecting() -> None:
    """Video-MME filters by duration after selecting, so its limit is applied there instead.

    Applied before that filter, `--limit 2 --duration long` truncates to the first two videos of
    any band and can then keep none of them.
    """
    items = (_Prepared("first"), _Prepared("second"))

    assert limit_units(items, None, label="units") == items
    assert limit_units(items, 1, label="units") == items[:1]
    with pytest.raises(ValueError, match="positive count of units"):
        limit_units(items, 0, label="units")


def test_the_shared_parser_carries_the_limit_through_to_every_runner() -> None:
    """Every benchmark CLI builds on `core_parser`, so this is the wiring all twelve share."""
    parser = core_parser(tenant_prefix="benchmark_probe")
    required = [
        "--dataset",
        "release.json",
        "--output",
        "out.jsonl",
        "--api-base-url",
        "http://localhost:8000",
        "--deployment-config",
        "deployment.json",
        "--run-id",
        "probe",
    ]

    assert core_arguments(CoreArguments, parser.parse_args([*required, "--limit", "3"])).limit == 3
    assert core_arguments(CoreArguments, parser.parse_args(required)).limit is None


def test_index_prepared_keys_every_prepared_unit() -> None:
    prepared = (_Prepared("second"), _Prepared("first"))

    indexed = index_prepared(
        ("first", "second"),
        prepared,
        key=lambda item: item.unit_id,
        label="test units",
    )

    assert indexed == {"second": prepared[0], "first": prepared[1]}


def test_index_prepared_rejects_a_run_missing_a_required_unit() -> None:
    with pytest.raises(ValueError, match="missing prepared test units: absent"):
        index_prepared(
            ("present", "absent"),
            (_Prepared("present"),),
            key=lambda item: item.unit_id,
            label="test units",
        )


def test_index_prepared_reports_integer_units_in_numeric_order() -> None:
    """String ordering would report these as "10, 100, 9" and read as a data error."""
    with pytest.raises(ValueError, match=r"missing prepared examples: 9, 10, 100$"):
        index_prepared(
            (9, 10, 100),
            (_Indexed(1),),
            key=lambda item: item.index,
            label="examples",
        )


def test_index_prepared_accepts_a_run_that_requires_nothing() -> None:
    assert index_prepared((), (_Prepared("spare"),), key=lambda i: i.unit_id, label="u") == {
        "spare": _Prepared("spare")
    }


def _arguments() -> CoreArguments:
    return CoreArguments(
        dataset_path=Path("dataset.json"),
        output_path=Path("predictions.json"),
        api_base_url="https://memory.example.test",
        deployment_config_path=Path("deployment.json"),
        run_id="run_01",
        tenant_prefix="benchmark_test",
        recall_limit=20,
        request_concurrency=4,
        unit_concurrency=1,
        request_timeout_seconds=1_800.0,
        limit=None,
        overwrite=False,
        predict_only=False,
        quiet=True,
    )


def test_progress_lines_go_to_stderr_so_stdout_stays_the_artifact(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A runner's stdout is its machine-readable output; progress must never land there."""
    report("running 3 conversations", quiet=False)
    report_unit("conversation conv-1", index=1, total=3, quiet=False)

    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err == "running 3 conversations\n[1/3] conversation conv-1\n"


def test_quiet_silences_every_progress_line(capsys: pytest.CaptureFixture[str]) -> None:
    report("running 3 conversations", quiet=True)
    report_unit("conversation conv-1", index=1, total=3, quiet=True)

    streams = capsys.readouterr()
    assert (streams.out, streams.err) == ("", "")


async def _peak_in_flight(units: int, unit_concurrency: int) -> tuple[int, tuple[int, ...]]:
    """Run `units` units and report the most that were ever in flight together, plus the order.

    Each unit yields a fixed number of times before returning, which is enough for every sibling
    a permit is free for to enter first. No wall-clock sleeps, so the peak is deterministic: with
    the units awaited one at a time it is 1 whatever `unit_concurrency` says.
    """
    in_flight = 0
    peak = 0

    async def run(unit: int) -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        # Later units yield fewer times, so they finish first and the result order is only
        # right if it comes from the argument order rather than from completion order.
        for _ in range(2 * (units - unit)):
            await asyncio.sleep(0)
        in_flight -= 1
        return unit

    results = await run_units(
        tuple(range(units)),
        label=lambda unit: f"unit {unit}",
        run=run,
        unit_concurrency=unit_concurrency,
        quiet=True,
    )
    return peak, results


async def test_units_run_together_up_to_their_permit_count() -> None:
    """A run's ceiling is unit_concurrency times request_concurrency, not one unit's budget.

    Awaiting each unit in turn held the whole run to one unit's own fan-out, and drained it to
    nothing between units -- and raising --request-concurrency could not lift that, because the
    ceiling was the loop rather than the flag.
    """
    peak, results = await _peak_in_flight(units=6, unit_concurrency=3)

    assert peak == 3, f"only {peak} units were ever in flight; three permits were free"
    assert results == (0, 1, 2, 3, 4, 5), "results must come back in the order units were given"


async def test_unit_concurrency_of_one_still_runs_the_units_one_at_a_time() -> None:
    """The serial shape stays reachable, which is what an operator drops to when debugging."""
    peak, results = await _peak_in_flight(units=4, unit_concurrency=1)

    assert peak == 1
    assert results == (0, 1, 2, 3)


async def test_run_units_refuses_a_ceiling_that_would_run_nothing() -> None:
    with pytest.raises(ValueError, match="unit_concurrency"):
        await _peak_in_flight(units=2, unit_concurrency=0)


async def test_finished_units_are_counted_rather_than_numbered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With units overlapping, the progress counter has to mean "done", not "starting".

    Several units are in flight at once, so none of them is "the second"; the completion line is
    what carries the count, and the start line names the unit without claiming a position.
    """
    await run_units(
        ("a", "b"),
        label=lambda unit: f"unit {unit}",
        run=lambda unit: asyncio.sleep(0, result=unit),
        unit_concurrency=2,
        quiet=False,
    )

    lines = capsys.readouterr().err.splitlines()
    assert sorted(line for line in lines if line.startswith("starting")) == [
        "starting unit a",
        "starting unit b",
    ]
    # Which unit finishes first is not fixed, so the counters are asserted against the set of
    # units rather than an order. Pinning only the `[n/2]` prefixes would pass on a regression
    # that printed one unit's label on both lines.
    counted = sorted(line for line in lines if line.startswith("["))
    assert counted == ["[1/2] unit a", "[2/2] unit b"] or counted == [
        "[1/2] unit b",
        "[2/2] unit a",
    ], counted


def test_every_runner_forwards_its_own_unit_ceiling() -> None:
    """No runner may pin its own ceiling, which is the one way to be silently serial.

    `run_units` takes `unit_concurrency` by keyword and typechecks, so a call that omits it does
    not build -- but a call passing a literal builds fine and quietly ignores the flag. Nine CLIs
    make this call and only one of them is reachable from a test with a fake client, so this
    reads the call sites instead. The count is asserted too: a guard that finds nothing to check
    passes for the wrong reason.
    """
    package = Path(cli_common.__file__).parent
    checked = 0
    for module in sorted(package.glob("*_cli.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            called = getattr(node, "func", None)
            if not isinstance(node, ast.Call) or getattr(called, "id", None) != "run_units":
                continue
            checked += 1
            passed = {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            assert passed.get("unit_concurrency") == "arguments.unit_concurrency", (
                f"{module.name} passes unit_concurrency={passed.get('unit_concurrency')!r} "
                "instead of the run's own ceiling"
            )
    assert checked == 9, f"expected nine run_units call sites, read {checked}"


async def test_a_failing_unit_takes_its_siblings_down_before_it_returns() -> None:
    """Nothing may still be in flight once `run_units` raises.

    The caller's next move is `connected_memory`'s `finally`, which closes the client. A sibling
    still running past that reaches a closed client from a task nobody awaits -- and a sibling
    that had already submitted observations leaves the Worker processing for a run that will
    never write predictions. Plain `gather` does not cancel siblings, so this is the check that
    the cancel-and-wait is really there.
    """
    started: list[int] = []
    finished: list[int] = []
    cancelled: list[int] = []

    async def run(unit: int) -> int:
        started.append(unit)
        if unit == 0:
            raise RuntimeError("unit 0 died")
        try:
            for _turn in range(50):
                await asyncio.sleep(0)
            finished.append(unit)
        except asyncio.CancelledError:
            cancelled.append(unit)
            raise
        return unit

    with pytest.raises(RuntimeError, match="unit 0 died"):
        await run_units(
            tuple(range(6)),
            label=lambda unit: f"unit {unit}",
            run=run,
            unit_concurrency=3,
            quiet=True,
        )

    assert not finished, f"units {finished} ran to completion after the run had already failed"
    assert sorted(cancelled) == sorted(unit for unit in started if unit != 0), (
        f"started {sorted(started)} but only {sorted(cancelled)} had unwound by the time "
        "run_units returned"
    )
    assert 5 not in started, "a unit still waiting for a permit must never start"
