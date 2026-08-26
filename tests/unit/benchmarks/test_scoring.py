"""What the ported lmms-eval scoring contract does, including the parts that cost something.

The 0.0 floor is the one worth reading twice: a judge that cannot be reached or cannot be parsed
scores the answer wrong, which is upstream's behaviour and was adopted deliberately. The tests
below pin it so nobody has to rediscover it from a suspiciously low number, and pin the tally
beside it that makes such a run tellable apart from one that genuinely answered badly.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from stub_judge import stub_judge  # noqa: F401 - autouse within this module

from mindbridge.benchmarks import cli_common
from mindbridge.benchmarks.cli import RUNNERS
from mindbridge.benchmarks.cli_common import CoreArguments, scoring_snapshot
from mindbridge.benchmarks.scoring import (
    BYPASS_METRIC,
    BYPASS_VALUE,
    DEFAULT_JUDGE_MODEL,
    JUDGE_API_KEY_PLACEHOLDER,
    JUDGE_API_KEY_VARIABLE,
    JUDGE_ENDPOINT_VARIABLE,
    JUDGE_METRIC,
    JUDGE_MODEL_VARIABLE,
    JUDGE_PATIENCE,
    JUDGE_TIMEOUT_SECONDS,
    SCORING,
    JudgedAnswer,
    build_judge,
    bypass_metrics,
    configured_judge_model,
    judge_answers,
    parse_score,
    require_scoring_is_possible,
)
from mindbridge.benchmarks.task_catalog import TASKS
from mindbridge.core import ModelReference
from mindbridge.models import GenerateRequest, GenerateResult

_JUDGE = ModelReference(model_id="stub-judge")


class _ScriptedJudge:
    """A judge that replies from a script, and raises where the script says to."""

    def __init__(self, replies: Sequence[str | Exception]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.prompts.append(_text(request))
        reply = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return GenerateResult(text=reply, model_reference=_JUDGE)


def _text(request: GenerateRequest) -> str:
    return "".join(getattr(part, "text", "") for part in request.input.parts)


def _answer(prediction: str = "a blue mug") -> JudgedAnswer:
    return JudgedAnswer("What was on the table?", "a blue mug", prediction)


def test_every_benchmark_the_catalog_can_name_declares_who_scores_it() -> None:
    """A benchmark absent from the table is one no run of it could report anything about.

    Read off the catalog rather than a list restated here, so a benchmark added to `--tasks`
    without a scoring declaration fails this instead of failing mid-sweep on a KeyError.
    """
    named = {task.benchmark for task in TASKS.values()}

    assert named <= set(SCORING), sorted(named - set(SCORING))
    assert set(SCORING) <= set(RUNNERS), sorted(set(SCORING) - set(RUNNERS))


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("0.0", 0.0),
        ("1.0", 1.0),
        ("0.7", 0.7),
        ("Correctness score: 0.7", 0.7),
        ("7", None),
        ("1.5", None),
        ("-0.5", None),
        ("perfect", None),
        ("", None),
    ],
)
def test_a_judge_reply_is_read_as_a_score_only_when_it_is_one(
    reply: str, expected: float | None
) -> None:
    """`7` is not `0.7`, and guessing which was meant would invent the number being attributed."""
    assert parse_score(reply) == expected


def test_a_judge_that_answers_on_the_first_try_is_asked_once() -> None:
    judge = _ScriptedJudge(["0.7"])

    outcome = judge_answers((_answer(),), judge=judge, concurrency=1)

    assert outcome.metrics[JUDGE_METRIC] == 0.7
    assert outcome.failure_count == 0
    assert len(judge.prompts) == 1


def test_an_unreadable_reply_is_retried_with_a_stricter_instruction_each_time() -> None:
    """The stand-in for MM-Vet's `temperature += 0.5`, which this protocol has nothing to raise.

    Retrying a deterministic request unchanged would be three identical unparseable answers, so
    the prompt is what escalates. Each attempt must therefore differ from the one before it.
    """
    judge = _ScriptedJudge(["perfect", "very good", "0.4"])

    outcome = judge_answers((_answer(),), judge=judge, concurrency=1)

    assert outcome.metrics[JUDGE_METRIC] == 0.4
    assert outcome.failure_count == 0
    assert len(judge.prompts) == JUDGE_PATIENCE
    assert len(set(judge.prompts)) == JUDGE_PATIENCE, "an identical retry would change nothing"


def test_a_judge_that_never_returns_a_score_floors_the_answer_to_zero() -> None:
    """lmms-eval's `score = 0.0` after exhausting patience, adopted verbatim.

    The consequence is that a wrong answer and an unreadable judge produce the same number, which
    is why the failure count exists beside it.
    """
    judge = _ScriptedJudge(["nope"])

    outcome = judge_answers((_answer(),), judge=judge, concurrency=1)

    assert outcome.metrics[JUDGE_METRIC] == 0.0
    assert outcome.failure_count == 1
    assert len(judge.prompts) == JUDGE_PATIENCE


def test_an_unreachable_judge_also_floors_to_zero_rather_than_failing_the_run() -> None:
    """Upstream's retry loop catches everything, so an outage reads as a low score, not an error.

    The attempt count is the load-bearing half: a raise that escaped the retry loop would also
    end at 0.0, by way of the gather below it, having asked once instead of three times.
    """
    judge = _ScriptedJudge([RuntimeError("connection refused")])

    outcome = judge_answers((_answer(),), judge=judge, concurrency=1)

    assert outcome.metrics[JUDGE_METRIC] == 0.0
    assert outcome.failure_count == 1
    assert len(judge.prompts) == JUDGE_PATIENCE, "an exception must be retried, not given up on"


def test_one_answer_whose_judge_dies_does_not_discard_the_answers_beside_it() -> None:
    """Without `return_exceptions` the first raise cancels every sibling still in flight.

    That shape has cost this repository whole paid-for passes before, so it is pinned rather than
    left to the next person to rediscover.
    """

    class _OneBadApple:
        async def generate(self, request: GenerateRequest) -> GenerateResult:
            if "poison" in _text(request):
                raise RuntimeError("judge exploded")
            return GenerateResult(text="1.0", model_reference=_JUDGE)

    answers = (_answer("fine"), _answer("poison"), _answer("also fine"))

    outcome = judge_answers(answers, judge=_OneBadApple(), concurrency=3)

    assert outcome.failure_count == 1
    assert outcome.metrics[JUDGE_METRIC] == pytest.approx(2 / 3)


def test_an_interrupt_mid_pass_ends_the_run_rather_than_scoring_the_answer_zero() -> None:
    """`Exception` is a judge failing; `BaseException` is the operator stopping the run.

    Flooring the second would turn Ctrl-C into a published zero, so it is deliberately outside
    what the retry loop catches -- and `gather` re-raises it regardless of `return_exceptions`.
    """

    class _Interrupted:
        async def generate(self, request: GenerateRequest) -> GenerateResult:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        judge_answers((_answer(),), judge=_Interrupted(), concurrency=1)


def test_the_bypass_metric_is_the_sentinel_upstream_reports() -> None:
    """`bypass_agg` returns `999`, in the same column as an accuracy. It is not a score."""
    assert bypass_metrics() == {BYPASS_METRIC: BYPASS_VALUE}
    assert BYPASS_VALUE == 999.0


def test_predict_only_contacts_no_judge_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """`override_metric("bypass")` replaces every declared metric before anything is computed.

    Proven by making the judge unbuildable rather than by inspecting the result: a snapshot that
    happened to say `bypass` while still having paid for a judge pass would pass a weaker test.
    """

    def refuse(**_: object) -> None:
        raise AssertionError("--predict-only must not reach a judge")

    monkeypatch.setattr(cli_common, "build_judge", refuse)

    snapshot = scoring_snapshot("locomo-refined", _arguments(predict_only=True))

    assert snapshot.mode == "bypass"
    assert snapshot.metrics == {BYPASS_METRIC: BYPASS_VALUE}
    assert snapshot.judge_model is None


def test_a_runner_scored_benchmark_carries_the_direction_each_metric_points() -> None:
    """`higher_is_better` travels with the number, as lmms-eval's `metric_list` declares it."""
    snapshot = scoring_snapshot(
        "video-mme", _arguments(), metrics={"accuracy": 0.61, "strict_accuracy": 0.6}
    )

    assert snapshot.mode == "runner"
    assert snapshot.metrics == {"accuracy": 0.61, "strict_accuracy": 0.6}
    assert snapshot.higher_is_better["accuracy"] is True
    assert snapshot.judge_model is None


def test_a_held_out_benchmark_reports_no_number_and_says_so_by_its_mode() -> None:
    """MMBench's test split upstream: `metric: submission`, an upload, and no score."""
    snapshot = scoring_snapshot("egomem", _arguments())

    assert snapshot.mode == "external"
    assert snapshot.metrics == {}


def test_a_judged_run_records_which_judge_produced_its_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs under different judges are not comparable, so the manifest has to name the judge."""
    monkeypatch.setenv(JUDGE_MODEL_VARIABLE, "some-other-judge")

    snapshot = scoring_snapshot("locomo-refined", _arguments(), answers=(_answer(),), metrics=None)

    assert snapshot.mode == "judge"
    assert snapshot.judge_model == "some-other-judge"
    assert snapshot.metrics[JUDGE_METRIC] == 1.0
    assert snapshot.judge_failure_count == 0


def test_a_judged_run_records_how_much_of_its_score_was_a_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without this count a run whose judge was down looks exactly like a run that answered badly.

    Through `scoring_snapshot` rather than `judge_answers`, because the tally has to survive the
    trip into the manifest and that is a separate assignment from computing it.
    """

    class _Mute:
        async def generate(self, request: GenerateRequest) -> GenerateResult:
            return GenerateResult(text="no idea", model_reference=_JUDGE)

    monkeypatch.setattr(cli_common, "build_judge", lambda **_: _Mute())

    snapshot = scoring_snapshot(
        "locomo-refined", _arguments(), answers=(_answer(), _answer("wrong"))
    )

    assert snapshot.metrics[JUDGE_METRIC] == 0.0
    assert snapshot.judge_failure_count == 2


def test_an_unconfigured_judge_is_refused_before_a_single_answer_is_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No judge at all is a configuration error, not a judge that failed.

    The 0.0 floor is for a judge that was asked and could not answer. Applying it to a run where
    nobody configured one would report a whole benchmark as zero for a missing environment
    variable, so this is the one place the port refuses instead of flooring.
    """
    monkeypatch.delenv(JUDGE_ENDPOINT_VARIABLE, raising=False)

    with pytest.raises(ValueError, match=JUDGE_ENDPOINT_VARIABLE):
        build_judge(request_timeout_seconds=60.0)


def test_a_judged_benchmark_with_no_judge_is_refused_before_the_run_not_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build_judge`'s refusal is correct and arrives far too late to be useful.

    It is first reached from `scoring_snapshot`, which every runner calls after its own
    `asyncio.run` has returned and before `write_run_artifacts` -- so an unset endpoint threw
    away every prediction a finished run had paid for, and a sweep repeated that for each of the
    seventeen judged tasks. `require_scoring_is_possible` is the same question asked where the
    answer is still free.
    """
    monkeypatch.delenv(JUDGE_ENDPOINT_VARIABLE, raising=False)

    with pytest.raises(ValueError, match=JUDGE_ENDPOINT_VARIABLE):
        require_scoring_is_possible("locomo-refined", predict_only=False)

    # Nothing to refuse: no judge is needed either way.
    require_scoring_is_possible("locomo-refined", predict_only=True)
    require_scoring_is_possible("video-mme", predict_only=False)
    require_scoring_is_possible("egomem", predict_only=False)


def test_the_judge_gets_its_own_deadline_and_no_transport_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 16-token grading call must not inherit a benchmark request's 1800-second deadline.

    Inherited, and multiplied by the OpenAI client's own default of two retries, `JUDGE_PATIENCE`
    became a nine-attempt ladder at half an hour each -- so one stalled judge could hold a pass
    for hours per answer while `_score_one` patiently tried again.
    """
    captured: dict[str, object] = {}
    monkeypatch.setenv(JUDGE_ENDPOINT_VARIABLE, "https://judge.example/v1")
    monkeypatch.delenv(JUDGE_API_KEY_VARIABLE, raising=False)
    monkeypatch.setattr(
        "mindbridge.models.openai.create_generator",
        lambda config: captured.update(config) or _ScriptedJudge(["1.0"]),
    )

    build_judge(request_timeout_seconds=1_800.0)

    assert captured["request_timeout_seconds"] == JUDGE_TIMEOUT_SECONDS
    assert captured["max_retries"] == 0
    # A local judge wants no key, and the generator contract requires a non-empty one.
    assert captured["api_key"] == JUDGE_API_KEY_PLACEHOLDER


def test_the_default_judge_is_the_one_mm_vet_uses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kept so a number produced here is comparable to one lmms-eval would produce."""
    monkeypatch.delenv(JUDGE_MODEL_VARIABLE, raising=False)

    assert configured_judge_model() == DEFAULT_JUDGE_MODEL == "gpt-4o-2024-11-20"


def _arguments(*, predict_only: bool = False) -> CoreArguments:
    from pathlib import Path

    return CoreArguments(
        dataset_path=Path("release.json"),
        output_path=Path("predictions.jsonl"),
        api_base_url="https://memory.example.test",
        deployment_config_path=Path("deployment.json"),
        run_id="run_01",
        tenant_prefix="benchmark_test",
        recall_limit=20,
        request_concurrency=4,
        request_timeout_seconds=1_800.0,
        limit=None,
        overwrite=False,
        predict_only=predict_only,
        quiet=True,
    )
