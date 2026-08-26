"""A judge stub for the runner tests whose benchmark scores its own free-text answers.

Seven runners now call a judge model from inside the run, the way lmms-eval's MM-Vet task does,
so a test that drives one of them end to end has to supply one. Importing `stub_judge` into a
test module makes it autouse *for that module*, which keeps the substitution visible in the file
that needs it rather than hidden in a directory-wide fixture.

Not a `conftest.py` on purpose: mypy maps modules by name and the tree already has one conftest,
so a second is a hard error under `strict`. Nothing is lost by being explicit -- `build_judge`
refuses to connect without `MINDBRIDGE_BENCH_JUDGE_ENDPOINT`, so a judged runner reached with no
stub fails loudly rather than scoring every answer 0.0.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mindbridge.benchmarks import cli_common, scoring
from mindbridge.core import ModelReference
from mindbridge.models import GenerateRequest, GenerateResult

_JUDGE = ModelReference(model_id="stub-judge")


class StubJudge:
    """A judge that reads every answer as fully correct, and records what it was asked."""

    def __init__(self, reply: str = "1.0") -> None:
        self.reply = reply
        self.requests: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.requests.append(request)
        return GenerateResult(text=self.reply, model_reference=_JUDGE)


@pytest.fixture(autouse=True)
def stub_judge(monkeypatch: pytest.MonkeyPatch) -> Iterator[StubJudge]:
    """Replace the judge a judged runner would connect to, and name it in the manifest."""
    judge = StubJudge()
    monkeypatch.setenv(scoring.JUDGE_ENDPOINT_VARIABLE, "https://judge.example.test/v1")
    monkeypatch.setenv(scoring.JUDGE_MODEL_VARIABLE, "stub-judge")
    monkeypatch.setattr(cli_common, "build_judge", lambda **_: judge)
    yield judge
