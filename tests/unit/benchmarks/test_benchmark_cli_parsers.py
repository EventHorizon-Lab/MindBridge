"""Every benchmark CLI still accepts the shared invocation after inheriting it.

`_parse_arguments` is the one part of a runner no other test reaches: it runs before any API
call, so a broken flag surfaces only when a long benchmark is launched. These parse a minimal
valid command line per CLI and assert the shared fields land on the shared dataclass.
"""

from importlib import import_module
from pathlib import Path

import pytest

from mindbridge.benchmarks.artifacts import CommonArguments

SHARED = [
    "--dataset",
    "data.json",
    "--output",
    "out.json",
    "--api-base-url",
    "https://memory.example.test",
    "--code-revision",
    "commit",
    "--deployment-config",
    "deployment.json",
    "--run-id",
    "run_01",
]
BENCHMARK_ARGUMENTS = {
    "egolife_cli": [
        "--prepared-media",
        "p.json",
        "--dataset-revision",
        "d",
        "--evaluator-revision",
        "e",
    ],
    "egomem_cli": [
        "--prepared-media",
        "p.json",
        "--dataset-revision",
        "d",
        "--evaluator-revision",
        "e",
    ],
    "egotempo_cli": [
        "--prepared-media",
        "p.json",
        "--source-revision",
        "s",
        "--evaluator-revision",
        "e",
    ],
    "locomo_cli": ["--source-revision", "s"],
    "m3_cli": [
        "--prepared-media",
        "p.json",
        "--source-revision",
        "s",
        "--media-revision",
        "m",
        "--subset",
        "robot",
    ],
    "memlens_cli": [
        "--dataset-revision",
        "d",
        "--evaluator-revision",
        "e",
        "--context-window",
        "32k",
    ],
    "mm_lifelong_cli": [
        "--prepared-media",
        "p.json",
        "--source-revision",
        "s",
        "--split",
        "day_test",
    ],
    "supermemory_cli": [
        "--prepared-media",
        "p.json",
        "--subject",
        "1",
        "--dataset-revision",
        "d",
        "--source-revision",
        "s",
    ],
    "video_mme_cli": [
        "--prepared-media",
        "p.json",
        "--dataset-revision",
        "d",
        "--evaluator-revision",
        "e",
        "--transcript-source",
        "none",
    ],
}


@pytest.mark.parametrize("module_name", sorted(BENCHMARK_ARGUMENTS))
def test_cli_parses_the_shared_invocation(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module(f"mindbridge.benchmarks.{module_name}")
    monkeypatch.setattr("sys.argv", [module_name, *SHARED, *BENCHMARK_ARGUMENTS[module_name]])

    arguments = module._parse_arguments()

    assert isinstance(arguments, CommonArguments)
    assert arguments.dataset_path == Path("data.json")
    assert arguments.output_path == Path("out.json")
    assert arguments.deployment_config_path == Path("deployment.json")
    assert arguments.api_base_url == "https://memory.example.test"
    assert arguments.code_revision == "commit"
    assert arguments.run_id == "run_01"
    assert arguments.recall_limit == 20
    assert arguments.request_concurrency == 4
    assert arguments.request_timeout_seconds == 1_800.0
    assert arguments.overwrite is False
    assert arguments.tenant_prefix.startswith("benchmark_")


@pytest.mark.parametrize("module_name", sorted(BENCHMARK_ARGUMENTS))
def test_cli_still_honours_overridden_shared_flags(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module(f"mindbridge.benchmarks.{module_name}")
    monkeypatch.setattr(
        "sys.argv",
        [
            module_name,
            *SHARED,
            *BENCHMARK_ARGUMENTS[module_name],
            "--tenant-prefix",
            "benchmark_custom",
            "--recall-limit",
            "5",
            "--request-concurrency",
            "2",
            "--request-timeout-seconds",
            "60",
            "--overwrite",
        ],
    )

    arguments = module._parse_arguments()

    assert arguments.tenant_prefix == "benchmark_custom"
    assert arguments.recall_limit == 5
    assert arguments.request_concurrency == 2
    assert arguments.request_timeout_seconds == 60.0
    assert arguments.overwrite is True


@pytest.mark.parametrize("module_name", sorted(BENCHMARK_ARGUMENTS))
def test_cli_rejects_an_invocation_missing_a_shared_flag(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module(f"mindbridge.benchmarks.{module_name}")
    monkeypatch.setattr("sys.argv", [module_name, *BENCHMARK_ARGUMENTS[module_name]])

    with pytest.raises(SystemExit):
        module._parse_arguments()
