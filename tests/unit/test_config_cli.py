"""Tests for the pre-flight configuration report."""

import os
from pathlib import Path

import pytest

from mindbridge.config_cli import ROLE_EXTRAS, ROLES, main


def _without_mindbridge_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from an environment that configures nothing, whatever the host has set."""
    for name in list(os.environ):
        if name.startswith("MINDBRIDGE_"):
            monkeypatch.delenv(name, raising=False)


def _complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure one API deployment fully, the way the shipped files intend."""
    config = tmp_path / "mindbridge.toml"
    config.write_text(
        "[object_storage]\nbucket = 'mindbridge-media'\n"
        "[embedding]\ndimension = 1024\nspace_id = 's'\nspace_revision = 'r'\n"
        "[generator]\nendpoint = 'https://g/v1'\nmodel_revision = 'gr'\n"
        "[embedder]\nendpoint = 'https://e/v1'\nmodel_id = 'm'\nmodel_revision = 'er'\n",
        encoding="utf-8",
    )
    for name, value in {
        "MINDBRIDGE_CONFIG_FILE": str(config),
        "MINDBRIDGE_DATABASE_URL": "postgresql://u:p@h/d",
        "MINDBRIDGE_TASK_BROKER_URL": "redis://h:6379/0",
        "MINDBRIDGE_GENERATOR_API_KEY": "sk-generator",
        "MINDBRIDGE_EMBEDDER_API_KEY": "sk-embedder",
        "MINDBRIDGE_TENANT_API_KEYS_JSON": '{"tenant_01":["' + "a" * 48 + '"]}',
    }.items():
        monkeypatch.setenv(name, value)


def test_every_documented_role_is_addressable() -> None:
    assert set(ROLES) == {"api", "mcp", "worker", "consolidate", "lifecycle", "edge-sync"}


def test_a_complete_configuration_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_mindbridge_environment(monkeypatch)
    _complete(tmp_path, monkeypatch)

    assert main(["--role", "api"], prog="mindbridge config check") == 0
    assert "ready" in capsys.readouterr().out


def test_all_missing_settings_are_reported_in_one_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)  # away from the repository's own mindbridge.toml
    _without_mindbridge_environment(monkeypatch)

    assert main(["--role", "api"], prog="mindbridge config check") == 1

    reported = capsys.readouterr().out
    # One restart per missing value is the cost this command exists to remove.
    for expected in (
        "MINDBRIDGE_DATABASE_URL",
        "MINDBRIDGE_TASK_BROKER_URL",
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET",
        "MINDBRIDGE_GENERATOR_API_KEY",
        "MINDBRIDGE_TENANT_API_KEYS_JSON",
    ):
        assert expected in reported


def test_no_configuration_value_is_ever_printed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_mindbridge_environment(monkeypatch)
    monkeypatch.setenv("MINDBRIDGE_DATABASE_URL", "postgresql://user:s3cr3t-do-not-print@h/d")
    monkeypatch.setenv("MINDBRIDGE_GENERATOR_API_KEY", "sk-do-not-print")

    main(["--role", "api"], prog="mindbridge config check")

    printed = capsys.readouterr().out
    assert "MINDBRIDGE_GENERATOR_API_KEY" in printed, "the variable itself must be reported"
    assert "s3cr3t-do-not-print" not in printed
    assert "sk-do-not-print" not in printed


def test_an_unloaded_env_file_is_named_as_a_likely_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_mindbridge_environment(monkeypatch)
    (tmp_path / ".env").write_text(
        "MINDBRIDGE_DATABASE_URL=postgresql://u:p@h/d\n", encoding="utf-8"
    )

    main(["--role", "api"], prog="mindbridge config check")

    assert "--env-file" in capsys.readouterr().out


def test_the_lifecycle_sweep_needs_a_database_not_only_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The sweep reads MINDBRIDGE_DATABASE_URL separately from the media access it builds only
    # for --reclaim-orphan-clips. Probing the storage alone would call it ready with no database.
    monkeypatch.chdir(tmp_path)
    _without_mindbridge_environment(monkeypatch)
    monkeypatch.setenv("MINDBRIDGE_OBJECT_STORAGE_BUCKET", "mindbridge-media")

    assert main(["--role", "lifecycle"], prog="mindbridge config check") == 1
    assert "MINDBRIDGE_DATABASE_URL" in capsys.readouterr().out


def test_a_source_is_named_for_every_resolved_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_mindbridge_environment(monkeypatch)
    _complete(tmp_path, monkeypatch)

    main(["--role", "api"], prog="mindbridge config check")

    reported = capsys.readouterr().out
    assert "MINDBRIDGE_DATABASE_URL  (from environment)" in reported
    assert "MINDBRIDGE_OBJECT_STORAGE_BUCKET  (from mindbridge.toml)" in reported


def test_a_role_whose_extra_is_absent_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # `mindbridge.cli` names an extra only for commands that declare one, and this command
    # deliberately declares none so it runs on a bare install. So it has to say so itself.
    import builtins

    from mindbridge.config_cli import _settings_probe

    real_import = builtins.__import__

    def _without_fastapi(name: str, *rest: object, **kwargs: object) -> object:
        if name.startswith("mindbridge.api"):
            raise ImportError("No module named 'fastapi'", name="fastapi")
        return real_import(name, *rest, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _without_fastapi)

    with pytest.raises(ImportError, match=r"needs fastapi; install it with .*--extra server"):
        _settings_probe("api")


def test_every_role_needing_an_extra_declares_which() -> None:
    assert set(ROLE_EXTRAS) == set(ROLES) - {"edge-sync"}
