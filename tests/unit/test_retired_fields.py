"""Tolerance for the field names migration 0021 retired.

Removing a field from a model whose config is `extra="forbid"` is as breaking as adding a
required one, and three readers meet values written before that migration: an operator's
`*_CONFIG_JSON` object, an edge device's spooled `ObserveRequest` payloads, and -- during a
rolling upgrade, where the server goes first -- `/v1/observe` bodies from a device still on
the previous release. These tests pin the tolerance and, just as importantly, pin its limit.
"""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from mindbridge.configuration import PluginConfigModel, PluginText
from mindbridge.contracts import IdentityObservationInput, ObserveRequest
from mindbridge.core import RETIRED_FIELD_NAMES
from mindbridge.edge.identity_schema import _TABLES_RETIRED_WITH_THEIR_REVISION_COLUMN


def _identity_span(**extra: object) -> dict[str, object]:
    return {
        "identity_id": "person-1",
        "kind": "voice",
        "start_ms": 0,
        "end_ms": 1_000,
        "confidence": 0.9,
        "model_id": "iic/speech_campplus_sv_zh-cn_16k-common",
        **extra,
    }


def test_an_identity_span_from_the_previous_release_still_validates() -> None:
    span = IdentityObservationInput.model_validate(_identity_span(model_revision="v2.0.2"))

    assert span.identity_id == "person-1"
    assert not hasattr(span, "model_revision")


def test_an_observe_request_from_the_previous_release_still_validates() -> None:
    """The nested span is what carried the retired name, so the whole body has to survive it."""
    started = datetime(2026, 8, 25, tzinfo=timezone.utc)
    request = ObserveRequest.model_validate(
        {
            "tenant_id": "tenant_01",
            "device_id": "camera_01",
            "boot_id": "boot_01",
            "sequence": 7,
            "sensor": "microphone",
            "media_objects": [
                {
                    "media_object_id": "media_01",
                    "kind": "audio",
                    "uri": "s3://memory/tenants/tenant_01/media_01.wav",
                    "sha256": "00" * 32,
                    "size_bytes": 5,
                    "created_at": started,
                    "duration_ms": 1_000,
                }
            ],
            "occurred_at": started,
            "ended_at": started + timedelta(seconds=1),
            "observed_at": started + timedelta(seconds=1),
            "identity_observations": [_identity_span(model_revision="v2.0.2")],
        }
    )

    assert len(request.identity_observations) == 1


def test_a_misspelled_field_is_still_rejected() -> None:
    """The tolerance is a closed list, not a hole in `extra="forbid"`.

    If this passed, the mechanism would have stopped catching the typo it exists to catch, and
    a value an operator meant to set would silently revert to its default.
    """
    with pytest.raises(ValidationError, match="extra_forbidden"):
        IdentityObservationInput.model_validate(_identity_span(model_revsion="v2.0.2"))


def test_a_plugin_config_ignores_a_retired_name_it_does_not_declare() -> None:
    class _Config(PluginConfigModel):
        model_id: PluginText = "m"

    assert _Config.model_validate({"model_id": "m", "space_revision": "gone"}).model_id == "m"


def test_a_plugin_config_keeps_a_retired_name_it_does_declare() -> None:
    """A reader that still declares one of these names is the reason the rule is per-model.

    The local Jina embedder's `model_revision` is a loader argument that selects which weights
    and which remote code run. Dropping it as retired would replace an operator's pin with a
    default, so the tolerance has to defer to what the model declares.
    """

    class _Config(PluginConfigModel):
        model_revision: PluginText = "default-pin"

    assert _Config.model_validate({"model_revision": "operator-pin"}).model_revision == (
        "operator-pin"
    )


def test_every_tolerated_name_was_actually_retired_somewhere() -> None:
    """Keeps the list from growing into a general-purpose escape from `extra="forbid"`.

    Read off the two places a retirement is recorded rather than restated as literals here: a
    copy of the constant cannot tell whether a name in it was ever a field, which is exactly
    the way this list would quietly become an amnesty. Migration 0021 names the two PostgreSQL
    columns it drops; `identity_schema` names the two the edge SQLite store dropped with their
    tables. Their union is the closed set, and a name added to `RETIRED_FIELD_NAMES` without a
    retirement behind it fails here.
    """
    migration = (Path(__file__).parents[2] / "migrations/0021_drop_model_revisions.sql").read_text(
        encoding="utf-8"
    )
    dropped_columns = set(re.findall(r"DROP COLUMN (\w+)", migration))
    retired_on_edge = {column for _table, column in _TABLES_RETIRED_WITH_THEIR_REVISION_COLUMN}

    assert frozenset(dropped_columns | retired_on_edge) == RETIRED_FIELD_NAMES
