"""Strict schema derivation for provider-constrained structured generation."""

import json
from typing import Annotated, Literal, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from mindbridge.application.pipelines.structured import output_schema, unwrap_json_code_fence


class _Nested(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: Annotated[str, StringConstraints(min_length=1, max_length=64)]


class _Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    required_field: str
    bounded: Annotated[float, Field(ge=0.0, le=1.0)]
    optional_value: int | None
    choice: Literal["a", "b"] = "a"
    listed: Annotated[tuple[_Nested, ...], Field(max_length=4)] = ()


def _schema() -> dict[str, object]:
    derived = output_schema("sample", _Sample)
    assert derived.name == "sample"
    return cast(dict[str, object], json.loads(derived.json_schema))


def test_every_property_becomes_required_including_defaulted_ones() -> None:
    """Strict decoding cannot express an optional key, so a default must still be emitted."""
    schema = _schema()

    assert schema["required"] == [
        "bounded",
        "choice",
        "listed",
        "optional_value",
        "required_field",
    ]


def test_nested_objects_are_closed_and_required_too() -> None:
    """A nested object left open is where an unexpected key gets through the constraint."""
    nested = json.loads(output_schema("sample", _Sample).json_schema)["$defs"]["_Nested"]

    assert nested["additionalProperties"] is False
    assert nested["required"] == ["label"]


def test_a_model_that_permits_extra_keys_is_still_closed_by_derivation() -> None:
    """Derivation, not the output model's config, is what guarantees a closed constraint.

    `extra="forbid"` happens to emit the same keyword today, so a model written without it
    would otherwise ship an open object and let the provider invent fields the parser drops.
    """

    class _Permissive(BaseModel):
        value: str

    schema = json.loads(output_schema("permissive", _Permissive).json_schema)

    assert schema["additionalProperties"] is False


def test_range_and_length_bounds_are_dropped_from_the_constraint() -> None:
    """The parsing model still enforces them; sending them only risks a rejected schema."""
    rendered = output_schema("sample", _Sample).json_schema

    assert not any(
        keyword in rendered
        for keyword in ("minimum", "maximum", "maxLength", "minLength", "maxItems", "default")
    )


def test_shapes_the_constraint_can_express_survive() -> None:
    schema = _schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)

    assert schema["additionalProperties"] is False
    assert properties["optional_value"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    # The default survives as a choice the model may pick, never as an absent key.
    assert properties["choice"] == {"enum": ["a", "b"], "type": "string"}
    assert properties["listed"] == {"items": {"$ref": "#/$defs/_Nested"}, "type": "array"}


def test_a_model_the_constraint_cannot_express_fails_derivation() -> None:
    """This package's own tests are where that has to surface, not a first observation."""

    class _Unexpressible(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        pair: tuple[int, str]

    with pytest.raises(ValueError, match="strict decoding rejects"):
        output_schema("unexpressible", _Unexpressible)


def test_derived_schema_admits_what_the_model_parses() -> None:
    """Schema and parser are derived from one declaration, so a payload valid to one is
    valid to the other -- which is the property that keeps them from drifting apart."""
    schema = _schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)

    assert set(properties) == set(_Sample.model_fields)
    assert (
        _Sample.model_validate(
            {
                "required_field": "x",
                "bounded": 0.5,
                "optional_value": None,
                "choice": "b",
                "listed": [{"label": "n"}],
            }
        ).choice
        == "b"
    )


def test_a_provider_added_fence_is_still_unwrapped() -> None:
    """Constrained decoding removes the fence; the fallback path can still produce one."""
    assert unwrap_json_code_fence('```json\n{"a":1}\n```') == '{"a":1}'
    assert unwrap_json_code_fence('{"a":1}') == '{"a":1}'
