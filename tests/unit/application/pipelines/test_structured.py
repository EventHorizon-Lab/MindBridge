"""Strict schema derivation for provider-constrained structured generation."""

import json
from typing import Annotated, Literal, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, StringConstraints

from mindbridge.application.pipelines import structured
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


class _KeywordNamed(BaseModel):
    """A model whose fields are spelled like the keywords the rewrite filters.

    Not contrived: `title`, `format` and `default` are ordinary names for a summary, a
    rendering hint and a fallback, and `then` is one for a rule.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    body: str
    default: str
    format: str
    then: str
    title: str


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


@pytest.mark.parametrize(
    "keyword",
    ["allOf", "not", "oneOf", "if", "then", "else", "dependentRequired", "dependentSchemas"],
)
def test_every_keyword_the_provider_contract_excludes_is_named(keyword: str) -> None:
    """A guard that names only some of them implies a coverage it does not have."""
    assert keyword in structured._UNSUPPORTED_KEYWORDS


def test_a_root_that_is_not_an_object_fails_derivation() -> None:
    """A strict root must be an object, and a rejected request would degrade to JSON mode.

    The provider fallback exists for an endpoint that cannot compile schemas at all, so
    letting an inexpressible root reach it would silently drop the constraint instead of
    naming the model that cannot carry one.
    """
    with pytest.raises(ValueError, match="object at its root"):
        output_schema("array_root", RootModel[list[int]])


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


def test_fields_named_like_schema_keywords_reach_the_provider() -> None:
    """Keys under `properties` are names the model chose, not keywords to filter.

    Dropping one is silent and total rather than partial: the field leaves `required` too,
    and `additionalProperties: false` then forbids the model from volunteering it, so the
    validator rejects every single response and the retry pays for a second one.
    """
    schema = json.loads(output_schema("keyword_named", _KeywordNamed).json_schema)

    assert sorted(cast(dict[str, object], schema["properties"])) == sorted(
        _KeywordNamed.model_fields
    )
    assert schema["required"] == sorted(_KeywordNamed.model_fields)


def test_a_field_named_like_an_unsupported_keyword_is_not_mistaken_for_one() -> None:
    """The fail-loud guard reads schema keywords, and a property name is not one.

    Mistaking a field called `then` for a conditional subschema fails derivation at import
    and takes the whole package down with it.
    """
    schema = json.loads(output_schema("keyword_named", _KeywordNamed).json_schema)

    assert "then" in cast(dict[str, object], schema["properties"])


def test_the_definitions_namespace_keeps_a_model_named_like_a_keyword() -> None:
    """`$defs` keys are model names, so the same filter must not reach them either."""

    class Format(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        label: str

    class Holder(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        nested: Format

    schema = json.loads(output_schema("holder", Holder).json_schema)

    assert sorted(cast(dict[str, object], schema["$defs"])) == ["Format"]
