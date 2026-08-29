"""The API schema must build for every route.

`app.openapi()` forces FastAPI to resolve every `response_model`, which is the
only cheap way to exercise all the Pydantic models at once. This is what would
have caught the v1 -> v2 migration issues:

* `class Config: orm_mode` silently doing nothing under v2;
* fields declared `x: str = None`, which v1 auto-promoted to Optional and v2 does
  not, so a nullable database column fails validation;
* the protected `model_` namespace colliding with `model_architecture` and
  friends.

Importing the routers alone does not catch any of those -- the models are only
built when the schema is generated or a request is served.
"""

import pytest
from fastapi import FastAPI


@pytest.fixture(scope="module")
def api_schema():
    from anylearning.routers import (
        dataset,
        labeling,
        model,
        project,
        training,
        window,
    )

    app = FastAPI()
    for module in (dataset, labeling, model, project, training, window):
        app.include_router(module.router)
    return app.openapi()


def test_every_route_has_a_schema(api_schema):
    assert api_schema["paths"], "no routes registered"
    # Guard against a router silently failing to import and shrinking the API.
    assert len(api_schema["paths"]) >= 40


def test_response_models_resolve(api_schema):
    schemas = api_schema.get("components", {}).get("schemas", {})
    for name in (
        "ProjectResponse",
        "ModelResponse",
        "DataItemResponse",
        "TrainingParams",
    ):
        assert name in schemas, f"{name} missing from the OpenAPI components"


def test_nullable_model_fields_are_optional(api_schema):
    """`model_architecture` and friends must accept null.

    Pydantic v1 turned `x: str = None` into an Optional field; v2 does not. These
    columns are nullable in the database, so losing that promotion would raise a
    ValidationError when serialising a model that has not been trained yet.
    """
    model_response = api_schema["components"]["schemas"]["ModelResponse"]
    for field in ("model_architecture", "model_size", "model_variant"):
        spec = model_response["properties"][field]
        allows_null = "anyOf" in spec and any(
            option.get("type") == "null" for option in spec["anyOf"]
        )
        assert allows_null, f"{field} is not nullable in the generated schema"


def test_protected_namespace_fields_survive(api_schema):
    """Pydantic v2 reserves `model_`; these names are part of the API contract."""
    params = api_schema["components"]["schemas"]["TrainingParams"]["properties"]
    for field in ("model_architecture", "model_size", "model_variant"):
        assert field in params
