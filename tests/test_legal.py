"""The licence notices have to reach the binary, not just the repository.

MIT, BSD and Apache 2.0 all require the notice to travel with what is
distributed. A build that quietly drops LICENSES.md is therefore a licensing
failure rather than a cosmetic one, and it is invisible unless something looks.
"""

import pathlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from anylearning import legal
from anylearning.routers.legal import router


def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_the_notices_exist_in_this_checkout():
    path = legal.notices_path()
    assert path is not None, "LICENSES.md is missing"
    assert "Apache License" in path.read_text(encoding="utf-8")


def test_the_endpoint_serves_them_component_by_component():
    """A list, not the raw file. The markdown source rendered in a <pre> shows
    "###" and code fences as literal characters, and two megabytes of it is not
    something anyone will scroll to find one package."""
    response = client().get("/api/legal/notices")
    assert response.status_code == 200

    components = response.json()["components"]
    assert len(components) > 100, "the build ships far more than a handful"

    by_name = {component["name"]: component for component in components}
    assert "torch" in by_name
    assert by_name["torch"]["version"]
    assert "###" not in by_name["torch"]["text"]
    assert "```" not in by_name["torch"]["text"]


def test_parsing_keeps_the_licence_body_intact():
    from anylearning import legal as legal_module

    parsed = legal_module.parse_notices(
        "\n".join(
            [
                "### example 1.2.3",
                "",
                "https://example.invalid",
                "",
                "```",
                "MIT License",
                "",
                "  indented and blank lines are part of the text",
                "```",
                "",
            ]
        )
    )
    assert parsed == [
        {
            "name": "example",
            "version": "1.2.3",
            "text": "https://example.invalid\nMIT License\n\n  indented and blank lines are part of the text",
        }
    ]


def test_a_build_without_them_says_so(monkeypatch):
    """Better an explicit packaging error than a panel that looks deliberate."""
    monkeypatch.setattr(
        legal, "_CANDIDATES", (pathlib.Path("/nonexistent/LICENSES.md"),)
    )
    response = client().get("/api/legal/notices")
    assert response.status_code == 404
    assert "LICENSES.md" in response.json()["detail"]


def test_the_packaged_copy_wins_over_the_checkout(tmp_path, monkeypatch):
    """Order matters: a build that failed to include the file must not read
    the developer's checkout and report success."""
    packaged = tmp_path / "packaged.md"
    packaged.write_text("packaged copy")
    fallback = tmp_path / "fallback.md"
    fallback.write_text("checkout copy")
    monkeypatch.setattr(legal, "_CANDIDATES", (packaged, fallback))
    assert legal.read_notices() == "packaged copy"


def test_the_project_license_ships_and_is_served():
    """The license shown by the installer remains readable afterwards."""
    from anylearning import legal as legal_module

    assert legal_module.read_license() is not None

    response = client().get("/api/legal/license")
    assert response.status_code == 200
    text = response.json()["text"]
    assert "Apache License" in text
    assert "Version 2.0" in text


def test_a_build_without_the_distribution_license_says_so(monkeypatch):
    import pathlib as pathlib_module

    from anylearning import legal as legal_module

    monkeypatch.setattr(
        legal_module,
        "_LICENSE_CANDIDATES",
        (pathlib_module.Path("/nonexistent/LICENSE"),),
    )
    response = client().get("/api/legal/license")
    assert response.status_code == 404
    assert "LICENSE" in response.json()["detail"]


def test_the_model_policy_ships_and_is_served():
    """The terms point users at it, so it has to be readable by them."""
    from anylearning import legal as legal_module

    assert legal_module.read_model_policy() is not None

    response = client().get("/api/legal/model-policy")
    assert response.status_code == 200
    assert "licence" in response.json()["text"].lower()
