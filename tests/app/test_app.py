import pytest
from anylearning.app import (
    create_app,
    get_random_port,
    is_anylearning_running,
    is_port_in_use,
    resolve_frontend_file,
)
from anylearning.app_info import __appname__, __description__
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def test_app():
    app = create_app()
    return app


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as client:
        yield client


def test_create_app(test_app):
    assert test_app.title == __appname__
    assert test_app.description == __description__


def test_resolve_frontend_file_maps_extensionless_static_routes(tmp_path):
    (tmp_path / "projects").mkdir()
    route = tmp_path / "projects" / "dataset.html"
    route.write_text("dataset page")
    asset = tmp_path / "asset.js"
    asset.write_text("asset")

    assert resolve_frontend_file(str(tmp_path), "projects/dataset") == route
    assert resolve_frontend_file(str(tmp_path), "asset.js") == asset
    assert resolve_frontend_file(str(tmp_path), "../outside.txt") is None


def test_is_anylearning_endpoint(client):
    response = client.get("/api/is_anylearning")
    assert response.status_code == 200
    assert response.json() == {"is_anylearning": True}


def test_is_port_in_use():
    # Test with unused port
    assert not is_port_in_use(65535)

    # Test with used port
    with patch("socket.socket") as mock_socket:
        mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0
        assert is_port_in_use(8000)


def test_get_random_port():
    with patch("anylearning.app.is_port_in_use", return_value=False):
        port = get_random_port()
        assert 1024 <= port <= 65535


def test_is_anylearning_running():
    # Test when service is running
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_anylearning": True}
        mock_get.return_value = mock_response

        assert is_anylearning_running("localhost", 8000)

    # Test when service is not running
    with patch("requests.get", side_effect=Exception()):
        assert not is_anylearning_running("localhost", 8000)
