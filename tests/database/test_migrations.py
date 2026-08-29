import sqlite3
from unittest.mock import patch

import pytest

from anylearning import migration_manager as migrations_module
from anylearning.migration_manager import MigrationManager


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A data root of its own.

    Without this the suite runs against ~/anylearning-data -- creating folders
    in it, and answering differently depending on whether the developer happens
    to have projects.
    """
    from anylearning import config

    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))
    return tmp_path


@pytest.fixture
def migration_manager(data_root):
    return MigrationManager()


def test_init(data_root):
    manager = MigrationManager()
    assert manager.projects_dir == data_root / "projects"
    assert manager.projects_dir.exists()


def test_verify_project_db(data_root):
    manager = MigrationManager()
    project_id = "123"
    db_path = manager.verify_project_db(project_id)

    assert db_path == data_root / "projects" / project_id / "database.db"
    assert (data_root / "projects" / project_id).exists()


def test_run_migrations_for_project_failure(migration_manager):
    with patch("alembic.command.upgrade", side_effect=Exception("Migration failed")):
        success = migration_manager.run_migrations_for_project("123")
        assert success is False


def test_run_all_migrations(migration_manager):
    with (
        patch("alembic.command.upgrade") as mock_upgrade,
        patch.object(migration_manager, "get_all_project_ids") as mock_get_ids,
    ):
        mock_get_ids.return_value = ["1", "2"]

        results = migration_manager.run_all_migrations()

        assert results == {"success": ["main", "1", "2"], "failed": []}
        assert mock_upgrade.call_count == 3  # main + 2 projects


def test_run_all_migrations_main_failure(migration_manager):
    with patch("alembic.command.upgrade", side_effect=Exception("Migration failed")):
        results = migration_manager.run_all_migrations()
        assert results == {"success": [], "failed": ["main"]}


def test_create_new_project(migration_manager):
    with patch("alembic.command.upgrade") as mock_upgrade:
        success = migration_manager.create_new_project("123")
        assert success is True
        mock_upgrade.assert_called_once()


# --------------------------------------------------------------------------
# Against real databases. The mocked tests above prove the manager calls
# alembic; these prove the schema ends up right, which is the part a user
# notices -- and the part that used to fail silently, because `Config(
# "alembic.ini")` is a relative path that resolves to nothing in a packaged app.
# --------------------------------------------------------------------------

def tables_in(path):
    connection = sqlite3.connect(path)
    try:
        return {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()


def stamped_revision(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    except sqlite3.OperationalError:
        return None
    finally:
        connection.close()


def test_the_scripts_are_found_without_a_working_directory(monkeypatch, tmp_path):
    """The whole reason migrations never ran in the packaged app."""
    monkeypatch.chdir(tmp_path)
    config = migrations_module.alembic_config()
    assert (tmp_path / "alembic.ini").exists() is False
    assert config.get_main_option("script_location") == str(migrations_module.MIGRATIONS_DIR)


def test_a_new_database_gets_the_schema_and_a_stamp(data_root):
    manager = MigrationManager()
    assert manager.run_all_migrations()["failed"] == []

    main = data_root / "main.db"
    assert "projects" in tables_in(main)
    assert stamped_revision(main) == "0001_baseline"


def test_an_existing_database_is_stamped_rather_than_rebuilt(data_root):
    """The upgrade path for every install that predates migrations.

    Its schema was created by `create_all`, so it is already current; running
    the baseline against it would fail on tables that already exist, and the
    failure is caught and logged, which is how this stayed invisible.
    """
    from anylearning.database import Base
    from sqlalchemy import create_engine

    main = data_root / "main.db"
    engine = create_engine(f"sqlite:///{main}")
    Base.metadata.create_all(engine)
    engine.dispose()

    connection = sqlite3.connect(main)
    connection.execute("INSERT INTO projects (name, type) VALUES ('Kept', 'Object Detection')")
    connection.commit()
    connection.close()

    manager = MigrationManager()
    assert manager.run_all_migrations()["failed"] == []

    assert stamped_revision(main) == "0001_baseline"
    connection = sqlite3.connect(main)
    try:
        assert connection.execute("SELECT name FROM projects").fetchall() == [("Kept",)]
    finally:
        connection.close()


def test_a_second_run_changes_nothing(data_root):
    manager = MigrationManager()
    manager.run_all_migrations()
    assert manager.run_all_migrations()["failed"] == []
    assert stamped_revision(data_root / "main.db") == "0001_baseline"
