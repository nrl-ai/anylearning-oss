"""Bringing databases up to the schema this build expects.

There are many databases: one main registry plus one per project, created at
different times by different versions of the app. Alembic keeps a version stamp
in each of them so a schema change can be applied exactly once, wherever that
database happens to be in its history.

Two decisions worth knowing about.

**Existing databases are stamped, not migrated.** Every database created before
this existed was built by `Base.metadata.create_all`, so its schema already
matches the models -- there is nothing to upgrade, and running the initial
revision against it would fail trying to create tables that are already there.
So a database with tables but no version stamp is recorded as being at the
baseline. From the next schema change onwards it upgrades like any other.

**The script location is resolved from the package.** It used to be
`Config("alembic.ini")`, a relative path: in the packaged app there is no
alembic.ini and no meaningful working directory, so every migration failed --
silently, because the failure is caught and logged.
"""

import logging
import os
import pathlib

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

import anylearning
from anylearning import config

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = pathlib.Path(anylearning.__file__).parent / "migrations"


def alembic_config() -> Config:
    """An alembic config that does not depend on the working directory."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # env.py builds the URL itself from ALEMBIC_PROJECT_ID, but alembic wants
    # something here or it complains before env.py ever runs.
    cfg.set_main_option("sqlalchemy.url", "sqlite://")
    return cfg


class MigrationManager:
    def __init__(self):
        self.projects_dir = pathlib.Path(config.PROJECTS_ROOT)
        self.main_db = config.DATABASE_PATH
        self.projects_dir.mkdir(exist_ok=True)

    def get_all_project_ids(self):
        """Get project IDs from main database"""
        engine = create_engine(f"sqlite:///{self.main_db}")
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT id FROM projects"))
                return [str(row[0]) for row in result]
        except Exception as error:
            # A main database that has no projects table yet is a first run, not
            # a fault: there is nothing to migrate and the caller should carry on.
            logger.info(f"No projects to migrate yet: {error}")
            return []
        finally:
            # Same as verify_project_db: closing the connection only returns it
            # to the pool, so the engine has to be disposed for the sqlite handle
            # to actually close.
            engine.dispose()

    def verify_project_db(self, project_id):
        """Verify project database exists or create it"""
        project_path = self.projects_dir / str(project_id)
        project_path.mkdir(exist_ok=True)
        db_path = project_path / "database.db"
        if not db_path.exists():
            engine = create_engine(f"sqlite:///{db_path}")
            try:
                with engine.connect():
                    # Create initial schema or markers if needed
                    pass
            finally:
                # dispose(), not just closing the connection: the connection goes
                # back to the engine's pool rather than being closed, so the
                # underlying sqlite3 handle stayed open for the life of the
                # process -- once per project, and this engine is never reused.
                engine.dispose()
        return db_path

    def _database_url(self, project_id):
        if project_id == "main":
            return f"sqlite:///{self.main_db}"
        return f"sqlite:///{self.projects_dir / str(project_id) / 'database.db'}"

    def _needs_stamping(self, project_id) -> bool:
        """True for a database built before migrations existed.

        It has tables and no version stamp. Upgrading it would try to create
        what is already there; the schema is current by construction, so the
        honest record is "already at the baseline".
        """
        engine = create_engine(self._database_url(project_id))
        try:
            with engine.connect() as connection:
                stamped = MigrationContext.configure(connection).get_current_revision()
                if stamped is not None:
                    return False
                tables = set(inspect(connection).get_table_names())
                return bool(tables - {"alembic_version"})
        except Exception as error:
            logger.warning(f"Could not inspect {project_id}: {error}")
            return False
        finally:
            engine.dispose()

    def _migrate(self, project_id) -> bool:
        os.environ["ALEMBIC_PROJECT_ID"] = str(project_id)
        cfg = alembic_config()
        if self._needs_stamping(project_id):
            logger.info(f"Stamping existing database {project_id} at the baseline")
            command.stamp(cfg, "head")
            return True
        command.upgrade(cfg, "head")
        return True

    def run_migrations_for_project(self, project_id):
        """Run migrations for a specific project"""
        try:
            self.verify_project_db(project_id)
            logger.info(f"Running migrations for project {project_id}")
            self._migrate(project_id)
            logger.info(f"Successfully migrated project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to migrate project {project_id}: {str(e)}")
            return False

    def run_all_migrations(self):
        """Run migrations on main DB and all project DBs"""
        results = {"success": [], "failed": []}

        # First migrate main database
        try:
            logger.info("Migrating main database")
            self._migrate("main")
            results["success"].append("main")
        except Exception as e:
            logger.error(f"Failed to migrate main database: {str(e)}")
            results["failed"].append("main")
            return results  # Stop if main DB migration fails

        # Then migrate all project databases
        project_ids = self.get_all_project_ids()
        for project_id in project_ids:
            success = self.run_migrations_for_project(project_id)
            if success:
                results["success"].append(project_id)
            else:
                results["failed"].append(project_id)

        return results

    def create_new_project(self, project_id):
        """Initialize a new project with current schema"""
        self.verify_project_db(project_id)
        return self.run_migrations_for_project(project_id)


# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = MigrationManager()
    results = manager.run_all_migrations()
    print(f"Migration Results: {results}")
