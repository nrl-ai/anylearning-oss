"""Recognize databases created by pre-open-source AnyLearning releases.

Some existing installations are stamped at ``0002_project_data_sources``.
The open-source schema does not use that optional feature, but Alembic still
has to know the revision so it can open those databases and preserve all user
projects. This compatibility marker intentionally performs no schema changes.

Revision ID: 0002_project_data_sources
Revises: 0001_baseline
Create Date: 2026-08-29

"""

revision = "0002_project_data_sources"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Keep the existing schema unchanged."""


def downgrade() -> None:
    """Keep the existing schema unchanged."""
