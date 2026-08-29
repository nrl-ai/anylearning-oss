"""The schema as it stood before there were any migrations.

Generated from the models rather than from a live database: every database on
disk was built by `Base.metadata.create_all`, so a diff against one of those is
empty by definition and would have produced a baseline that creates nothing.

Databases that already exist are stamped at this revision rather than run
through it -- see MigrationManager. It is here so that databases created from
now on have a history to build changes on, and so the first real schema change
has something to say "revises" about.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-08-16

"""

import sqlalchemy as sa
from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("train_version", sa.String(), nullable=True),
        sa.Column("val_version", sa.String(), nullable=True),
        sa.Column("test_version", sa.String(), nullable=True),
        sa.Column("modified_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_datasets_id"), "datasets", ["id"], unique=False)
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("type", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("path", sa.String(), nullable=True),
        sa.Column("size", sa.Float(), nullable=True),
        sa.Column("dataset", sa.String(), nullable=True),
        sa.Column("num_train", sa.Integer(), nullable=True),
        sa.Column("num_val", sa.Integer(), nullable=True),
        sa.Column("num_test", sa.Integer(), nullable=True),
        sa.Column("num_trained_models", sa.Integer(), nullable=True),
        sa.Column("new_models_this_month", sa.Integer(), nullable=True),
        sa.Column("labels", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_projects_id"), "projects", ["id"], unique=False)
    op.create_index(op.f("ix_projects_name"), "projects", ["name"], unique=False)
    op.create_index(op.f("ix_projects_type"), "projects", ["type"], unique=False)
    op.create_table(
        "training_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("metric_logs", sa.JSON(), nullable=True),
        sa.Column("training_logs", sa.String(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("config_file", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_training_sessions_id"), "training_sessions", ["id"], unique=False
    )
    op.create_table(
        "data_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=True),
        sa.Column("subset", sa.Integer(), nullable=True),
        sa.Column("path", sa.String(), nullable=True),
        sa.Column("labeled", sa.Boolean(), nullable=True),
        sa.Column("class_id", sa.Integer(), nullable=True),
        sa.Column("annotation", sa.JSON(), nullable=True),
        sa.Column("original_name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("modified_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_data_items_id"), "data_items", ["id"], unique=False)
    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("training_session_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("path", sa.String(), nullable=True),
        sa.Column("config_file", sa.String(), nullable=True),
        sa.Column("exported_path", sa.String(), nullable=True),
        sa.Column("model_architecture", sa.String(), nullable=True),
        sa.Column("model_size", sa.String(), nullable=True),
        sa.Column("test_version", sa.String(), nullable=True),
        sa.Column("test_result", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("modified_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["training_session_id"],
            ["training_sessions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_models_id"), "models", ["id"], unique=False)
    op.create_table(
        "training_processes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("training_session_id", sa.Integer(), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["training_session_id"],
            ["training_sessions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_training_processes_id"), "training_processes", ["id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_training_processes_id"), table_name="training_processes")
    op.drop_table("training_processes")
    op.drop_index(op.f("ix_models_id"), table_name="models")
    op.drop_table("models")
    op.drop_index(op.f("ix_data_items_id"), table_name="data_items")
    op.drop_table("data_items")
    op.drop_index(op.f("ix_training_sessions_id"), table_name="training_sessions")
    op.drop_table("training_sessions")
    op.drop_index(op.f("ix_projects_type"), table_name="projects")
    op.drop_index(op.f("ix_projects_name"), table_name="projects")
    op.drop_index(op.f("ix_projects_id"), table_name="projects")
    op.drop_table("projects")
    op.drop_index(op.f("ix_datasets_id"), table_name="datasets")
    op.drop_table("datasets")
