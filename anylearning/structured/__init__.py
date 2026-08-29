"""Portable structured-data projects: tables, text ML, and response evaluation."""

TEXT_AI_PROJECT_TYPE = "Text AI"
# Both names were used by preview builds.  They remain valid so projects and
# exported archives created before the terminology correction still open.
LEGACY_TEXT_PROJECT_TYPES = frozenset(
    {"Text AI & LLM Evaluation", "Text & LLM", "Sentiment Analysis"}
)
TEXT_PROJECT_TYPES = frozenset({TEXT_AI_PROJECT_TYPE, *LEGACY_TEXT_PROJECT_TYPES})
PROJECT_TYPES = frozenset({"Tabular AI", *TEXT_PROJECT_TYPES})


def is_structured_project(project_type: str | None) -> bool:
    return project_type in PROJECT_TYPES


def is_text_project(project_type: str | None) -> bool:
    return project_type in TEXT_PROJECT_TYPES
