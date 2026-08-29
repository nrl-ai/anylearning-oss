from importlib import resources

# importlib.resources rather than pkg_resources: setuptools no longer ships
# pkg_resources by default, so importing it breaks on fresh environments.
# Kept dependency-free here so `handpose` stays installable on its own.
HAND_LANDMARK_MODEL_DIR = str(
    resources.files("anylearning").joinpath("models").joinpath("hand_landmarker.task")
)
