__appname__ = "AnyLearning"
# Product name shown by the application and installers.
__product__ = "AnyLearning"
__description__ = "Build your own AI models"
# Calendar-versioned: the minor is the year (26 = 2026). This is the single
# source of truth -- build_app.sh reads it, and the Inno Setup scripts include
# the version file that build_app.sh generates from it. Do not hardcode the
# version anywhere else; it used to appear in six places and nothing caught a
# mismatch, so an installer could be named for one version and contain another.
__version__ = "0.26.3"

# Whether this checkout was stamped by the packaging script.
try:
    from anylearning.build_stamp import IS_RELEASE_BUILD  # noqa: F401
except ImportError:
    IS_RELEASE_BUILD = False
