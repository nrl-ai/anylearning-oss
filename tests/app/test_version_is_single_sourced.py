"""The version must be defined once.

It used to appear in six places -- app_info.py, build_app.sh, and twice in each
of the two Inno Setup scripts -- with nothing checking they agreed. A missed
edit shipped an installer *named* for one version containing another, which is
close to undiagnosable from a bug report.
"""

import pathlib
import re

import pytest

from anylearning.app_info import __appname__, __product__, __version__

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FRONTEND_APP_INFO = REPO_ROOT / "frontend" / "src" / "lib" / "app-info.ts"


def test_version_looks_like_a_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__


def test_version_is_calendar_versioned():
    """The minor is the year: 0.26.x means 2026. Documented in app_info.py."""
    _major, minor, _patch = __version__.split(".")
    assert 24 <= int(minor) <= 99, (
        f"minor {minor} does not read as a year; if the scheme changed, "
        "update app_info.py's comment and this test together"
    )


def test_build_script_does_not_hardcode_the_version():
    build_script = (REPO_ROOT / "build_app.sh").read_text()
    assert __version__ not in build_script, (
        "build_app.sh hardcodes the version; it should read app_info.py"
    )
    assert "app_info.py" in build_script


@pytest.mark.parametrize(
    "installer",
    ["AnyLearning-Windows-Setup.iss", "AnyLearning-GPU-Windows-Setup.iss"],
)
def test_installers_do_not_hardcode_the_version(installer):
    text = (REPO_ROOT / installer).read_text(encoding="utf-8", errors="replace")
    assert __version__ not in text, f"{installer} hardcodes the version"
    assert "installer_version.iss" in text, (
        f"{installer} should include the generated version file"
    )
    # The output filename must interpolate too, or the installer ships with a
    # name that disagrees with its contents.
    assert "{#MyAppVersion}" in text


@pytest.mark.parametrize(
    "installer",
    ["AnyLearning-Windows-Setup.iss", "AnyLearning-GPU-Windows-Setup.iss"],
)
def test_installer_app_name_matches_app_info(installer):
    text = (REPO_ROOT / installer).read_text(encoding="utf-8", errors="replace")
    match = re.search(r'#define MyAppName "([^"]+)"', text)
    assert match, f"{installer} has no MyAppName"
    assert match.group(1) == __appname__, (
        f"{installer} calls the app {match.group(1)!r} but app_info says "
        f"{__appname__!r}"
    )


def test_the_frontend_mirrors_the_same_version():
    """The UI cannot import app_info.py, so it copies it -- and copies drift.

    0.26.1 shipped its first build with the sidebar still reading v0.26.0,
    because the Python side was bumped and this one was not. The frontend is
    static-exported, so the value is baked in at build time and no amount of
    checking at runtime would have caught it.
    """
    text = FRONTEND_APP_INFO.read_text(encoding="utf-8")
    match = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', text)
    assert match, f"no APP_VERSION in {FRONTEND_APP_INFO.name}"
    assert match.group(1) == __version__, (
        f"{FRONTEND_APP_INFO.name} says {match.group(1)}, app_info.py says "
        f"{__version__}"
    )


def test_the_frontend_mirrors_the_same_product_name():
    text = FRONTEND_APP_INFO.read_text(encoding="utf-8")
    match = re.search(r'PRODUCT_NAME\s*=\s*"([^"]+)"', text)
    assert match, f"no PRODUCT_NAME in {FRONTEND_APP_INFO.name}"
    assert match.group(1) == __product__, (
        f"{FRONTEND_APP_INFO.name} says {match.group(1)!r}, app_info.py says "
        f"{__product__!r}"
    )


def test_no_stray_version_literals_in_python_sources():
    """Nothing but app_info.py may spell the version out."""
    offenders = []
    for path in (REPO_ROOT / "anylearning").rglob("*.py"):
        if path.name == "app_info.py" or "models/nanodet" in path.as_posix():
            continue
        if __version__ in path.read_text(encoding="utf-8", errors="replace"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"hardcoded version in: {offenders}"
