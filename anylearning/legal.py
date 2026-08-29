"""The third-party licence notices, and where to find them at runtime.

AnyLearning redistributes other people's code -- NanoDet, detectron2, torch and
a long tail behind them. The permissive licences that makes possible (MIT, BSD,
Apache 2.0) all require the notice to travel *with the binary*: a copy sitting
in the source repository does not discharge the obligation for a user who only
ever downloads an installer.

So `LICENSES.md` is a shipped artefact, not documentation. It is included in
the build, offered by the Windows installer, and readable inside the app.
"""

from __future__ import annotations

import importlib.metadata
import pathlib

import anylearning

#: Packaged first: `build_app.sh` copies LICENSES.md next to the package, and
#: that copy is the one a user has. The repository root is the development
#: fallback, and the order matters -- a build that failed to include the file
#: must not silently read the developer's checkout instead.
_CANDIDATES = (
    pathlib.Path(anylearning.__file__).parent / "LICENSES.md",
    pathlib.Path(anylearning.__file__).parent.parent / "LICENSES.md",
)

#: The project's Apache-2.0 license, separate from third-party notices.
_LICENSE_CANDIDATES = (
    pathlib.Path(anylearning.__file__).parent / "LICENSE",
    pathlib.Path(anylearning.__file__).parent.parent / "LICENSE",
)

#: Which weights ship and what their licences allow, written for the person
#: using the application.
#:
#: Deliberately not `docs/model_license_policy.md`, which used to be what this
#: served: that document is the rule we apply when *choosing* a model, and it
#: reads like it -- repository paths, the installer's filename, and tables of
#: models we rejected or have not adopted. Shown in Settings it advertised
#: capabilities the product does not have and leaked our build layout.
_MODEL_POLICY_CANDIDATES = (
    pathlib.Path(anylearning.__file__).parent / "MODEL_LICENCES.md",
    pathlib.Path(anylearning.__file__).parent.parent / "MODEL_LICENCES.md",
)


def _metadata_license_path(filename: str) -> pathlib.Path | None:
    """Find a notice installed in the wheel's ``dist-info/licenses`` folder."""
    try:
        distribution = importlib.metadata.distribution("anylearning")
    except importlib.metadata.PackageNotFoundError:
        return None

    for entry in distribution.files or ():
        if entry.name == filename and "licenses" in entry.parts:
            candidate = pathlib.Path(distribution.locate_file(entry))
            if candidate.is_file():
                return candidate
    return None


def notices_path() -> pathlib.Path | None:
    """Where the notices are, or None if this build did not ship them."""
    for candidate in _CANDIDATES:
        if candidate.is_file():
            return candidate
    return _metadata_license_path("LICENSES.md")


def read_notices() -> str | None:
    path = notices_path()
    if path is None:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def read_license() -> str | None:
    """The project license, or None if this build did not ship it."""
    for candidate in _LICENSE_CANDIDATES:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    metadata_path = _metadata_license_path("LICENSE")
    if metadata_path is not None:
        return metadata_path.read_text(encoding="utf-8", errors="replace")
    return None


def read_model_policy() -> str | None:
    """The model and weight licence policy, or None if not shipped."""
    for candidate in _MODEL_POLICY_CANDIDATES:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    metadata_path = _metadata_license_path("MODEL_LICENCES.md")
    if metadata_path is not None:
        return metadata_path.read_text(encoding="utf-8", errors="replace")
    return None


def parse_notices(text: str) -> list[dict]:
    """Split the generated file into one entry per component.

    The file is markdown because a human reads it in the repository and the
    Windows installer displays it as text. The app should not: dumping it into
    a <pre> shows "###" headings and ``` fences as literal characters, which
    looks like a bug and buries the one thing a reader wants -- which component
    they are looking at.

    Parsed rather than served as structured data from the generator, so what
    the app shows and what the installer shows cannot drift apart.
    """
    components: list[dict] = []
    current: dict | None = None
    in_fence = False

    for line in (text or "").splitlines():
        if line.startswith("### "):
            heading = line[4:].strip()
            name, _, version = heading.rpartition(" ")
            current = {
                "name": name or heading,
                "version": version if name else "",
                "lines": [],
            }
            components.append(current)
            in_fence = False
            continue
        if current is None:
            continue
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or line.strip():
            current["lines"].append(line)

    return [
        {
            "name": entry["name"],
            "version": entry["version"],
            "text": "\n".join(entry["lines"]).strip(),
        }
        for entry in components
    ]
