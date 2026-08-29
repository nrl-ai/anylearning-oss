#! /usr/bin/env python
"""Regenerate LICENSES.md from what this environment actually ships.

The hand-written file listed four components. The build ships around 150
distributions -- torch, numpy, OpenCV, FastAPI, detectron2 and everything they
pull in -- and the permissive licences they carry (MIT, BSD, Apache 2.0) each
require their notice to travel with the binary. A notice that names four of
them does not discharge that for the other hundred and forty.

So the list is generated from the installed environment rather than maintained
by hand, because a hand-maintained list is out of date the first time someone
adds a dependency. Run it in the environment a release is built from, and give
it that build's Nuitka report:

    python generate_licenses.py --from-report compilation-report.xml

The report is worth the extra step. Without it the list is everything installed,
which on a development machine means pre-commit, virtualenv and nodeenv appear in
a document about what the installer contains -- eleven such entries, last time.
It also cuts the other way: the report is what proved PyGObject (LGPL-2.1) and
the CUDA compiler packages are *not* bundled, so their notices belong to the
machine rather than to us. Regenerate per release, from that release's report;
a report from another build describes another artefact.

What it cannot do is judge licences. It reports what each package declares; a
package that declares something copyleft is flagged in the summary and needs a
human decision -- see docs/model_license_policy.md.

LICENSES.md is output, not source: do not edit it by hand, add the component to
the environment and run this again. That instruction used to be printed in the
file's own header, where the only people who could not act on it -- the ones
reading it in Settings -- were the ones being told.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import pathlib
import re
import sys
import xml.etree.ElementTree as ElementTree

HEADER = """# Third-party licences

AnyLearning includes the open-source components listed below. Their licences
and copyright notices are reproduced here, as those licences require.

Two things are not in this list:

* **Vendored trainers.** NanoDet, the handpose classifier, the segmentation
  trainers and detectron2's Mask R-CNN are included as source. Their licence
  texts follow, in full.
* **Model weights.** Pretrained weights are licensed separately from the code
  that loads them, and are listed under **Model licences**.
"""

#: Our own package is not a third-party component, and its metadata is often
#: stale anyway in an editable install.
OURS = {"anylearning"}

#: Licences that mean someone has to make a decision before shipping. Not a
#: blocker here -- this script reports, it does not refuse -- but they are
#: listed first in the summary so nobody has to scan 150 rows to find them.
NEEDS_REVIEW = re.compile(r"\b(GPL|AGPL|LGPL|MPL|CDDL|EPL|CC BY-NC|SSPL)\b", re.I)

# The generated notice embeds the licences of source trees vendored by this
# repository.  Keep a stable content marker so regenerating the notice can
# recover that payload without copying the previous generated header into
# itself.
VENDORED_START = "NanoDet - Apache License 2.0"

#: Files a wheel keeps its licence text in, most specific first.
LICENCE_FILE_NAMES = re.compile(
    r"(LICENSE|LICENCE|COPYING|NOTICE)(\.(txt|md|rst))?$", re.I
)


def declared_licence(dist: metadata.Distribution) -> str:
    """The licence a package claims, from whichever field it used."""
    meta = dist.metadata
    # Modern wheels use License-Expression (PEP 639); older ones a classifier;
    # older still a free-text License field that is sometimes the whole licence.
    expression = meta.get("License-Expression")
    if expression:
        return expression.strip()

    classifiers = [
        value.split("::")[-1].strip()
        for value in meta.get_all("Classifier") or []
        if value.startswith("License ::")
    ]
    if classifiers:
        return ", ".join(sorted(set(classifiers)))

    declared = (meta.get("License") or "").strip()
    if declared and len(declared) < 80 and "\n" not in declared:
        return declared
    if declared:
        return "see text below"
    return "not declared"


def licence_text(dist: metadata.Distribution) -> str | None:
    """The licence file a wheel shipped, if it shipped one."""
    for file in dist.files or []:
        name = pathlib.PurePath(str(file)).name
        if not LICENCE_FILE_NAMES.match(name):
            continue
        try:
            text = file.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if text and text.strip():
            return text.strip()
    return None


def included_by_build(report: pathlib.Path) -> set[str] | None:
    """The distributions a Nuitka build actually bundled, from its report.

    Without this the list is everything installed in the environment, which
    includes the tools used to build and test -- black, pytest, ruff -- that no
    user ever receives. Over-inclusive is safe and under-inclusive is not, so
    this is an accuracy improvement rather than a correctness fix, and it is
    optional: the report only exists after a build.
    """
    try:
        root = ElementTree.parse(report).getroot()
    except (OSError, ElementTree.ParseError) as error:
        print(f"Could not read {report}: {error}", file=sys.stderr)
        return None
    distributions = root.find("distributions")
    if distributions is None:
        return None
    return {
        (item.get("name") or "").lower() for item in distributions if item.get("name")
    }


def collect(included: set[str] | None = None) -> list[dict]:
    seen: dict[str, dict] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        key = name.lower()
        if key in OURS:
            continue
        if included is not None and key not in included:
            continue
        # An environment can hold two copies of a distribution (a stale
        # .dist-info beside a fresh one); keep whichever declares a licence.
        text = licence_text(dist)
        licence = declared_licence(dist)
        # A wheel can ship the licence and never name it in metadata:
        # `faster-coco-eval` includes the full Apache 2.0 text and declares no
        # License field, no expression and no classifier. The obligation is
        # discharged either way -- the text below is what Apache 2.0 asks us to
        # carry -- but a summary row reading "not declared" beside a reproduced
        # licence is misleading in the more alarming direction. Not guessed from
        # the text: naming a licence a package did not claim is the one error
        # worth avoiding in a legal document.
        if licence == "not declared" and text:
            licence = "not declared; see text below"
        entry = {
            "name": name,
            "version": dist.version or "",
            "licence": licence,
            "text": text,
            "url": dist.metadata.get("Home-page") or "",
        }
        if key not in seen or (not seen[key]["text"] and entry["text"]):
            seen[key] = entry
    return sorted(seen.values(), key=lambda entry: entry["name"].lower())


def render(entries: list[dict], vendored: str) -> str:
    review = [e for e in entries if NEEDS_REVIEW.search(e["licence"])]

    lines = [HEADER, "", "## Summary", ""]
    if review:
        lines += [
            "Licences below that carry conditions worth checking before "
            "redistribution:",
            "",
        ]
        lines += [f"- **{e['name']} {e['version']}** — {e['licence']}" for e in review]
        lines += [""]

    lines += ["| Component | Version | Licence |", "|---|---|---|"]
    lines += [f"| {e['name']} | {e['version']} | {e['licence']} |" for e in entries]
    lines += ["", vendored, "", "## Full licence texts", ""]

    for entry in entries:
        lines += [f"### {entry['name']} {entry['version']}", ""]
        if entry["url"]:
            lines += [entry["url"], ""]
        if entry["text"]:
            lines += ["```", entry["text"], "```", ""]
        else:
            lines += [
                f"Declared licence: {entry['licence']}. "
                "This package shipped no licence file in its distribution.",
                "",
            ]

    # License files occasionally contain editor-only trailing spaces. Preserve
    # every visible character while keeping the generated repository artifact
    # clean enough for ``git diff --check`` and cross-platform rebuilds.
    rendered = "\n".join(lines)
    return "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"


def extract_vendored_licences(existing: str) -> str:
    """Return only the hand-maintained vendored licence payload.

    Older versions took everything before ``Full licence texts``.  Since that
    range also contained the generated header, every invocation nested another
    complete summary.  Anchor on the actual vendored payload and fail closed if
    the source notice is malformed instead of silently dropping a licence.
    """
    heading = "## Vendored components"
    heading_at = existing.rfind(heading)
    payload_at = existing.find(VENDORED_START, heading_at)
    payload_end = existing.find("\n```", payload_at)
    if heading_at < 0 or payload_at < 0 or payload_end < 0:
        raise RuntimeError(
            "Could not recover vendored licence payload from LICENSES.md"
        )
    return existing[payload_at:payload_end].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-report",
        type=pathlib.Path,
        default=None,
        help="A Nuitka compilation-report.xml, to list only what the build "
        "bundles rather than everything installed.",
    )
    arguments = parser.parse_args()

    root = pathlib.Path(__file__).parent
    existing = (root / "LICENSES.md").read_text(encoding="utf-8")

    # The hand-written vendored section is kept: those components are copied
    # into this repository rather than installed, so nothing in the environment
    # describes them.
    vendored = "## Vendored components\n\nThese are copied into "
    vendored += "`anylearning/training/models/` rather than installed.\n\n```\n"
    vendored += extract_vendored_licences(existing)
    vendored += "\n```"

    included = (
        included_by_build(arguments.from_report) if arguments.from_report else None
    )
    entries = collect(included)
    (root / "LICENSES.md").write_text(render(entries, vendored), encoding="utf-8")
    print(f"Wrote LICENSES.md for {len(entries)} distributions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
