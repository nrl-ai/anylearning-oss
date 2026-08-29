"""A project archive is a file someone else made.

`routers/project.py` imports a project by untarring an archive the user chose.
Without `filter="data"` tarfile runs in its legacy fully-trusted mode, and a
member named `../../x` writes outside the temporary directory -- on a file
people email each other. Python 3.13 warns about this and 3.14 makes filtering
the default; the warning was silenced here by an ignore whose comment said the
only unfiltered extraction was torch's, which had stopped being true.
"""

import pathlib
import tarfile

import pytest


def build_malicious_archive(path: pathlib.Path, tmp_path: pathlib.Path):
    """An archive with a member that climbs out of the extraction directory."""
    escapee = tmp_path / "payload.txt"
    escapee.write_text("owned")
    with tarfile.open(path, "w:gz") as tar:
        tar.add(escapee, arcname="../../escaped.txt")
    return path


def test_the_import_path_refuses_a_traversal_member(tmp_path):
    archive = build_malicious_archive(tmp_path / "evil.tar.gz", tmp_path)
    destination = tmp_path / "extract_here"
    destination.mkdir()

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(tarfile.OutsideDestinationError):
            tar.extractall(destination, filter="data")

    assert not (tmp_path / "escaped.txt").exists()


def test_the_router_passes_a_filter():
    """The assertion that would have caught the shipped code.

    Read as source rather than executed: reaching the extraction needs a whole
    import in flight, and the property worth pinning is that no call site loses
    the argument again.
    """
    source = pathlib.Path("anylearning/routers/project.py").read_text()

    for line in source.splitlines():
        if ".extractall(" in line:
            assert "filter=" in line, f"unfiltered extraction: {line.strip()}"


def test_every_tar_extraction_in_the_application_is_filtered():
    """Not just this router -- anywhere we untar something a user supplied.

    Scoped to tarfile on purpose. `zipfile.extractall` sanitises member names
    itself and takes no `filter=`, so `auto_labeling/model_manager.py` is not
    the same hazard and must not be "fixed" with an argument that does not
    exist.
    """
    offenders = []
    for path in pathlib.Path("anylearning").rglob("*.py"):
        if "models/nanodet" in path.as_posix():
            continue
        lines = path.read_text(errors="replace").splitlines()
        for number, line in enumerate(lines, 1):
            if ".extractall(" not in line or "filter=" in line:
                continue
            # Which library opened it: look back a few lines for the `with`.
            context = " ".join(lines[max(0, number - 5) : number])
            if "tarfile.open" in context:
                offenders.append(f"{path}:{number}")
    assert not offenders, f"unfiltered tar extraction in: {offenders}"
