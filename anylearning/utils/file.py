import itertools
import logging
import os
import pathlib
import platform
import shutil
import stat
import subprocess
import tempfile

import requests
from tqdm import tqdm

from anylearning.utils.resources import resource_path


def open_file(path):
    """
    Open file in default application
    """
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def make_writable(root):
    """Give the owner write permission on a tree, so it can be replaced later.

    `shutil.copytree` copies permission bits, and an installation the user does
    not own is read-only: /opt/anylearning, Program Files, an app bundle in
    /Applications. Copying its `frontend-dist` into the data root therefore
    produced 0555 directories and 0444 files *in the user's own folder*, and the
    next launch could not replace them.
    """
    for path in itertools.chain([root], pathlib.Path(root).rglob("*")):
        try:
            mode = os.stat(path).st_mode
            os.chmod(
                path, mode | stat.S_IWUSR | (stat.S_IXUSR if os.path.isdir(path) else 0)
            )
        except OSError:
            # Best effort: a file we cannot chmod is one rmtree will report on,
            # which is now checked rather than ignored.
            pass


def extract_frontend_dist(static_folder):
    """
    Extract folder frontend/dist from package anylearning
    and put it in the same static folder for serving

    Read-only installations are the reason this is more than a copytree. The
    app shipped a version that started exactly once from `/opt` or from a
    read-only `.app`: copytree carried the installation's 0555/0444 modes into
    the data root, `rmtree(..., ignore_errors=True)` then could not delete
    inside those directories *and said nothing*, and copytree raised
    `FileExistsError` on the second launch. Exit 1, no server, no window, and a
    user with no way to guess that `chmod -R u+w` on a cache directory is the
    cure.
    """
    dist_folder = resource_path("anylearning", "frontend-dist")
    if os.path.exists(dist_folder):
        if os.path.exists(static_folder):
            logging.info(f"Refreshing {static_folder}...")
            # Writable first, or the delete below silently does nothing.
            make_writable(static_folder)
            shutil.rmtree(static_folder, ignore_errors=True)
            if os.path.exists(static_folder):
                # Do not fall through to copytree: it raises FileExistsError
                # and that is what the user would see instead of the cause.
                raise RuntimeError(
                    f"Could not replace {static_folder}. Delete it and start "
                    "again; the frontend is extracted there on every launch."
                )
        pathlib.Path(static_folder).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dist_folder, static_folder)
        # And leave it replaceable next time, whatever the installation's modes.
        make_writable(static_folder)
    else:
        logging.warning("frontend-dist not found in package anylearning")
        if not os.path.exists(static_folder):
            pathlib.Path(static_folder).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(static_folder, "index.html"), "w") as f:
                f.write(
                    "<b>frontend-dist</b> not found in package anylearning. "
                    "Please run: <code>bash build_frontend.sh</code>"
                )
            return


def download_file(url, file_path):
    """Download to a temporary file beside the destination, then move it in.

    Two things this has to get right, and got wrong on Windows.

    The temporary file is created *closed*. `NamedTemporaryFile` hands back an
    open handle, and Windows refuses to move a file that anyone still has open:
    every model download failed with

        PermissionError: [WinError 32] The process cannot access the file
        because it is being used by another process

    And it is created next to the destination rather than in the system temp
    directory, so the move is a rename on the same filesystem -- atomic, so a
    reader never sees a half-written model -- instead of a copy across devices.

    A failed download leaves nothing behind: a half-file that survives looks
    exactly like a model to the next caller.
    """
    destination = pathlib.Path(file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    os.close(handle)

    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            logging.info("Failed to download file.")
            return

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192  # Chunk size in bytes
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

        with open(temporary, "wb") as file:
            # Iterate over the response content in chunks
            for chunk in response.iter_content(chunk_size=block_size):
                file.write(chunk)
                progress_bar.update(len(chunk))

        progress_bar.close()
        os.replace(temporary, destination)
        logging.info("File downloaded successfully.")
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
