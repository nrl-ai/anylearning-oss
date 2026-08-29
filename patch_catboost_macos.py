"""Fix CatBoost's misleading Mach-O install name before Nuitka packaging.

CatBoost 1.2.10's universal macOS wheel ships ``catboost/_catboost.so`` as a
Mach-O dynamic library whose ``LC_ID_DYLIB`` is ``@rpath/lib_catboost.dylib``.
There is no separate ``lib_catboost.dylib`` in the wheel: the extension itself
contains the library and imports normally. Nuitka 4.1.3 nevertheless resolves
that install name as an external dependency and aborts because the invented
path does not exist.

Give the binary an install name matching the file that actually exists. The
ID is not a load dependency, so this does not change CatBoost's runtime links.
Changing a Mach-O invalidates its signature; re-sign it ad-hoc for Apple
Silicon. On non-macOS hosts, or wheel layouts without this defect, do nothing.
"""

from __future__ import annotations

import importlib.util
import pathlib
import platform
import subprocess

BROKEN_ID = "@rpath/lib_catboost.dylib"
FIXED_ID = "@rpath/_catboost.so"


def catboost_extension() -> pathlib.Path | None:
    try:
        spec = importlib.util.find_spec("catboost")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    extension = pathlib.Path(spec.origin).parent / "_catboost.so"
    return extension if extension.is_file() else None


def install_name(binary: pathlib.Path) -> str | None:
    output = subprocess.run(
        ["otool", "-D", str(binary)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return output[1].strip() if len(output) > 1 else None


def main() -> int:
    if platform.system() != "Darwin":
        return 0

    extension = catboost_extension()
    if extension is None:
        print("CatBoost extension is not installed; nothing to patch")
        return 0

    current = install_name(extension)
    if current != BROKEN_ID:
        print(f"CatBoost extension install name needs no patching: {current or 'none'}")
        return 0

    subprocess.run(["install_name_tool", "-id", FIXED_ID, str(extension)], check=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(extension)], check=True)
    print(f"Repointed CatBoost extension install name: {BROKEN_ID} -> {FIXED_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
