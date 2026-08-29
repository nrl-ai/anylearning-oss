"""Point detectron2's extension at the torch dylibs it actually loads.

detectron2 builds `_C.cpython-*-darwin.so` against torch and records the
dependency as `@rpath/libc10.dylib` (and libtorch, libtorch_cpu,
libtorch_python). The only LC_RPATH it carries is the environment's `lib`
directory -- which is where a *conda* torch keeps its dylibs. A pip-installed
torch, which is what setup.py pins, keeps them in `site-packages/torch/lib`
instead, so that rpath resolves to nothing.

Nothing notices at runtime. `import torch` always happens first, so dyld has
already loaded libc10 under that install name and the reference binds to the
image in memory rather than to a file. Packaging is where it matters, because
Nuitka has to resolve the dependency on disk before it can copy it:

    FATAL: Error, failed to find path '@rpath/libc10.dylib' (resolved DLL to
    '.../detectron2/libc10.dylib') for binary '.../detectron2/_C...so' from
    package 'detectron2', please report the bug.

That is where every macOS build has died so far.

The fix is to rewrite those four references to `@loader_path`-relative paths,
which are true in both layouts that matter: in the environment, where
`detectron2/` and `torch/` are siblings in site-packages, and in the standalone
tree Nuitka produces, where they are siblings again. Run before Nuitka, from
`build_app.sh`; on anything other than macOS it does nothing.

Rewriting a Mach-O invalidates its signature, and on Apple Silicon an invalid
signature is a load error rather than a warning, so each patched binary is
re-signed ad-hoc afterwards.
"""

from __future__ import annotations

import importlib.util
import pathlib
import platform
import re
import subprocess

# The prefix that cannot be resolved from disk, and what it should become.
RPATH_PREFIX = "@rpath/"
TORCH_RELATIVE = "@loader_path/../torch/lib/"


def package_directory(name: str) -> pathlib.Path | None:
    """Where a package lives, without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return pathlib.Path(spec.origin).parent


def linked_dylibs(binary: pathlib.Path) -> list[str]:
    output = subprocess.run(
        ["otool", "-L", str(binary)], capture_output=True, text=True, check=True
    ).stdout
    return re.findall(r"^\s+(\S+)\s+\(compatibility", output, re.MULTILINE)


def repoint(binary: pathlib.Path, torch_lib: pathlib.Path) -> list[str]:
    """Rewrite every @rpath reference that torch/lib can satisfy."""
    changed = []
    for reference in linked_dylibs(binary):
        if not reference.startswith(RPATH_PREFIX):
            continue
        name = reference[len(RPATH_PREFIX) :]
        if not (torch_lib / name).exists():
            # Something else's @rpath dependency; leave it for Nuitka to find.
            continue
        subprocess.run(
            [
                "install_name_tool",
                "-change",
                reference,
                TORCH_RELATIVE + name,
                str(binary),
            ],
            check=True,
        )
        changed.append(name)

    if changed:
        # install_name_tool leaves the signature stale, which arm64 treats as
        # fatal at load time.
        subprocess.run(["codesign", "--force", "--sign", "-", str(binary)], check=True)
    return changed


def main() -> int:
    if platform.system() != "Darwin":
        return 0

    detectron2 = package_directory("detectron2")
    torch = package_directory("torch")
    if detectron2 is None or torch is None:
        print("detectron2 or torch is not installed; nothing to patch")
        return 0

    torch_lib = torch / "lib"
    extensions = sorted(detectron2.glob("_C*.so"))
    if not extensions:
        print(f"No compiled extension in {detectron2}; nothing to patch")
        return 0

    for extension in extensions:
        changed = repoint(extension, torch_lib)
        if changed:
            print(f"Repointed {extension.name} at torch/lib: {', '.join(changed)}")
        else:
            print(f"{extension.name} needs no patching")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
