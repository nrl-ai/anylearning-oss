import os
import platform
import shutil
import sys


def patch_libtorch(root=None):
    """Fix up the torch libraries inside a freshly built tree.

    `root` is where Nuitka left `app.dist` / `app.app`, and defaults to this
    script's directory because that is where a release build puts them. A twin
    build (`ANYLEARNING_TWIN_BUILD=1`) passes its own output directory: without
    it this reached for a sibling `app.dist` that belongs to a different build,
    and on Linux -- where the whole function is a no-op -- nothing would have
    said so.
    """
    # A twin build passes ``twin-out`` here.  Keep it absolute: the old macOS
    # branch changed directory before walking torch/lib, which made that
    # relative root point inside torch/bin/torch/lib and failed only after a
    # full native build had completed.
    script_dir = (
        os.path.abspath(root)
        if root is not None
        else os.path.dirname(os.path.abspath(__file__))
    )

    if platform.system() == "Darwin":
        # Create target directory for macOS
        target_dir = os.path.join(
            script_dir, "app.app/Contents/MacOS/torch/bin/torch/lib"
        )
        shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)

        # Library directory for macOS
        lib_dir = os.path.join(script_dir, "app.app/Contents/MacOS/torch/lib")

        # Create symbolic links for libraries on macOS
        if os.path.exists(lib_dir):
            for lib in os.listdir(lib_dir):
                if lib.endswith(".dylib"):
                    # Use relative path for symlink
                    rel_source = "../../../lib/" + lib
                    target = os.path.join(target_dir, lib)
                    if not os.path.lexists(target):
                        os.symlink(rel_source, target)
                        # Verify symlink was created successfully
                        if not os.path.islink(target):
                            raise RuntimeError(f"Failed to create symlink for {lib}")
                        # Make library executable
                        os.chmod(os.path.join(lib_dir, lib), 0o755)

    elif platform.system() == "Windows":
        # Create target directory for Windows
        target_dir = os.path.join(script_dir, "app.dist/torch/bin/torch/lib")
        shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)

        # Library directory for Windows
        lib_dir = os.path.join(script_dir, "app.dist/torch/lib")

        # Copy DLL files on Windows
        if os.path.exists(lib_dir):
            for lib in os.listdir(lib_dir):
                if lib.endswith(".dll"):
                    source = os.path.join(lib_dir, lib)
                    target = os.path.join(target_dir, lib)
                    shutil.copy2(source, target)


if __name__ == "__main__":
    patch_libtorch(sys.argv[1] if len(sys.argv) > 1 else None)
