import os

import patch_libtorch


def test_macos_twin_root_can_be_relative(tmp_path, monkeypatch):
    """The postprocessor must work with build_app.sh's ``twin-out`` value."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(patch_libtorch.platform, "system", lambda: "Darwin")
    library_dir = tmp_path / "twin-out/app.app/Contents/MacOS/torch/lib"
    library_dir.mkdir(parents=True)
    library = library_dir / "libtorch_test.dylib"
    library.write_bytes(b"test")

    patch_libtorch.patch_libtorch("twin-out")

    link = (
        tmp_path
        / "twin-out/app.app/Contents/MacOS/torch/bin/torch/lib/libtorch_test.dylib"
    )
    assert link.is_symlink()
    assert os.readlink(link) == "../../../lib/libtorch_test.dylib"
    assert link.resolve() == library
