#! /bin/bash

# Package the Linux build into something downloadable.
#
# build_app.sh leaves app.dist/ with app.bin inside it. Nothing turned that
# into a release artefact, which is why the update feed has had `"linux": null`
# in it since the beginning.
#
# A tarball rather than an AppImage, deliberately. The window is drawn by
# pywebview through the system's GTK/WebKit -- webkit2gtk is a system library
# with its own certificate store, GPU drivers and a large dependency graph, and
# bundling it is how AppImages of browsers end up broken on the distributions
# they were not built on. The tarball says what it needs and lets the package
# manager provide it, which is also what every other local-first Linux app of
# this shape does.
#
# Usage: bash make_linux_release.sh [path-to-app.dist]

set -euo pipefail

DIST="${1:-app.dist}"
if [ ! -d "$DIST" ]; then
    echo "no build at '$DIST' -- run build_app.sh first" >&2
    exit 1
fi
if [ ! -x "$DIST/app.bin" ]; then
    echo "'$DIST' has no app.bin in it" >&2
    exit 1
fi

VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' anylearning/app_info.py)"
if [ -z "$VERSION" ]; then
    echo "Could not read __version__ from anylearning/app_info.py" >&2
    exit 1
fi

# The architecture is in the name from 0.26.0 on. Releases up to 0.24.13 did
# not carry it -- AnyLearning-macOS-0.24.13.zip -- which was fine while there
# was one build per platform and stops being fine the moment there are two.
ARCH="$(uname -m)"
NAME="AnyLearning-Linux-${ARCH}-${VERSION}"
# Staged beside the build rather than in $TMPDIR: the tree is around 7 GB, and
# on a machine where /tmp is a tmpfs -- most of them -- that is 7 GB of RAM, or
# a "Disk quota exceeded" halfway through a copy that then tars up whatever
# happened to make it.
STAGING="$(mktemp -d -p "$(dirname "$(readlink -f "$DIST")")" .release-staging.XXXXXX)"
trap 'rm -rf "$STAGING"' EXIT

ROOT="$STAGING/$NAME"
mkdir -p "$ROOT"
cp -R "$DIST" "$ROOT/lib"
# Nuitka copies anylearning/migrations verbatim, __pycache__ and all, so the
# archive carried .pyc files -- including one compiled by a Python 3.10 that
# has nothing to do with this build.
find "$ROOT" -name __pycache__ -type d -prune -exec rm -rf {} +
cp LICENSES.md "$ROOT/"

# The launcher, so nobody has to know the binary is called app.bin and lives in
# lib/. It also resolves its own directory rather than assuming the working
# one: double-clicking from a file manager starts the process in $HOME.
cat > "$ROOT/AnyLearning" <<'LAUNCHER'
#! /bin/sh
here="$(dirname "$(readlink -f "$0")")"
exec "$here/lib/app.bin" "$@"
LAUNCHER
chmod +x "$ROOT/AnyLearning"

# A .desktop file for whoever wants a menu entry. Not installed automatically:
# the tarball can be unpacked anywhere, so Exec is filled in by install.sh.
cat > "$ROOT/anylearning.desktop.in" <<DESKTOP
[Desktop Entry]
Type=Application
Name=AnyLearning
Comment=Label images and train models on your own machine
Exec=@INSTALL_DIR@/AnyLearning
Icon=@INSTALL_DIR@/app_icon.png
Categories=Development;Science;
Terminal=false
DESKTOP
cp app_icon.png "$ROOT/app_icon.png"

cat > "$ROOT/install.sh" <<'INSTALL'
#! /bin/sh
# Optional: adds a menu entry pointing at wherever this folder already is.
# Nothing is copied -- move the folder and run this again.
set -eu
here="$(dirname "$(readlink -f "$0")")"
applications="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$applications"
sed "s|@INSTALL_DIR@|$here|g" "$here/anylearning.desktop.in" \
    > "$applications/anylearning.desktop"
echo "Menu entry written to $applications/anylearning.desktop"
INSTALL
chmod +x "$ROOT/install.sh"

cat > "$ROOT/README.txt" <<README
AnyLearning ${VERSION} for Linux (${ARCH})

Run ./AnyLearning

The window is drawn with your system's GTK and WebKit, which are not bundled --
see the note below. Everything else is included, including PyTorch, so the
download is large and needs no Python on the machine.

Requirements
------------
  Debian / Ubuntu   sudo apt install libwebkit2gtk-4.1-0 gir1.2-webkit2-4.1
  Fedora            sudo dnf install webkit2gtk4.1
  Arch              sudo pacman -S webkit2gtk-4.1

Without those the app starts and serves its API, but the window is blank --
that is the one failure this dependency produces, and it looks like a bug
rather than a missing package.

An NVIDIA GPU is used automatically when the driver is present. Without one,
training runs on the CPU; you can also pick CPU explicitly per run.

Your data
---------
Projects, images, models and databases live in ~/anylearning-data and are never
uploaded anywhere.

Licences
--------
LICENSES.md lists every third-party component and its licence.
README

echo "Packing ${NAME}.tar.gz"
rm -f "${NAME}.tar.gz"
tar -C "$STAGING" -czf "${NAME}.tar.gz" "$NAME"

echo
echo "Built ${NAME}.tar.gz ($(du -h "${NAME}.tar.gz" | cut -f1))"
echo "Unpacked size: $(du -sh "$ROOT" | cut -f1)"
