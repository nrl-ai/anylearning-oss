#! /bin/bash

# Package AnyLearning.app into a .dmg for the website.
#
# build_app.sh produces AnyLearning.app; nothing turned that into something
# downloadable, so macOS releases had no artefact to publish. This makes the
# disk image people expect: the app on the left, a link to /Applications on the
# right, drag across.
#
# Usage: bash make_dmg.sh [path-to-.app]

set -euo pipefail

APP="${1:-AnyLearning.app}"
if [ ! -d "$APP" ]; then
    echo "no app bundle at '$APP' -- run build_app.sh first" >&2
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
DMG="AnyLearning-macOS-${ARCH}-${VERSION}.dmg"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "Staging ${APP} for ${DMG}"
# Copied, not moved: the build output stays where the smoke tests expect it.
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

rm -f "$DMG"
# UDZO is the compressed read-only format every macOS can mount. The app is
# ~1.6 GB of mostly-compressible Python and shared libraries, so this matters
# for a download.
hdiutil create \
    -volname "AnyLearning ${VERSION}" \
    -srcfolder "$STAGING" \
    -ov \
    -format UDZO \
    "$DMG"

echo
echo "Built ${DMG} ($(du -h "$DMG" | cut -f1))"
echo
echo "Before publishing this, read the signing note in docs/release_testing.md:"
echo "an unsigned, un-notarised build is refused by Gatekeeper on any Mac but"
echo "the one that built it -- the user is told the app is damaged, not that it"
echo "is unsigned, so it looks like a broken download rather than a policy."
