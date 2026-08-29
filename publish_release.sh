#! /bin/bash

# Put the built artefacts where the download page can reach them.
#
# The bucket is the only thing between a build and a customer, so this script
# does three things and refuses to guess at any of them:
#
#   1. checks each artefact exists, is plausibly sized, and records its sha256;
#   2. uploads it to r2://anylearning/releases/ under the name the feed will
#      use, and reads the object back to confirm the size matches;
#   3. prints the `check-for-update.json` entry to paste into the website
#      repository -- it does *not* publish that itself, because that file is
#      what makes a release visible, and it should be a deliberate commit.
#
# Credentials come from the publishing checkout's .env (S3_ENDPOINT,
# S3_ACCESS_TOKEN, S3_SECRET_TOKEN) and are never passed on a command line.
#
# Usage:
#   bash publish_release.sh [--dry-run] <artefact> [<artefact> ...]
#
# Artefacts are named by convention, and the convention is load-bearing --
# check-for-update.json points straight at them:
#   AnyLearning-Windows-Setup-<version>.exe
#   AnyLearning-macOS-<arch>-<version>.dmg
#   AnyLearning-Linux-<arch>-<version>.tar.gz
#
# The architecture appears from 0.26.0 on; releases up to 0.24.13 have no such
# segment, so the version is parsed out of the name rather than assumed to sit
# at a fixed position.

set -euo pipefail

# Wrapped in a function on purpose. Bash reads a script incrementally as it
# executes, so editing this file during a run shifts the offsets it is still
# reading from -- which happened mid-upload and ended a nine-gigabyte publish
# with `syntax error near unexpected token 'fi'` after the last artefact had
# already gone up. Bash parses a function completely before running any of it,
# so a running publish holds its own copy. build_app.sh is wrapped for the same
# reason; do not unwrap either.
main() {
ENV_FILE="${ANYLEARNING_PUBLISH_ENV:-/home/vietanhdev/Workspaces/AnyLearning/publishing/.env}"
BUCKET="s3://anylearning/releases"
CDN="https://cdn.anylearning.nrl.ai/releases"
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

if [ "$#" -eq 0 ]; then
    echo "usage: bash publish_release.sh [--dry-run] <artefact> [<artefact> ...]" >&2
    exit 2
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "no credentials at $ENV_FILE" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

export AWS_ACCESS_KEY_ID="$S3_ACCESS_TOKEN"
export AWS_SECRET_ACCESS_KEY="$S3_SECRET_TOKEN"
export AWS_DEFAULT_REGION=auto
# R2 does not accept the streaming checksum headers newer AWS CLIs send by
# default; without these every upload fails with an unhelpful 400.
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required

r2() { aws s3 --endpoint-url "$S3_ENDPOINT" "$@"; }
r2api() { aws s3api --endpoint-url "$S3_ENDPOINT" "$@"; }

human() {
    python3 - "$1" <<'PY'
import sys

size = float(sys.argv[1])
for unit in ("B", "KB", "MB", "GB"):
    if size < 1024 or unit == "GB":
        print(f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}")
        break
    size /= 1024
PY
}

# The feed distinguishes the two macOS artefacts: `macos` is the disk image the
# download page offers, `macos_zip` is the .app the page offers beside it. They
# had one key between them here, so whichever file was passed last won and the
# printed suggestion silently dropped the other -- paste that into the feed and
# the DMG disappears from the site.
platform_of() {
    case "$1" in
        *Windows*)     echo windows ;;
        *macOS*.zip)   echo macos_zip ;;
        *macOS*)       echo macos ;;
        *Linux*)       echo linux ;;
        *)             echo unknown ;;
    esac
}

declare -A URLS SIZES
VERSION=""

for artefact in "$@"; do
    name="$(basename "$artefact")"
    if [ ! -f "$artefact" ]; then
        echo "missing: $artefact" >&2
        exit 1
    fi

    bytes=$(stat -c %s "$artefact" 2>/dev/null || stat -f %z "$artefact")
    # A truncated or half-copied artefact is the failure this catches: every
    # real one is hundreds of megabytes.
    if [ "$bytes" -lt 100000000 ]; then
        echo "$name is only $bytes bytes -- refusing to publish it" >&2
        exit 1
    fi

    platform="$(platform_of "$name")"
    if [ "$platform" = "unknown" ]; then
        echo "cannot tell which platform $name is for" >&2
        exit 1
    fi

    # The version is in the filename, and every artefact must agree.
    found="$(echo "$name" | sed -n 's/.*-\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)\..*/\1/p')"
    if [ -z "$found" ]; then
        echo "no version in $name" >&2
        exit 1
    fi
    if [ -n "$VERSION" ] && [ "$VERSION" != "$found" ]; then
        echo "version mismatch: $VERSION vs $found ($name)" >&2
        exit 1
    fi
    VERSION="$found"

    echo "== $name"
    echo "   $(human "$bytes") ($bytes bytes)"
    echo "   sha256 $(sha256sum "$artefact" | cut -d' ' -f1)"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "   (dry run, not uploaded)"
    else
        # --no-progress: the transfer meter writes a line per 256 KB, which
        # for a five-gigabyte artefact is a megabyte of log that hides the one
        # line that matters.
        r2 cp --no-progress "$artefact" "$BUCKET/$name"
        remote=$(r2api head-object --bucket anylearning --key "releases/$name" \
                    --query ContentLength --output text)
        if [ "$remote" != "$bytes" ]; then
            echo "   UPLOAD MISMATCH: local $bytes, remote $remote" >&2
            exit 1
        fi
        echo "   uploaded, remote size matches"
    fi

    URLS[$platform]="$CDN/$name"
    SIZES[$platform]="$(human "$bytes")"
done

echo
echo "check-for-update.json entry for AnyLearning ${VERSION}:"
python3 - "$VERSION" "${URLS[macos]:-}" "${URLS[macos_zip]:-}" "${URLS[windows]:-}" "${URLS[linux]:-}" \
                     "${SIZES[macos]:-}" "${SIZES[macos_zip]:-}" "${SIZES[windows]:-}" "${SIZES[linux]:-}" <<'PY'
import json
import sys

(
    version,
    macos,
    macos_zip,
    windows,
    linux,
    macos_size,
    macos_zip_size,
    windows_size,
    linux_size,
) = sys.argv[1:10]
entry = {
    "version": version,
    "url": "https://anylearning-oss.nrl.ai/download",
    "download_urls": {
        "macos": macos or None,
        "macos_zip": macos_zip or None,
        "windows": windows or None,
        "linux": linux or None,
    },
    "download_sizes": {
        "macos": macos_size or None,
        "macos_zip": macos_zip_size or None,
        "windows": windows_size or None,
        "linux": linux_size or None,
    },
}
print(json.dumps({"latest_version": version, "versions": [entry]}, indent=4))
PY
}

main "$@"
