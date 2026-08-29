#! /bin/bash

# Acceptance test for a *packaged* build.
#
# Nuitka failures do not all look like build failures. The build that excluded
# torch._dynamo compiled and linked cleanly, produced a 696 MB binary, and then
# died on startup because torchvision.ops imports the excluded module -- the
# kind of break that only exists in the packaged tree, which is exactly the
# thing worth testing before publishing an artefact.
#
# Usage: bash smoke_test_build.sh <path-to-binary>

set -uo pipefail

BINARY="${1:-}"
if [ -z "$BINARY" ]; then
    echo "usage: bash smoke_test_build.sh <path-to-binary>" >&2
    exit 2
fi
if [ ! -f "$BINARY" ]; then
    echo "FAIL: no binary at '$BINARY'" >&2
    exit 1
fi

# A high, unlikely-to-clash port. The app silently falls back to a random port
# when its preferred one is taken, so a stale process would otherwise have us
# testing something else entirely.
PORT="${SMOKE_PORT:-5799}"
BASE="http://127.0.0.1:${PORT}"
LOG="$(mktemp)"
failures=0

pass() { echo "  ok    $1"; }
fail() { echo "  FAIL  $1" >&2; failures=$((failures + 1)); }

# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2329
cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    # In the same handler, not a second `trap ... EXIT`: bash keeps one EXIT
    # trap, so registering another silently replaces this one. That left the
    # server running between runs, and the next run then talked to the previous
    # server -- whose data directory had just been deleted, so it reported the
    # frontend missing. A confusing way to learn how traps work.
    if [ -n "${SMOKE_DATA_ROOT:-}" ]; then
        rm -rf "$SMOKE_DATA_ROOT"
    fi
}
trap cleanup EXIT

echo "Smoke testing: $BINARY"

# ---------------------------------------------------------------------------
# 1. It starts at all. --version imports the whole application, which is where
#    a missing or excluded module surfaces.
# ---------------------------------------------------------------------------
# macOS has no `timeout` -- it is GNU coreutils, where it arrives as `gtimeout`
# if anyone installed it. Without this the whole script died on the first check
# with "timeout: command not found", reported as "the packaged app cannot
# start": the one machine where a macOS build can be tested was the one machine
# the test could not run on.
run_with_limit() {
    local seconds="$1"
    shift
    if command -v timeout > /dev/null 2>&1; then
        timeout "$seconds" "$@"
    elif command -v gtimeout > /dev/null 2>&1; then
        gtimeout "$seconds" "$@"
    else
        # Last resort: a watchdog that kills the command if it outlives its
        # budget. $! is the command; the watchdog exits with it either way.
        "$@" &
        local pid=$!
        { sleep "$seconds"; kill -9 "$pid" 2> /dev/null; } 2> /dev/null &
        local watchdog=$!
        wait "$pid"
        local status=$?
        # Killed quietly: the shell announces a terminated job otherwise, and
        # a stray "Terminated: 15" in the middle of a pass/fail list reads as
        # a failure.
        kill "$watchdog" 2> /dev/null
        wait "$watchdog" 2> /dev/null
        return $status
    fi
}

if run_with_limit 300 "$BINARY" --version > "$LOG" 2>&1; then
    pass "--version ($(tr -d '\r\n' < "$LOG" | tail -c 60))"
else
    fail "--version exited $? -- the packaged app cannot start"
    tail -30 "$LOG" >&2
    exit 1
fi

# A macOS app is identified by Info.plist, not by the executable's --version.
# Nuitka defaults these fields to the input filename and "1.0" unless the build
# supplies them explicitly, which produced a technically runnable but
# release-unready `app` bundle.  Check the sealed metadata on the actual bundle.
if [[ "$BINARY" == *.app/Contents/MacOS/* ]] && [ -x /usr/libexec/PlistBuddy ]; then
    app_root="$(dirname "$(dirname "$(dirname "$BINARY")")")"
    plist="$app_root/Contents/Info.plist"
    bundle_id=$(/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" "$plist" 2>/dev/null || true)
    bundle_name=$(/usr/libexec/PlistBuddy -c "Print :CFBundleName" "$plist" 2>/dev/null || true)
    bundle_version=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$plist" 2>/dev/null || true)
    binary_version=$(sed -n 's/.* v//p' "$LOG" | tail -1)
    if [ "$bundle_id" = "ai.nrl.anylearning" ] && \
       [ "$bundle_name" = "AnyLearning" ] && \
       [ "$bundle_version" = "$binary_version" ]; then
        pass "macOS metadata ($bundle_id, $bundle_name $bundle_version)"
    else
        fail "macOS metadata is '$bundle_id', '$bundle_name', '$bundle_version'; expected ai.nrl.anylearning, AnyLearning, $binary_version"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 2. It serves. --server runs the API without the desktop window, so this works
#    on a headless runner.
# ---------------------------------------------------------------------------
# ANYLEARNING_DEVELOPMENT makes verify_token return early, which is what lets
# this script call the authenticated routes without the per-window token.
# Deliberately not --development: that flag switches uvicorn to reload mode,
# which re-imports "anylearning.app:create_app" in a subprocess and cannot work
# inside a frozen binary. The env var only affects the token check.
# A data root of its own: the migration check below has to look at a database
# this build created, not at whatever is in the tester's home directory.
SMOKE_DATA_ROOT="$(mktemp -d)"
ANYLEARNING_DEVELOPMENT=TRUE ANYLEARNING_DATA_ROOT="$SMOKE_DATA_ROOT" \
    "$BINARY" --server --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 90); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        fail "server exited early"
        tail -30 "$LOG" >&2
        exit 1
    fi
    if curl -fsS "${BASE}/openapi.json" -o /dev/null 2>/dev/null; then
        ready=1
        break
    fi
    sleep 2
done

if [ "$ready" != 1 ]; then
    fail "server did not answer on ${BASE} within 180s"
    tail -30 "$LOG" >&2
    exit 1
fi
pass "server responds on ${BASE}"

# ---------------------------------------------------------------------------
# 3. The routes the app is made of. Each is a separate check so a failure names
#    what broke rather than just "the app is down".
# ---------------------------------------------------------------------------
code() { curl -fsS -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || echo 000; }

# The project registry lives in the main database; reaching it proves the
# two-tier SQLite setup initialised.
if [ "$(code "${BASE}/api/projects")" = 200 ]; then
    pass "GET /api/projects"
else
    fail "GET /api/projects returned $(code "${BASE}/api/projects")"
fi

# Model variants come from config.py, and the UI cannot render without them.
if [ "$(code "${BASE}/api/model-variants")" = 200 ]; then
    pass "GET /api/model-variants"
else
    fail "GET /api/model-variants returned $(code "${BASE}/api/model-variants")"
fi

# The frontend is extracted from the bundle at startup; serving index.html
# proves --include-data-dir actually carried it.
body="$(curl -fsS "${BASE}/" 2>/dev/null | head -c 400)"
if printf '%s' "$body" | grep -qi "<!doctype html\|<html"; then
    pass "GET / serves the frontend"
else
    fail "GET / did not return HTML (got: $(printf '%s' "$body" | head -c 80))"
fi

# ---------------------------------------------------------------------------
# 4. The heavy imports, in-process. These are the ones a packaging mistake
#    breaks: torch and torchvision.ops (which is what the excluded
#    torch._dynamo took down), plus every trainer the UI can select.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The migrations have to be *in* the build, not merely in the repository.
#
# Alembic reads env.py and its revisions from disk by path, and Nuitka's
# --include-data-dir silently skips .py files -- so a build once shipped a
# migrations folder holding only a README. Nothing failed: MigrationManager
# catches its own errors and logs them, so every database went unstamped and
# the next schema change would have broken every install. A stamped database is
# the only proof that the files arrived.
# ---------------------------------------------------------------------------
stamp=$(sqlite3 "${SMOKE_DATA_ROOT}/anylearning.db" \
    "select version_num from alembic_version" 2>/dev/null || echo "")
if [ -n "$stamp" ]; then
    pass "migrations ran (database stamped ${stamp})"
elif ! command -v sqlite3 > /dev/null; then
    echo "  skip  migration check (sqlite3 not installed)"
else
    fail "the database this build created has no alembic stamp -- the revisions are missing from the build"
fi

if [ "$(code "${BASE}/api/health/imports")" = 200 ]; then
    pass "GET /api/health/imports"
elif [ "$(code "${BASE}/api/health/imports")" = 404 ]; then
    echo "  skip  /api/health/imports (endpoint not present in this build)"
else
    fail "GET /api/health/imports returned $(code "${BASE}/api/health/imports")"
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "PASS: packaged build is functional"
    exit 0
fi
echo "FAIL: $failures check(s) failed" >&2
tail -40 "$LOG" >&2
exit 1
