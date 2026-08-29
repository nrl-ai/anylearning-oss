#!/bin/sh
set -e

# Build first, swap afterwards. The previous version removed
# anylearning/frontend-dist before running pnpm, so any build failure -- a bad
# dependency install, a type error, an out-of-memory -- left the tree with no
# frontend at all, and a build_app.sh running at the same time would package an
# app without a UI.
cd frontend && pnpm install && pnpm run build && cd ..

# pnpm run build writes frontend/out. Only once that exists do we replace the
# packaged copy.
if [ ! -d frontend/out ]; then
    echo "frontend build produced no output directory (frontend/out)" >&2
    exit 1
fi

rm -rf anylearning/frontend-dist
mv frontend/out anylearning/frontend-dist
touch anylearning/frontend-dist/__init__.py
