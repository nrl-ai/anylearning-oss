#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
files=()

for file in "$@"; do
    files+=("${file#frontend/}")
done

cd "$repo_root/frontend"
pnpm exec prettier --write --ignore-unknown "${files[@]}"
