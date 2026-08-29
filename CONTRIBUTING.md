# Contributing to AnyLearning

Thank you for helping improve AnyLearning. Contributions should be focused,
tested, and safe for an offline-first desktop application.

## Development setup

Follow the platform and application setup in [README.md](README.md), then
install the repository hooks once in your clone:

```shell
python -m pip install pre-commit==4.6.2
pre-commit install --install-hooks
```

This installs both pre-commit and pre-push hooks. Pre-commit performs fast,
staged-file checks and applies safe formatting fixes. Pre-push runs the fast
Python test suite, excluding the heavyweight end-to-end and vendored NanoDet
tests.

Run every repository check at any time with:

```shell
pre-commit run --all-files
```

If a formatter changes files, review the result, stage the changes, and run the
command again. Hooks should pass without modifying files on the second run.

## Quality checks

The shared pre-commit configuration is the source of truth for local and CI
checks:

- Gitleaks scans staged changes for credentials and CI scans Git history.
- Ruff formats Python and checks syntax, imports, undefined names, and common
  bug patterns.
- Prettier formats JavaScript, TypeScript, Markdown, MDX, JSON, YAML, and CSS.
- Oxlint checks JavaScript and TypeScript correctness.
- ShellCheck validates shell scripts.
- yamllint and actionlint validate YAML and GitHub Actions workflows.
- Repository hygiene hooks catch merge markers, broken links, private keys,
  malformed configuration files, and accidentally added large files.

Run the history-level secret scan locally with:

```shell
pre-commit run gitleaks-history --hook-stage manual --all-files
```

Never silence a secret finding until you have established that it is a
non-secret test fixture. If a real credential was committed, revoke it first;
removing it from the latest revision does not remove it from Git history.

## Testing

Run the checks relevant to your change before opening a pull request:

```shell
./run_tests.sh

cd frontend
pnpm lint
pnpm format
pnpm typecheck
pnpm test

cd ../website
pnpm build
```

Add regression coverage for bug fixes and tests for new behavior. Keep tests
offline by default; network- or dataset-dependent tests belong in the explicit
end-to-end suite.

## Pull requests

1. Search existing issues and pull requests before starting.
2. Open an issue first for large features or changes to public behavior.
3. Keep each pull request focused and explain both the change and its rationale.
4. Update user-facing documentation when behavior changes.
5. Include screenshots for visible frontend or website changes.
6. Confirm that `pre-commit run --all-files` and relevant tests pass.

## Licensing and provenance

By submitting a contribution, you agree that it is licensed under the Apache
License 2.0 and that you have the right to submit it. Do not copy code whose
license is incompatible with Apache-2.0. Preserve copyright and attribution
notices for third-party code.

Model weights and datasets require separate review. Record their source,
version, license, attribution requirements, and checksum in the appropriate
notice file. Do not commit private datasets, credentials, generated application
bundles, or downloaded model weights.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
