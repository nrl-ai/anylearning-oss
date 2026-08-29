# AnyLearning website

This directory contains the public AnyLearning documentation website. It is
part of the main AnyLearning repository and is deployed to Vercel from the
`website/` project root.

## Local development

Requirements: Node.js 22 and pnpm 10.30.3.

```shell
corepack enable
pnpm install --frozen-lockfile
pnpm dev
```

Open <http://localhost:3000>. Run `pnpm build` before submitting website
changes; the repository-wide pre-commit hooks handle formatting and source
linting.

## Product screenshots and videos

Published UI media is generated from the real app, not edited by hand. Use a
dedicated copy of the demo data because model-inference captures and the
auto-labelling recording can update project state.

Requirements:

- the Python development environment from the repository root;
- the frontend dependencies in `frontend/`;
- Chrome or Chromium, `ffmpeg`, and `ffprobe`; and
- demo projects with finished models for every task being captured.

Start the backend and app frontend in separate terminals:

```shell
ANYLEARNING_DATA_ROOT=/path/to/capture-data \
  .venv/bin/python -m anylearning.app --port 5678 --development

cd frontend
corepack pnpm dev
```

Then regenerate and verify the computer-vision guides, hero tour, and
auto-labelling recordings:

```shell
cd website
corepack pnpm capture:docs
corepack pnpm capture:auto-labeling:probe
corepack pnpm capture:auto-labeling
corepack pnpm capture:social
corepack pnpm verify:media
```

The probe writes a temporary screenshot and control inventory without changing
published media. Run it first after UI changes; if a visible label changed, fix
the recorder selector before replacing the videos.

`capture:social` expects the website development server at
<http://localhost:3000> and writes the light-theme landing-page preview used by
Open Graph and X cards.

Structured-data screenshots use a separate deterministic data root. Restart
the backend with that capture dataset, leave the frontend running, then run:

```shell
cd website
corepack pnpm capture:structured
corepack pnpm verify:media
```

Capture scripts force a consistent viewport and theme, remove development-only
chrome, and replace files in `public/` in place. Review every generated image
and video before committing; a successful command proves the file decodes, not
that a tooltip or dialog is visually correct.

## Deployment

Vercel is connected to `nrl-ai/anylearning-oss` and uses `website/` as its root
directory. See [DEPLOYMENT.md](DEPLOYMENT.md) for the production configuration.
