#!/usr/bin/env node
/**
 * Regenerates the tutorial screenshots from the running app.
 *
 * Why a script and not a folder of hand-made PNGs: the tutorials carry ~43
 * step images, and every one of them goes stale the moment the UI moves. The
 * old set still showed a light-mode labelling screen with a toolbar that no
 * longer exists. This turns "the screenshots are wrong again" from a day of
 * work into one command.
 *
 *   1. start the app:  python -m anylearning.app --port 5678 --development
 *   2. start its frontend:            cd frontend && pnpm dev          (port 3021)
 *   3. node scripts/capture-docs.mjs [tutorial…] [--dry]
 *
 * Screenshots are captured at 2x against the real app and real local data, then
 * annotated and compressed. Nothing is mocked: if a screen cannot be reached —
 * usually because the local database has no trained model for that project —
 * the shot is skipped and reported rather than faked.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

import { annotate } from "./lib/annotate.mjs";
import { launch } from "./lib/browser.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(HERE, "..", "public");
const APP = process.env.APP_URL || "http://localhost:3021";

const args = process.argv.slice(2);
const dryRun = args.includes("--dry");
const shotFilter = args.find((a) => a.startsWith("--shot="))?.slice("--shot=".length);
const only = args.filter((a) => !a.startsWith("--"));

const TUTORIALS = [
  "marketing",
  "object-detection",
  "image-classification",
  "handpose-classification",
  "semantic-segmentation",
  "instance-segmentation",
  "keypoint-detection",
];

/** Hides the dev-server overlay so it never lands in a published image. */
const HIDE_DEV_CHROME = `
  const s = document.createElement('style');
  s.textContent = 'nextjs-portal,#__next-build-watcher,[data-nextjs-toast]{display:none!important}';
  document.head.appendChild(s);
  true`;

/** Forces the theme, so a stale localStorage value cannot change the look. */
const forceTheme = (theme) => `
  try { localStorage.setItem('theme', '${theme}'); } catch (e) {}
  document.documentElement.classList.toggle('dark', ${theme === "dark"});
  true`;

async function resolveProject(page, wanted) {
  const projects = await page.evaluate(`
    fetch('/api/projects').then(r => r.json()).then(list =>
      list.map(p => ({ id: p.id, name: p.name, type: p.type })))`);
  const match =
    projects?.find((p) => p.name === wanted.name) ?? projects?.find((p) => p.type === wanted.type);
  if (!match) {
    throw new Error(
      `No local project matching "${wanted.name}" (${wanted.type}).\n` +
        `Available: ${(projects || []).map((p) => `${p.name} [${p.type}]`).join(", ") || "none"}`
    );
  }
  return match;
}

async function hasModels(page, projectId) {
  const data = await page.evaluate(
    `fetch('/api/projects/${projectId}/models').then(r => r.json()).then(d => d.total_count ?? 0)`
  );
  return Number(data) > 0;
}

async function run() {
  const page = await launch();
  const report = { written: [], skipped: [] };

  try {
    await page.setViewport(1440, 900);
    await page.goto(`${APP}/projects`, 6000);
    await page.evaluate(forceTheme("dark"));

    for (const name of TUTORIALS.filter((t) => !only.length || only.includes(t))) {
      const mod = await import(`./shots/${name}.mjs`);
      const project = await resolveProject(page, mod.project);
      const modelsExist = await hasModels(page, project.id);
      console.log(
        `\n${name}  →  project #${project.id} "${project.name}"${modelsExist ? "" : "  (no trained models)"}`
      );

      for (const shot of mod.shots) {
        if (shotFilter && !shot.file.includes(shotFilter)) continue;
        if (shot.requiresModel && !modelsExist) {
          report.skipped.push({ file: shot.file, why: "project has no trained model yet" });
          console.log(`  skip  ${shot.file}  — needs a trained model`);
          continue;
        }

        const [w, h] = shot.viewport || [1440, 900];
        await page.setViewport(w, h);
        await page.goto(`${APP}${shot.path}?projectId=${project.id}`, shot.wait ?? 9000);
        await page.evaluate(HIDE_DEV_CHROME);
        await page.evaluate(forceTheme("dark"));
        await page.sleep(600);

        for (const step of shot.steps || []) {
          await page.evaluate(step.eval);
          await page.sleep(step.wait ?? 800);
        }

        const raw = await page.screenshot();
        const marked = await annotate(raw, shot.notes);
        const out = path.join(PUBLIC, shot.file);

        if (dryRun) {
          console.log(`  dry   ${shot.file}  (${(marked.length / 1024).toFixed(0)} KB)`);
        } else {
          fs.mkdirSync(path.dirname(out), { recursive: true });
          const optimised = await sharp(marked)
            .resize({ width: 1600, withoutEnlargement: true })
            .png({ compressionLevel: 9, effort: 10, palette: true, quality: 92, dither: 1 })
            .toBuffer();
          fs.writeFileSync(out, optimised);
          console.log(`  ok    ${shot.file}  (${(optimised.length / 1024).toFixed(0)} KB)`);
        }
        report.written.push(shot.file);
      }
    }
  } finally {
    await page.close();
  }

  if (report.skipped.length) {
    console.log("\nSkipped:");
    for (const s of report.skipped) console.log(`  ${s.file} — ${s.why}`);
  }
  console.log(`\n${report.written.length} captured, ${report.skipped.length} skipped.`);
}

run().catch((err) => {
  console.error("\nCapture failed:", err.message);
  process.exit(1);
});
