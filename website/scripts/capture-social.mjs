#!/usr/bin/env node
/** Capture the current light-theme landing page used by social previews. */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

import { launch } from "./lib/browser.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const output = path.join(here, "..", "public", "screenshot.png");
const site = process.env.SITE_URL || "http://localhost:3000";
const browser = await launch({ width: 1600, height: 900, scale: 1 });

try {
  await browser.setViewport(1600, 900);
  await browser.goto(site, 6000);
  await browser.evaluate(`
    try { localStorage.setItem('theme', 'light'); } catch (error) {}
    document.documentElement.classList.remove('dark');
    document.documentElement.style.colorScheme = 'light';
    const style = document.createElement('style');
    style.textContent = 'nextjs-portal,#__next-build-watcher,[data-nextjs-toast]{display:none!important}';
    document.head.appendChild(style);
    true
  `);
  await browser.sleep(800);

  const image = await sharp(await browser.screenshot())
    .resize(1600, 900, { fit: "cover", position: "top" })
    .png({ compressionLevel: 9, effort: 10, palette: true, quality: 94 })
    .toBuffer();
  fs.writeFileSync(output, image);
  process.stdout.write(`Captured ${output} (${Math.round(image.length / 1024)} KB)\n`);
} finally {
  await browser.close();
}
