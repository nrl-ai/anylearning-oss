#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PUBLIC = path.join(ROOT, "public");
const IMAGE_EXTENSIONS = new Set([".jpeg", ".jpg", ".png"]);
const SOURCE_EXTENSIONS = new Set([".js", ".jsx", ".md", ".mdx", ".mjs", ".ts", ".tsx"]);

function walk(directory, ignored = new Set()) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    if (ignored.has(entry.name)) return [];
    const fullPath = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(fullPath, ignored) : [fullPath];
  });
}

const failures = [];
const media = walk(PUBLIC).filter((file) => {
  const extension = path.extname(file).toLowerCase();
  return IMAGE_EXTENSIONS.has(extension) || extension === ".mp4";
});

for (const file of media) {
  const relative = path.relative(ROOT, file);
  try {
    if (fs.statSync(file).size === 0) throw new Error("file is empty");

    if (IMAGE_EXTENSIONS.has(path.extname(file).toLowerCase())) {
      const metadata = await sharp(file).metadata();
      if (!metadata.width || !metadata.height) throw new Error("image has no dimensions");
    } else {
      const probe = JSON.parse(
        execFileSync(
          "ffprobe",
          [
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt:format=duration",
            "-of",
            "json",
            file,
          ],
          { encoding: "utf8" }
        )
      );
      const video = probe.streams?.[0];
      if (!video?.width || !video?.height || Number(probe.format?.duration) <= 0) {
        throw new Error("video has no decodable stream or duration");
      }
      if (video.codec_name !== "h264" || video.pix_fmt !== "yuv420p") {
        throw new Error(`expected H.264/yuv420p, found ${video.codec_name}/${video.pix_fmt}`);
      }
    }
  } catch (error) {
    failures.push(`${relative}: ${error.message}`);
  }
}

const sourceFiles = walk(ROOT, new Set([".next", "node_modules", "public"])).filter((file) =>
  SOURCE_EXTENSIONS.has(path.extname(file).toLowerCase())
);
const references = new Set();
const mediaReference = /["'(]\/([^"')\s]+\.(?:jpe?g|mp4|png))/gi;

for (const file of sourceFiles) {
  const source = fs.readFileSync(file, "utf8");
  for (const match of source.matchAll(mediaReference)) references.add(match[1]);
}

for (const reference of references) {
  if (reference.startsWith("path/to/")) continue;
  if (!fs.existsSync(path.join(PUBLIC, reference)))
    failures.push(`missing referenced asset: ${reference}`);
}

for (let index = 1; index <= 5; index += 1) {
  const file = path.join(PUBLIC, "screenshots", `${index}.png`);
  const metadata = await sharp(file).metadata();
  if (metadata.width !== 1600 || metadata.height !== 1000) {
    failures.push(
      `screenshots/${index}.png: expected 1600×1000, found ${metadata.width}×${metadata.height}`
    );
  }
}

if (failures.length) {
  process.stderr.write(`Media verification failed:\n- ${failures.join("\n- ")}\n`);
  process.exit(1);
}

process.stdout.write(`Verified ${media.length} media files and ${references.size} references.\n`);
