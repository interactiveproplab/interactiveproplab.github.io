import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const ROOT = process.cwd();

const required = [
  "public/models/face_landmarker.task",
];

const wasmDir = path.join(ROOT, "public", "wasm");

if (!fs.existsSync(wasmDir)) {
  throw new Error(
    "Vendored MediaPipe WASM directory is missing. Run: npm install && npm run vendor"
  );
}

const wasmFiles = fs
  .readdirSync(wasmDir)
  .filter((name) => name.endsWith(".js") || name.endsWith(".wasm"));

if (wasmFiles.length < 2) {
  throw new Error(
    "Vendored MediaPipe WASM files are missing/incomplete. Run: npm run vendor"
  );
}

for (const relative of required) {
  const file = path.join(ROOT, relative);

  if (!fs.existsSync(file)) {
    throw new Error(
      `${relative} is missing. Run: npm install && npm run vendor`
    );
  }

  const size = fs.statSync(file).size;

  if (size < 1024 * 100) {
    throw new Error(
      `${relative} looks incomplete (${size} bytes). Re-run: npm run vendor`
    );
  }
}

const manifest = {};

for (const relative of [
  ...wasmFiles.map((name) => `public/wasm/${name}`),
  ...required,
]) {
  const file = path.join(ROOT, relative);
  const bytes = fs.readFileSync(file);

  manifest[relative] = {
    bytes: bytes.length,
    sha256: crypto
      .createHash("sha256")
      .update(bytes)
      .digest("hex"),
  };
}

fs.writeFileSync(
  path.join(ROOT, "assets-lock.json"),
  JSON.stringify(manifest, null, 2) + "\n"
);

console.log(
  `Verified ${Object.keys(manifest).length} vendored runtime assets.`
);
