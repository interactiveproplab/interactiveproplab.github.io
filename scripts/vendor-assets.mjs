import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const WASM_SOURCE = path.join(
  ROOT,
  "node_modules",
  "@mediapipe",
  "tasks-vision",
  "wasm"
);
const WASM_DEST = path.join(ROOT, "public", "wasm");
const MODEL_DEST_DIR = path.join(ROOT, "public", "models");
const MODEL_DEST = path.join(
  MODEL_DEST_DIR,
  "face_landmarker.task"
);

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

if (!fs.existsSync(WASM_SOURCE)) {
  throw new Error(
    "Pinned @mediapipe/tasks-vision is not installed. Run npm install first."
  );
}

fs.mkdirSync(WASM_DEST, { recursive: true });
fs.mkdirSync(MODEL_DEST_DIR, { recursive: true });

for (const oldFile of fs.readdirSync(WASM_DEST)) {
  fs.rmSync(path.join(WASM_DEST, oldFile), {
    recursive: true,
    force: true,
  });
}

const copied = [];

for (const file of fs.readdirSync(WASM_SOURCE)) {
  const source = path.join(WASM_SOURCE, file);
  const dest = path.join(WASM_DEST, file);

  if (!fs.statSync(source).isFile()) continue;

  fs.copyFileSync(source, dest);
  copied.push(file);
}

console.log(
  `Vendored ${copied.length} MediaPipe WASM files from the pinned npm package.`
);

const response = await fetch(MODEL_URL);

if (!response.ok) {
  throw new Error(
    `Face model download failed: HTTP ${response.status}`
  );
}

const modelBytes = new Uint8Array(
  await response.arrayBuffer()
);

fs.writeFileSync(MODEL_DEST, modelBytes);

console.log(
  `Vendored face_landmarker.task (${modelBytes.byteLength} bytes).`
);

console.log("");
console.log("Vendoring complete.");
console.log("Commit public/wasm and public/models to the repository.");
console.log("Normal GitHub Pages builds will not download them again.");
