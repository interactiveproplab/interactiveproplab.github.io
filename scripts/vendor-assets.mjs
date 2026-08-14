import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const source = path.join(
  ROOT, "node_modules", "@mediapipe", "tasks-vision", "wasm"
);
const wasmDest = path.join(ROOT, "public", "wasm");
const modelDir = path.join(ROOT, "public", "models");
const modelDest = path.join(modelDir, "face_landmarker.task");

const modelUrl =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

if (!fs.existsSync(source)) {
  throw new Error(
    "Run npm install first. The pinned @mediapipe/tasks-vision package is missing."
  );
}

fs.mkdirSync(wasmDest, { recursive: true });
fs.mkdirSync(modelDir, { recursive: true });

for (const entry of fs.readdirSync(wasmDest)) {
  fs.rmSync(path.join(wasmDest, entry), { recursive: true, force: true });
}

for (const name of fs.readdirSync(source)) {
  const from = path.join(source, name);
  const to = path.join(wasmDest, name);
  if (fs.statSync(from).isFile()) fs.copyFileSync(from, to);
}

const response = await fetch(modelUrl);
if (!response.ok) {
  throw new Error(`Face model download failed: HTTP ${response.status}`);
}
fs.writeFileSync(
  modelDest,
  new Uint8Array(await response.arrayBuffer())
);

console.log("Vendored pinned MediaPipe 0.10.32 WASM + face_landmarker.task.");
console.log("Commit public/wasm and public/models to the repository.");
