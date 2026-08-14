import fs from "node:fs";
import path from "node:path";

const publicDir = path.resolve("public");
const wasmSource = path.resolve(
  "node_modules/@mediapipe/tasks-vision/wasm"
);
const wasmDest = path.join(publicDir, "wasm");
const modelDestDir = path.join(publicDir, "models");
const modelDest = path.join(
  modelDestDir,
  "face_landmarker.task"
);

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

fs.mkdirSync(wasmDest, { recursive: true });
fs.mkdirSync(modelDestDir, { recursive: true });

if (!fs.existsSync(wasmSource)) {
  throw new Error(
    `Missing MediaPipe WASM directory: ${wasmSource}`
  );
}

for (const file of fs.readdirSync(wasmSource)) {
  fs.copyFileSync(
    path.join(wasmSource, file),
    path.join(wasmDest, file)
  );
}

const response = await fetch(MODEL_URL);
if (!response.ok) {
  throw new Error(
    `Could not download face model (${response.status}).`
  );
}

const modelBytes = new Uint8Array(
  await response.arrayBuffer()
);
fs.writeFileSync(modelDest, modelBytes);

console.log(
  `Prepared MediaPipe WASM + face model (${modelBytes.length} bytes).`
);
