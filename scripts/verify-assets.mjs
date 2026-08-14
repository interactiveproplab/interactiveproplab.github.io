import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const required = [
  "public/wasm/vision_wasm_nosimd_internal.js",
  "public/wasm/vision_wasm_nosimd_internal.wasm",
  "public/models/face_landmarker.task",
];

for (const relative of required) {
  const file = path.join(root, relative);
  if (!fs.existsSync(file)) {
    throw new Error(
      `${relative} is missing. Run: npm install && npm run vendor`
    );
  }
  const size = fs.statSync(file).size;
  if (size < 1024) {
    throw new Error(`${relative} looks incomplete (${size} bytes).`);
  }
}

console.log("Verified vendored MediaPipe runtime/model assets.");
