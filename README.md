# Interactive Prop Lab — v18 main-thread ImageData tracker

This version deliberately removes the face-tracking Web Worker.

The repeated production failure:

`Last stage: attempt 3: INIT sent. Face tracker worker failed before initialization`

occurred before MediaPipe ever ran. The worker layer itself was the failure boundary.

This build returns to the detector architecture that was actually observed working:

camera
→ fixed 2D processing canvas
→ getImageData()
→ MediaPipe FaceLandmarker.detect(ImageData)
→ expression mapping
→ 64×32 protogen matrix

Key points:

- no Web Worker;
- no worker chunk;
- no worker INIT/READY handshake;
- no worker startup timeout;
- @mediapipe/tasks-vision pinned to 0.10.32;
- explicit no-SIMD WASM fileset, matching the proven build;
- WASM + model are vendored under public/;
- detector input is ImageData, avoiding the video/ImageBitmap/WebGL bridge;
- approved layout/calibration/matrix preserved;
- Firefox/LibreWolf gets the Chromium compatibility message;
- detector recovery failures keep the camera visible.

## One-time asset vendoring

Run:

```bash
npm install
npm run vendor
npm run verify-assets
```

Then commit everything:

```bash
git add -A
git commit -m "Deploy v18 main-thread tracker"
git fetch origin
git push -u origin main --force-with-lease
```

Normal GitHub Pages builds then use the committed local runtime/model assets.
