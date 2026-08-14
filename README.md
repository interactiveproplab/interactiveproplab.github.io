# Interactive Prop Lab — live demo v15

v15 keeps the same visible layout, calibration, and LED matrix as v14.

The change is limited to the official MediaPipe webcam/worker path.

## What changed

The current official 2026 MediaPipe sample does two things v14 did not match
closely enough:

1. It cache-busts the copied same-origin WASM loader on worker initialization.
2. It sends one webcam `ImageBitmap`, waits for the detection result, then
   schedules the next frame.

v15 now does both.

It also uses the official sample's simple camera request:

`navigator.mediaDevices.getUserMedia({ video: true })`

instead of imposing a specific 1280×720 preference.

## Console diagnostics

The visible site stays clean, but every ~30 frames the console reports:

- webcam/video dimensions;
- transferred ImageBitmap dimensions;
- number of faces returned by Face Landmarker.

Example:

`[IPL tracker] sending frame 30 1280×720 video=1280×720`

`[IPL tracker] result 30 faces=1 bitmap=1280×720`

If detection still returns zero, those lines tell us whether the worker is
receiving a real-sized camera image or an unexpected blank/zero-sized source.

## Deploy

Same as v14:

1. Replace the repository contents with this project.
2. Settings → Pages → Source → GitHub Actions.
3. Push to `main`.
4. Let the included workflow build/deploy `dist/`.

Pinned:
- `@mediapipe/tasks-vision` 1.0.1
- Vite 8.2.1
