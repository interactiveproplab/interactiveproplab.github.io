# Interactive Prop Lab — v18 vendored runtime

This build combines:

- the working Chromium tracker architecture;
- three fresh-worker startup attempts;
- 10 second timeout per attempt;
- explicit "Facial landmarking source unavailable" errors;
- LibreWolf/Firefox compatibility handling;
- the Chromium compatibility footer;
- a permanently vendored MediaPipe WASM runtime and Face Landmarker model.

## One-time vendoring

The runtime assets are intended to be committed to this repository so normal
deployments never download or copy them again.

On CachyOS / Arch:

```bash
npm install
npm run vendor
npm run verify-assets
```

That creates:

```text
public/
├── wasm/
│   └── <the exact WASM files from @mediapipe/tasks-vision 1.0.0>
└── models/
    └── face_landmarker.task
```

Then commit them:

```bash
git add -A
git commit -m "Vendor MediaPipe runtime assets"
git push
```

After that, GitHub Actions only verifies and deploys the committed files. It does
not fetch the model or copy WASM from node_modules during deployment.

## Why this is useful

It removes build/deploy asset drift:

- package version is pinned;
- WASM bytes are committed;
- model bytes are committed;
- `assets-lock.json` records size + SHA-256 hashes;
- a build fails if runtime files are absent or incomplete.

This does not guarantee MediaPipe itself can never hang during initialization,
which is why the fresh-worker retry logic remains.

## Browser support

For best compatibility, use a Chromium-based browser.

Firefox/LibreWolf receive a clear landmark-source unavailable message instead of
entering the MediaPipe runtime path.

## Deploy

GitHub Pages should use **GitHub Actions** as its source.

Once `public/wasm` and `public/models` are committed, ordinary deployment is:

```bash
git add -A
git commit -m "Update Interactive Prop Lab"
git push
```


## Inline-worker hardening

The face-tracking worker is now imported using Vite's inline worker mode:

```js
import FaceWorker from "./face-worker.js?worker&inline";
```

and constructed with:

```js
const worker = new FaceWorker();
```

This deliberately removes the separately emitted/fetched hashed worker asset
(`assets/face-worker-....js`) from the startup path.

The observed failure:

```text
Last stage: attempt 3: INIT sent.
Last error: Face tracker worker failed to load.
```

means the worker script failed before executing enough code to post its first
`INIT_STAGE` message. It therefore occurred before WASM resolution, model loading,
FaceLandmarker creation, camera-frame inference, or calibration.

Vendored WASM/model assets remain unchanged and local to the site.
