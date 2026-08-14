# Interactive Prop Lab — v18 stable static build

This intentionally returns to the last browser architecture that was actually
confirmed working:

- plain static HTML/CSS/JS;
- no Vite;
- no npm runtime import;
- no Web Worker;
- no worker handshake;
- no worker chunk;
- no inline-worker transform;
- MediaPipe Tasks Vision 0.10.32;
- main-thread fixed processing canvas;
- `getImageData()` -> `FaceLandmarker.detect(ImageData)`;
- approved calibration and 64x32 matrix output.

The only additions are:
- Chromium compatibility wording in the privacy/footer copy;
- a Firefox/LibreWolf guard with a clear landmark-source unavailable message;
- a simple GitHub Actions workflow that deploys the static files as-is.

## Deploy

Replace the repo contents with this folder, then:

```bash
git init
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/interactiveproplab/interactiveproplab.github.io.git

git add -A
git commit -m "Deploy stable static tracker"
git fetch origin
git push -u origin main --force-with-lease
```

GitHub Pages should remain set to **GitHub Actions**.

There is no `npm install`, `npm run build`, vendoring step, or generated `dist/`.
