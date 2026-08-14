# Interactive Prop Lab — v18 static-clean

This build is intentionally plain static HTML/CSS/JS.

Critical properties:
- no Vite;
- no npm package import;
- no bare module specifier;
- no Web Worker;
- no generated asset chunks;
- MediaPipe Tasks Vision is imported from a full browser-resolvable jsDelivr URL;
- script filename is unique: `ipl-tracker-static-v18.js`;
- index references it with `?v=18-static-clean` to prevent stale browser cache;
- GitHub Actions deploys only four explicit static files.

The browser import is:

```js
import { FaceLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs";
```

## Replace the repository contents

From the unzipped folder:

```bash
git init
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/interactiveproplab/interactiveproplab.github.io.git

git add -A
git commit -m "Deploy v18 static clean"
git fetch origin
git push -u origin main --force-with-lease
```

If the remote repo previously contained Vite files, the GitHub Actions workflow still
deploys only:
- index.html
- style.css
- ipl-tracker-static-v18.js
- favicon.ico

After Actions completes, open a fresh Chromium Incognito window or hard reload.
