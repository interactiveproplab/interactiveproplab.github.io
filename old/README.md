# Interactive Prop Lab — live demo v11

v11 starts from the good v7 layout. The visible HTML and CSS are unchanged.

Upload:

- `index.html`
- `style.css`
- `face-demo-v11.js`

Delete older `face-demo-v*.js` and `face-worker-v*.js`.

## Detector change

v11 uses MediaPipe Tasks Vision 0.10.32 with:

- CPU delegate;
- no-SIMD WASM runtime;
- Face Landmarker in IMAGE mode;
- 640px maximum detector width;
- roughly 12 inference updates/sec.

The critical difference is the input type.

Previous builds passed video/canvas/ImageBitmap objects into browser ML runtimes.
Those runtimes can internally convert those objects through a WebGL texture path,
even when the inference delegate itself is CPU.

v11 instead does:

1. draw the webcam frame to an ordinary 2D canvas;
2. call `getImageData()`;
3. pass the resulting `ImageData` directly to Face Landmarker.

`ImageData` is already CPU pixel memory and avoids the browser image-to-WebGL
conversion path.

The tracker also runs a blank `ImageData` inference during initialization before
camera permission is requested. If the runtime cannot consume this input path, it
fails immediately instead of hanging on FINDING FACE.
