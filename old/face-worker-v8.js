const TASKS_VISION_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/+esm";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let faceLandmarker = null;

self.addEventListener("message", async (event) => {
  const message = event.data;

  if (message.type === "INIT") {
    try {
      if (!faceLandmarker) {
        const visionModule = await import(TASKS_VISION_URL);

        const FaceLandmarker =
          visionModule.FaceLandmarker ??
          visionModule.default?.FaceLandmarker;

        const FilesetResolver =
          visionModule.FilesetResolver ??
          visionModule.default?.FilesetResolver;

        if (!FaceLandmarker || !FilesetResolver) {
          throw new Error("MediaPipe Tasks Vision exports were not available.");
        }

        const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

        const modelResponse = await fetch(MODEL_URL);
        if (!modelResponse.ok) {
          throw new Error(
            `Face Landmarker model download failed (${modelResponse.status}).`
          );
        }

        const modelBuffer = await modelResponse.arrayBuffer();

        faceLandmarker = await FaceLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetBuffer: new Uint8Array(modelBuffer),
              delegate: "CPU",
            },
            runningMode: "IMAGE",
            numFaces: 1,
            minFaceDetectionConfidence: 0.45,
            minFacePresenceConfidence: 0.45,
            minTrackingConfidence: 0.45,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
          }
        );
      }

      self.postMessage({ type: "READY" });
    } catch (error) {
      console.error("Face Landmarker init failed:", error);
      self.postMessage({
        type: "INIT_ERROR",
        error: error?.message || String(error),
      });
    }

    return;
  }

  if (message.type !== "FRAME") return;

  const bitmap = message.bitmap;

  try {
    if (!faceLandmarker) {
      throw new Error("Face Landmarker is not initialized.");
    }

    // IMAGE mode is intentional: CPU/WASM only, no WebGL video pipeline.
    const result = faceLandmarker.detect(bitmap);
    const face = result?.faceLandmarks?.[0] || null;

    self.postMessage({
      type: "RESULT",
      face,
    });
  } catch (error) {
    console.error("Face Landmarker inference failed:", error);

    self.postMessage({
      type: "DETECT_ERROR",
      error: error?.message || String(error),
    });
  } finally {
    bitmap?.close?.();
  }
});
