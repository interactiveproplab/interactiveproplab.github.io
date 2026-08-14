import {
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

let faceLandmarker = null;
let processing = false;

self.addEventListener("message", async (event) => {
  const message = event.data;

  if (message?.type === "INIT") {
    try {
      if (faceLandmarker) {
        faceLandmarker.close?.();
        faceLandmarker = null;
      }

      const wasmPath = message.wasmRoot.replace(/\/$/, "");
      const vision = await FilesetResolver.forVisionTasks(
        wasmPath,
        true
      );

      // Match the current official MediaPipe worker: force the copied
      // same-origin loader to reload instead of reusing a stale cached loader.
      vision.wasmLoaderPath =
        `${vision.wasmLoaderPath}?cb=${Date.now()}`;

      const response = await fetch(message.modelUrl);
      if (!response.ok) {
        throw new Error(
          `Face model failed to load (${response.status}).`
        );
      }

      const modelBuffer = await response.arrayBuffer();

      faceLandmarker = await FaceLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetBuffer: new Uint8Array(modelBuffer),
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        }
      );

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

  if (message?.type !== "FRAME") return;

  if (processing) {
    message.bitmap?.close?.();
    return;
  }

  processing = true;

  try {
    if (!faceLandmarker) {
      throw new Error("Face Landmarker is not initialized.");
    }

    const bitmapSize = {
      width: message.bitmap.width,
      height: message.bitmap.height,
    };

    const result = faceLandmarker.detectForVideo(
      message.bitmap,
      message.timestampMs
    );

    const faceLandmarks = result?.faceLandmarks || [];
    const face = faceLandmarks[0] || null;

    self.postMessage({
      type: "RESULT",
      face,
      faceCount: faceLandmarks.length,
      bitmapSize,
    });
  } catch (error) {
    console.error("Face Landmarker inference failed:", error);
    self.postMessage({
      type: "DETECT_ERROR",
      error: error?.message || String(error),
    });
  } finally {
    message.bitmap?.close?.();
    processing = false;
  }
});
