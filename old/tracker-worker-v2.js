import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let landmarker = null;

self.onmessage = async (event) => {
  const message = event.data;

  if (message.type === "INIT") {
    try {
      if (!landmarker) {
        const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

        const response = await fetch(MODEL_URL);
        if (!response.ok) {
          throw new Error(`model download failed (${response.status})`);
        }

        const modelBuffer = await response.arrayBuffer();

        landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetBuffer: new Uint8Array(modelBuffer),
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.6,
          minFacePresenceConfidence: 0.6,
          minTrackingConfidence: 0.6,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });
      }

      self.postMessage({ type: "READY" });
    } catch (error) {
      self.postMessage({
        type: "ERROR",
        error: error?.message || String(error),
      });
    }
    return;
  }

  if (message.type === "FRAME") {
    const bitmap = message.bitmap;

    try {
      if (!landmarker) {
        throw new Error("tracker is not initialized");
      }

      const result = landmarker.detectForVideo(
        bitmap,
        message.timestampMs
      );

      const face = result.faceLandmarks?.[0] || null;

      self.postMessage({
        type: "RESULT",
        face,
        timestampMs: message.timestampMs,
      });
    } catch (error) {
      self.postMessage({
        type: "ERROR",
        error: error?.message || String(error),
      });
    } finally {
      bitmap?.close?.();
    }
  }
};
