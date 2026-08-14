import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const CALIBRATION_MS = 1000;
const FRAME_INTERVAL_MS = 70; // ~14 fps: enough for the demo without hammering the UI thread.

const TUNING = {
  gazeGain: 3.25,
  browHeightGain: 6.0,
  browTiltGain: 3.5,
  mouthOpenRange: 0.035,
  mouthWidthRange: 0.30,
  mouthRoundnessRange: 0.06,
  mouthCurvatureRange: 0.15,
};

const ui = {
  video: document.getElementById("demoVideo"),
  overlay: document.getElementById("trackingOverlay"),
  start: document.getElementById("startTracking"),
  retry: document.getElementById("retryTracking"),
  stop: document.getElementById("stopTracking"),
  startPanel: document.getElementById("trackingStart"),
  calibration: document.getElementById("trackingCalibration"),
  calibrationFill: document.getElementById("calibrationFill"),
  status: document.getElementById("trackingStatus"),
  dot: document.getElementById("trackingDot"),
  signalStrip: document.getElementById("signalStrip"),
  error: document.getElementById("trackingError"),
  errorText: document.getElementById("trackingErrorText"),
};

const signalEls = {
  eyes: document.querySelector('[data-signal="eyes"]'),
  brows: document.querySelector('[data-signal="brows"]'),
  gaze: document.querySelector('[data-signal="gaze"]'),
  mouth: document.querySelector('[data-signal="mouth"]'),
};

const processingCanvas = document.createElement("canvas");
const processingContext = processingCanvas.getContext("2d", {
  alpha: false,
  willReadFrequently: true,
});

const overlayContext = ui.overlay.getContext("2d");

let faceLandmarker = null;
let stream = null;
let running = false;
let rafId = 0;
let lastInferenceAt = 0;

let baseline = null;
let calibrationStartedAt = null;
let calibrationSamples = [];

let state = neutralState();

ui.start.addEventListener("click", start);
ui.retry.addEventListener("click", start);
ui.stop.addEventListener("click", stop);
window.addEventListener("pagehide", stop);

async function start() {
  if (running) return;

  hideError();
  ui.start.disabled = true;
  ui.retry.disabled = true;
  setStatus("LOADING TRACKER", false);

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access is not available in this browser.");
    }

    await ensureLandmarker();

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
      audio: false,
    });

    ui.video.srcObject = stream;
    await ui.video.play();
    await waitForVideo(ui.video);

    sizeCanvases();
    resetTracking();

    running = true;
    ui.startPanel.hidden = true;
    ui.stop.hidden = false;
    ui.signalStrip.hidden = false;
    ui.video.classList.add("live");

    setStatus("FINDING FACE", true);
    rafId = requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    stopTracks();
    showError(readableError(error));
    setStatus("UNAVAILABLE", false);
  } finally {
    ui.start.disabled = false;
    ui.retry.disabled = false;
  }
}

async function ensureLandmarker() {
  if (faceLandmarker) return;

  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "CPU",
    },
    runningMode: "IMAGE",
    numFaces: 1,
    minFaceDetectionConfidence: 0.55,
    minFacePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

function loop(now) {
  if (!running) return;
  rafId = requestAnimationFrame(loop);

  if (now - lastInferenceAt < FRAME_INTERVAL_MS) return;
  if (ui.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  if (!ui.video.videoWidth || !ui.video.videoHeight) return;

  lastInferenceAt = now;

  try {
    sizeCanvases();

    processingContext.drawImage(
      ui.video,
      0,
      0,
      processingCanvas.width,
      processingCanvas.height
    );

    // Deliberately use IMAGE mode here. We do not need MediaPipe's temporal
    // video graph for a portfolio demo; our own calibration/smoothing sits above it.
    const result = faceLandmarker.detect(processingCanvas);
    const landmarks = result?.faceLandmarks?.[0];

    clearOverlay();

    if (!landmarks) {
      handleFaceLost();
      return;
    }

    drawFeatureOverlay(landmarks);

    const raw = measureLandmarks(landmarks, performance.now() / 1000);

    if (!baseline) {
      collectCalibration(raw, now);
      return;
    }

    const target = normalise(raw, baseline);
    state = smoothState(state, target);

    updateSignals(state);
    ui.calibration.hidden = true;
    setStatus("TRACKING", true);
  } catch (error) {
    console.error("Face Landmarker inference failed:", error);
    failRuntime(error);
  }
}

function collectCalibration(raw, now) {
  if (calibrationStartedAt === null) {
    calibrationStartedAt = now;
    calibrationSamples = [];
    ui.calibration.hidden = false;
    ui.calibrationFill.style.width = "0%";
    setStatus("CALIBRATING", true);
  }

  calibrationSamples.push(raw);

  const elapsed = now - calibrationStartedAt;
  const progress = clamp(elapsed / CALIBRATION_MS, 0, 1);
  ui.calibrationFill.style.width = `${progress * 100}%`;

  if (elapsed < CALIBRATION_MS) return;

  baseline = meanMeasurements(calibrationSamples);
  ui.calibration.hidden = true;

  state = {
    ...neutralState(),
    trackingValid: true,
    measuredAt: raw.measuredAt,
  };

  updateSignals(state);
  setStatus("TRACKING", true);
}

function handleFaceLost() {
  if (!baseline) {
    calibrationStartedAt = null;
    calibrationSamples = [];
    ui.calibration.hidden = true;
    ui.calibrationFill.style.width = "0%";
    setStatus("FINDING FACE", true);
    return;
  }

  state = smoothState(state, neutralState());
  updateSignals(state);
  setStatus("FACE LOST", false);
}

function failRuntime(error) {
  running = false;
  cancelAnimationFrame(rafId);
  rafId = 0;

  stopTracks();

  ui.video.pause();
  ui.video.srcObject = null;
  ui.video.classList.remove("live");

  clearOverlay();
  ui.stop.hidden = true;
  ui.signalStrip.hidden = true;
  ui.calibration.hidden = true;

  showError(
    `Face tracking failed: ${error?.message || String(error)}`
  );
  setStatus("TRACKER ERROR", false);
}

function stop() {
  running = false;
  cancelAnimationFrame(rafId);
  rafId = 0;

  stopTracks();

  ui.video.pause();
  ui.video.srcObject = null;
  ui.video.classList.remove("live");

  clearOverlay();
  resetTracking();

  ui.startPanel.hidden = false;
  ui.stop.hidden = true;
  ui.signalStrip.hidden = true;
  ui.calibration.hidden = true;

  setStatus("CAMERA OFF", false);
}

function stopTracks() {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    track.stop();
  }
  stream = null;
}

function resetTracking() {
  baseline = null;
  calibrationStartedAt = null;
  calibrationSamples = [];
  state = neutralState();
  lastInferenceAt = 0;
  updateSignals(state);
  hideError();
}

function sizeCanvases() {
  const width = ui.video.videoWidth;
  const height = ui.video.videoHeight;

  if (!width || !height) return;

  if (
    processingCanvas.width !== width ||
    processingCanvas.height !== height
  ) {
    processingCanvas.width = width;
    processingCanvas.height = height;
  }

  if (
    ui.overlay.width !== width ||
    ui.overlay.height !== height
  ) {
    ui.overlay.width = width;
    ui.overlay.height = height;
  }
}

function clearOverlay() {
  overlayContext.clearRect(
    0,
    0,
    ui.overlay.width,
    ui.overlay.height
  );
}

function drawFeatureOverlay(points) {
  const w = ui.overlay.width;
  const h = ui.overlay.height;

  overlayContext.save();
  overlayContext.lineWidth = Math.max(2, w / 600);
  overlayContext.strokeStyle = "#ff6b5e";
  overlayContext.fillStyle = "#ff6b5e";
  overlayContext.lineCap = "round";
  overlayContext.lineJoin = "round";
  overlayContext.shadowColor = "rgba(255,107,94,.45)";
  overlayContext.shadowBlur = Math.max(4, w / 160);

  // Deliberately only show control-relevant features, not a full face mesh.
  drawLoop(points, [33, 160, 158, 133, 153, 144], w, h);
  drawLoop(points, [362, 385, 387, 263, 373, 380], w, h);

  drawLine(points, [70, 63, 105, 66, 107], w, h);
  drawLine(points, [336, 296, 334, 293, 300], w, h);

  drawLoop(
    points,
    [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91],
    w,
    h
  );

  if (points.length > 477) {
    drawIris(points, [468, 469, 470, 471, 472], w, h);
    drawIris(points, [473, 474, 475, 476, 477], w, h);
  }

  overlayContext.restore();
}

function drawLoop(points, indices, w, h) {
  overlayContext.beginPath();

  indices.forEach((index, position) => {
    const point = points[index];
    const x = point.x * w;
    const y = point.y * h;

    if (position === 0) {
      overlayContext.moveTo(x, y);
    } else {
      overlayContext.lineTo(x, y);
    }
  });

  overlayContext.closePath();
  overlayContext.stroke();
}

function drawLine(points, indices, w, h) {
  overlayContext.beginPath();

  indices.forEach((index, position) => {
    const point = points[index];
    const x = point.x * w;
    const y = point.y * h;

    if (position === 0) {
      overlayContext.moveTo(x, y);
    } else {
      overlayContext.lineTo(x, y);
    }
  });

  overlayContext.stroke();
}

function drawIris(points, indices, w, h) {
  const iris = averagePoint(points, indices);
  const radius = Math.max(3, w / 220);

  overlayContext.beginPath();
  overlayContext.arc(
    iris.x * w,
    iris.y * h,
    radius,
    0,
    Math.PI * 2
  );
  overlayContext.fill();
}

function updateSignals(current) {
  const eyeActivity =
    1 - Math.min(current.leftEyeOpen, current.rightEyeOpen);

  const browActivity = Math.max(
    Math.abs(current.leftBrowHeight),
    Math.abs(current.rightBrowHeight),
    Math.abs(current.leftBrowTilt),
    Math.abs(current.rightBrowTilt)
  );

  const gazeActivity = Math.max(
    Math.abs(current.gazeX),
    Math.abs(current.gazeY)
  );

  const mouthActivity = Math.max(
    current.mouthOpen,
    Math.abs(current.mouthWidth - 0.5) * 2,
    Math.abs(current.mouthCurvature)
  );

  setSignal("eyes", eyeActivity > 0.25);
  setSignal("brows", browActivity > 0.18);
  setSignal("gaze", gazeActivity > 0.15);
  setSignal("mouth", mouthActivity > 0.18);
}

function setSignal(name, active) {
  signalEls[name]?.classList.toggle("active", active);
}

function setStatus(text, live) {
  ui.status.textContent = text;
  ui.dot.classList.toggle("live", live);
}

function showError(message) {
  ui.errorText.textContent = message;
  ui.error.hidden = false;
}

function hideError() {
  ui.error.hidden = true;
  ui.errorText.textContent = "";
}

function readableError(error) {
  if (error?.name === "NotAllowedError") {
    return "Camera permission was denied.";
  }

  if (error?.name === "NotFoundError") {
    return "No camera was found on this device.";
  }

  return error?.message || String(error);
}

function waitForVideo(video) {
  if (
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.videoWidth > 0 &&
    video.videoHeight > 0
  ) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(
        new Error("Camera opened but did not produce a usable video frame.")
      );
    }, 8000);

    const check = () => {
      if (
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        video.videoWidth > 0 &&
        video.videoHeight > 0
      ) {
        cleanup();
        resolve();
      }
    };

    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", check);
      video.removeEventListener("canplay", check);
      video.removeEventListener("resize", check);
    };

    video.addEventListener("loadeddata", check);
    video.addEventListener("canplay", check);
    video.addEventListener("resize", check);
    check();
  });
}

/* Same generic measurements used by the standalone tracker. */

function measureLandmarks(points, measuredAt) {
  const faceWidth = distance(points[234], points[454]);

  const leftEyeOpen = ratio(
    distance(points[159], points[145]),
    distance(points[33], points[133])
  );

  const rightEyeOpen = ratio(
    distance(points[386], points[374]),
    distance(points[362], points[263])
  );

  const mouthWidth =
    average(
      distance(points[61], points[291]),
      distance(points[78], points[308])
    ) / Math.max(faceWidth, 1e-9);

  const mouthOpen =
    ratio(distance(points[13], points[14]), faceWidth);

  const mouthRoundnessRatio =
    mouthOpen / Math.max(mouthWidth, 1e-9);

  const mouthCompression =
    clamp(1 - mouthOpen * 18, 0, 1);

  const leftBrowHeight = browHeight(
    points[70],
    points[105],
    points[33],
    points[133],
    faceWidth
  );

  const rightBrowHeight = browHeight(
    points[300],
    points[334],
    points[362],
    points[263],
    faceWidth
  );

  const leftBrowTilt = browTilt(
    points[70],
    points[105],
    points[33],
    points[133]
  );

  const rightBrowTilt = browTilt(
    points[300],
    points[334],
    points[362],
    points[263]
  );

  let gazeX = 0;
  let gazeY = 0;

  if (points.length > 477) {
    const leftIris =
      averagePoint(points, [468,469,470,471,472]);

    const rightIris =
      averagePoint(points, [473,474,475,476,477]);

    gazeX = average(
      relativeInterval(
        leftIris.x,
        points[33].x,
        points[133].x
      ),
      relativeInterval(
        rightIris.x,
        points[362].x,
        points[263].x
      )
    );

    gazeY = average(
      relativeInterval(
        leftIris.y,
        points[159].y,
        points[145].y
      ),
      relativeInterval(
        rightIris.y,
        points[386].y,
        points[374].y
      )
    );
  }

  const cornersY =
    (points[61].y + points[291].y) / 2;

  const mouthCurvature = clamp(
    (points[13].y - cornersY) /
      Math.max(faceWidth * 0.03, 1e-9),
    -1,
    1
  );

  return {
    leftEyeOpen,
    rightEyeOpen,
    gazeX,
    gazeY,
    leftBrowHeight,
    rightBrowHeight,
    leftBrowTilt,
    rightBrowTilt,
    mouthOpen,
    mouthWidth,
    mouthRoundnessRatio,
    mouthCompression,
    mouthCurvature,
    measuredAt,
  };
}

function normalise(raw, base) {
  const leftEyeOpen = clamp(
    raw.leftEyeOpen /
      Math.max(base.leftEyeOpen, 1e-9),
    0,
    1
  );

  const rightEyeOpen = clamp(
    raw.rightEyeOpen /
      Math.max(base.rightEyeOpen, 1e-9),
    0,
    1
  );

  const mouthOpen = clamp(
    (raw.mouthOpen - base.mouthOpen) /
      TUNING.mouthOpenRange,
    0,
    1
  );

  const stretch =
    (raw.mouthWidth /
      Math.max(base.mouthWidth, 1e-9) -
      1) /
    TUNING.mouthWidthRange;

  const mouthWidth =
    clamp(0.5 + stretch * 0.5, 0, 1);

  const mouthRoundness = clamp(
    (raw.mouthRoundnessRatio -
      base.mouthRoundnessRatio) /
      TUNING.mouthRoundnessRange,
    0,
    1
  );

  const mouthCompression = clamp(
    1 -
      raw.mouthOpen /
        Math.max(base.mouthOpen, 1e-9),
    0,
    1
  );

  const mouthCurvature = clamp(
    (raw.mouthCurvature -
      base.mouthCurvature) /
      TUNING.mouthCurvatureRange,
    -1,
    1
  );

  return {
    leftEyeOpen,
    rightEyeOpen,

    gazeX: deadZone(
      clamp(
        (raw.gazeX - base.gazeX) *
          TUNING.gazeGain,
        -1,
        1
      ),
      0.04
    ),

    gazeY: deadZone(
      clamp(
        (raw.gazeY - base.gazeY) *
          TUNING.gazeGain,
        -1,
        1
      ),
      0.04
    ),

    leftBrowHeight: deadZone(
      clamp(
        ((raw.leftBrowHeight -
          base.leftBrowHeight) /
          0.03) *
          TUNING.browHeightGain,
        -1,
        1
      ),
      0.06
    ),

    rightBrowHeight: deadZone(
      clamp(
        ((raw.rightBrowHeight -
          base.rightBrowHeight) /
          0.03) *
          TUNING.browHeightGain,
        -1,
        1
      ),
      0.06
    ),

    leftBrowTilt: deadZone(
      clamp(
        (raw.leftBrowTilt -
          base.leftBrowTilt) *
          TUNING.browTiltGain,
        -1,
        1
      ),
      0.06
    ),

    rightBrowTilt: deadZone(
      clamp(
        (raw.rightBrowTilt -
          base.rightBrowTilt) *
          TUNING.browTiltGain,
        -1,
        1
      ),
      0.06
    ),

    mouthOpen,
    mouthWidth,
    mouthRoundness,
    mouthCompression,
    mouthCurvature,

    trackingValid: true,
    measuredAt: raw.measuredAt,
  };
}

function smoothState(previous, target) {
  const blend = (a, b, alpha) =>
    a + (b - a) * alpha;

  const eyeBlend = (a, b) => {
    if (b < 0.35) return b;
    return blend(a, b, b > a ? 0.85 : 0.8);
  };

  return {
    leftEyeOpen:
      eyeBlend(
        previous.leftEyeOpen,
        target.leftEyeOpen
      ),

    rightEyeOpen:
      eyeBlend(
        previous.rightEyeOpen,
        target.rightEyeOpen
      ),

    gazeX:
      blend(previous.gazeX, target.gazeX, 0.35),

    gazeY:
      blend(previous.gazeY, target.gazeY, 0.35),

    leftBrowHeight:
      blend(
        previous.leftBrowHeight,
        target.leftBrowHeight,
        0.7
      ),

    rightBrowHeight:
      blend(
        previous.rightBrowHeight,
        target.rightBrowHeight,
        0.7
      ),

    leftBrowTilt:
      blend(
        previous.leftBrowTilt,
        target.leftBrowTilt,
        0.7
      ),

    rightBrowTilt:
      blend(
        previous.rightBrowTilt,
        target.rightBrowTilt,
        0.7
      ),

    mouthOpen:
      blend(
        previous.mouthOpen,
        target.mouthOpen,
        0.45
      ),

    mouthWidth:
      blend(
        previous.mouthWidth,
        target.mouthWidth,
        0.45
      ),

    mouthRoundness:
      blend(
        previous.mouthRoundness,
        target.mouthRoundness,
        0.45
      ),

    mouthCompression:
      blend(
        previous.mouthCompression,
        target.mouthCompression,
        0.45
      ),

    mouthCurvature:
      blend(
        previous.mouthCurvature,
        target.mouthCurvature,
        0.45
      ),

    trackingValid:
      target.trackingValid,

    measuredAt:
      target.measuredAt,
  };
}

function meanMeasurements(samples) {
  const keys = [
    "leftEyeOpen",
    "rightEyeOpen",
    "gazeX",
    "gazeY",
    "leftBrowHeight",
    "rightBrowHeight",
    "leftBrowTilt",
    "rightBrowTilt",
    "mouthOpen",
    "mouthWidth",
    "mouthRoundnessRatio",
    "mouthCompression",
    "mouthCurvature",
  ];

  const result = {};

  for (const key of keys) {
    result[key] =
      samples.reduce(
        (sum, sample) => sum + sample[key],
        0
      ) / samples.length;
  }

  return result;
}

function neutralState() {
  return {
    leftEyeOpen: 0.75,
    rightEyeOpen: 0.75,
    gazeX: 0,
    gazeY: 0,
    leftBrowHeight: 0,
    rightBrowHeight: 0,
    leftBrowTilt: 0,
    rightBrowTilt: 0,
    mouthOpen: 0,
    mouthWidth: 0.5,
    mouthRoundness: 0,
    mouthCompression: 0,
    mouthCurvature: 0,
    trackingValid: false,
    measuredAt: 0,
  };
}

function distance(a, b) {
  return Math.hypot(
    b.x - a.x,
    b.y - a.y
  );
}

function ratio(a, b) {
  return Math.abs(b) < 1e-9
    ? 0
    : a / b;
}

function average(...values) {
  return (
    values.reduce(
      (sum, value) => sum + value,
      0
    ) / values.length
  );
}

function averagePoint(points, indices) {
  return {
    x: average(
      ...indices.map(
        (index) => points[index].x
      )
    ),
    y: average(
      ...indices.map(
        (index) => points[index].y
      )
    ),
  };
}

function relativeInterval(
  value,
  first,
  second
) {
  const low = Math.min(first, second);
  const high = Math.max(first, second);
  const span = high - low;

  if (span < 1e-9) return 0;

  return clamp(
    ((value - low) /
      span -
      0.5) *
      2,
    -1,
    1
  );
}

function browHeight(
  browA,
  browB,
  eyeA,
  eyeB,
  faceWidth
) {
  const browCentre = {
    x: (browA.x + browB.x) / 2,
    y: (browA.y + browB.y) / 2,
  };

  const dx = eyeB.x - eyeA.x;
  const dy = eyeB.y - eyeA.y;
  const length = Math.hypot(dx, dy);

  if (length < 1e-9) return 0;

  const perpendicular =
    -(
      dx *
        (browCentre.y - eyeA.y) -
      dy *
        (browCentre.x - eyeA.x)
    ) /
    length;

  return ratio(
    perpendicular,
    faceWidth
  );
}

function browTilt(
  browA,
  browB,
  eyeA,
  eyeB
) {
  const slope = (a, b) => {
    const dx = b.x - a.x;

    return Math.abs(dx) < 1e-9
      ? 0
      : (b.y - a.y) / dx;
  };

  return clamp(
    (
      slope(browA, browB) -
      slope(eyeA, eyeB)
    ) /
      0.25,
    -1,
    1
  );
}

function deadZone(value, threshold) {
  return Math.abs(value) < threshold
    ? 0
    : value;
}

function clamp(value, low, high) {
  return Math.max(
    low,
    Math.min(high, value)
  );
}

updateSignals(state);
