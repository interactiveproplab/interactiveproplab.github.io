import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const CALIBRATION_SECONDS = 1.0;
const INFERENCE_INTERVAL_MS = 33;

const TUNING = {
  gazeGain: 3.25,
  browHeightGain: 6.0,
  browTiltGain: 3.5,
  mouthOpenRange: 0.035,
  mouthWidthRange: 0.30,
  mouthRoundnessRange: 0.06,
  mouthCurvatureRange: 0.15,
};

const SMOOTHING = {
  eyes: 0.55,
  gaze: 0.35,
  brows: 0.70,
  mouth: 0.45,
};

const ui = {
  video: document.getElementById("demoVideo"),
  start: document.getElementById("startDemo"),
  stop: document.getElementById("stopDemo"),
  idle: document.getElementById("cameraIdle"),
  calibration: document.getElementById("calibrationOverlay"),
  calibrationFill: document.getElementById("calibrationFill"),
  status: document.getElementById("demoStatus"),
  tracking: document.getElementById("trackingLabel"),
  message: document.getElementById("demoMessage"),

  eyeLBar: document.getElementById("eyeLBar"),
  eyeLValue: document.getElementById("eyeLValue"),
  eyeRBar: document.getElementById("eyeRBar"),
  eyeRValue: document.getElementById("eyeRValue"),

  browLBar: document.getElementById("browLBar"),
  browLValue: document.getElementById("browLValue"),
  browRBar: document.getElementById("browRBar"),
  browRValue: document.getElementById("browRValue"),

  gazeXBar: document.getElementById("gazeXBar"),
  gazeXValue: document.getElementById("gazeXValue"),
  gazeYBar: document.getElementById("gazeYBar"),
  gazeYValue: document.getElementById("gazeYValue"),

  mouthOpenBar: document.getElementById("mouthOpenBar"),
  mouthOpenValue: document.getElementById("mouthOpenValue"),
  mouthWidthBar: document.getElementById("mouthWidthBar"),
  mouthWidthValue: document.getElementById("mouthWidthValue"),
  curvatureBar: document.getElementById("curvatureBar"),
  curvatureValue: document.getElementById("curvatureValue"),
};

let landmarker = null;
let stream = null;
let running = false;
let animationFrame = 0;
let lastInferenceAt = 0;
let lastVideoTime = -1;

let calibrationStartedAt = null;
let calibrationSamples = [];
let baseline = null;

let state = neutralState();

const frameCanvas = document.createElement("canvas");
const frameContext = frameCanvas.getContext("2d", { alpha: false });

let lastTimestampMs = -1;

ui.start?.addEventListener("click", startDemo);
ui.stop?.addEventListener("click", stopDemo);

window.addEventListener("pagehide", stopDemo);

async function startDemo() {
  if (running) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    fail(
      "This browser does not expose webcam access here. Open the HTTPS site in a current browser."
    );
    return;
  }

  ui.start.disabled = true;
  setStatus("LOADING TRACKER", "idle");
  setMessage("Loading the local face-tracking model…");

  try {
    const trackerPromise = ensureLandmarker();

    setMessage("Waiting for camera permission…");
    const cameraPromise = navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
      audio: false,
    });

    [landmarker, stream] = await Promise.all([trackerPromise, cameraPromise]);

    ui.video.srcObject = stream;
    await ui.video.play();
    await waitForUsableVideo(ui.video);

    running = true;
    resetCalibration();
    ui.idle.hidden = true;
    ui.stop.hidden = false;
    ui.video.classList.add("live");

    setStatus("CAMERA LOCAL", "live");
    setTracking("FINDING FACE");
    setMessage("Move into frame. Calibration begins when a face is detected.");

    animationFrame = requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);

    if (error?.name === "NotAllowedError") {
      fail("Camera permission was denied. Nothing was captured or uploaded.");
    } else if (error?.name === "NotFoundError") {
      fail("No camera was found on this device.");
    } else {
      fail("The live tracker could not start on this browser/device.");
    }

    stopTracks();
  } finally {
    ui.start.disabled = false;
  }
}

async function ensureLandmarker() {
  if (landmarker) return landmarker;

  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

  const options = (delegate) => ({
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate,
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.6,
    minFacePresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  try {
    return await FaceLandmarker.createFromOptions(vision, options("GPU"));
  } catch (gpuError) {
    console.warn("GPU delegate unavailable; falling back to CPU.", gpuError);
    return FaceLandmarker.createFromOptions(vision, options("CPU"));
  }
}

function loop(now) {
  if (!running) return;

  animationFrame = requestAnimationFrame(loop);

  if (now - lastInferenceAt < INFERENCE_INTERVAL_MS) return;
  if (ui.video.readyState < 2) return;
  if (ui.video.currentTime === lastVideoTime) return;

  lastInferenceAt = now;
  lastVideoTime = ui.video.currentTime;

  const width = ui.video.videoWidth;
  const height = ui.video.videoHeight;

  if (!width || !height || !frameContext) {
    setTracking("WAITING FOR VIDEO");
    setMessage("Camera is active but has not produced a usable frame yet.");
    return;
  }

  if (frameCanvas.width !== width || frameCanvas.height !== height) {
    frameCanvas.width = width;
    frameCanvas.height = height;
  }

  frameContext.drawImage(ui.video, 0, 0, width, height);

  // MediaPipe requires monotonically increasing timestamps in milliseconds.
  const timestampMs = Math.max(Math.floor(performance.now()), lastTimestampMs + 1);
  lastTimestampMs = timestampMs;

  let result;
  try {
    result = landmarker.detectForVideo(frameCanvas, timestampMs);
  } catch (error) {
    console.error(error);
    running = false;
    cancelAnimationFrame(animationFrame);
    stopTracks();
    ui.video.pause();
    ui.video.srcObject = null;
    ui.video.classList.remove("live");
    ui.idle.hidden = false;
    ui.stop.hidden = true;
    const detail = error?.message ? ` ${error.message}` : "";
    fail(`Tracking stopped after an inference error.${detail}`);
    return;
  }

  const landmarks = result?.faceLandmarks?.[0];

  if (!landmarks) {
    onTrackingLost();
    return;
  }

  const raw = measureLandmarks(landmarks, now / 1000);

  if (!baseline) {
    collectCalibration(raw, now);
    return;
  }

  const target = normalise(raw, baseline);
  state = smoothState(state, target);
  renderState(state);

  setTracking("LOCKED");
  setMessage("Live semantic controls — ready for character-specific mapping.");
}

function collectCalibration(raw, now) {
  if (calibrationStartedAt === null) {
    calibrationStartedAt = now;
    calibrationSamples = [];
    ui.calibration.hidden = false;
    setTracking("CALIBRATING");
    setMessage("Hold a neutral expression for one second.");
  }

  calibrationSamples.push(raw);

  const elapsed = (now - calibrationStartedAt) / 1000;
  const progress = clamp(elapsed / CALIBRATION_SECONDS, 0, 1);

  ui.calibrationFill.style.width = `${progress * 100}%`;

  if (elapsed < CALIBRATION_SECONDS) return;

  baseline = meanMeasurements(calibrationSamples);
  ui.calibration.hidden = true;
  ui.calibrationFill.style.width = "100%";

  state = {
    ...neutralState(),
    trackingValid: true,
    measuredAt: raw.measuredAt,
  };

  setTracking("LOCKED");
  setMessage("Calibration complete. Blink, look around, raise a brow, smile or open your mouth.");
}

function onTrackingLost() {
  if (!baseline) {
    calibrationStartedAt = null;
    calibrationSamples = [];
    ui.calibration.hidden = true;
    ui.calibrationFill.style.width = "0%";
    setTracking("FINDING FACE");
    setMessage("Move into frame. Calibration begins when a face is detected.");
    return;
  }

  const target = neutralState();
  state = smoothState(state, target);
  renderState(state);
  setTracking("LOST");
  setMessage("Face lost — controls return toward neutral.");
}

function stopDemo() {
  if (!running && !stream) return;

  running = false;
  cancelAnimationFrame(animationFrame);
  animationFrame = 0;

  stopTracks();

  ui.video.pause();
  ui.video.srcObject = null;
  ui.video.classList.remove("live");

  ui.idle.hidden = false;
  ui.stop.hidden = true;
  ui.calibration.hidden = true;
  ui.calibrationFill.style.width = "0%";

  resetCalibration();

  state = neutralState();
  renderState(state);

  setStatus("CAMERA OFF", "idle");
  setTracking("WAITING");
  setMessage("Camera stopped. No video is retained by the page.");
}

function stopTracks() {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
  stream = null;
}

function resetCalibration() {
  calibrationStartedAt = null;
  calibrationSamples = [];
  baseline = null;
  lastInferenceAt = 0;
  lastVideoTime = -1;
  lastTimestampMs = -1;
}

function fail(message) {
  setStatus("UNAVAILABLE", "error");
  setTracking("ERROR");
  setMessage(message);
}

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

  const mouthOpen = ratio(distance(points[13], points[14]), faceWidth);
  const mouthRoundnessRatio = mouthOpen / Math.max(mouthWidth, 1e-9);
  const mouthCompression = clamp(1 - mouthOpen * 18, 0, 1);

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
    const leftIris = averagePoint(points, [468, 469, 470, 471, 472]);
    const rightIris = averagePoint(points, [473, 474, 475, 476, 477]);

    gazeX = average(
      relativeInterval(leftIris.x, points[33].x, points[133].x),
      relativeInterval(rightIris.x, points[362].x, points[263].x)
    );

    gazeY = average(
      relativeInterval(leftIris.y, points[159].y, points[145].y),
      relativeInterval(rightIris.y, points[386].y, points[374].y)
    );
  }

  const cornersY = (points[61].y + points[291].y) / 2;
  const mouthCurvature = clamp(
    (points[13].y - cornersY) / Math.max(faceWidth * 0.03, 1e-9),
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
    raw.leftEyeOpen / Math.max(base.leftEyeOpen, 1e-9),
    0,
    1
  );

  const rightEyeOpen = clamp(
    raw.rightEyeOpen / Math.max(base.rightEyeOpen, 1e-9),
    0,
    1
  );

  const mouthOpen = clamp(
    (raw.mouthOpen - base.mouthOpen) / TUNING.mouthOpenRange,
    0,
    1
  );

  const stretch =
    (raw.mouthWidth / Math.max(base.mouthWidth, 1e-9) - 1) /
    TUNING.mouthWidthRange;

  const mouthWidth = clamp(0.5 + stretch * 0.5, 0, 1);

  const mouthRoundness = clamp(
    (raw.mouthRoundnessRatio - base.mouthRoundnessRatio) /
      TUNING.mouthRoundnessRange,
    0,
    1
  );

  const mouthCompression = clamp(
    1 - raw.mouthOpen / Math.max(base.mouthOpen, 1e-9),
    0,
    1
  );

  const mouthCurvature = clamp(
    (raw.mouthCurvature - base.mouthCurvature) /
      TUNING.mouthCurvatureRange,
    -1,
    1
  );

  return {
    leftEyeOpen,
    rightEyeOpen,

    gazeX: deadZone(
      clamp((raw.gazeX - base.gazeX) * TUNING.gazeGain, -1, 1),
      0.04
    ),
    gazeY: deadZone(
      clamp((raw.gazeY - base.gazeY) * TUNING.gazeGain, -1, 1),
      0.04
    ),

    leftBrowHeight: deadZone(
      clamp(
        ((raw.leftBrowHeight - base.leftBrowHeight) / 0.03) *
          TUNING.browHeightGain,
        -1,
        1
      ),
      0.06
    ),

    rightBrowHeight: deadZone(
      clamp(
        ((raw.rightBrowHeight - base.rightBrowHeight) / 0.03) *
          TUNING.browHeightGain,
        -1,
        1
      ),
      0.06
    ),

    leftBrowTilt: deadZone(
      clamp(
        (raw.leftBrowTilt - base.leftBrowTilt) * TUNING.browTiltGain,
        -1,
        1
      ),
      0.06
    ),

    rightBrowTilt: deadZone(
      clamp(
        (raw.rightBrowTilt - base.rightBrowTilt) * TUNING.browTiltGain,
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
  const blend = (a, b, alpha) => a + (b - a) * alpha;

  const eyeBlend = (a, b) => {
    // Preserve fast blinks while smoothing normal eye motion.
    if (b < 0.35) return b;
    const alpha = b > a ? 0.85 : Math.max(SMOOTHING.eyes, 0.8);
    return blend(a, b, alpha);
  };

  return {
    leftEyeOpen: eyeBlend(previous.leftEyeOpen, target.leftEyeOpen),
    rightEyeOpen: eyeBlend(previous.rightEyeOpen, target.rightEyeOpen),

    gazeX: blend(previous.gazeX, target.gazeX, SMOOTHING.gaze),
    gazeY: blend(previous.gazeY, target.gazeY, SMOOTHING.gaze),

    leftBrowHeight: blend(
      previous.leftBrowHeight,
      target.leftBrowHeight,
      SMOOTHING.brows
    ),
    rightBrowHeight: blend(
      previous.rightBrowHeight,
      target.rightBrowHeight,
      SMOOTHING.brows
    ),
    leftBrowTilt: blend(
      previous.leftBrowTilt,
      target.leftBrowTilt,
      SMOOTHING.brows
    ),
    rightBrowTilt: blend(
      previous.rightBrowTilt,
      target.rightBrowTilt,
      SMOOTHING.brows
    ),

    mouthOpen: blend(previous.mouthOpen, target.mouthOpen, SMOOTHING.mouth),
    mouthWidth: blend(previous.mouthWidth, target.mouthWidth, SMOOTHING.mouth),
    mouthRoundness: blend(
      previous.mouthRoundness,
      target.mouthRoundness,
      SMOOTHING.mouth
    ),
    mouthCompression: blend(
      previous.mouthCompression,
      target.mouthCompression,
      SMOOTHING.mouth
    ),
    mouthCurvature: blend(
      previous.mouthCurvature,
      target.mouthCurvature,
      SMOOTHING.mouth
    ),

    trackingValid: target.trackingValid,
    measuredAt: target.measuredAt,
  };
}

function neutralState() {
  return {
    leftEyeOpen: 0.6,
    rightEyeOpen: 0.6,

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
      samples.reduce((total, sample) => total + sample[key], 0) /
      samples.length;
  }

  return result;
}

function renderState(current) {
  setUnit(ui.eyeLBar, ui.eyeLValue, current.leftEyeOpen);
  setUnit(ui.eyeRBar, ui.eyeRValue, current.rightEyeOpen);

  setSigned(ui.browLBar, ui.browLValue, current.leftBrowHeight);
  setSigned(ui.browRBar, ui.browRValue, current.rightBrowHeight);

  setSigned(ui.gazeXBar, ui.gazeXValue, current.gazeX);
  setSigned(ui.gazeYBar, ui.gazeYValue, current.gazeY);

  setUnit(ui.mouthOpenBar, ui.mouthOpenValue, current.mouthOpen);
  setUnit(ui.mouthWidthBar, ui.mouthWidthValue, current.mouthWidth);
  setSigned(ui.curvatureBar, ui.curvatureValue, current.mouthCurvature);
}

function setUnit(bar, output, value) {
  const clamped = clamp(value, 0, 1);
  bar.style.width = `${clamped * 100}%`;
  output.value = clamped.toFixed(2);
  output.textContent = clamped.toFixed(2);
}

function setSigned(bar, output, value) {
  const clamped = clamp(value, -1, 1);

  if (clamped >= 0) {
    bar.style.left = "50%";
    bar.style.width = `${clamped * 50}%`;
  } else {
    bar.style.left = `${50 + clamped * 50}%`;
    bar.style.width = `${-clamped * 50}%`;
  }

  const formatted = `${clamped >= 0 ? "+" : ""}${clamped.toFixed(2)}`;
  output.value = formatted;
  output.textContent = formatted;
}

function setStatus(text, stateName) {
  ui.status.textContent = text;
  ui.status.dataset.state = stateName;
}

function setTracking(text) {
  ui.tracking.textContent = text;
}

function setMessage(text) {
  ui.message.textContent = text;
}

function waitForUsableVideo(video) {
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
      reject(new Error("Camera opened but never produced a usable video frame."));
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
      video.removeEventListener("loadedmetadata", check);
      video.removeEventListener("loadeddata", check);
      video.removeEventListener("canplay", check);
      video.removeEventListener("resize", check);
    };

    video.addEventListener("loadedmetadata", check);
    video.addEventListener("loadeddata", check);
    video.addEventListener("canplay", check);
    video.addEventListener("resize", check);

    check();
  });
}

function distance(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function ratio(a, b) {
  return Math.abs(b) < 1e-9 ? 0 : a / b;
}

function average(...values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function averagePoint(points, indices) {
  return {
    x: average(...indices.map((index) => points[index].x)),
    y: average(...indices.map((index) => points[index].y)),
  };
}

function relativeInterval(value, first, second) {
  const low = Math.min(first, second);
  const high = Math.max(first, second);
  const span = high - low;

  if (span < 1e-9) return 0;

  return clamp(((value - low) / span - 0.5) * 2, -1, 1);
}

function browHeight(browA, browB, eyeA, eyeB, faceWidth) {
  const browCentre = {
    x: (browA.x + browB.x) / 2,
    y: (browA.y + browB.y) / 2,
  };

  const dx = eyeB.x - eyeA.x;
  const dy = eyeB.y - eyeA.y;
  const length = Math.hypot(dx, dy);

  if (length < 1e-9) return 0;

  const perpendicular =
    -(dx * (browCentre.y - eyeA.y) - dy * (browCentre.x - eyeA.x)) /
    length;

  return ratio(perpendicular, faceWidth);
}

function browTilt(browA, browB, eyeA, eyeB) {
  const slope = (a, b) => {
    const dx = b.x - a.x;
    return Math.abs(dx) < 1e-9 ? 0 : (b.y - a.y) / dx;
  };

  return clamp((slope(browA, browB) - slope(eyeA, eyeB)) / 0.25, -1, 1);
}

function deadZone(value, threshold) {
  return Math.abs(value) < threshold ? 0 : value;
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

renderState(state);
