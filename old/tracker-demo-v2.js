const CALIBRATION_MS = 1000;
const FRAME_INTERVAL_MS = 55; // ~18 fps: enough for a portfolio demo, easier on laptops/phones.

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
  start: document.getElementById("startDemo"),
  stop: document.getElementById("stopDemo"),
  idle: document.getElementById("demoIdle"),
  calibration: document.getElementById("demoCalibration"),
  calibrationFill: document.getElementById("calibrationFill"),
  panel: document.getElementById("puppetPanel"),
  status: document.getElementById("demoStatus"),
  statusDot: document.getElementById("statusDot"),
  error: document.getElementById("demoError"),

  eyeLeft: document.getElementById("puppetEyeLeft"),
  eyeRight: document.getElementById("puppetEyeRight"),
  pupilLeft: document.getElementById("puppetPupilLeft"),
  pupilRight: document.getElementById("puppetPupilRight"),
  browLeft: document.getElementById("puppetBrowLeft"),
  browRight: document.getElementById("puppetBrowRight"),
  mouth: document.getElementById("puppetMouth"),
};

let worker = null;
let workerReady = false;
let workerBusy = false;

let stream = null;
let running = false;
let rafId = 0;
let lastFrameAt = 0;
let lastTimestampMs = -1;

let baseline = null;
let calibrationStartedAt = null;
let samples = [];

let state = neutralState();

ui.start.addEventListener("click", startDemo);
ui.stop.addEventListener("click", stopDemo);
window.addEventListener("pagehide", stopDemo);

async function startDemo() {
  if (running) return;

  ui.start.disabled = true;
  hideError();
  setStatus("LOADING TRACKER", false);

  try {
    await ensureWorker();

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

    resetTrackingState();

    running = true;
    ui.idle.hidden = true;
    ui.stop.hidden = false;
    ui.panel.hidden = false;
    ui.video.classList.add("live");

    setStatus("FINDING FACE", true);
    rafId = requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    showError(readableStartupError(error));
    stopTracks();
    setStatus("UNAVAILABLE", false);
  } finally {
    ui.start.disabled = false;
  }
}

function ensureWorker() {
  if (workerReady) return Promise.resolve();

  if (!worker) {
    worker = new Worker(
      new URL("tracker-worker-v2.js", import.meta.url),
      { type: "module" }
    );

    worker.addEventListener("message", onWorkerMessage);

    worker.addEventListener("error", (event) => {
      workerBusy = false;
      workerReady = false;
      showError(event.message || "Tracker worker failed to load.");
      setStatus("TRACKER ERROR", false);
    });
  }

  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("tracker initialization timed out"));
    }, 15000);

    const ready = (event) => {
      if (event.data?.type === "READY") {
        cleanup();
        workerReady = true;
        resolve();
      }

      if (event.data?.type === "ERROR") {
        cleanup();
        reject(new Error(event.data.error || "tracker initialization failed"));
      }
    };

    const cleanup = () => {
      window.clearTimeout(timeout);
      worker.removeEventListener("message", ready);
    };

    worker.addEventListener("message", ready);
    worker.postMessage({ type: "INIT" });
  });
}

async function loop(now) {
  if (!running) return;

  rafId = requestAnimationFrame(loop);

  if (!workerReady || workerBusy) return;
  if (now - lastFrameAt < FRAME_INTERVAL_MS) return;
  if (ui.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  if (!ui.video.videoWidth || !ui.video.videoHeight) return;

  lastFrameAt = now;

  try {
    const bitmap = await createImageBitmap(ui.video);

    const rawTimestamp = performance.now();
    const timestampMs =
      rawTimestamp > lastTimestampMs
        ? rawTimestamp
        : lastTimestampMs + 1;

    lastTimestampMs = timestampMs;
    workerBusy = true;

    worker.postMessage(
      {
        type: "FRAME",
        bitmap,
        timestampMs,
      },
      [bitmap]
    );
  } catch (error) {
    console.warn("Could not extract camera frame", error);
  }
}

function onWorkerMessage(event) {
  const message = event.data;

  if (message.type === "READY") {
    workerReady = true;
    return;
  }

  if (message.type === "ERROR") {
    workerBusy = false;
    showError(`Tracker error: ${message.error}`);
    setStatus("TRACKER ERROR", false);
    stopDemo();
    return;
  }

  if (message.type !== "RESULT") return;

  workerBusy = false;

  if (!running) return;

  if (!message.face) {
    onFaceLost();
    return;
  }

  const raw = measureLandmarks(
    message.face,
    message.timestampMs / 1000
  );

  if (!baseline) {
    collectCalibration(raw);
    return;
  }

  const target = normalise(raw, baseline);
  state = smoothState(state, target);
  renderPuppet(state);

  ui.calibration.hidden = true;
  setStatus("TRACKING", true);
}

function collectCalibration(raw) {
  const now = performance.now();

  if (calibrationStartedAt === null) {
    calibrationStartedAt = now;
    samples = [];
    ui.calibration.hidden = false;
    ui.calibrationFill.style.width = "0%";
    setStatus("CALIBRATING", true);
  }

  samples.push(raw);

  const elapsed = now - calibrationStartedAt;
  const progress = clamp(elapsed / CALIBRATION_MS, 0, 1);
  ui.calibrationFill.style.width = `${progress * 100}%`;

  if (elapsed < CALIBRATION_MS) return;

  baseline = meanMeasurements(samples);
  ui.calibration.hidden = true;

  state = {
    ...neutralState(),
    trackingValid: true,
    measuredAt: raw.measuredAt,
  };

  renderPuppet(state);
  setStatus("TRACKING", true);
}

function onFaceLost() {
  if (!baseline) {
    calibrationStartedAt = null;
    samples = [];
    ui.calibration.hidden = true;
    ui.calibrationFill.style.width = "0%";
    setStatus("FINDING FACE", true);
    return;
  }

  state = smoothState(state, neutralState());
  renderPuppet(state);
  setStatus("FACE LOST", false);
}

function stopDemo() {
  running = false;
  cancelAnimationFrame(rafId);
  rafId = 0;
  workerBusy = false;

  stopTracks();

  ui.video.pause();
  ui.video.srcObject = null;
  ui.video.classList.remove("live");

  ui.idle.hidden = false;
  ui.stop.hidden = true;
  ui.panel.hidden = true;
  ui.calibration.hidden = true;
  ui.calibrationFill.style.width = "0%";

  resetTrackingState();
  renderPuppet(state);
  setStatus("CAMERA OFF", false);
}

function stopTracks() {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
  stream = null;
}

function resetTrackingState() {
  baseline = null;
  calibrationStartedAt = null;
  samples = [];
  state = neutralState();
  lastFrameAt = 0;
  lastTimestampMs = -1;
  hideError();
}

function renderPuppet(current) {
  const eyeL = clamp(current.leftEyeOpen, 0.08, 1);
  const eyeR = clamp(current.rightEyeOpen, 0.08, 1);

  ui.eyeLeft.style.transform = `scaleY(${eyeL})`;
  ui.eyeRight.style.transform = `scaleY(${eyeR})`;

  // Invert horizontal gaze to visually match the mirrored webcam preview.
  const pupilX = clamp(-current.gazeX, -1, 1) * 20;
  const pupilY = clamp(current.gazeY, -1, 1) * 15;

  ui.pupilLeft.style.transform =
    `translate(calc(-50% + ${pupilX}px), calc(-50% + ${pupilY}px))`;
  ui.pupilRight.style.transform =
    `translate(calc(-50% + ${pupilX}px), calc(-50% + ${pupilY}px))`;

  const browLeftY = clamp(-current.leftBrowHeight, -1, 1) * 12;
  const browRightY = clamp(-current.rightBrowHeight, -1, 1) * 12;

  ui.browLeft.style.transform =
    `translateY(${browLeftY}px) rotate(${current.leftBrowTilt * 13}deg)`;

  ui.browRight.style.transform =
    `translateY(${browRightY}px) rotate(${current.rightBrowTilt * 13}deg)`;

  const width = 28 + current.mouthWidth * 26;
  const height = 6 + current.mouthOpen * 25;

  ui.mouth.style.width = `${width}%`;
  ui.mouth.style.height = `${height}%`;

  if (current.mouthCurvature < -0.15) {
    ui.mouth.style.borderRadius = "999px 999px 0 0";
  } else if (current.mouthCurvature > 0.15) {
    ui.mouth.style.borderRadius = "0 0 999px 999px";
  } else {
    ui.mouth.style.borderRadius = "999px";
  }
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

  const mouthOpen =
    ratio(distance(points[13], points[14]), faceWidth);

  const mouthRoundnessRatio =
    mouthOpen / Math.max(mouthWidth, 1e-9);

  const mouthCompression =
    clamp(1 - mouthOpen * 18, 0, 1);

  const leftBrowHeight = browHeight(
    points[70], points[105], points[33], points[133], faceWidth
  );

  const rightBrowHeight = browHeight(
    points[300], points[334], points[362], points[263], faceWidth
  );

  const leftBrowTilt = browTilt(
    points[70], points[105], points[33], points[133]
  );

  const rightBrowTilt = browTilt(
    points[300], points[334], points[362], points[263]
  );

  let gazeX = 0;
  let gazeY = 0;

  if (points.length > 477) {
    const leftIris = averagePoint(points, [468,469,470,471,472]);
    const rightIris = averagePoint(points, [473,474,475,476,477]);

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
    (raw.mouthOpen - base.mouthOpen) /
      TUNING.mouthOpenRange,
    0,
    1
  );

  const stretch =
    (raw.mouthWidth / Math.max(base.mouthWidth, 1e-9) - 1) /
    TUNING.mouthWidthRange;

  const mouthWidth =
    clamp(0.5 + stretch * 0.5, 0, 1);

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
        (raw.leftBrowTilt - base.leftBrowTilt) *
          TUNING.browTiltGain,
        -1,
        1
      ),
      0.06
    ),

    rightBrowTilt: deadZone(
      clamp(
        (raw.rightBrowTilt - base.rightBrowTilt) *
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
    leftEyeOpen: eyeBlend(
      previous.leftEyeOpen,
      target.leftEyeOpen
    ),
    rightEyeOpen: eyeBlend(
      previous.rightEyeOpen,
      target.rightEyeOpen
    ),

    gazeX: blend(previous.gazeX, target.gazeX, 0.35),
    gazeY: blend(previous.gazeY, target.gazeY, 0.35),

    leftBrowHeight: blend(
      previous.leftBrowHeight,
      target.leftBrowHeight,
      0.7
    ),
    rightBrowHeight: blend(
      previous.rightBrowHeight,
      target.rightBrowHeight,
      0.7
    ),
    leftBrowTilt: blend(
      previous.leftBrowTilt,
      target.leftBrowTilt,
      0.7
    ),
    rightBrowTilt: blend(
      previous.rightBrowTilt,
      target.rightBrowTilt,
      0.7
    ),

    mouthOpen: blend(
      previous.mouthOpen,
      target.mouthOpen,
      0.45
    ),
    mouthWidth: blend(
      previous.mouthWidth,
      target.mouthWidth,
      0.45
    ),
    mouthRoundness: blend(
      previous.mouthRoundness,
      target.mouthRoundness,
      0.45
    ),
    mouthCompression: blend(
      previous.mouthCompression,
      target.mouthCompression,
      0.45
    ),
    mouthCurvature: blend(
      previous.mouthCurvature,
      target.mouthCurvature,
      0.45
    ),

    trackingValid: target.trackingValid,
    measuredAt: target.measuredAt,
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
      samples.reduce((sum, sample) => sum + sample[key], 0) /
      samples.length;
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

function waitForVideo(video) {
  if (
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.videoWidth > 0 &&
    video.videoHeight > 0
  ) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error("camera opened but produced no usable frames"));
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
      clearTimeout(timeout);
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

function setStatus(text, live) {
  ui.status.textContent = text;
  ui.statusDot.classList.toggle("live", live);
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function hideError() {
  ui.error.hidden = true;
  ui.error.textContent = "";
}

function readableStartupError(error) {
  if (error?.name === "NotAllowedError") {
    return "Camera permission was denied.";
  }

  if (error?.name === "NotFoundError") {
    return "No camera was found on this device.";
  }

  return error?.message || String(error);
}

function distance(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function ratio(a, b) {
  return Math.abs(b) < 1e-9 ? 0 : a / b;
}

function average(...values) {
  return values.reduce((sum, value) => sum + value, 0) /
    values.length;
}

function averagePoint(points, indices) {
  return {
    x: average(...indices.map((i) => points[i].x)),
    y: average(...indices.map((i) => points[i].y)),
  };
}

function relativeInterval(value, first, second) {
  const low = Math.min(first, second);
  const high = Math.max(first, second);
  const span = high - low;

  if (span < 1e-9) return 0;

  return clamp(
    ((value - low) / span - 0.5) * 2,
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
      dx * (browCentre.y - eyeA.y) -
      dy * (browCentre.x - eyeA.x)
    ) / length;

  return ratio(perpendicular, faceWidth);
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
    (slope(browA, browB) - slope(eyeA, eyeB)) / 0.25,
    -1,
    1
  );
}

function deadZone(value, threshold) {
  return Math.abs(value) < threshold ? 0 : value;
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

renderPuppet(state);
