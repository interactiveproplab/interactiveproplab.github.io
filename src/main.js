import FaceWorker from "./face-worker.js?worker&inline";

const CALIBRATION_MS = 1600;
const CALIBRATION_SETTLE_MS = 350;

const TUNING = {
  gazeGain: 2.4,
  browHeightGain: 1.0,
  browTiltGain: 2.2,
  mouthOpenRange: 0.045,
  mouthWidthRange: 0.30,
  mouthRoundnessRange: 0.06,
  mouthCurvatureRange: 0.35,
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
  matrix: document.getElementById("protogenMatrix"),
  error: document.getElementById("trackingError"),
  errorText: document.getElementById("trackingErrorText"),
};

const signalEls = {
  eyes: document.querySelector('[data-signal="eyes"]'),
  brows: document.querySelector('[data-signal="brows"]'),
  gaze: document.querySelector('[data-signal="gaze"]'),
  mouth: document.querySelector('[data-signal="mouth"]'),
};

const overlayContext = ui.overlay.getContext("2d");
const matrixContext = ui.matrix.getContext("2d");

let worker = null;
let workerReady = false;
let workerBusy = false;
let initPromise = null;
let trackerInitStage = "worker not started";
const TRACKER_INIT_ATTEMPTS = 3;
const TRACKER_INIT_TIMEOUT_MS = 10000;

let stream = null;
let running = false;
let rafId = 0;
let lastVideoTime = -1;
let lastTimestampMs = -1;
let consecutiveInferenceErrors = 0;
let framesSent = 0;
let resultsReceived = 0;

let baseline = null;
let calibrationStartedAt = null;
let calibrationSamples = [];
let consecutiveNoFaceFrames = 0;

let state = neutralState();

ui.start.addEventListener("click", start);
ui.retry.addEventListener("click", start);
ui.stop.addEventListener("click", stop);
window.addEventListener("pagehide", stop);

updateSignals(state);

async function start() {
  if (running) return;

  hideError();

  if (isFirefoxFamily()) {
    showError(
      "Facial landmarking source unavailable in this browser. " +
      "For best compatibility, open the live demo in a Chromium-based browser."
    );
    setStatus("CHROMIUM REQUIRED", false);
    return;
  }
  ui.start.disabled = true;
  ui.retry.disabled = true;
  setStatus("LOADING TRACKER", false);

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access is not available in this browser.");
    }

    await ensureWorker();

    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    ui.video.srcObject = stream;
    await ui.video.play();
    await waitForVideo(ui.video);

    configureProcessingFrame();
    resetTracking();

    running = true;
    ui.startPanel.hidden = true;
    ui.stop.hidden = false;
    ui.video.classList.add("live");

    setStatus("FINDING FACE", true);
    rafId = requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    stopTracks();

    const detail = readableError(error);

    showError(
      detail.startsWith("Facial landmarking source unavailable.")
        ? detail
        : `Facial landmarking source unavailable. ${detail}`
    );

    setStatus(
      "LANDMARK SOURCE UNAVAILABLE",
      false
    );
  } finally {
    ui.start.disabled = false;
    ui.retry.disabled = false;
  }
}

async function ensureWorker() {
  if (workerReady && worker) return;

  let lastError = null;

  for (
    let attempt = 1;
    attempt <= TRACKER_INIT_ATTEMPTS;
    attempt += 1
  ) {
    setStatus(
      attempt === 1
        ? "LOADING TRACKER"
        : `RETRYING TRACKER ${attempt}/${TRACKER_INIT_ATTEMPTS}`,
      false
    );

    try {
      await initializeFreshWorker(attempt);
      return;
    } catch (error) {
      lastError = error;
      console.warn(
        `[IPL tracker] initialization attempt ${attempt} failed:`,
        error
      );
      destroyWorker();

      if (attempt < TRACKER_INIT_ATTEMPTS) {
        await delay(250);
      }
    }
  }

  throw new Error(
    `Facial landmarking source unavailable. ` +
    `The MediaPipe tracking runtime did not initialize after ` +
    `${TRACKER_INIT_ATTEMPTS} attempts. ` +
    `Last stage: ${trackerInitStage}. ` +
    `Last error: ${lastError?.message || String(lastError)}`
  );
}

function initializeFreshWorker(attempt) {
  destroyWorker();

  return new Promise((resolve, reject) => {
    trackerInitStage =
      `attempt ${attempt}: worker starting`;

    // Vite embeds the complete face worker into the production bundle.
    // This removes the separately fetched hashed worker chunk, which was the
    // failure point behind intermittent "Face tracker worker failed to load".
    const candidate = new FaceWorker();

    worker = candidate;
    workerReady = false;

    let settled = false;

    const cleanup = () => {
      window.clearTimeout(timeout);
      candidate.removeEventListener("message", onMessage);
      candidate.removeEventListener("error", onError);
    };

    const succeed = () => {
      if (settled) return;
      settled = true;
      cleanup();
      workerReady = true;
      trackerInitStage =
        `attempt ${attempt}: ready`;
      resolve();
    };

    const fail = (error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };

    const onMessage = (event) => {
      const message = event.data;

      if (message?.type === "INIT_STAGE") {
        trackerInitStage =
          `attempt ${attempt}: ${message.stage || "unknown stage"}`;
        console.info(
          "[IPL tracker] init:",
          trackerInitStage
        );
        return;
      }

      // Support both v17 READY and official-style INIT_DONE protocols.
      if (
        message?.type === "READY" ||
        message?.type === "INIT_DONE"
      ) {
        succeed();
        return;
      }

      if (
        message?.type === "INIT_ERROR" ||
        message?.type === "ERROR"
      ) {
        fail(
          new Error(
            message.error ||
            "Face tracker worker initialization failed."
          )
        );
      }
    };

    const onError = (event) => {
      const location = [
        event.filename,
        event.lineno ? `line ${event.lineno}` : "",
        event.colno ? `column ${event.colno}` : "",
      ]
        .filter(Boolean)
        .join(", ");

      fail(
        new Error(
          `Face tracker worker failed before initialization` +
          `${location ? ` (${location})` : ""}: ` +
          `${event.message || "unknown worker startup error"}`
        )
      );
    };

    candidate.addEventListener("message", onMessage);
    candidate.addEventListener("error", onError);

    // Keep the normal runtime handler attached as well.
    candidate.addEventListener(
      "message",
      handleWorkerMessage
    );

    const timeout = window.setTimeout(() => {
      fail(
        new Error(
          `Initialization timed out at ${trackerInitStage}`
        )
      );
    }, TRACKER_INIT_TIMEOUT_MS);

    trackerInitStage =
      `attempt ${attempt}: INIT sent`;

    candidate.postMessage({
      type: "INIT",
      wasmRoot: new URL(
        "./wasm/",
        window.location.href
      ).href,
      modelUrl: new URL(
        "./models/face_landmarker.task",
        window.location.href
      ).href,
    });
  });
}

function destroyWorker() {
  workerReady = false;
  workerBusy = false;
  initPromise = null;

  if (!worker) return;

  try {
    worker.terminate();
  } catch (error) {
    console.warn(
      "[IPL tracker] worker termination failed:",
      error
    );
  }

  worker = null;
}

function delay(ms) {
  return new Promise(
    (resolve) => window.setTimeout(resolve, ms)
  );
}

async function loop() {
  if (!running) return;

  if (
    !workerReady ||
    workerBusy ||
    ui.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    !ui.video.videoWidth ||
    !ui.video.videoHeight
  ) {
    rafId = requestAnimationFrame(loop);
    return;
  }

  if (ui.video.currentTime === lastVideoTime) {
    rafId = requestAnimationFrame(loop);
    return;
  }

  lastVideoTime = ui.video.currentTime;

  try {
    const bitmap = await createImageBitmap(ui.video);

    const now = performance.now();
    const timestampMs =
      now > lastTimestampMs ? now : lastTimestampMs + 1;
    lastTimestampMs = timestampMs;

    workerBusy = true;
    framesSent += 1;

    if (framesSent === 1 || framesSent % 30 === 0) {
      console.info(
        "[IPL tracker] sending frame",
        framesSent,
        `${bitmap.width}×${bitmap.height}`,
        `video=${ui.video.videoWidth}×${ui.video.videoHeight}`
      );
    }

    worker.postMessage(
      {
        type: "FRAME",
        bitmap,
        timestampMs,
      },
      [bitmap]
    );
  } catch (error) {
    console.error("Failed to create ImageBitmap from video:", error);
    consecutiveInferenceErrors += 1;

    if (consecutiveInferenceErrors >= 4) {
      showError(`Camera frame conversion failed: ${error?.message || String(error)}`);
      setStatus("TRACKER ERROR", false);
    }

    rafId = requestAnimationFrame(loop);
  }
}

function scheduleNextFrame() {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
}

function handleWorkerMessage(event) {
  const message = event.data;

  if (message?.type === "READY") {
    workerReady = true;
    return;
  }

  if (message?.type === "RESULT") {
    workerBusy = false;
    consecutiveInferenceErrors = 0;
    hideError();
    resultsReceived += 1;

    if (
      resultsReceived === 1 ||
      resultsReceived % 30 === 0
    ) {
      console.info(
        "[IPL tracker] result",
        resultsReceived,
        `faces=${message.faceCount ?? (message.face ? 1 : 0)}`,
        message.bitmapSize
          ? `bitmap=${message.bitmapSize.width}×${message.bitmapSize.height}`
          : ""
      );
    }

    if (!running) return;

    sizeCanvases();
    clearOverlay();

    const landmarks = message.face;

    if (!landmarks) {
      consecutiveNoFaceFrames += 1;
      handleFaceLost();
      scheduleNextFrame();
      return;
    }

    consecutiveNoFaceFrames = 0;
    drawFeatureOverlay(landmarks);

    const now = performance.now();
    const raw = measureLandmarks(landmarks, now / 1000);

    if (!baseline) {
      collectCalibration(raw, now);
      scheduleNextFrame();
      return;
    }

    const target = normalise(raw, baseline);
    state = smoothState(state, target);

    updateSignals(state);
    ui.calibration.hidden = true;
    setStatus("TRACKING", true);
    scheduleNextFrame();
    return;
  }

  if (message?.type === "DETECT_ERROR") {
    workerBusy = false;
    consecutiveInferenceErrors += 1;
    console.warn(
      `Face Landmarker frame failed (${consecutiveInferenceErrors}):`,
      message.error
    );

    if (consecutiveInferenceErrors < 4) {
      setStatus("RECOVERING", false);
      scheduleNextFrame();
      return;
    }

    showError(
      `Face tracker error: ${
        message.error || "Repeated Face Landmarker inference failure."
      }`
    );
    setStatus("TRACKER ERROR", false);

    // Keep the webcam running. A detector failure should not make the camera
    // disappear and look like camera permission failed.
    scheduleNextFrame();
    return;
  }

  if (message?.type === "ERROR") {
    workerBusy = false;

    showError(
      "Facial landmarking source unavailable. " +
      (message.error || "The face tracker worker failed.")
    );
    setStatus("LANDMARK SOURCE UNAVAILABLE", false);

    // Camera remains visible; only the landmark source has failed.
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

  const elapsed = now - calibrationStartedAt;
  const progress = clamp(elapsed / CALIBRATION_MS, 0, 1);
  ui.calibrationFill.style.width = `${progress * 100}%`;

  // Ignore the first fraction of a second while landmark geometry settles.
  if (elapsed >= CALIBRATION_SETTLE_MS) {
    calibrationSamples.push(raw);
  }

  if (elapsed < CALIBRATION_MS) return;

  if (calibrationSamples.length < 6) {
    calibrationStartedAt = now;
    calibrationSamples = [];
    ui.calibrationFill.style.width = "0%";
    return;
  }

  baseline = medianMeasurements(calibrationSamples);
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

    if (consecutiveNoFaceFrames > 35) {
      setStatus("NO FACE — MOVE CLOSER", false);
    } else {
      setStatus("FINDING FACE", true);
    }
    return;
  }

  state = smoothState(state, neutralState());
  updateSignals(state);
  setStatus("FACE LOST", false);
}

function failRuntime(error) {
  if (!running && !stream) return;

  running = false;
  cancelAnimationFrame(rafId);
  rafId = 0;
  workerBusy = false;

  stopTracks();

  ui.video.pause();
  ui.video.srcObject = null;
  ui.video.classList.remove("live");

  clearOverlay();
  ui.stop.hidden = true;
  ui.calibration.hidden = true;

  const detail = error?.message || String(error);
  showError(`Face tracking failed: ${detail}`);
  setStatus("TRACKER ERROR", false);
}


function stop() {
  running = false;
  cancelAnimationFrame(rafId);
  rafId = 0;
  workerBusy = false;

  stopTracks();

  ui.video.pause();
  ui.video.srcObject = null;
  ui.video.classList.remove("live");

  clearOverlay();
  resetTracking();

  ui.startPanel.hidden = false;
  ui.stop.hidden = true;
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
  consecutiveNoFaceFrames = 0;
  state = neutralState();
  lastVideoTime = -1;
  lastTimestampMs = -1;
  consecutiveInferenceErrors = 0;
  framesSent = 0;
  resultsReceived = 0;
  updateSignals(state);
  hideError();
}


function configureProcessingFrame() {
  sizeCanvases();
}


function sizeCanvases() {
  const width = ui.video.videoWidth;
  const height = ui.video.videoHeight;

  if (!width || !height) return;

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
  drawProtogenMatrix(current);

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


function drawProtogenMatrix(current) {
  const ctx = matrixContext;
  const cols = 64;
  const rows = 32;
  const cellW = ui.matrix.width / cols;
  const cellH = ui.matrix.height / rows;

  ctx.clearRect(0, 0, ui.matrix.width, ui.matrix.height);
  ctx.fillStyle = "#050404";
  ctx.fillRect(0, 0, ui.matrix.width, ui.matrix.height);

  // Dim physical matrix points, so the output reads as hardware rather than vector art.
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      drawLed(ctx, x, y, cellW, cellH, 0.10);
    }
  }

  const gazeX = clamp(current.gazeX, -1, 1);
  const gazeY = clamp(current.gazeY, -1, 1);

  drawMatrixEye(
    ctx,
    18 + Math.round(gazeX * 1.5),
    11 + Math.round(gazeY * 0.8),
    current.leftEyeOpen,
    gazeX,
    gazeY,
    cellW,
    cellH
  );

  drawMatrixEye(
    ctx,
    46 + Math.round(gazeX * 1.5),
    11 + Math.round(gazeY * 0.8),
    current.rightEyeOpen,
    gazeX,
    gazeY,
    cellW,
    cellH
  );

  drawMatrixBrow(
    ctx,
    18,
    5,
    current.leftBrowHeight,
    current.leftBrowTilt,
    cellW,
    cellH
  );

  drawMatrixBrow(
    ctx,
    46,
    5,
    current.rightBrowHeight,
    current.rightBrowTilt,
    cellW,
    cellH
  );

  drawMatrixMouth(
    ctx,
    current.mouthOpen,
    current.mouthWidth,
    current.mouthCurvature,
    cellW,
    cellH
  );
}

function drawMatrixEye(
  ctx,
  centreX,
  centreY,
  openness,
  gazeX,
  gazeY,
  cellW,
  cellH
) {
  const halfWidth = 7;
  const halfHeight = Math.max(1, Math.round(clamp(openness, 0, 1) * 3.5));

  for (let dx = -halfWidth; dx <= halfWidth; dx += 1) {
    const norm = Math.abs(dx) / halfWidth;
    const heightHere = Math.max(
      0,
      Math.round(halfHeight * (1 - norm * 0.72))
    );

    for (let dy = -heightHere; dy <= heightHere; dy += 1) {
      drawLed(
        ctx,
        centreX + dx,
        centreY + dy,
        cellW,
        cellH,
        0.88
      );
    }
  }

  // Dark pupil/cut-out creates a readable gaze direction within the lit eye.
  const pupilX = centreX + Math.round(gazeX * 3);
  const pupilY = centreY + Math.round(gazeY * 1.5);
  const pupilRadius = openness < 0.3 ? 0 : 1;

  for (let dx = -pupilRadius; dx <= pupilRadius; dx += 1) {
    for (let dy = -pupilRadius; dy <= pupilRadius; dy += 1) {
      drawLed(
        ctx,
        pupilX + dx,
        pupilY + dy,
        cellW,
        cellH,
        0.02
      );
    }
  }
}

function drawMatrixBrow(
  ctx,
  centreX,
  baseY,
  height,
  tilt,
  cellW,
  cellH
) {
  const yOffset = Math.round(clamp(-height, -1, 1) * 2.2);
  const slope = clamp(tilt, -1, 1) * 0.45;

  for (let dx = -6; dx <= 6; dx += 1) {
    const y = Math.round(baseY + yOffset + dx * slope);
    drawLed(ctx, centreX + dx, y, cellW, cellH, 0.92);
  }
}

function drawMatrixMouth(
  ctx,
  mouthOpen,
  mouthWidth,
  curvature,
  cellW,
  cellH
) {
  const centreX = 32;
  const centreY = 23;
  const halfWidth = Math.round(7 + clamp(mouthWidth, 0, 1) * 6);
  const openHeight = Math.round(clamp(mouthOpen, 0, 1) * 4);

  for (let dx = -halfWidth; dx <= halfWidth; dx += 1) {
    const norm = halfWidth === 0 ? 0 : dx / halfWidth;
    const curve = Math.round(
      clamp(curvature, -1, 1) * (1 - norm * norm) * 2.6
    );

    const topY = centreY - openHeight + curve;
    const bottomY = centreY + openHeight + curve;

    drawLed(ctx, centreX + dx, topY, cellW, cellH, 0.96);

    if (openHeight > 0) {
      drawLed(ctx, centreX + dx, bottomY, cellW, cellH, 0.96);
    }
  }

  // Light the corners so even subtle smiles/frowns read clearly at matrix scale.
  const cornerLift = Math.round(clamp(curvature, -1, 1) * 3);
  drawLed(
    ctx,
    centreX - halfWidth - 1,
    centreY + cornerLift,
    cellW,
    cellH,
    1
  );
  drawLed(
    ctx,
    centreX + halfWidth + 1,
    centreY + cornerLift,
    cellW,
    cellH,
    1
  );
}

function drawLed(ctx, x, y, cellW, cellH, intensity) {
  if (x < 0 || x >= 64 || y < 0 || y >= 32) return;

  const active = intensity > 0.2;
  const radius = Math.max(1.35, Math.min(cellW, cellH) * 0.31);
  const cx = x * cellW + cellW / 2;
  const cy = y * cellH + cellH / 2;

  ctx.save();

  if (active) {
    ctx.fillStyle =
      `rgba(255, 107, 94, ${clamp(intensity, 0.72, 1)})`;
    ctx.shadowColor = "rgba(255, 107, 94, .55)";
    ctx.shadowBlur = Math.max(2, radius * 1.5);
  } else {
    ctx.fillStyle = "rgba(112, 42, 37, .34)";
    ctx.shadowBlur = 0;
  }

  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
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


function isFirefoxFamily() {
  return /Firefox\//i.test(navigator.userAgent);
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
  const leftEyeRatio =
    raw.leftEyeOpen / Math.max(base.leftEyeOpen, 1e-9);
  const rightEyeRatio =
    raw.rightEyeOpen / Math.max(base.rightEyeOpen, 1e-9);

  // Neutral should look neutral, not maximally open.
  const leftEyeOpen = clamp(
    0.72 + (leftEyeRatio - 1) * 2.2,
    0,
    1
  );

  const rightEyeOpen = clamp(
    0.72 + (rightEyeRatio - 1) * 2.2,
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
        (raw.leftBrowHeight - base.leftBrowHeight) * 55 *
          TUNING.browHeightGain,
        -1,
        1
      ),
      0.08
    ),

    rightBrowHeight: deadZone(
      clamp(
        (raw.rightBrowHeight - base.rightBrowHeight) * 55 *
          TUNING.browHeightGain,
        -1,
        1
      ),
      0.08
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

function medianMeasurements(samples) {
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
    const values = samples
      .map((sample) => sample[key])
      .sort((a, b) => a - b);

    const middle = Math.floor(values.length / 2);

    result[key] = values.length % 2
      ? values[middle]
      : (values[middle - 1] + values[middle]) / 2;
  }

  return result;
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
    leftEyeOpen: 0.72,
    rightEyeOpen: 0.72,
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
