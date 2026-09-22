import { useCallback, useEffect, useRef, useState } from "react";
import Box from "@mui/material/Box";
import { useGame, gameApi } from "../store.js";
import { ORANGE, MONO } from "../theme.js";
import { ANTON, COMIC_INK, CREAM } from "./comic.jsx";

const KNOWN_CATS = new Set(["Nooby", "Jaina"]);
const FEEDER_ASSIGNMENTS = { left: "Nooby", right: "Jaina" };
const FEEDING_CONFIRMATION_FRAMES = 2;
const FEEDER_CACHE_MS = 8_000;
const CAMERA_FRAME_TIMEOUT_MS = 4_000;
const VIRTUAL_CAMERA_PATTERN = /virtual|vcam|cameraextension|obs|snap camera|mmhmm|screen capture|ndi/i;

const buttonSx = {
  appearance: "none",
  border: `2px solid ${COMIC_INK}`,
  borderRadius: 0,
  background: CREAM,
  color: COMIC_INK,
  px: "0.7rem",
  py: "0.45rem",
  fontFamily: ANTON,
  fontSize: "0.72rem",
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  boxShadow: `2px 2px 0 ${COMIC_INK}`,
  cursor: "pointer",
  "&:hover": { background: ORANGE },
  "&:disabled": { opacity: 0.42, cursor: "not-allowed" },
};

function cameraPriority(camera) {
  if (VIRTUAL_CAMERA_PATTERN.test(camera.label)) return 2;
  if (/macbook|facetime|logitech|c920|webcam/i.test(camera.label)) return 0;
  return 1;
}

async function enumerateVideoInputs({ requestPermission = false } = {}) {
  let cameras = (await navigator.mediaDevices.enumerateDevices())
    .filter((device) => device.kind === "videoinput");

  if (requestPermission && cameras.length && cameras.every((camera) => !camera.label)) {
    const permissionProbe = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    for (const track of permissionProbe.getTracks()) track.stop();
    cameras = (await navigator.mediaDevices.enumerateDevices())
      .filter((device) => device.kind === "videoinput");
  }

  return cameras.sort((left, right) => cameraPriority(left) - cameraPriority(right));
}

function attachCameraStream(video, stream) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      window.clearTimeout(timer);
      video.removeEventListener("loadedmetadata", checkFrame);
      video.removeEventListener("canplay", checkFrame);
      video.removeEventListener("error", fail);
    };
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback(value);
    };
    const checkFrame = () => {
      if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        finish(resolve);
      }
    };
    const fail = () => finish(reject, new Error("CameraPlaybackError"));
    const timer = window.setTimeout(
      () => finish(reject, new Error("CameraFrameTimeout")),
      CAMERA_FRAME_TIMEOUT_MS,
    );

    video.addEventListener("loadedmetadata", checkFrame);
    video.addEventListener("canplay", checkFrame);
    video.addEventListener("error", fail);
    video.srcObject = stream;
    video.play().then(checkFrame).catch((error) => finish(reject, error));
  });
}

function DetectionBox({ prediction, width, height }) {
  if (!width || !height) return null;
  const [x1, y1, x2, y2] = prediction.box;
  const known = KNOWN_CATS.has(prediction.label);
  return (
    <Box
      sx={{
        position: "absolute",
        left: `${(x1 / width) * 100}%`,
        top: `${(y1 / height) * 100}%`,
        width: `${((x2 - x1) / width) * 100}%`,
        height: `${((y2 - y1) / height) * 100}%`,
        border: `2px solid ${known ? ORANGE : "#b5bac5"}`,
        boxShadow: "0 0 0 1px rgba(8,8,12,0.7)",
        pointerEvents: "none",
      }}
    >
      <Box
        sx={{
          position: "absolute",
          left: -2,
          top: -23,
          background: known ? ORANGE : "#b5bac5",
          color: COMIC_INK,
          px: "5px",
          py: "2px",
          fontFamily: MONO,
          fontWeight: 800,
          fontSize: "0.6rem",
          whiteSpace: "nowrap",
        }}
      >
        {prediction.label.toUpperCase()} {Math.round(prediction.confidence * 100)}%
      </Box>
    </Box>
  );
}

function FeederZone({ feeder, alert }) {
  const [x1, y1, x2, y2] = feeder.box;
  const colour = alert ? "#ff4d38" : feeder.side === "left" ? "#f9a23b" : "#63d8ff";
  return (
    <Box
      sx={{
        position: "absolute",
        left: `${x1 * 100}%`,
        top: `${y1 * 100}%`,
        width: `${(x2 - x1) * 100}%`,
        height: `${(y2 - y1) * 100}%`,
        border: `2px dashed ${colour}`,
        background: alert ? "rgba(255,77,56,.13)" : "rgba(0,0,0,.06)",
        pointerEvents: "none",
      }}
    >
      <Box
        sx={{
          position: "absolute",
          left: -2,
          bottom: -20,
          background: colour,
          color: COMIC_INK,
          px: "5px",
          py: "2px",
          fontFamily: MONO,
          fontWeight: 900,
          fontSize: "0.5rem",
          whiteSpace: "nowrap",
        }}
      >
        {feeder.side === "left" ? "GAUCHE" : "DROITE"} · {feeder.assignedTo.toUpperCase()}
      </Box>
    </Box>
  );
}

function normaliseFeeder(feeder, width, height, observedAt) {
  if (!Array.isArray(feeder?.box) || !width || !height) return null;
  const [x1, y1, x2, y2] = feeder.box;
  return {
    side: feeder.side,
    assignedTo: feeder.assigned_to || FEEDER_ASSIGNMENTS[feeder.side],
    box: [x1 / width, y1 / height, x2 / width, y2 / height],
    observedAt,
  };
}

function boxesOverlapEnough(catBox, feederBox, width, height) {
  const [cx1, cy1, cx2, fullCy2] = [
    catBox[0] / width,
    catBox[1] / height,
    catBox[2] / width,
    catBox[3] / height,
  ];
  // The fixed feeder view sees a cat from behind: the head/shoulders are
  // the upper part of its detection box, nearest the wall and the tray.
  // Testing only that leading 58% avoids treating a tail or hindquarters
  // crossing the zone as eating.
  const cy2 = cy1 + (fullCy2 - cy1) * 0.58;
  const [fx1, fy1, fx2, fy2] = feederBox;
  const intersection = Math.max(0, Math.min(cx2, fx2) - Math.max(cx1, fx1))
    * Math.max(0, Math.min(cy2, fy2) - Math.max(cy1, fy1));
  const feederArea = Math.max(0.0001, (fx2 - fx1) * (fy2 - fy1));
  const catArea = Math.max(0.0001, (cx2 - cx1) * (cy2 - cy1));
  return intersection / Math.min(feederArea, catArea) >= 0.12;
}

export function wrongFeederCandidates(predictions, feeders, width, height) {
  const candidates = [];
  for (const prediction of predictions) {
    if (!KNOWN_CATS.has(prediction.label)) continue;
    for (const feeder of feeders) {
      if (prediction.label === feeder.assignedTo) continue;
      if (!boxesOverlapEnough(prediction.box, feeder.box, width, height)) continue;
      candidates.push({
        label: prediction.label,
        side: feeder.side,
        bearing: ((prediction.box[0] + prediction.box[2]) / 2) / width,
      });
    }
  }
  return candidates;
}

export default function CatVisionPanel() {
  const entered = useGame((state) => state.entered);
  const menuOpen = useGame((state) => state.menuOpen);
  const [expanded, setExpanded] = useState(true);
  const [service, setService] = useState({ state: "checking", detail: "Connexion…" });
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewSize, setPreviewSize] = useState({ width: 0, height: 0 });
  const [predictions, setPredictions] = useState([]);
  const [feeders, setFeeders] = useState([]);
  const [feederViolations, setFeederViolations] = useState([]);
  const [busy, setBusy] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStarting, setCameraStarting] = useState(false);
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState("");
  const [lastReaction, setLastReaction] = useState("En attente d’un chat");
  const fileRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const cameraStartActive = useRef(false);
  const requestActive = useRef(false);
  const feederZonesRef = useRef([]);
  const feedingEvidenceRef = useRef(new Map());

  useEffect(() => {
    let cancelled = false;
    fetch("/api/health")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((health) => {
        if (!cancelled) {
          setService({
            state: "ready",
            detail: `${health.device.toUpperCase()} · TEST ${Math.round(health.test_accuracy * 100)}%`,
          });
        }
      })
      .catch(() => {
        if (!cancelled) setService({ state: "error", detail: "Service vision indisponible" });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!navigator.mediaDevices?.enumerateDevices) return undefined;
    enumerateVideoInputs()
      .then((availableCameras) => {
        if (cancelled) return;
        setCameras(availableCameras);
        if (availableCameras.some((camera) => camera.label)) {
          setSelectedCameraId((current) => current || availableCameras[0]?.deviceId || "");
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const updateFeederZones = useCallback((detected, width, height, { live }) => {
    const now = performance.now();
    const bySide = new Map(
      (live ? feederZonesRef.current : [])
        .filter((feeder) => now - feeder.observedAt <= FEEDER_CACHE_MS)
        .map((feeder) => [feeder.side, feeder]),
    );
    for (const rawFeeder of detected ?? []) {
      const feeder = normaliseFeeder(rawFeeder, width, height, now);
      if (feeder?.assignedTo) bySide.set(feeder.side, feeder);
    }
    const next = ["left", "right"].map((side) => bySide.get(side)).filter(Boolean);
    feederZonesRef.current = next;
    setFeeders(next);
    return next;
  }, []);

  const react = useCallback((nextPredictions, { live, feeders: nextFeeders, width, height }) => {
    const recognized = nextPredictions.filter((prediction) => KNOWN_CATS.has(prediction.label));
    const labels = [...new Set(recognized.map((prediction) => prediction.label))];
    gameApi.observeCats?.(labels, { live });

    let confirmedViolations = [];
    if (live && nextFeeders.length === 2) {
      const candidates = wrongFeederCandidates(nextPredictions, nextFeeders, width, height);
      const previous = feedingEvidenceRef.current;
      const nextEvidence = new Map();
      for (const candidate of candidates) {
        const key = `${candidate.label}:${candidate.side}`;
        const count = (previous.get(key) || 0) + 1;
        nextEvidence.set(key, count);
        if (count >= FEEDING_CONFIRMATION_FRAMES) confirmedViolations.push(candidate);
      }
      feedingEvidenceRef.current = nextEvidence;
    } else {
      feedingEvidenceRef.current.clear();
    }
    setFeederViolations(confirmedViolations);
    gameApi.observeFeederViolations?.(confirmedViolations, { live });

    if (confirmedViolations.length) {
      const names = [...new Set(confirmedViolations.map((violation) => violation.label))].join(" + ");
      const sides = [...new Set(confirmedViolations.map((violation) =>
        violation.side === "left" ? "gauche" : "droite"
      ))].join(" + ");
      setLastReaction(`ALERTE · ${names} mange à ${sides} · poursuite`);
      return;
    }
    if (!recognized.length) {
      setLastReaction(nextFeeders.length < 2
        ? "Cadre les deux distributeurs dans la webcam"
        : nextPredictions.length ? "Chat inconnu — silence" : "Distributeurs surveillés");
      return;
    }
    if (labels.includes("Nooby") && labels.includes("Jaina")) {
      setLastReaction(live
        ? "Jaina + Nooby reconnues · deux annonces + sauts Nooby"
        : "Jaina + Nooby reconnues · deux annonces + saut Nooby");
    } else if (labels.includes("Nooby")) {
      setLastReaction(live
        ? "Nooby reconnue · annonce + sauts continus"
        : "Nooby reconnue · annonce + saut");
    } else {
      setLastReaction("Jaina reconnue · annonce vocale");
    }
  }, []);

  const predictBlob = useCallback(async (blob, { live = false } = {}) => {
    if (!blob || requestActive.current) return;
    requestActive.current = true;
    setBusy(true);
    try {
      const body = new FormData();
      body.append("image", blob, "camera.jpg");
      const response = await fetch("/api/predict", { method: "POST", body });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();
      setPreviewSize({ width: result.width, height: result.height });
      setPredictions(result.predictions);
      const nextFeeders = updateFeederZones(
        result.feeders,
        result.width,
        result.height,
        { live },
      );
      react(result.predictions, {
        live,
        feeders: nextFeeders,
        width: result.width,
        height: result.height,
      });
    } catch {
      if (live) gameApi.clearCatPresence?.();
      setLastReaction("Analyse impossible");
    } finally {
      requestActive.current = false;
      setBusy(false);
    }
  }, [react, updateFeederZones]);

  const stopCamera = useCallback(() => {
    for (const track of streamRef.current?.getTracks?.() ?? []) track.stop();
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    gameApi.clearCatPresence?.();
    feederZonesRef.current = [];
    feedingEvidenceRef.current.clear();
    setFeeders([]);
    setFeederViolations([]);
    setCameraActive(false);
  }, []);

  const startCamera = useCallback(async (requestedCameraId = "") => {
    if (cameraStartActive.current) return;
    cameraStartActive.current = true;
    setCameraStarting(true);
    stopCamera();
    setLastReaction("Connexion à la caméra…");
    try {
      if (!navigator.mediaDevices?.getUserMedia || !videoRef.current) {
        throw new Error("CameraUnsupported");
      }

      const discovered = await enumerateVideoInputs({ requestPermission: true });
      if (!discovered.length) throw new DOMException("No camera", "NotFoundError");
      setCameras(discovered);

      const preferredId = requestedCameraId || selectedCameraId;
      const selected = discovered.find((camera) => camera.deviceId === preferredId)
        || discovered[0];
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: selected.deviceId },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      await attachCameraStream(videoRef.current, stream);
      setSelectedCameraId(selected.deviceId);
      setPreviewSize({ width: videoRef.current.videoWidth, height: videoRef.current.videoHeight });
      setPredictions([]);
      setFeeders([]);
      setFeederViolations([]);
      setPreviewUrl(null);
      setCameraActive(true);
      setLastReaction(`Caméra active · ${selected.label || "source vidéo"}`);
    } catch (error) {
      stopCamera();
      if (error?.name === "NotAllowedError") {
        setLastReaction("Autorise la caméra dans le navigateur");
      } else if (error?.name === "NotFoundError") {
        setLastReaction("Aucune caméra détectée");
      } else if (error?.message === "CameraFrameTimeout") {
        setLastReaction("Caméra connectée, mais aucune image reçue");
      } else {
        setLastReaction("Caméra indisponible");
      }
      console.warn("[Cat vision] Camera unavailable", error);
    } finally {
      cameraStartActive.current = false;
      setCameraStarting(false);
    }
  }, [selectedCameraId, stopCamera]);

  const changeCamera = useCallback((event) => {
    const nextCameraId = event.target.value;
    setSelectedCameraId(nextCameraId);
    if (cameraActive) startCamera(nextCameraId);
  }, [cameraActive, startCamera]);

  useEffect(() => {
    if (!cameraActive) return undefined;
    const analyzeFrame = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2 || requestActive.current) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      canvas.toBlob((blob) => predictBlob(blob, { live: true }), "image/jpeg", 0.86);
    };
    analyzeFrame();
    const timer = window.setInterval(analyzeFrame, 1_200);
    return () => window.clearInterval(timer);
  }, [cameraActive, predictBlob]);

  useEffect(() => () => {
    stopCamera();
  }, [stopCamera]);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const selectPhoto = useCallback((event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    stopCamera();
    setPredictions([]);
    setPreviewSize({ width: 0, height: 0 });
    setPreviewUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return URL.createObjectURL(file);
    });
    setLastReaction("Analyse de la photo…");
    predictBlob(file, { live: false });
    event.target.value = "";
  }, [predictBlob, stopCamera]);

  if (!entered || menuOpen) return null;

  if (!expanded) {
    return (
      <Box
        component="button"
        type="button"
        onClick={() => setExpanded(true)}
        sx={{ ...buttonSx, position: "fixed", top: "5.8rem", left: "1.5rem", zIndex: 12, background: ORANGE }}
      >
        Cat vision
      </Box>
    );
  }

  return (
    <Box
      component="section"
      aria-label="Reconnaissance de Nooby et Jaina"
      sx={{
        position: "fixed",
        top: "5.8rem",
        left: "1.5rem",
        zIndex: 12,
        width: "min(330px, calc(100vw - 3rem))",
        border: `3px solid ${CREAM}`,
        borderRadius: 0,
        background: "rgba(8, 8, 12, 0.86)",
        boxShadow: `inset 0 0 0 1px ${COMIC_INK}, 5px 5px 0 ${COMIC_INK}`,
        p: "10px",
        color: CREAM,
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: "8px" }}>
        <Box>
          <Box sx={{ fontFamily: ANTON, fontSize: "1.05rem", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            Cat vision
          </Box>
          <Box sx={{ fontFamily: MONO, fontSize: "0.55rem", color: service.state === "ready" ? ORANGE : "rgba(255,255,255,.55)", letterSpacing: ".08em" }}>
            {service.detail}
          </Box>
        </Box>
        <Box component="button" type="button" aria-label="Réduire" onClick={() => setExpanded(false)} sx={{ ...buttonSx, px: "0.5rem", py: "0.25rem" }}>
          —
        </Box>
      </Box>

      <Box
        sx={{
          position: "relative",
          aspectRatio: previewSize.width && previewSize.height
            ? `${previewSize.width} / ${previewSize.height}`
            : "4 / 3",
          maxHeight: "42vh",
          overflow: "hidden",
          background: "#15151a",
          border: `2px solid ${COMIC_INK}`,
        }}
      >
        <Box
          component="video"
          ref={videoRef}
          muted
          playsInline
          sx={{ width: "100%", height: "100%", objectFit: "contain", display: previewUrl ? "none" : "block" }}
        />
        {previewUrl ? (
          <Box component="img" src={previewUrl} alt="Photo analysée" sx={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }} />
        ) : null}
        {predictions.map((prediction, index) => (
          <DetectionBox key={`${prediction.box.join("-")}-${index}`} prediction={prediction} width={previewSize.width} height={previewSize.height} />
        ))}
        {feeders.map((feeder) => (
          <FeederZone
            key={feeder.side}
            feeder={feeder}
            alert={feederViolations.some((violation) => violation.side === feeder.side)}
          />
        ))}
        {!previewUrl && !cameraActive ? (
          <Box sx={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", textAlign: "center", px: 3, fontFamily: MONO, fontSize: "0.65rem", color: "rgba(255,255,255,.45)" }}>
            Choisis une photo ou active la webcam
          </Box>
        ) : null}
        {busy ? (
          <Box sx={{ position: "absolute", right: 7, bottom: 7, background: ORANGE, color: COMIC_INK, px: "6px", py: "3px", fontFamily: MONO, fontSize: "0.55rem", fontWeight: 800 }}>
            ANALYSE
          </Box>
        ) : null}
      </Box>

      <Box sx={{ minHeight: 29, display: "flex", alignItems: "center", mt: "7px", fontFamily: MONO, fontSize: "0.62rem", color: "rgba(255,255,255,.7)" }}>
        {lastReaction}
      </Box>
      <Box sx={{ display: "flex", gap: "8px", mt: "5px" }}>
        <Box component="input" ref={fileRef} type="file" accept="image/*" onChange={selectPhoto} sx={{ display: "none" }} />
        <Box component="button" type="button" disabled={service.state !== "ready"} onClick={() => fileRef.current?.click()} sx={{ ...buttonSx, flex: 1 }}>
          Photo
        </Box>
        <Box
          component="button"
          type="button"
          disabled={service.state !== "ready" || cameraStarting}
          onClick={cameraActive ? stopCamera : () => startCamera()}
          sx={{ ...buttonSx, flex: 1, background: cameraActive ? ORANGE : CREAM }}
        >
          {cameraStarting ? "Connexion…" : cameraActive ? "Stop" : "Webcam"}
        </Box>
      </Box>
      {cameras.length ? (
        <Box
          component="select"
          aria-label="Caméra"
          value={selectedCameraId}
          onChange={changeCamera}
          disabled={cameraStarting}
          sx={{ ...buttonSx, width: "100%", mt: "8px", fontFamily: MONO, fontSize: "0.58rem" }}
        >
          {!selectedCameraId ? <option value="">Choisir une caméra</option> : null}
          {cameras.map((camera, index) => (
            <option key={camera.deviceId} value={camera.deviceId}>
              {camera.label || `Caméra ${index + 1}`}
            </option>
          ))}
        </Box>
      ) : null}
      <canvas ref={canvasRef} hidden />
    </Box>
  );
}
