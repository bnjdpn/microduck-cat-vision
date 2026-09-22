from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from threading import Lock

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from ultralytics import YOLO

from .common import MODEL_ROOT, SIMULATOR_DIST, choose_device


CAT_CLASS_ID = 15
FEEDER_CLASS_IDS = {45, 61, 71}  # bowl, toilet and sink in COCO
FEEDER_ASSIGNMENTS = {"left": "Nooby", "right": "Jaina"}
FEEDER_PRODUCT = "Xiaomi Smart Pet Food Feeder (1st generation)"
FOUNTAIN_PRODUCT = "Petlibro Dockstream 2"
DETECTION_CONFIDENCE = 0.05
DETECTION_IMAGE_SIZE = 640


def _tile_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Full frame plus overlapping zoomed regions for small/occluded cats."""
    candidates = [
        (0, 0, width, height),
        (0, 0, round(width * 0.68), height),
        (round(width * 0.16), 0, round(width * 0.84), height),
        (round(width * 0.32), 0, width, height),
        (0, 0, width, round(height * 0.72)),
        (0, round(height * 0.28), width, height),
    ]
    return list(dict.fromkeys(candidates))


def _overlap(box_a: list[float], box_b: list[float]) -> tuple[float, float]:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    containment = intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0
    return iou, containment


def _deduplicate_detections(
    detections: list[tuple[list[float], float]],
    limit: int = 6,
) -> list[tuple[list[float], float]]:
    kept: list[tuple[list[float], float]] = []
    for location, confidence in sorted(detections, key=lambda item: item[1], reverse=True):
        duplicate = False
        for kept_location, _ in kept:
            iou, containment = _overlap(location, kept_location)
            if iou >= 0.50 or containment >= 0.78:
                duplicate = True
                break
        if not duplicate:
            kept.append((location, confidence))
        if len(kept) >= limit:
            break
    return kept


def _feeder_zone(
    location: list[float],
    width: int,
    height: int,
) -> list[float]:
    """Turn a detected appliance into the smaller tray/eating area."""
    x1, y1, x2, y2 = location
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    if box_height > box_width * 1.12:
        # YOLO often calls the complete tall feeder a toilet. Only its
        # lower tray is relevant when deciding whether a cat is eating.
        y1 = y1 + box_height * 0.55
    else:
        # A sink/bowl detection is normally already the metal tray. Give it
        # a little head room because the cat's muzzle hides its rear edge.
        y1 = y1 - box_height * 0.20
    x1 -= box_width * 0.10
    x2 += box_width * 0.10
    y2 += box_height * 0.10
    return [
        max(0.0, x1),
        max(0.0, y1),
        min(float(width), x2),
        min(float(height), y2),
    ]


def _select_feeders(
    detections: list[tuple[list[float], float, int]],
    width: int,
    height: int,
) -> list[dict]:
    """Select the outer feeder on each side, excluding the centre fountain."""
    selected: dict[str, tuple[float, list[float], float, int]] = {}
    for location, confidence, class_id in detections:
        x1, _, x2, _ = location
        centre = ((x1 + x2) / 2) / max(1, width)
        if centre <= 0.42:
            side = "left"
        elif centre >= 0.58:
            side = "right"
        else:
            continue

        # Confidence remains the main signal. A small outer-position bonus
        # favours the two white feeders over the black fountain between them.
        score = confidence + abs(centre - 0.5) * 0.08
        if class_id == 61:  # complete feeder, commonly classified as toilet
            score += 0.02
        previous = selected.get(side)
        if previous is None or score > previous[0]:
            selected[side] = (score, location, confidence, class_id)

    feeders = []
    for side in ("left", "right"):
        candidate = selected.get(side)
        if candidate is None:
            continue
        _, location, confidence, class_id = candidate
        feeders.append(
            {
                "side": side,
                "assigned_to": FEEDER_ASSIGNMENTS[side],
                "box": _feeder_zone(location, width, height),
                "detection_confidence": confidence,
                "detected_as": {45: "bowl", 61: "toilet", 71: "sink"}[class_id],
            }
        )
    return feeders


class VisionRuntime:
    def __init__(self, model_root: Path, device: str) -> None:
        metadata_path = model_root / "metadata.json"
        weights_path = model_root / "cat_identity.pt"
        detector_path = model_root.parents[1] / "yolo11s.pt"
        if not metadata_path.is_file() or not weights_path.is_file() or not detector_path.is_file():
            raise RuntimeError("Modèle absent. Lance microduck-prepare puis microduck-train.")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.detector = YOLO(str(detector_path))
        self.classifier = YOLO(str(weights_path))
        self.detector_name = detector_path.name
        self.device = device
        self.lock = Lock()

    def predict(self, image: Image.Image) -> dict:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        tile_boxes = _tile_boxes(width, height)
        sources = [np.asarray(image.crop(tile_box)) for tile_box in tile_boxes]
        with self.lock:
            detection_results = self.detector.predict(
                source=sources,
                classes=[CAT_CLASS_ID, *sorted(FEEDER_CLASS_IDS)],
                conf=DETECTION_CONFIDENCE,
                imgsz=DETECTION_IMAGE_SIZE,
                device=self.device,
                max_det=10,
                verbose=False,
            )
            raw_detections: list[tuple[list[float], float]] = []
            raw_feeder_detections: list[tuple[list[float], float, int]] = []
            for tile_box, detected in zip(tile_boxes, detection_results, strict=True):
                offset_x, offset_y = tile_box[0], tile_box[1]
                boxes = [] if detected.boxes is None else detected.boxes
                for box in boxes:
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    location = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if class_id == CAT_CLASS_ID:
                        raw_detections.append((location, confidence))
                    elif class_id in FEEDER_CLASS_IDS:
                        raw_feeder_detections.append((location, confidence, class_id))

            detections = _deduplicate_detections(raw_detections)
            feeders = _select_feeders(raw_feeder_detections, width, height)
            crops: list[Image.Image] = []
            locations: list[list[float]] = []
            detection_confidences: list[float] = []
            for location, detection_confidence in detections:
                x1, y1, x2, y2 = location
                box_width, box_height = x2 - x1, y2 - y1
                margin = 0.14 * max(box_width, box_height)
                crop_box = (
                    max(0, int(x1 - margin)),
                    max(0, int(y1 - margin)),
                    min(width, int(x2 + margin)),
                    min(height, int(y2 + margin)),
                )
                crops.append(image.crop(crop_box))
                locations.append(location)
                detection_confidences.append(detection_confidence)

            classifications = []
            if crops:
                classifications = self.classifier.predict(
                    source=[np.asarray(crop) for crop in crops],
                    imgsz=224,
                    device=self.device,
                    verbose=False,
                )

        predictions = []
        confidence_threshold = float(self.metadata["confidence_threshold"])
        margin_threshold = float(self.metadata["margin_threshold"])
        for location, detection_confidence, result in zip(
            locations, detection_confidences, classifications, strict=True
        ):
            probabilities = result.probs.data.detach().cpu().numpy().astype(float)
            order = np.argsort(probabilities)[::-1]
            top_index, second_index = int(order[0]), int(order[1])
            confidence = float(probabilities[top_index])
            margin = float(probabilities[top_index] - probabilities[second_index])
            raw_label = self.classifier.names[top_index]
            accepted = confidence >= confidence_threshold and margin >= margin_threshold
            predictions.append(
                {
                    "label": raw_label if accepted else "Inconnu",
                    "candidate": raw_label,
                    "confidence": confidence,
                    "margin": margin,
                    "detection_confidence": detection_confidence,
                    "box": location,
                }
            )
        return {
            "width": width,
            "height": height,
            "predictions": predictions,
            "feeders": feeders,
        }


def create_app(model_root: Path = MODEL_ROOT, simulator_dist: Path = SIMULATOR_DIST, device: str = "auto") -> FastAPI:
    app = FastAPI(title="Microduck Cat Vision", version="0.1.0")
    runtime = VisionRuntime(model_root, choose_device(device))

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ready": True,
            "device": runtime.device,
            "detector": runtime.detector_name,
            "detection_strategy": "full-frame + overlapping tiles + left/right feeder zones",
            "feeder_assignments": FEEDER_ASSIGNMENTS,
            "feeder_product": FEEDER_PRODUCT,
            "centre_fountain": FOUNTAIN_PRODUCT,
            "labels": runtime.metadata["labels"],
            "test_accuracy": runtime.metadata["accuracy"],
        }

    @app.get("/api/config")
    def config() -> dict:
        return {
            "labels": runtime.metadata["labels"],
            "sounds": runtime.metadata["sounds"],
            "confidence_threshold": runtime.metadata["confidence_threshold"],
            "margin_threshold": runtime.metadata["margin_threshold"],
        }

    @app.post("/api/predict")
    async def predict(image: UploadFile = File(...)) -> dict:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Le fichier doit être une image.")
        payload = await image.read()
        if len(payload) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image trop volumineuse.")
        try:
            with Image.open(io.BytesIO(payload)) as opened:
                return runtime.predict(opened.copy())
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail="Image illisible.") from error

    if simulator_dist.is_dir():
        @app.get("/favicon.ico", include_in_schema=False)
        def favicon() -> FileResponse:
            return FileResponse(simulator_dist / "assets" / "duck-head-mark.webp")

        app.mount("/", StaticFiles(directory=simulator_dist, html=True), name="simulator")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve cat recognition and the Microduck simulator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(device=args.device), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
