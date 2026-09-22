from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import imagehash
import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

from .common import DATASET_ROOT, PROJECT_ROOT, choose_device, write_json


LABELS = ("Jaina", "Nooby")
CAT_CLASS_ID = 15  # COCO's cat class.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


@dataclass
class Candidate:
    label: str
    source: Path
    digest: str
    session: str
    crop: Image.Image | None = None
    phash: imagehash.ImageHash | None = None
    confidence: float = 0.0
    reason: str = ""


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_session(image: Image.Image, path: Path) -> str:
    try:
        exif = image.getexif()
        reverse_tags = {name: tag for tag, name in ExifTags.TAGS.items()}
        value = exif.get(reverse_tags.get("DateTimeOriginal")) or exif.get(reverse_tags.get("DateTime"))
        if value:
            return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S").date().isoformat()
    except (TypeError, ValueError, OSError):
        pass
    return f"unknown-{path.stem}"


def load_candidates(source_root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for label in LABELS:
        folder = source_root / label
        if not folder.is_dir():
            raise SystemExit(f"Dossier manquant : {folder}")
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                with Image.open(path) as raw:
                    candidates.append(
                        Candidate(
                            label=label,
                            source=path,
                            digest=sha256(path),
                            session=capture_session(raw, path),
                        )
                    )
    return candidates


def mark_duplicate_conflicts(candidates: list[Candidate]) -> None:
    by_digest: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_digest[candidate.digest].append(candidate)
    for duplicates in by_digest.values():
        labels = {candidate.label for candidate in duplicates}
        if len(labels) > 1:
            for candidate in duplicates:
                candidate.reason = "duplicate_cross_label"
        elif len(duplicates) > 1:
            for candidate in duplicates[1:]:
                candidate.reason = "duplicate_same_label"


def detect_single_cat(
    model: YOLO,
    candidate: Candidate,
    device: str,
    confidence: float,
    image_size: int,
) -> None:
    if candidate.reason:
        return
    with Image.open(candidate.source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    image_array = np.asarray(image)
    result = model.predict(
        source=image_array,
        classes=[CAT_CLASS_ID],
        conf=confidence,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]
    boxes = result.boxes
    count = 0 if boxes is None else len(boxes)
    if count == 0:
        candidate.reason = "no_cat_detected"
        return
    if count > 1:
        candidate.reason = f"multiple_cats_detected:{count}"
        return

    x1, y1, x2, y2 = [float(value) for value in boxes.xyxy[0].tolist()]
    width, height = image.size
    box_width, box_height = x2 - x1, y2 - y1
    if box_width * box_height < width * height * 0.025:
        candidate.reason = "cat_too_small"
        return
    margin = 0.14 * max(box_width, box_height)
    crop_box = (
        max(0, int(x1 - margin)),
        max(0, int(y1 - margin)),
        min(width, int(x2 + margin)),
        min(height, int(y2 + margin)),
    )
    candidate.crop = image.crop(crop_box)
    candidate.phash = imagehash.phash(candidate.crop.resize((256, 256)))
    candidate.confidence = float(boxes.conf[0])


def group_candidates(candidates: list[Candidate], threshold: int) -> list[list[Candidate]]:
    union = UnionFind(len(candidates))
    sessions: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        sessions[candidate.session].append(index)
    for members in sessions.values():
        for member in members[1:]:
            union.union(members[0], member)
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if candidates[left].phash - candidates[right].phash <= threshold:
                union.union(left, right)
    groups: dict[int, list[Candidate]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[union.find(index)].append(candidate)
    return list(groups.values())


def split_groups(groups: list[list[Candidate]], seed: int) -> dict[str, list[Candidate]]:
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = sum(len(group) for group in groups)
    targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
    splits: dict[str, list[Candidate]] = {"train": [], "val": [], "test": []}
    for group in groups:
        split = min(splits, key=lambda name: len(splits[name]) / max(targets[name], 1))
        splits[split].extend(group)
    return splits


def save_contact_sheet(paths: list[Path], destination: Path, title: str) -> None:
    if not paths:
        return
    thumb_size = 180
    columns = 6
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + 22) + 42), "#f6f1e7")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 12), title, fill="#111111", font=font)
    for index, path in enumerate(paths):
        with Image.open(path) as raw:
            image = ImageOps.contain(raw.convert("RGB"), (thumb_size - 8, thumb_size - 8))
        x = (index % columns) * thumb_size + (thumb_size - image.width) // 2
        y = 42 + (index // columns) * (thumb_size + 22) + (thumb_size - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((index % columns * thumb_size + 4, y + image.height + 2), path.stem[:20], fill="#222222", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=88)


def prepare(
    source_root: Path,
    dataset_root: Path,
    device: str,
    confidence: float,
    image_size: int,
    seed: int,
) -> dict:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    (dataset_root / "crops").mkdir(parents=True)
    (dataset_root / "splits").mkdir(parents=True)
    (dataset_root / "reports").mkdir(parents=True)

    candidates = load_candidates(source_root)
    mark_duplicate_conflicts(candidates)
    detector = YOLO("yolo11n.pt")
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index:03}/{len(candidates):03}] {candidate.label}/{candidate.source.name}")
        detect_single_cat(detector, candidate, device, confidence, image_size)

    accepted = [candidate for candidate in candidates if candidate.crop is not None]
    split_counts: dict[str, Counter] = {}
    saved_by_label: dict[str, list[Path]] = defaultdict(list)

    for label in LABELS:
        label_candidates = [candidate for candidate in accepted if candidate.label == label]
        groups = group_candidates(label_candidates, threshold=6)
        splits = split_groups(groups, seed=seed + LABELS.index(label))
        split_counts[label] = Counter({name: len(items) for name, items in splits.items()})
        for split, items in splits.items():
            target_dir = dataset_root / "splits" / split / label
            crop_dir = dataset_root / "crops" / label
            target_dir.mkdir(parents=True, exist_ok=True)
            crop_dir.mkdir(parents=True, exist_ok=True)
            for candidate in items:
                filename = f"{candidate.digest[:12]}-{candidate.source.stem}.jpg"
                crop_path = crop_dir / filename
                candidate.crop.save(crop_path, format="JPEG", quality=94, optimize=True)
                target_path = target_dir / filename
                shutil.copyfile(crop_path, target_path)
                saved_by_label[label].append(crop_path)

    with (dataset_root / "reports" / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "source", "status", "detector_confidence", "session"])
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.label,
                    str(candidate.source.relative_to(source_root)),
                    candidate.reason or "accepted",
                    f"{candidate.confidence:.6f}" if candidate.confidence else "",
                    candidate.session,
                ]
            )

    report = {
        "source_counts": dict(Counter(candidate.label for candidate in candidates)),
        "accepted_counts": dict(Counter(candidate.label for candidate in accepted)),
        "rejected_reasons": dict(Counter(candidate.reason for candidate in candidates if candidate.reason)),
        "split_counts": {label: dict(counts) for label, counts in split_counts.items()},
        "device": device,
        "detector": "yolo11n.pt",
        "detector_confidence": confidence,
        "detector_image_size": image_size,
        "seed": seed,
    }
    write_json(dataset_root / "reports" / "dataset_report.json", report)
    for label, paths in saved_by_label.items():
        save_contact_sheet(paths, dataset_root / "reports" / f"{label.lower()}-accepted.jpg", f"{label} — accepted crops")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect, crop and split Nooby/Jaina photos.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare(
        source_root=args.source.resolve(),
        dataset_root=args.output.resolve(),
        device=choose_device(args.device),
        confidence=args.confidence,
        image_size=args.image_size,
        seed=args.seed,
    )
    print("Dataset prêt :")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
