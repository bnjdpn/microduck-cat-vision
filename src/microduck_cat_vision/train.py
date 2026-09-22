from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .common import ARTIFACT_ROOT, DATASET_ROOT, MODEL_ROOT, choose_device, write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def class_paths(root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(label_dir.glob("*.jpg")):
            pairs.append((path, label_dir.name))
    return pairs


def prepare_training_dataset(dataset_root: Path, artifact_root: Path) -> tuple[Path, dict[str, int]]:
    """Build a disposable training view while keeping validation and test fully real."""
    source_root = dataset_root / "splits"
    training_root = artifact_root / "training_dataset"
    if training_root.exists():
        shutil.rmtree(training_root)
    shutil.copytree(source_root, training_root, ignore=shutil.ignore_patterns("*.cache"))

    counts: Counter[str] = Counter()
    synthetic_root = dataset_root / "synthetic"
    if synthetic_root.is_dir():
        for label_dir in sorted(path for path in synthetic_root.iterdir() if path.is_dir()):
            target_dir = training_root / "train" / label_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(label_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                shutil.copyfile(image_path, target_dir / f"synthetic-{image_path.name}")
                counts[label_dir.name] += 1

    return training_root, dict(sorted(counts.items()))


def evaluate(model: YOLO, dataset_root: Path, device: str) -> tuple[dict, list[dict]]:
    test_pairs = class_paths(dataset_root / "test")
    labels = [name for _, name in test_pairs]
    paths = [str(path) for path, _ in test_pairs]
    if not paths:
        raise SystemExit("Le jeu de test est vide.")
    results = model.predict(source=paths, imgsz=224, device=device, verbose=False)
    names = model.names
    rows: list[dict] = []
    per_class = defaultdict(lambda: Counter(total=0, correct=0))
    correct_confidences: list[float] = []
    margins: list[float] = []
    for (path, expected), result in zip(test_pairs, results, strict=True):
        probabilities = result.probs.data.detach().cpu().numpy().astype(float)
        order = np.argsort(probabilities)[::-1]
        predicted = names[int(order[0])]
        confidence = float(probabilities[order[0]])
        margin = float(probabilities[order[0]] - probabilities[order[1]])
        correct = predicted == expected
        per_class[expected]["total"] += 1
        per_class[expected]["correct"] += int(correct)
        if correct:
            correct_confidences.append(confidence)
            margins.append(margin)
        rows.append(
            {
                "path": str(path),
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "margin": margin,
                "correct": correct,
            }
        )

    accuracy = sum(row["correct"] for row in rows) / len(rows)
    # This is a conservative known-class threshold, not a calibrated open-set guarantee.
    confidence_threshold = max(0.72, min(0.84, float(np.percentile(correct_confidences, 10)) - 0.03))
    margin_threshold = max(0.25, min(0.68, float(np.percentile(margins, 10)) - 0.05))
    metrics = {
        "test_images": len(rows),
        "accuracy": accuracy,
        "per_class": {
            label: {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": counts["correct"] / counts["total"],
            }
            for label, counts in sorted(per_class.items())
        },
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "threshold_note": "Thresholds reject uncertain known-class examples; unknown cats were not available for calibration.",
    }
    return metrics, rows


def train(
    dataset_root: Path,
    artifact_root: Path,
    epochs: int,
    device: str,
    seed: int,
    reuse_best: bool = False,
) -> dict:
    split_root = dataset_root / "splits"
    for required in ("train", "val", "test"):
        if not (split_root / required).is_dir():
            raise SystemExit(f"Dataset absent : {split_root / required}. Lance d'abord microduck-prepare.")

    training_root, synthetic_counts = prepare_training_dataset(dataset_root, artifact_root)

    run_root = artifact_root / "runs"
    model_root = artifact_root / "model"
    run_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    target_pt = model_root / "cat_identity.pt"
    fine_tuning = bool(synthetic_counts) and target_pt.is_file()
    training_strategy = "frozen low-rate fine-tune from deployed model" if fine_tuning else "ImageNet transfer learning"

    best_path = run_root / "cat_identity_augmented" / "weights" / "best.pt"
    if reuse_best:
        if not best_path.is_file():
            raise SystemExit(f"Modèle existant absent : {best_path}")
    else:
        model = YOLO(str(target_pt) if fine_tuning else "yolo11n-cls.pt")
        result = model.train(
            data=str(training_root),
            epochs=epochs,
            patience=6 if fine_tuning else 12,
            imgsz=224,
            batch=16,
            workers=0,
            cache=True,
            device=device,
            seed=seed,
            deterministic=True,
            optimizer="AdamW" if fine_tuning else "auto",
            lr0=0.0001 if fine_tuning else 0.01,
            lrf=0.1 if fine_tuning else 0.01,
            freeze=8 if fine_tuning else None,
            project=str(run_root),
            name="cat_identity_augmented",
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        best_path = Path(result.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_path))
    metrics, rows = evaluate(best_model, split_root, device)

    shutil.copyfile(best_path, target_pt)
    export_path = Path(
        best_model.export(format="onnx", imgsz=224, simplify=True, opset=17, device="cpu")
    )
    target_onnx = model_root / "cat_identity.onnx"
    shutil.copyfile(export_path, target_onnx)

    with (model_root / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        **metrics,
        "labels": [best_model.names[index] for index in sorted(best_model.names)],
        "input_size": 224,
        "training_device": device,
        "epochs_requested": epochs,
        "seed": seed,
        "synthetic_training_images": synthetic_counts,
        "evaluation_dataset": "real held-out photos only",
        "training_strategy": training_strategy,
        "weights": str(target_pt.relative_to(artifact_root.parent)),
        "onnx": str(target_onnx.relative_to(artifact_root.parent)),
        "sounds": {
            "Nooby": "duck2/chirp_c.wav",
            "Jaina": "duck4/chirp_k.wav",
        },
    }
    write_json(model_root / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the Nooby/Jaina classifier.")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--reuse-best", action="store_true", help="Évalue et exporte le best.pt existant.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = train(
        dataset_root=args.dataset.resolve(),
        artifact_root=args.artifacts.resolve(),
        epochs=args.epochs,
        device=choose_device(args.device),
        seed=args.seed,
        reuse_best=args.reuse_best,
    )
    print("Modèle prêt :")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
