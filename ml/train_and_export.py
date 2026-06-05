from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "ml" / "data.yaml"
ASSETS_MODELS = ROOT / "assets" / "models"
OUTPUT_TFLITE = ASSETS_MODELS / "yolov8_tcg.tflite"
OUTPUT_LABELS = ASSETS_MODELS / "labels.txt"
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640


def ensure_dataset_exists() -> None:
    splits = {
        "train": (
            ROOT / "datasets" / "tcg_vision" / "images" / "train",
            ROOT / "datasets" / "tcg_vision" / "labels" / "train",
        ),
        "val": (
            ROOT / "datasets" / "tcg_vision" / "images" / "val",
            ROOT / "datasets" / "tcg_vision" / "labels" / "val",
        ),
        "test": (
            ROOT / "datasets" / "tcg_vision" / "images" / "test",
            ROOT / "datasets" / "tcg_vision" / "labels" / "test",
        ),
    }

    missing: list[str] = []
    for split_name, (images_dir, labels_dir) in splits.items():
        if not images_dir.exists():
            missing.append(str(images_dir))
        if not labels_dir.exists():
            missing.append(str(labels_dir))

    if missing:
        raise FileNotFoundError(
            "Expected YOLO dataset folders are missing:\n  "
            + "\n  ".join(missing)
        )


def build_labels_file() -> None:
    ASSETS_MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.write_text("hand\ncard\ndeck\n", encoding="utf-8")


def run_test_evaluation(model_path: Path) -> None:
    from ultralytics import YOLO

    print("\n--- Running test-set evaluation ---")
    test_model = YOLO(str(model_path))
    metrics = test_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=DEFAULT_IMGSZ,
    )
    print(f"Test mAP50      : {metrics.box.map50:.4f}")
    print(f"Test mAP50-95   : {metrics.box.map:.4f}")
    print(f"Test Precision  : {metrics.box.mp:.4f}")
    print(f"Test Recall     : {metrics.box.mr:.4f}")
    print("--- Test evaluation complete ---\n")


def main() -> int:
    ensure_dataset_exists()
    build_labels_file()

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install ultralytics"
        ) from error

    model = YOLO(DEFAULT_MODEL)
    model.train(
        data=str(DATA_YAML),
        imgsz=DEFAULT_IMGSZ,
        epochs=DEFAULT_EPOCHS,
        project="runs/tcg",
        name="yolov8_tcg",
    )

    best_pt = ROOT / "runs" / "tcg" / "yolov8_tcg" / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training did not produce {best_pt}")

    run_test_evaluation(best_pt)

    trained_model = YOLO(str(best_pt))
    exported_path = trained_model.export(format="tflite", imgsz=DEFAULT_IMGSZ)
    if isinstance(exported_path, (str, Path)):
        exported_path = Path(exported_path)
    else:
        exported_path = best_pt.with_suffix(".tflite")

    if not exported_path.exists():
        fallback_candidates = sorted(
            best_pt.parent.glob("*.tflite"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not fallback_candidates:
            raise FileNotFoundError(f"No .tflite file was exported from {best_pt.parent}")
        exported_path = fallback_candidates[0]

    OUTPUT_TFLITE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exported_path, OUTPUT_TFLITE)
    print(f"Copied model to {OUTPUT_TFLITE}")
    print("Done. Add the model to your Flutter app and rebuild.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())