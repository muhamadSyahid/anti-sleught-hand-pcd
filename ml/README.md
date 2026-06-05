# YOLOv8 TCG model training

This folder contains the files needed to build the TFLite model used by the Flutter app.

## Expected dataset layout

Place your dataset here:

```text
datasets/tcg_vision/
  images/
    train/
    val/
  labels/
    train/
    val/
```

The labels must use this class order:

1. hand
2. card
3. deck

## How to get the dataset

You usually need to build this dataset yourself from recorded table footage, because the exact TCG dealing patterns you care about are specific to your competition rules and camera angle.

Recommended workflow:

1. Record short videos of normal dealing and suspicious dealing from the same camera angle you will use in the app.
2. Extract frames from those videos, then keep frames where the hand, card, and deck are clearly visible.
3. Label each frame in YOLO format with bounding boxes for `hand`, `card`, and `deck`.
4. Split the labeled images into `train` and `val` folders, usually around 80/20.

Good tools for labeling:

- CVAT
- Roboflow
- LabelImg

You can also bootstrap the dataset by mixing your own footage with public hand/object datasets, but for this project the important part is your own table footage, because bottom dealing and second dealing need the same viewpoint and dealing style you will judge in competition.

If you want a quick start, record 10 to 20 minutes of video and extract every 5th or 10th frame. That is usually enough to build an initial prototype dataset.

## Landmark-based alternative

If you do not want to label bounding boxes, you can extract MediaPipe Hands landmarks from your videos instead. That produces a much smaller dataset and is a good fit for sequence classification.

See [ml/landmarks/README.md](ml/landmarks/README.md) and [ml/extract_landmark_sequences.py](ml/extract_landmark_sequences.py) for the landmark pipeline.

## Train and export

Install Ultralytics and run the script:

```bash
python ml/train_and_export.py
```

The script will:

- train a YOLOv8n model on the dataset in `ml/data.yaml`
- export the trained weights to TFLite
- copy the final model to `assets/models/yolov8_tcg.tflite`
- refresh `assets/models/labels.txt`

## Notes

- The app currently uses OpenCV preprocessing plus a YOLOv8 TFLite hook.
- If `assets/models/yolov8_tcg.tflite` is missing, the app falls back to preprocessing-only behavior.
- For better mobile speed, start with `yolov8n`.
