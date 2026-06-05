# TCG Vision Dataset

Use YOLO format for three classes:

1. hand
2. card
3. deck

Expected layout:

```text
datasets/tcg_vision/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each image must have a matching `.txt` label file with normalized YOLO boxes.