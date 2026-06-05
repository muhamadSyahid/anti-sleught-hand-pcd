# Landmark-based sequence extraction

This folder is for lightweight datasets built from MediaPipe Hands landmarks instead of raw video pixels.

## What to record

Capture short clips of:

- normal dealing
- second dealing
- bottom dealing

Keep the same camera position, distance, and lighting you expect during the competition.

## Input layout

You can either put videos directly in `ml/videos` or use label folders like this:

```text
videos/
  normal/
  second_dealing/
  bottom_dealing/
```

The script will also infer labels from the filename, so names like `normal_01.mp4` or `bottom_dealing_trial2.mov` work too.

## Run the extractor

```bash
pip install -r ml/requirements.txt
python ml/extract_landmark_sequences.py --input videos --output ml/landmarks_dataset
```

## Output

The script writes:

- `sequences.jsonl` for one record per clip
- `frames.csv` for frame-level landmark rows

Each frame contains 21 hand landmarks with normalized `x`, `y`, `z`, and visibility-like values.

## Next step

Use the exported landmark sequences to train a small classifier such as:

- RandomForest
- XGBoost
- 1D CNN
- LSTM or GRU

For a quick prototype, start with `RandomForest` or `XGBoost` on flattened sequences.

## Training idea

After extraction, you can build one sample per clip by concatenating the landmark frames in time order, then train a classifier to predict:

- normal
- second_dealing
- bottom_dealing

That is the whole point of landmark-based extraction: you skip training on raw pixels and instead learn from motion patterns in the hand landmarks.
