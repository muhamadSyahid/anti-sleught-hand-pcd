from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp

LANDMARK_COUNT = 21
POSE_CLASSES = {"normal", "second_dealing", "bottom_dealing"}


@dataclass
class FrameLandmarks:
    frame_index: int
    timestamp_ms: float
    label: str
    visibility: float
    handedness: str
    landmarks: list[float]


@dataclass
class SequenceRecord:
    clip_name: str
    label: str
    video_path: str
    frame_count: int
    detected_frames: int
    frames: list[FrameLandmarks]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Hands landmark sequences from labeled videos.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("videos"),
        help="Folder containing labeled video clips or nested label folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("landmarks_dataset"),
        help="Folder where the JSONL/CSV outputs will be written.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.6,
        help="MediaPipe Hands minimum detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe Hands minimum tracking confidence.",
    )
    return parser.parse_args()


def infer_label(video_path: Path) -> str:
    for part in [video_path.parent.name, video_path.stem]:
        normalized = part.lower().strip()
        for label in POSE_CLASSES:
            if label in normalized:
                return label
    return "unknown"


def flatten_landmarks(hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> list[float]:
    values: list[float] = []
    for landmark in hand_landmarks.landmark:
        values.extend(
            [
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
                float(landmark.visibility) if hasattr(landmark, "visibility") else 0.0,
            ]
        )
    return values


def normalize_landmarks(landmarks: list[float]) -> list[float]:
    if len(landmarks) != LANDMARK_COUNT * 4:
        return landmarks

    xs = landmarks[0::4]
    ys = landmarks[1::4]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)

    normalized: list[float] = []
    for index in range(LANDMARK_COUNT):
        x = landmarks[index * 4]
        y = landmarks[index * 4 + 1]
        z = landmarks[index * 4 + 2]
        visibility = landmarks[index * 4 + 3]
        normalized.extend(
            [
                (x - min_x) / width,
                (y - min_y) / height,
                z,
                visibility,
            ]
        )
    return normalized


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.mp4")):
        yield path
    for path in sorted(root.rglob("*.mov")):
        yield path
    for path in sorted(root.rglob("*.avi")):
        yield path


def extract_sequences(args: argparse.Namespace) -> list[SequenceRecord]:
    mp_hands = mp.solutions.hands
    records: list[SequenceRecord] = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        for video_path in iter_video_files(args.input):
            record = extract_sequence_for_video(video_path, hands, args.stride)
            if record is not None:
                records.append(record)

    return records


def extract_sequence_for_video(
    video_path: Path,
    hands: mp.solutions.hands.Hands,
    stride: int,
) -> SequenceRecord | None:
    label = infer_label(video_path)
    clip_name = video_path.stem
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    try:
        frame_index = 0
        detected_frames = 0
        frames: list[FrameLandmarks] = []
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            detected = detect_landmarks_in_frame(frame, hands, label, frame_index, capture.get(cv2.CAP_PROP_POS_MSEC))
            if detected is not None:
                detected_frames += 1
                frames.append(detected)

            frame_index += 1

        return SequenceRecord(
            clip_name=clip_name,
            label=label,
            video_path=str(video_path),
            frame_count=total_frames,
            detected_frames=detected_frames,
            frames=frames,
        )
    finally:
        capture.release()


def detect_landmarks_in_frame(
    frame,
    hands: mp.solutions.hands.Hands,
    label: str,
    frame_index: int,
    timestamp_ms: float,
) -> FrameLandmarks | None:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]
    handedness = "unknown"
    if result.multi_handedness:
        handedness = result.multi_handedness[0].classification[0].label.lower()

    raw_landmarks = flatten_landmarks(hand_landmarks)
    return FrameLandmarks(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        label=label,
        visibility=sum(raw_landmarks[3::4]) / LANDMARK_COUNT if raw_landmarks else 0.0,
        handedness=handedness,
        landmarks=normalize_landmarks(raw_landmarks),
    )


def save_outputs(records: list[SequenceRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "sequences.jsonl"
    csv_path = output_dir / "frames.csv"

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(asdict(record), ensure_ascii=False))
            jsonl_file.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "clip_name",
            "label",
            "video_path",
            "frame_index",
            "timestamp_ms",
            "handedness",
            "visibility",
            "landmarks",
        ])
        for record in records:
            for frame in record.frames:
                writer.writerow(
                    [
                        record.clip_name,
                        frame.label,
                        record.video_path,
                        frame.frame_index,
                        frame.timestamp_ms,
                        frame.handedness,
                        frame.visibility,
                        json.dumps(frame.landmarks),
                    ]
                )


def main() -> int:
    args = parse_args()
    records = extract_sequences(args)
    save_outputs(records, args.output)
    print(f"Saved {len(records)} sequence(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
