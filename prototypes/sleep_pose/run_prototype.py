"""Run the YOLO-Pose sleep prototype on a local video."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from sleep_logic import SleepState, SleepStateMachine, estimate_head_pose


STATE_COLORS = {
    SleepState.NORMAL: (80, 220, 80),
    SleepState.SUSPECTED: (0, 200, 255),
    SleepState.SLEEPING: (40, 40, 255),
    SleepState.RECOVERING: (255, 180, 40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-Pose sleep detection prototype")
    parser.add_argument("--source", required=True, help="input video path")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose checkpoint")
    parser.add_argument("--output-dir", default="outputs/latest", help="annotated video and JSONL output directory")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker configuration")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--pitch-threshold-deg", type=float, default=18.0)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--recovery-seconds", type=float, default=1.5)
    parser.add_argument("--frame-stride", type=int, default=1, help="run inference every Nth source frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means the complete video")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _safe_timestamp_ms(capture: cv2.VideoCapture, frame_index: int, fps: float) -> int:
    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
    if timestamp > 0:
        return int(round(timestamp))
    return int(round(frame_index * 1000.0 / fps))


def _draw_label(frame: np.ndarray, x: int, y: int, lines: list[str], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    line_height = 22
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    box_width = max(widths, default=120) + 12
    box_height = line_height * len(lines) + 8
    top = max(0, y - box_height)
    left = max(0, x)
    cv2.rectangle(frame, (left, top), (left + box_width, top + box_height), (20, 20, 20), -1)
    cv2.rectangle(frame, (left, top), (left + box_width, top + box_height), color, 2)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (left + 6, top + 19 + index * line_height), font, scale, color, thickness, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "annotated.mp4"
    events_path = output_dir / "frames.jsonl"
    summary_path = output_dir / "summary.json"

    if not source.is_file():
        raise FileNotFoundError(f"video not found: {source}")
    if args.frame_stride <= 0:
        raise ValueError("frame stride must be positive")

    model = YOLO(args.model)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = fps / args.frame_stride
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {video_path}")

    state_machine = SleepStateMachine(
        sleep_duration_ms=max(1, int(args.sleep_seconds * 1000)),
        recovery_duration_ms=max(1, int(args.recovery_seconds * 1000)),
    )
    state_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    sleep_events = 0
    observed_tracks: set[int] = set()
    source_frame_index = 0
    processed_frames = 0
    valid_pose_observations = 0
    low_head_observations = 0
    inference_ms_total = 0.0
    started = time.perf_counter()

    try:
        with events_path.open("w", encoding="utf-8") as event_stream:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                current_frame_index = source_frame_index
                source_frame_index += 1
                if current_frame_index % args.frame_stride != 0:
                    continue
                if args.max_frames > 0 and processed_frames >= args.max_frames:
                    break

                timestamp_ms = _safe_timestamp_ms(capture, current_frame_index, fps)
                infer_started = time.perf_counter()
                result = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.confidence,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                inference_ms_total += (time.perf_counter() - infer_started) * 1000.0
                annotated = result.plot(labels=False, conf=False)
                seen_this_frame: set[int] = set()

                if result.boxes is not None and result.keypoints is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id
                    ids = track_ids.int().cpu().tolist() if track_ids is not None else list(range(len(boxes)))
                    keypoints = result.keypoints.data.cpu().numpy()

                    for box, track_id, person_keypoints in zip(boxes, ids, keypoints):
                        track_id = int(track_id)
                        seen_this_frame.add(track_id)
                        observed_tracks.add(track_id)
                        evidence = estimate_head_pose(
                            person_keypoints,
                            min_confidence=args.keypoint_confidence,
                            pitch_threshold_deg=args.pitch_threshold_deg,
                        )
                        if evidence.valid:
                            valid_pose_observations += 1
                            if evidence.low_head:
                                low_head_observations += 1
                        else:
                            invalid_reasons[evidence.reason] += 1
                        update = state_machine.update(track_id, timestamp_ms, evidence.low_head if evidence.valid else None)
                        state_counts[update.state.value] += 1
                        if update.sleep_event:
                            sleep_events += 1

                        pitch_text = f"pitch={evidence.pitch_proxy_deg:.1f}deg" if evidence.pitch_proxy_deg is not None else "pitch=n/a"
                        low_text = "LOW_HEAD" if evidence.low_head else ("HEAD_OK" if evidence.valid else "POSE_INVALID")
                        color = STATE_COLORS[update.state]
                        _draw_label(
                            annotated,
                            int(box[0]),
                            int(box[1]),
                            [f"ID {track_id} {update.state.value}", f"{pitch_text} {low_text}"],
                            color,
                        )

                        row = {
                            "frame": current_frame_index,
                            "timestamp_ms": timestamp_ms,
                            "track_id": track_id,
                            "bbox_xyxy": [round(float(value), 2) for value in box],
                            "state": update.state.value,
                            "state_changed": update.changed,
                            "sleep_event": update.sleep_event,
                            "pose_valid": evidence.valid,
                            "low_head": evidence.low_head,
                            "pitch_proxy_deg": None if evidence.pitch_proxy_deg is None else round(evidence.pitch_proxy_deg, 3),
                            "head_offset_deg": None if evidence.head_offset_deg is None else round(evidence.head_offset_deg, 3),
                            "pose_confidence": round(evidence.confidence, 3),
                            "invalid_reason": evidence.reason,
                        }
                        event_stream.write(json.dumps(row, ensure_ascii=False) + "\n")

                state_machine.prune(timestamp_ms)
                writer.write(annotated)
                if args.show:
                    cv2.imshow("sleep-pose prototype", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                processed_frames += 1
                if processed_frames % 100 == 0:
                    print(f"processed {processed_frames} frames (source frame {current_frame_index})", flush=True)
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed_seconds = time.perf_counter() - started
    summary = {
        "source": str(source),
        "model": args.model,
        "device": args.device,
        "source_frames": total_frames,
        "processed_frames": processed_frames,
        "frame_stride": args.frame_stride,
        "source_fps": round(fps, 3),
        "output_fps": round(output_fps, 3),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "processing_fps": round(processed_frames / elapsed_seconds, 3) if elapsed_seconds else 0.0,
        "average_inference_ms": round(inference_ms_total / processed_frames, 3) if processed_frames else 0.0,
        "tracks": sorted(observed_tracks),
        "sleep_events": sleep_events,
        "state_observations": dict(state_counts),
        "pose_observations": {
            "valid": valid_pose_observations,
            "invalid": sum(invalid_reasons.values()),
            "low_head": low_head_observations,
            "invalid_reasons": dict(invalid_reasons),
        },
        "thresholds": {
            "detection_confidence": args.confidence,
            "keypoint_confidence": args.keypoint_confidence,
            "pitch_threshold_deg": args.pitch_threshold_deg,
            "sleep_seconds": args.sleep_seconds,
            "recovery_seconds": args.recovery_seconds,
        },
        "outputs": {
            "video": str(video_path),
            "frames_jsonl": str(events_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
