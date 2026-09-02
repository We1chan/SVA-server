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

from eye_inference import EyeInferenceResult, EyeStateClassifier
from sleep_logic import (
    EyeClosureTracker,
    EyeEvidence,
    EyeInferenceScheduler,
    HybridEvidenceTracker,
    PoseActivityTracker,
    SleepState,
    SleepStateMachine,
    estimate_head_pose,
)


STATE_COLORS = {
    SleepState.NORMAL: (80, 220, 80),
    SleepState.SUSPECTED: (0, 200, 255),
    SleepState.SLEEPING: (40, 40, 255),
    SleepState.RECOVERING: (255, 180, 40),
}
DEFAULT_EYE_MODEL = Path(__file__).resolve().parent / "models" / "open-closed-eye-0001.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-Pose sleep detection prototype")
    parser.add_argument("--source", required=True, help="input video path")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose checkpoint")
    parser.add_argument("--output-dir", default="outputs/latest", help="annotated video and JSONL output directory")
    parser.add_argument("--device", default="0", help="Ultralytics CUDA device index; use cpu only for explicit fallback")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker configuration")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--pitch-threshold-deg", type=float, default=28.0)
    parser.add_argument("--max-head-offset-deg", type=float, default=50.0)
    parser.add_argument("--desk-rest-face-ratio", type=float, default=0.04)
    parser.add_argument("--desk-rest-wrist-ratio", type=float, default=0.35)
    parser.add_argument("--strict-pitch-threshold-deg", type=float, default=35.0)
    parser.add_argument("--strict-desk-rest-face-ratio", type=float, default=0.08)
    parser.add_argument("--strict-desk-rest-wrist-ratio", type=float, default=0.25)
    parser.add_argument("--activity-window-seconds", type=float, default=3.0)
    parser.add_argument("--activity-threshold", type=float, default=0.18, help="maximum elbow/wrist travel in shoulder-width units")
    parser.add_argument("--disable-motion-gate", action="store_true", help="use head geometry without the inactivity check")
    parser.add_argument("--sleep-seconds", type=float, default=15.0)
    parser.add_argument("--eye-model", default=str(DEFAULT_EYE_MODEL), help="open/closed-eye ONNX model")
    parser.add_argument("--eye-keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--eye-min-distance-px", type=float, default=40.0)
    parser.add_argument("--eye-crop-scale", type=float, default=0.72)
    parser.add_argument("--eye-closed-threshold", type=float, default=0.60)
    parser.add_argument("--eye-sleep-seconds", type=float, default=3.0)
    parser.add_argument("--eye-grace-seconds", type=float, default=1.5)
    parser.add_argument("--eye-probe-interval-seconds", type=float, default=0.2)
    parser.add_argument("--eye-window-seconds", type=float, default=2.0)
    parser.add_argument("--eye-min-history-seconds", type=float, default=0.8)
    parser.add_argument("--eye-closed-ratio", type=float, default=0.60)
    parser.add_argument("--disable-eye", action="store_true", help="disable eye-state inference and use pose only")
    parser.add_argument("--recovery-seconds", type=float, default=2.0)
    parser.add_argument("--start-seconds", type=float, default=0.0, help="seek to this source timestamp before processing")
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="0 means process until the end")
    parser.add_argument("--roi", default="", help="optional inference ROI as x1,y1,x2,y2 in source pixels")
    parser.add_argument("--frame-stride", type=int, default=1, help="run inference every Nth source frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means the complete video")
    parser.add_argument("--no-video", action="store_true", help="skip overlay rendering and video encoding")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _safe_timestamp_ms(capture: cv2.VideoCapture, frame_index: int, fps: float) -> int:
    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
    if timestamp > 0:
        return int(round(timestamp))
    return int(round(frame_index * 1000.0 / fps))


def _parse_roi(value: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    try:
        x1, y1, x2, y2 = (int(part.strip()) for part in value.split(","))
    except (TypeError, ValueError) as exc:
        raise ValueError("ROI must contain four integers: x1,y1,x2,y2") from exc
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"ROI {(x1, y1, x2, y2)} is outside the {width}x{height} source")
    return x1, y1, x2, y2


def _cuda_device_id(value: str) -> int:
    normalized = str(value).strip().lower()
    if normalized.startswith("cuda:"):
        normalized = normalized.split(":", 1)[1]
    if not normalized.isdigit():
        raise ValueError("eye inference requires a numeric CUDA device such as --device 0")
    return int(normalized)


def _strict_pose_signal(evidence, activity, args: argparse.Namespace) -> bool:
    if not evidence.valid:
        return False
    if not args.disable_motion_gate and not (activity.valid and activity.inactive):
        return False
    if evidence.posture_mode == "desk_rest":
        return (
            evidence.face_below_shoulder_ratio is not None
            and evidence.face_below_shoulder_ratio >= args.strict_desk_rest_face_ratio
            and evidence.head_to_wrist_ratio is not None
            and evidence.head_to_wrist_ratio <= args.strict_desk_rest_wrist_ratio
        )
    return (
        evidence.pitch_proxy_deg is not None
        and evidence.pitch_proxy_deg >= args.strict_pitch_threshold_deg
        and evidence.head_offset_deg is not None
        and abs(evidence.head_offset_deg) <= args.max_head_offset_deg
    )


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


def _draw_eye_boxes(
    frame: np.ndarray,
    result: EyeInferenceResult,
    offset_x: int,
    offset_y: int,
    color: tuple[int, int, int],
) -> None:
    for box in (result.left_box, result.right_box):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), color, 1)


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
    if args.start_seconds < 0 or args.duration_seconds < 0:
        raise ValueError("start and duration must not be negative")
    if args.activity_window_seconds <= 0:
        raise ValueError("activity window must be positive")
    if args.max_head_offset_deg <= 0:
        raise ValueError("maximum head offset must be positive")
    if min(
        args.eye_sleep_seconds,
        args.eye_grace_seconds,
        args.eye_probe_interval_seconds,
        args.eye_window_seconds,
        args.eye_min_history_seconds,
    ) <= 0:
        raise ValueError("eye-state durations must be positive")

    model = YOLO(args.model)
    eye_classifier = None
    if not args.disable_eye:
        eye_classifier = EyeStateClassifier(
            args.eye_model,
            device_id=_cuda_device_id(args.device),
            min_keypoint_confidence=args.eye_keypoint_confidence,
            min_inter_eye_distance_px=args.eye_min_distance_px,
            crop_scale=args.eye_crop_scale,
            closed_threshold=args.eye_closed_threshold,
        )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    roi = _parse_roi(args.roi, width, height)
    start_frame = min(total_frames, int(round(args.start_seconds * fps)))
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_fps = fps / args.frame_stride
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot create output video: {video_path}")

    state_machine = SleepStateMachine(
        sleep_duration_ms=max(1, int(args.sleep_seconds * 1000)),
        recovery_duration_ms=max(1, int(args.recovery_seconds * 1000)),
    )
    activity_tracker = PoseActivityTracker(
        window_ms=max(1, int(args.activity_window_seconds * 1000)),
        min_history_ms=max(1, int(args.activity_window_seconds * 500)),
        min_confidence=args.keypoint_confidence,
        inactivity_threshold=args.activity_threshold,
    )
    hybrid_tracker = HybridEvidenceTracker(
        eye_sleep_duration_ms=max(1, int(args.eye_sleep_seconds * 1000)),
        pose_sleep_duration_ms=max(1, int(args.sleep_seconds * 1000)),
        eye_grace_ms=max(1, int(args.eye_grace_seconds * 1000)),
    )
    eye_scheduler = EyeInferenceScheduler(
        probe_interval_ms=max(1, int(args.eye_probe_interval_seconds * 1000)),
        candidate_hold_ms=max(1, int(args.eye_grace_seconds * 1000)),
    )
    eye_closure_tracker = EyeClosureTracker(
        window_ms=max(1, int(args.eye_window_seconds * 1000)),
        min_history_ms=max(1, int(args.eye_min_history_seconds * 1000)),
        closed_ratio_threshold=args.eye_closed_ratio,
    )
    state_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    sleep_events = 0
    observed_tracks: set[int] = set()
    source_frame_index = start_frame
    processed_frames = 0
    valid_pose_observations = 0
    low_head_observations = 0
    activity_valid_observations = 0
    inactive_observations = 0
    sleep_signal_observations = 0
    eye_valid_observations = 0
    eye_closed_observations = 0
    hybrid_sources: Counter[str] = Counter()
    inference_ms_total = 0.0
    eye_inference_ms_total = 0.0
    eye_inference_attempts = 0
    eye_model_inferences = 0
    started = time.perf_counter()

    try:
        with events_path.open("w", encoding="utf-8") as event_stream:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                current_frame_index = source_frame_index
                source_frame_index += 1
                if args.duration_seconds > 0 and current_frame_index >= start_frame + int(round(args.duration_seconds * fps)):
                    break
                if current_frame_index % args.frame_stride != 0:
                    continue
                if args.max_frames > 0 and processed_frames >= args.max_frames:
                    break

                timestamp_ms = _safe_timestamp_ms(capture, current_frame_index, fps)
                infer_started = time.perf_counter()
                if roi is None:
                    inference_frame = frame
                    roi_x = roi_y = 0
                else:
                    roi_x, roi_y, roi_x2, roi_y2 = roi
                    inference_frame = frame[roi_y:roi_y2, roi_x:roi_x2]
                result = model.track(
                    inference_frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.confidence,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                inference_ms_total += (time.perf_counter() - infer_started) * 1000.0
                render_frame = writer is not None or args.show
                if render_frame:
                    plotted = result.plot(labels=False, conf=False)
                    if roi is None:
                        annotated = plotted
                    else:
                        annotated = frame.copy()
                        annotated[roi_y:roi_y2, roi_x:roi_x2] = plotted
                        cv2.rectangle(annotated, (roi_x, roi_y), (roi_x2, roi_y2), (255, 200, 0), 2)
                else:
                    annotated = None
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
                            max_head_offset_deg=args.max_head_offset_deg,
                            desk_rest_face_ratio=args.desk_rest_face_ratio,
                            desk_rest_wrist_ratio=args.desk_rest_wrist_ratio,
                        )
                        activity = activity_tracker.update(track_id, timestamp_ms, person_keypoints)
                        if activity.valid:
                            activity_valid_observations += 1
                            if activity.inactive:
                                inactive_observations += 1
                        posture_candidate = evidence.valid and evidence.low_head
                        strict_pose_signal = _strict_pose_signal(evidence, activity, args)
                        motion_gate_applied = not args.disable_motion_gate
                        if strict_pose_signal:
                            sleep_signal_observations += 1
                        if evidence.valid:
                            valid_pose_observations += 1
                            if evidence.low_head:
                                low_head_observations += 1
                        else:
                            invalid_reasons[evidence.reason] += 1

                        if eye_classifier is None:
                            eye_result = EyeInferenceResult(EyeEvidence(False, False, None, "eye inference disabled"))
                            raw_eye_evidence = eye_result.evidence
                        elif eye_scheduler.should_infer(track_id, timestamp_ms, posture_candidate):
                            eye_started = time.perf_counter()
                            raw_eye_result = eye_classifier.infer(inference_frame, person_keypoints)
                            eye_inference_ms_total += (time.perf_counter() - eye_started) * 1000.0
                            eye_inference_attempts += 1
                            raw_eye_evidence = raw_eye_result.evidence
                            if raw_eye_evidence.valid:
                                eye_model_inferences += 1
                            eye_scheduler.observe(track_id, timestamp_ms, raw_eye_evidence)
                            smoothed_eye = eye_closure_tracker.update(track_id, timestamp_ms, raw_eye_evidence)
                            eye_result = EyeInferenceResult(
                                smoothed_eye,
                                raw_eye_result.left_probability,
                                raw_eye_result.right_probability,
                                raw_eye_result.left_box,
                                raw_eye_result.right_box,
                                raw_eye_result.inter_eye_distance_px,
                            )
                        else:
                            raw_eye_evidence = EyeEvidence(False, False, None, "eye probe not scheduled")
                            eye_result = EyeInferenceResult(raw_eye_evidence)
                        if eye_result.evidence.valid:
                            eye_valid_observations += 1
                            if eye_result.evidence.closed:
                                eye_closed_observations += 1

                        hybrid = hybrid_tracker.update(
                            track_id,
                            timestamp_ms,
                            eye_result.evidence,
                            strict_pose_signal if evidence.valid else None,
                        )
                        hybrid_sources[hybrid.source] += 1
                        if hybrid.source in ("eye", "eye_grace"):
                            suspect_signal = hybrid.sleep_signal
                        else:
                            suspect_signal = posture_candidate
                        update = state_machine.update(
                            track_id,
                            timestamp_ms,
                            hybrid.sleep_signal if hybrid.valid else None,
                            sleep_duration_ms=hybrid.sleep_duration_ms,
                            suspect_signal=suspect_signal,
                        )
                        state_counts[update.state.value] += 1
                        if update.sleep_event:
                            sleep_events += 1

                        pitch_text = f"pitch={evidence.pitch_proxy_deg:.1f}deg" if evidence.pitch_proxy_deg is not None else "pitch=n/a"
                        low_text = "LOW_HEAD" if evidence.low_head else ("HEAD_OK" if evidence.valid else "POSE_INVALID")
                        if activity.score is None:
                            activity_text = "activity=n/a"
                        else:
                            activity_text = f"activity={activity.score:.2f} {'INACTIVE' if activity.inactive else 'ACTIVE'}"
                        if eye_result.evidence.closed_probability is None:
                            eye_text = f"eye=n/a ({hybrid.source})"
                        else:
                            eye_state = "CLOSED" if eye_result.evidence.closed else "OPEN"
                            eye_text = f"eye={eye_result.evidence.closed_probability:.2f} {eye_state} ({hybrid.source})"
                        color = STATE_COLORS[update.state]
                        if annotated is not None:
                            _draw_eye_boxes(annotated, eye_result, roi_x, roi_y, color)
                            _draw_label(
                                annotated,
                                int(box[0]) + roi_x,
                                int(box[1]) + roi_y,
                                [
                                    f"ID {track_id} {update.state.value}",
                                    eye_text,
                                    f"{evidence.posture_mode} {pitch_text} {low_text}",
                                    activity_text,
                                ],
                                color,
                            )

                        row = {
                            "frame": current_frame_index,
                            "timestamp_ms": timestamp_ms,
                            "track_id": track_id,
                            "bbox_xyxy": [
                                round(float(box[0]) + roi_x, 2),
                                round(float(box[1]) + roi_y, 2),
                                round(float(box[2]) + roi_x, 2),
                                round(float(box[3]) + roi_y, 2),
                            ],
                            "state": update.state.value,
                            "state_changed": update.changed,
                            "sleep_event": update.sleep_event,
                            "pose_valid": evidence.valid,
                            "low_head": evidence.low_head,
                            "pitch_proxy_deg": None if evidence.pitch_proxy_deg is None else round(evidence.pitch_proxy_deg, 3),
                            "head_offset_deg": None if evidence.head_offset_deg is None else round(evidence.head_offset_deg, 3),
                            "posture_mode": evidence.posture_mode,
                            "face_below_shoulder_ratio": None if evidence.face_below_shoulder_ratio is None else round(evidence.face_below_shoulder_ratio, 4),
                            "head_to_wrist_ratio": None if evidence.head_to_wrist_ratio is None else round(evidence.head_to_wrist_ratio, 4),
                            "activity_score": None if activity.score is None else round(activity.score, 4),
                            "activity_valid": activity.valid,
                            "inactive": activity.inactive,
                            "posture_candidate": posture_candidate,
                            "strict_pose_signal": strict_pose_signal,
                            "motion_gate_applied": motion_gate_applied,
                            "pose_confidence": round(evidence.confidence, 3),
                            "invalid_reason": evidence.reason,
                            "eye_valid": eye_result.evidence.valid,
                            "eye_closed": eye_result.evidence.closed,
                            "raw_eye_valid": raw_eye_evidence.valid,
                            "raw_eye_closed": raw_eye_evidence.closed,
                            "raw_eye_closed_probability": (
                                None
                                if raw_eye_evidence.closed_probability is None
                                else round(raw_eye_evidence.closed_probability, 4)
                            ),
                            "eye_closed_probability": (
                                None
                                if eye_result.evidence.closed_probability is None
                                else round(eye_result.evidence.closed_probability, 4)
                            ),
                            "left_eye_closed_probability": (
                                None if eye_result.left_probability is None else round(eye_result.left_probability, 4)
                            ),
                            "right_eye_closed_probability": (
                                None if eye_result.right_probability is None else round(eye_result.right_probability, 4)
                            ),
                            "inter_eye_distance_px": (
                                None
                                if eye_result.inter_eye_distance_px is None
                                else round(eye_result.inter_eye_distance_px, 3)
                            ),
                            "eye_invalid_reason": eye_result.evidence.reason,
                            "evidence_source": hybrid.source,
                            "hybrid_sleep_signal": hybrid.sleep_signal,
                            "required_sleep_duration_ms": hybrid.sleep_duration_ms,
                        }
                        event_stream.write(json.dumps(row, ensure_ascii=False) + "\n")

                state_machine.prune(timestamp_ms)
                activity_tracker.prune(timestamp_ms)
                hybrid_tracker.prune(timestamp_ms)
                eye_scheduler.prune(timestamp_ms)
                eye_closure_tracker.prune(timestamp_ms)
                if writer is not None and annotated is not None:
                    writer.write(annotated)
                if args.show and annotated is not None:
                    cv2.imshow("sleep-pose prototype", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                processed_frames += 1
                if processed_frames % 100 == 0:
                    print(f"processed {processed_frames} frames (source frame {current_frame_index})", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed_seconds = time.perf_counter() - started
    summary = {
        "source": str(source),
        "model": args.model,
        "device": args.device,
        "source_frames": total_frames,
        "processed_frames": processed_frames,
        "start_seconds": args.start_seconds,
        "duration_seconds": args.duration_seconds,
        "roi": None if roi is None else list(roi),
        "frame_stride": args.frame_stride,
        "source_fps": round(fps, 3),
        "output_fps": round(output_fps, 3),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "processing_fps": round(processed_frames / elapsed_seconds, 3) if elapsed_seconds else 0.0,
        "average_inference_ms": round(inference_ms_total / processed_frames, 3) if processed_frames else 0.0,
        "average_eye_attempt_ms": (
            round(eye_inference_ms_total / eye_inference_attempts, 3) if eye_inference_attempts else 0.0
        ),
        "tracks": sorted(observed_tracks),
        "sleep_events": sleep_events,
        "state_observations": dict(state_counts),
        "pose_observations": {
            "valid": valid_pose_observations,
            "invalid": sum(invalid_reasons.values()),
            "low_head": low_head_observations,
            "activity_valid": activity_valid_observations,
            "inactive": inactive_observations,
            "sleep_signal": sleep_signal_observations,
            "invalid_reasons": dict(invalid_reasons),
        },
        "eye_observations": {
            "enabled": eye_classifier is not None,
            "model": None if eye_classifier is None else str(Path(args.eye_model).expanduser().resolve()),
            "valid": eye_valid_observations,
            "closed": eye_closed_observations,
            "attempts": eye_inference_attempts,
            "model_inferences": eye_model_inferences,
            "evidence_sources": dict(hybrid_sources),
        },
        "thresholds": {
            "detection_confidence": args.confidence,
            "keypoint_confidence": args.keypoint_confidence,
            "pitch_threshold_deg": args.pitch_threshold_deg,
            "strict_pitch_threshold_deg": args.strict_pitch_threshold_deg,
            "max_head_offset_deg": args.max_head_offset_deg,
            "desk_rest_face_ratio": args.desk_rest_face_ratio,
            "desk_rest_wrist_ratio": args.desk_rest_wrist_ratio,
            "strict_desk_rest_face_ratio": args.strict_desk_rest_face_ratio,
            "strict_desk_rest_wrist_ratio": args.strict_desk_rest_wrist_ratio,
            "activity_window_seconds": args.activity_window_seconds,
            "activity_threshold": args.activity_threshold,
            "motion_gate_enabled": not args.disable_motion_gate,
            "sleep_seconds": args.sleep_seconds,
            "eye_keypoint_confidence": args.eye_keypoint_confidence,
            "eye_min_distance_px": args.eye_min_distance_px,
            "eye_crop_scale": args.eye_crop_scale,
            "eye_closed_threshold": args.eye_closed_threshold,
            "eye_sleep_seconds": args.eye_sleep_seconds,
            "eye_grace_seconds": args.eye_grace_seconds,
            "eye_probe_interval_seconds": args.eye_probe_interval_seconds,
            "eye_window_seconds": args.eye_window_seconds,
            "eye_min_history_seconds": args.eye_min_history_seconds,
            "eye_closed_ratio": args.eye_closed_ratio,
            "recovery_seconds": args.recovery_seconds,
        },
        "outputs": {
            "video": None if writer is None else str(video_path),
            "frames_jsonl": str(events_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
