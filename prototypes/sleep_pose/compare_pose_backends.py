"""Compare YOLO-Pose PyTorch and ONNX Runtime CUDA raw outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO-Pose PyTorch and ONNX CUDA outputs")
    parser.add_argument("--source", action="append", required=True, help="video path; repeat for multiple videos")
    parser.add_argument("--pt-model", default="yolo11n-pose.pt")
    parser.add_argument("--onnx-model", default="yolo11n-pose.onnx")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--samples-per-source", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.70)
    parser.add_argument("--strict-confidence", type=float, default=0.50)
    parser.add_argument("--coordinate-tolerance-px", type=float, default=1.0)
    parser.add_argument("--confidence-tolerance", type=float, default=0.02)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def preprocess(frame: np.ndarray, imgsz: int) -> np.ndarray:
    letterboxed = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=frame)
    rgb_chw = letterboxed[..., ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb_chw[None], dtype=np.float32) / 255.0


def sample_frames(source: Path, count: int) -> list[tuple[int, int, np.ndarray]]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"video has no frames: {source}")
    indices = np.linspace(0, total_frames - 1, min(count, total_frames), dtype=np.int64)
    frames: list[tuple[int, int, np.ndarray]] = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot read frame {frame_index} from {source}")
            timestamp_ms = int(round(int(frame_index) * 1000.0 / fps))
            frames.append((int(frame_index), timestamp_ms, frame))
    finally:
        capture.release()
    return frames


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def postprocess(output: np.ndarray, confidence: float, iou_threshold: float) -> np.ndarray:
    detections = non_max_suppression(
        torch.from_numpy(output),
        conf_thres=confidence,
        iou_thres=iou_threshold,
        nc=1,
        max_det=300,
    )[0]
    return detections.cpu().numpy()


def compare_detections(
    pt: np.ndarray,
    onnx: np.ndarray,
    match_iou: float,
    strict_confidence: float,
) -> dict[str, object]:
    available_pt = set(range(len(pt)))
    available_onnx = set(range(len(onnx)))
    matches: list[tuple[int, int, float]] = []
    while available_pt and available_onnx:
        best = max(
            ((left, right, box_iou(pt[left, :4], onnx[right, :4])) for left in available_pt for right in available_onnx),
            key=lambda item: item[2],
        )
        if best[2] < match_iou:
            break
        matches.append(best)
        available_pt.remove(best[0])
        available_onnx.remove(best[1])

    box_max = confidence_max = keypoint_coordinate_max = keypoint_confidence_max = 0.0
    strict_matches = 0
    for pt_index, onnx_index, _ in matches:
        difference = np.abs(pt[pt_index] - onnx[onnx_index])
        confidence_max = max(confidence_max, float(difference[4]))
        keypoints = difference[6:].reshape(17, 3)
        if min(float(pt[pt_index, 4]), float(onnx[onnx_index, 4])) >= strict_confidence:
            strict_matches += 1
            box_max = max(box_max, float(difference[:4].max()))
            keypoint_coordinate_max = max(keypoint_coordinate_max, float(keypoints[:, :2].max()))
            keypoint_confidence_max = max(keypoint_confidence_max, float(keypoints[:, 2].max()))

    return {
        "pytorch_detections": len(pt),
        "onnx_detections": len(onnx),
        "matched": len(matches),
        "unmatched_pytorch": len(available_pt),
        "unmatched_onnx": len(available_onnx),
        "minimum_iou": min((item[2] for item in matches), default=1.0),
        "strict_matches": strict_matches,
        "high_confidence_box_max_abs_px": box_max,
        "confidence_max_abs": confidence_max,
        "high_confidence_keypoint_coordinate_max_abs_px": keypoint_coordinate_max,
        "high_confidence_keypoint_confidence_max_abs": keypoint_confidence_max,
    }


def main() -> int:
    args = parse_args()
    if args.samples_per_source <= 0 or args.imgsz <= 0:
        raise ValueError("sample count and image size must be positive")

    pt_path = Path(args.pt_model).expanduser().resolve()
    onnx_path = Path(args.onnx_model).expanduser().resolve()
    sources = [Path(value).expanduser().resolve() for value in args.source]
    for path in (pt_path, onnx_path, *sources):
        if not path.is_file():
            raise FileNotFoundError(path)

    cuda_device = torch.device(f"cuda:{args.device}")
    pt_model = YOLO(str(pt_path)).model.to(cuda_device).eval()

    session_options = ort.SessionOptions()
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=[("CUDAExecutionProvider", {"device_id": args.device})],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    comparisons: list[dict[str, object]] = []
    for source in sources:
        for frame_index, timestamp_ms, frame in sample_frames(source, args.samples_per_source):
            tensor = preprocess(frame, args.imgsz)
            with torch.inference_mode():
                pt_output = pt_model(torch.from_numpy(tensor).to(cuda_device))[0].float().cpu().numpy()
            onnx_output = session.run([output_name], {input_name: tensor})[0]
            if pt_output.shape != onnx_output.shape:
                raise RuntimeError(f"shape mismatch: PyTorch {pt_output.shape}, ONNX {onnx_output.shape}")

            difference = np.abs(pt_output - onnx_output)
            detection_metrics = compare_detections(
                postprocess(pt_output, args.confidence, args.iou_threshold),
                postprocess(onnx_output, args.confidence, args.iou_threshold),
                args.match_iou,
                args.strict_confidence,
            )
            comparisons.append(
                {
                    "source": str(source),
                    "frame": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "max_abs": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                    "p99_abs": float(np.percentile(difference, 99)),
                    "bbox_max_abs": float(difference[:, :4, :].max()),
                    "class_max_abs": float(difference[:, 4:5, :].max()),
                    "keypoint_max_abs": float(difference[:, 5:, :].max()),
                    "detections": detection_metrics,
                }
            )

    maximum = max(item["max_abs"] for item in comparisons)
    mean = float(np.mean([item["mean_abs"] for item in comparisons]))
    unmatched = sum(
        item["detections"]["unmatched_pytorch"] + item["detections"]["unmatched_onnx"]
        for item in comparisons
    )
    minimum_iou = min(item["detections"]["minimum_iou"] for item in comparisons)
    box_max = max(item["detections"]["high_confidence_box_max_abs_px"] for item in comparisons)
    keypoint_coordinate_max = max(
        item["detections"]["high_confidence_keypoint_coordinate_max_abs_px"] for item in comparisons
    )
    confidence_max = max(item["detections"]["confidence_max_abs"] for item in comparisons)
    keypoint_confidence_max = max(
        item["detections"]["high_confidence_keypoint_confidence_max_abs"] for item in comparisons
    )
    passed = (
        unmatched == 0
        and minimum_iou >= args.match_iou
        and max(box_max, keypoint_coordinate_max) <= args.coordinate_tolerance_px
        and max(confidence_max, keypoint_confidence_max) <= args.confidence_tolerance
    )
    summary = {
        "pt_model": str(pt_path),
        "onnx_model": str(onnx_path),
        "device": args.device,
        "providers": session.get_providers(),
        "input": {"name": input_name, "shape": session.get_inputs()[0].shape},
        "output": {"name": output_name, "shape": session.get_outputs()[0].shape},
        "samples": len(comparisons),
        "raw_tensor_diagnostics": {"max_abs": maximum, "mean_abs": mean},
        "detection_summary": {
            "unmatched": unmatched,
            "minimum_iou": minimum_iou,
            "high_confidence_box_max_abs_px": box_max,
            "high_confidence_keypoint_coordinate_max_abs_px": keypoint_coordinate_max,
            "confidence_max_abs": confidence_max,
            "high_confidence_keypoint_confidence_max_abs": keypoint_confidence_max,
        },
        "tolerances": {
            "match_iou": args.match_iou,
            "strict_confidence": args.strict_confidence,
            "coordinate_px": args.coordinate_tolerance_px,
            "confidence": args.confidence_tolerance,
        },
        "passed": passed,
        "comparisons": comparisons,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
