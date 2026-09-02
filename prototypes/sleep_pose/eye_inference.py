"""GPU eye-state inference using Open Model Zoo open-closed-eye-0001."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import torch  # Preloads the CUDA/cuDNN DLLs used by ONNX Runtime on Windows.
import onnxruntime as ort

from sleep_logic import CocoKeypoint, EyeEvidence


@dataclass(frozen=True)
class EyeInferenceResult:
    evidence: EyeEvidence
    left_probability: float | None = None
    right_probability: float | None = None
    left_box: tuple[int, int, int, int] | None = None
    right_box: tuple[int, int, int, int] | None = None
    inter_eye_distance_px: float | None = None


class EyeStateClassifier:
    """Crop both eyes from COCO keypoints and classify them on NVIDIA CUDA."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device_id: int = 0,
        min_keypoint_confidence: float = 0.35,
        min_inter_eye_distance_px: float = 12.0,
        crop_scale: float = 0.72,
        closed_threshold: float = 0.60,
    ) -> None:
        model = Path(model_path).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"eye-state model not found: {model}")
        if min_inter_eye_distance_px <= 0 or crop_scale <= 0:
            raise ValueError("eye crop dimensions must be positive")
        if not 0.0 < closed_threshold < 1.0:
            raise ValueError("closed-eye threshold must be between zero and one")

        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort.InferenceSession(
            str(model),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"device_id": device_id})],
        )
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.min_keypoint_confidence = min_keypoint_confidence
        self.min_inter_eye_distance_px = min_inter_eye_distance_px
        self.crop_scale = crop_scale
        self.closed_threshold = closed_threshold

    @staticmethod
    def _crop_square(
        frame: np.ndarray,
        center: np.ndarray,
        side: int,
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        height, width = frame.shape[:2]
        half = side / 2.0
        x1 = int(math.floor(float(center[0]) - half))
        y1 = int(math.floor(float(center[1]) - half))
        x2 = x1 + side
        y2 = y1 + side
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return None, None
        crop = frame[y1:y2, x1:x2]
        if crop.shape[:2] != (side, side):
            return None, None
        return crop, (x1, y1, x2, y2)

    def _closed_probability(self, crop: np.ndarray) -> float:
        resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32)
        tensor = (tensor - 127.0) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        probabilities = np.asarray(output, dtype=np.float32).reshape(-1)
        if probabilities.size != 2:
            raise RuntimeError(f"unexpected eye-state output shape: {output.shape}")
        return float(probabilities[1])

    def infer(self, frame: np.ndarray, keypoints: np.ndarray) -> EyeInferenceResult:
        points = np.asarray(keypoints, dtype=np.float32)
        if points.shape != (17, 3):
            return EyeInferenceResult(EyeEvidence(False, False, None, f"expected (17, 3), got {points.shape}"))

        left = points[CocoKeypoint.LEFT_EYE]
        right = points[CocoKeypoint.RIGHT_EYE]
        eye_confidence = min(float(left[2]), float(right[2]))
        if eye_confidence < self.min_keypoint_confidence:
            return EyeInferenceResult(EyeEvidence(False, False, None, "eye keypoint confidence too low"))

        distance = float(np.linalg.norm(left[:2] - right[:2]))
        if distance < self.min_inter_eye_distance_px:
            return EyeInferenceResult(
                EyeEvidence(False, False, None, "inter-eye distance too small"),
                inter_eye_distance_px=distance,
            )

        side = max(8, int(round(distance * self.crop_scale)))
        left_crop, left_box = self._crop_square(frame, left[:2], side)
        right_crop, right_box = self._crop_square(frame, right[:2], side)
        if left_crop is None or right_crop is None:
            return EyeInferenceResult(
                EyeEvidence(False, False, None, "eye crop crosses the frame boundary"),
                left_box=left_box,
                right_box=right_box,
                inter_eye_distance_px=distance,
            )

        left_probability = self._closed_probability(left_crop)
        right_probability = self._closed_probability(right_crop)
        combined_probability = (left_probability + right_probability) / 2.0
        closed = min(left_probability, right_probability) >= self.closed_threshold
        return EyeInferenceResult(
            EyeEvidence(True, closed, combined_probability),
            left_probability=left_probability,
            right_probability=right_probability,
            left_box=left_box,
            right_box=right_box,
            inter_eye_distance_px=distance,
        )
