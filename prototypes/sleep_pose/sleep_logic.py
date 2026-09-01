"""Pure sleep-detection logic shared by the Python prototype and tests.

The COCO pose model only provides 2D image keypoints, so ``pitch_proxy_deg``
is deliberately named a proxy. It measures how far the nose drops below the
eye line relative to the eye-to-shoulder vertical span. Camera placement and
person scale therefore still require threshold calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import math
from typing import Optional

import numpy as np


class CocoKeypoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6


class SleepState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    SLEEPING = "SLEEPING"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class PoseEvidence:
    valid: bool
    low_head: bool
    pitch_proxy_deg: Optional[float]
    head_offset_deg: Optional[float]
    shoulder_width_px: Optional[float]
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class StateUpdate:
    state: SleepState
    changed: bool
    sleep_event: bool
    state_duration_ms: int


@dataclass
class _TrackRuntime:
    state: SleepState
    state_since_ms: int
    last_update_ms: int
    low_since_ms: Optional[int] = None
    normal_since_ms: Optional[int] = None
    invalid_since_ms: Optional[int] = None


def _visible_mean(keypoints: np.ndarray, indices: tuple[int, ...], min_confidence: float) -> tuple[Optional[np.ndarray], float]:
    visible = [keypoints[index] for index in indices if keypoints[index, 2] >= min_confidence]
    if not visible:
        return None, 0.0
    values = np.asarray(visible, dtype=np.float64)
    return values[:, :2].mean(axis=0), float(values[:, 2].mean())


def estimate_head_pose(
    keypoints: np.ndarray,
    *,
    min_confidence: float = 0.35,
    pitch_threshold_deg: float = 18.0,
) -> PoseEvidence:
    """Estimate a camera-relative head-down signal from 17 COCO keypoints.

    Required landmarks are the nose, both shoulders, and at least one eye.
    Ears are used as a fallback face line when neither eye is reliable.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (17, 3):
        return PoseEvidence(False, False, None, None, None, 0.0, f"expected (17, 3), got {points.shape}")

    nose = points[CocoKeypoint.NOSE]
    left_shoulder = points[CocoKeypoint.LEFT_SHOULDER]
    right_shoulder = points[CocoKeypoint.RIGHT_SHOULDER]
    required_confidence = min(float(nose[2]), float(left_shoulder[2]), float(right_shoulder[2]))
    if required_confidence < min_confidence:
        return PoseEvidence(False, False, None, None, None, required_confidence, "nose or shoulder confidence too low")

    face_anchor, face_confidence = _visible_mean(
        points,
        (CocoKeypoint.LEFT_EYE, CocoKeypoint.RIGHT_EYE),
        min_confidence,
    )
    if face_anchor is None:
        face_anchor, face_confidence = _visible_mean(
            points,
            (CocoKeypoint.LEFT_EAR, CocoKeypoint.RIGHT_EAR),
            min_confidence,
        )
    if face_anchor is None:
        return PoseEvidence(False, False, None, None, None, 0.0, "eye and ear confidence too low")

    shoulder_mid = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    vertical_span = float(shoulder_mid[1] - face_anchor[1])
    if shoulder_width < 8.0 or vertical_span < 8.0:
        return PoseEvidence(False, False, None, None, shoulder_width, 0.0, "pose geometry is too small or inverted")

    nose_drop = float(nose[1] - face_anchor[1])
    pitch_proxy_deg = math.degrees(math.atan2(nose_drop, vertical_span))

    head_vector_x = float(nose[0] - shoulder_mid[0])
    head_vector_up = float(shoulder_mid[1] - nose[1])
    head_offset_deg = math.degrees(math.atan2(head_vector_x, max(1.0, head_vector_up)))

    confidence = min(required_confidence, face_confidence)
    return PoseEvidence(
        valid=True,
        low_head=pitch_proxy_deg >= pitch_threshold_deg,
        pitch_proxy_deg=pitch_proxy_deg,
        head_offset_deg=head_offset_deg,
        shoulder_width_px=shoulder_width,
        confidence=confidence,
    )


class SleepStateMachine:
    """Per-track temporal filter for transient low-head suppression."""

    def __init__(
        self,
        *,
        sleep_duration_ms: int = 5_000,
        recovery_duration_ms: int = 1_500,
        invalid_reset_ms: int = 2_000,
        track_timeout_ms: int = 3_000,
    ) -> None:
        if min(sleep_duration_ms, recovery_duration_ms, invalid_reset_ms, track_timeout_ms) <= 0:
            raise ValueError("all durations must be positive")
        self.sleep_duration_ms = sleep_duration_ms
        self.recovery_duration_ms = recovery_duration_ms
        self.invalid_reset_ms = invalid_reset_ms
        self.track_timeout_ms = track_timeout_ms
        self._tracks: dict[int, _TrackRuntime] = {}

    def _runtime(self, track_id: int, timestamp_ms: int) -> _TrackRuntime:
        runtime = self._tracks.get(track_id)
        if runtime is None:
            runtime = _TrackRuntime(SleepState.NORMAL, timestamp_ms, timestamp_ms)
            self._tracks[track_id] = runtime
        return runtime

    @staticmethod
    def _set_state(runtime: _TrackRuntime, state: SleepState, timestamp_ms: int) -> bool:
        if runtime.state == state:
            return False
        runtime.state = state
        runtime.state_since_ms = timestamp_ms
        return True

    def _reset(self, runtime: _TrackRuntime, timestamp_ms: int) -> bool:
        changed = self._set_state(runtime, SleepState.NORMAL, timestamp_ms)
        runtime.low_since_ms = None
        runtime.normal_since_ms = None
        runtime.invalid_since_ms = None
        return changed

    def update(self, track_id: int, timestamp_ms: int, low_head: Optional[bool]) -> StateUpdate:
        runtime = self._runtime(track_id, timestamp_ms)
        runtime.last_update_ms = timestamp_ms
        changed = False
        sleep_event = False

        if low_head is None:
            if runtime.invalid_since_ms is None:
                runtime.invalid_since_ms = timestamp_ms
            elif timestamp_ms - runtime.invalid_since_ms >= self.invalid_reset_ms:
                changed = self._reset(runtime, timestamp_ms)
            return StateUpdate(runtime.state, changed, False, timestamp_ms - runtime.state_since_ms)

        runtime.invalid_since_ms = None

        if runtime.state == SleepState.NORMAL:
            if low_head:
                runtime.low_since_ms = timestamp_ms
                changed = self._set_state(runtime, SleepState.SUSPECTED, timestamp_ms)

        elif runtime.state == SleepState.SUSPECTED:
            if not low_head:
                changed = self._reset(runtime, timestamp_ms)
            elif runtime.low_since_ms is not None and timestamp_ms - runtime.low_since_ms >= self.sleep_duration_ms:
                changed = self._set_state(runtime, SleepState.SLEEPING, timestamp_ms)
                sleep_event = changed
                runtime.normal_since_ms = None

        elif runtime.state == SleepState.SLEEPING:
            if not low_head:
                runtime.normal_since_ms = timestamp_ms
                changed = self._set_state(runtime, SleepState.RECOVERING, timestamp_ms)

        elif runtime.state == SleepState.RECOVERING:
            if low_head:
                runtime.normal_since_ms = None
                changed = self._set_state(runtime, SleepState.SLEEPING, timestamp_ms)
            elif runtime.normal_since_ms is not None and timestamp_ms - runtime.normal_since_ms >= self.recovery_duration_ms:
                changed = self._reset(runtime, timestamp_ms)

        return StateUpdate(runtime.state, changed, sleep_event, timestamp_ms - runtime.state_since_ms)

    def prune(self, timestamp_ms: int) -> list[int]:
        expired = [
            track_id
            for track_id, runtime in self._tracks.items()
            if timestamp_ms - runtime.last_update_ms >= self.track_timeout_ms
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return expired
