"""Pure sleep-detection logic shared by the Python prototype and tests.

The COCO pose model only provides 2D image keypoints, so ``pitch_proxy_deg``
is deliberately named a proxy. It measures how far the nose drops below the
eye line relative to the eye-to-shoulder vertical span. Camera placement and
person scale therefore still require threshold calibration.
"""

from __future__ import annotations

from collections import deque
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
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10


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
    posture_mode: str = "unknown"
    face_below_shoulder_ratio: Optional[float] = None
    head_to_wrist_ratio: Optional[float] = None


@dataclass(frozen=True)
class ActivityEvidence:
    valid: bool
    score: Optional[float]
    inactive: bool
    reason: str = ""


@dataclass(frozen=True)
class EyeEvidence:
    valid: bool
    closed: bool
    closed_probability: Optional[float]
    reason: str = ""


@dataclass(frozen=True)
class HybridEvidence:
    valid: bool
    sleep_signal: bool
    source: str
    sleep_duration_ms: int
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


@dataclass
class _HybridRuntime:
    last_eye_ms: int
    last_eye_closed: bool
    last_update_ms: int


@dataclass
class _EyeScheduleRuntime:
    last_probe_ms: Optional[int]
    active_until_ms: int
    last_update_ms: int


@dataclass(frozen=True)
class _EyeSample:
    timestamp_ms: int
    closed: bool
    probability: float


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
    pitch_threshold_deg: float = 28.0,
    max_head_offset_deg: float = 50.0,
    desk_rest_face_ratio: float = 0.04,
    desk_rest_wrist_ratio: float = 0.35,
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
    if shoulder_width < 8.0:
        return PoseEvidence(False, False, None, None, shoulder_width, 0.0, "pose geometry is too small or inverted")

    face_below_shoulder_ratio = float((face_anchor[1] - shoulder_mid[1]) / shoulder_width)
    visible_wrists = [
        points[index]
        for index in (CocoKeypoint.LEFT_WRIST, CocoKeypoint.RIGHT_WRIST)
        if points[index, 2] >= min_confidence
    ]
    head_to_wrist_ratio = None
    if visible_wrists:
        head_to_wrist_ratio = min(
            float(np.linalg.norm(nose[:2] - wrist[:2]) / shoulder_width)
            for wrist in visible_wrists
        )

    head_vector_x = float(nose[0] - shoulder_mid[0])
    head_vector_up = float(shoulder_mid[1] - nose[1])
    head_offset_deg = math.degrees(math.atan2(head_vector_x, max(1.0, head_vector_up)))
    confidence = min(required_confidence, face_confidence)

    desk_rest = (
        face_below_shoulder_ratio >= desk_rest_face_ratio
        and head_to_wrist_ratio is not None
        and head_to_wrist_ratio <= desk_rest_wrist_ratio
    )
    if desk_rest:
        return PoseEvidence(
            valid=True,
            low_head=True,
            pitch_proxy_deg=90.0,
            head_offset_deg=head_offset_deg,
            shoulder_width_px=shoulder_width,
            confidence=confidence,
            posture_mode="desk_rest",
            face_below_shoulder_ratio=face_below_shoulder_ratio,
            head_to_wrist_ratio=head_to_wrist_ratio,
        )
    if vertical_span < 8.0:
        return PoseEvidence(
            False,
            False,
            None,
            head_offset_deg,
            shoulder_width,
            confidence,
            "pose geometry is too small or inverted",
            face_below_shoulder_ratio=face_below_shoulder_ratio,
            head_to_wrist_ratio=head_to_wrist_ratio,
        )

    nose_drop = float(nose[1] - face_anchor[1])
    pitch_proxy_deg = math.degrees(math.atan2(nose_drop, vertical_span))
    low_head = pitch_proxy_deg >= pitch_threshold_deg and abs(head_offset_deg) <= max_head_offset_deg
    return PoseEvidence(
        valid=True,
        low_head=low_head,
        pitch_proxy_deg=pitch_proxy_deg,
        head_offset_deg=head_offset_deg,
        shoulder_width_px=shoulder_width,
        confidence=confidence,
        posture_mode="head_pitch",
        face_below_shoulder_ratio=face_below_shoulder_ratio,
        head_to_wrist_ratio=head_to_wrist_ratio,
    )


@dataclass(frozen=True)
class _ActivitySample:
    timestamp_ms: int
    points: np.ndarray


class PoseActivityTracker:
    """Measure recent upper-body motion in shoulder-width units.

    Coordinates are normalized around the shoulder midpoint, which removes
    most whole-person translation and scale changes. The score is the 90th
    percentile of elbow/wrist travel over the rolling window. A low score is
    evidence of inactivity; it is not a sleep decision on its own.
    """

    _MOTION_POINTS = (
        CocoKeypoint.LEFT_ELBOW,
        CocoKeypoint.RIGHT_ELBOW,
        CocoKeypoint.LEFT_WRIST,
        CocoKeypoint.RIGHT_WRIST,
    )

    def __init__(
        self,
        *,
        window_ms: int = 3_000,
        min_history_ms: int = 1_500,
        min_confidence: float = 0.35,
        inactivity_threshold: float = 0.18,
        track_timeout_ms: int = 3_000,
    ) -> None:
        if min(window_ms, min_history_ms, track_timeout_ms) <= 0:
            raise ValueError("activity durations must be positive")
        if min_history_ms > window_ms:
            raise ValueError("minimum activity history must not exceed the window")
        if inactivity_threshold < 0:
            raise ValueError("activity threshold must not be negative")
        self.window_ms = window_ms
        self.min_history_ms = min_history_ms
        self.min_confidence = min_confidence
        self.inactivity_threshold = inactivity_threshold
        self.track_timeout_ms = track_timeout_ms
        self._samples: dict[int, deque[_ActivitySample]] = {}
        self._last_update_ms: dict[int, int] = {}

    def update(self, track_id: int, timestamp_ms: int, keypoints: np.ndarray) -> ActivityEvidence:
        points = np.asarray(keypoints, dtype=np.float64)
        if points.shape != (17, 3):
            return ActivityEvidence(False, None, False, f"expected (17, 3), got {points.shape}")

        left_shoulder = points[CocoKeypoint.LEFT_SHOULDER]
        right_shoulder = points[CocoKeypoint.RIGHT_SHOULDER]
        if min(left_shoulder[2], right_shoulder[2]) < self.min_confidence:
            return ActivityEvidence(False, None, False, "shoulder confidence too low")
        shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
        if shoulder_width < 8.0:
            return ActivityEvidence(False, None, False, "shoulder width too small")

        shoulder_mid = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
        normalized = np.full((17, 3), np.nan, dtype=np.float64)
        visible = points[:, 2] >= self.min_confidence
        normalized[visible, :2] = (points[visible, :2] - shoulder_mid) / shoulder_width
        normalized[visible, 2] = points[visible, 2]

        samples = self._samples.setdefault(track_id, deque())
        samples.append(_ActivitySample(timestamp_ms, normalized))
        self._last_update_ms[track_id] = timestamp_ms
        cutoff = timestamp_ms - self.window_ms
        while len(samples) > 1 and samples[0].timestamp_ms < cutoff:
            samples.popleft()

        history_ms = timestamp_ms - samples[0].timestamp_ms
        if history_ms < self.min_history_ms:
            return ActivityEvidence(False, None, False, "activity window warming up")

        spans: list[float] = []
        for index in self._MOTION_POINTS:
            coordinates = np.asarray(
                [sample.points[index, :2] for sample in samples if np.isfinite(sample.points[index, 0])]
            )
            if len(coordinates) < 3:
                continue
            low = np.percentile(coordinates, 10, axis=0)
            high = np.percentile(coordinates, 90, axis=0)
            spans.append(float(np.linalg.norm(high - low)))

        if not spans:
            return ActivityEvidence(False, None, False, "elbow and wrist confidence too low")
        score = float(np.percentile(spans, 90))
        return ActivityEvidence(True, score, score <= self.inactivity_threshold)

    def prune(self, timestamp_ms: int) -> list[int]:
        expired = [
            track_id
            for track_id, last_update_ms in self._last_update_ms.items()
            if timestamp_ms - last_update_ms >= self.track_timeout_ms
        ]
        for track_id in expired:
            self._last_update_ms.pop(track_id, None)
            self._samples.pop(track_id, None)
        return expired


class EyeInferenceScheduler:
    """Run sparse eye probes normally and full-rate inference for candidates."""

    def __init__(
        self,
        *,
        probe_interval_ms: int = 200,
        candidate_hold_ms: int = 1_500,
        track_timeout_ms: int = 3_000,
    ) -> None:
        if min(probe_interval_ms, candidate_hold_ms, track_timeout_ms) <= 0:
            raise ValueError("eye scheduling durations must be positive")
        self.probe_interval_ms = probe_interval_ms
        self.candidate_hold_ms = candidate_hold_ms
        self.track_timeout_ms = track_timeout_ms
        self._tracks: dict[int, _EyeScheduleRuntime] = {}

    def should_infer(self, track_id: int, timestamp_ms: int, posture_candidate: bool) -> bool:
        runtime = self._tracks.get(track_id)
        if runtime is None:
            runtime = _EyeScheduleRuntime(None, -1, timestamp_ms)
            self._tracks[track_id] = runtime
        runtime.last_update_ms = timestamp_ms
        if posture_candidate:
            runtime.active_until_ms = max(runtime.active_until_ms, timestamp_ms + self.candidate_hold_ms)
        active = timestamp_ms <= runtime.active_until_ms
        probe_due = runtime.last_probe_ms is None or timestamp_ms - runtime.last_probe_ms >= self.probe_interval_ms
        if active or probe_due:
            runtime.last_probe_ms = timestamp_ms
            return True
        return False

    def observe(self, track_id: int, timestamp_ms: int, eye: EyeEvidence) -> None:
        runtime = self._tracks.get(track_id)
        if runtime is None:
            runtime = _EyeScheduleRuntime(timestamp_ms, -1, timestamp_ms)
            self._tracks[track_id] = runtime
        runtime.last_update_ms = timestamp_ms
        if eye.valid and eye.closed:
            runtime.active_until_ms = max(runtime.active_until_ms, timestamp_ms + self.candidate_hold_ms)

    def prune(self, timestamp_ms: int) -> list[int]:
        expired = [
            track_id
            for track_id, runtime in self._tracks.items()
            if timestamp_ms - runtime.last_update_ms >= self.track_timeout_ms
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return expired


class EyeClosureTracker:
    """Convert noisy frame classifications into a short-window PERCLOS signal."""

    def __init__(
        self,
        *,
        window_ms: int = 2_000,
        min_history_ms: int = 800,
        closed_ratio_threshold: float = 0.60,
        track_timeout_ms: int = 3_000,
    ) -> None:
        if min(window_ms, min_history_ms, track_timeout_ms) <= 0:
            raise ValueError("eye closure durations must be positive")
        if min_history_ms > window_ms:
            raise ValueError("minimum eye history must not exceed the window")
        if not 0.0 < closed_ratio_threshold <= 1.0:
            raise ValueError("closed-eye ratio threshold must be in (0, 1]")
        self.window_ms = window_ms
        self.min_history_ms = min_history_ms
        self.closed_ratio_threshold = closed_ratio_threshold
        self.track_timeout_ms = track_timeout_ms
        self._samples: dict[int, deque[_EyeSample]] = {}
        self._last_update_ms: dict[int, int] = {}

    def update(self, track_id: int, timestamp_ms: int, eye: EyeEvidence) -> EyeEvidence:
        self._last_update_ms[track_id] = timestamp_ms
        if not eye.valid or eye.closed_probability is None:
            return eye
        samples = self._samples.setdefault(track_id, deque())
        samples.append(_EyeSample(timestamp_ms, eye.closed, eye.closed_probability))
        cutoff = timestamp_ms - self.window_ms
        while len(samples) > 1 and samples[0].timestamp_ms < cutoff:
            samples.popleft()
        history_ms = timestamp_ms - samples[0].timestamp_ms
        if history_ms < self.min_history_ms:
            return EyeEvidence(True, False, eye.closed_probability, "eye closure window warming up")
        closed_ratio = sum(sample.closed for sample in samples) / len(samples)
        mean_probability = float(sum(sample.probability for sample in samples) / len(samples))
        return EyeEvidence(
            True,
            closed_ratio >= self.closed_ratio_threshold,
            mean_probability,
            f"perclos={closed_ratio:.3f}",
        )

    def prune(self, timestamp_ms: int) -> list[int]:
        expired = [
            track_id
            for track_id, last_update_ms in self._last_update_ms.items()
            if timestamp_ms - last_update_ms >= self.track_timeout_ms
        ]
        for track_id in expired:
            self._last_update_ms.pop(track_id, None)
            self._samples.pop(track_id, None)
        return expired


class HybridEvidenceTracker:
    """Prefer reliable eye state, then fall back to conservative pose evidence.

    The last reliable eye observation is retained briefly so a single missed
    frame does not switch evidence sources or reset a sustained eye closure.
    """

    def __init__(
        self,
        *,
        eye_sleep_duration_ms: int = 5_000,
        pose_sleep_duration_ms: int = 10_000,
        eye_grace_ms: int = 1_500,
        track_timeout_ms: int = 3_000,
    ) -> None:
        if min(eye_sleep_duration_ms, pose_sleep_duration_ms, eye_grace_ms, track_timeout_ms) <= 0:
            raise ValueError("hybrid evidence durations must be positive")
        self.eye_sleep_duration_ms = eye_sleep_duration_ms
        self.pose_sleep_duration_ms = pose_sleep_duration_ms
        self.eye_grace_ms = eye_grace_ms
        self.track_timeout_ms = track_timeout_ms
        self._tracks: dict[int, _HybridRuntime] = {}

    def update(
        self,
        track_id: int,
        timestamp_ms: int,
        eye: EyeEvidence,
        pose_signal: Optional[bool],
    ) -> HybridEvidence:
        runtime = self._tracks.get(track_id)
        if eye.valid:
            self._tracks[track_id] = _HybridRuntime(timestamp_ms, eye.closed, timestamp_ms)
            return HybridEvidence(
                True,
                eye.closed,
                "eye",
                self.eye_sleep_duration_ms,
                eye.reason,
            )

        if runtime is not None:
            runtime.last_update_ms = timestamp_ms
            if timestamp_ms - runtime.last_eye_ms <= self.eye_grace_ms:
                return HybridEvidence(
                    True,
                    runtime.last_eye_closed,
                    "eye_grace",
                    self.eye_sleep_duration_ms,
                    eye.reason or "eye observation temporarily unavailable",
                )

        if pose_signal is not None:
            return HybridEvidence(
                True,
                pose_signal,
                "pose",
                self.pose_sleep_duration_ms,
                eye.reason,
            )
        return HybridEvidence(False, False, "none", self.pose_sleep_duration_ms, eye.reason)

    def prune(self, timestamp_ms: int) -> list[int]:
        expired = [
            track_id
            for track_id, runtime in self._tracks.items()
            if timestamp_ms - runtime.last_update_ms >= self.track_timeout_ms
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return expired


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

    def update(
        self,
        track_id: int,
        timestamp_ms: int,
        low_head: Optional[bool],
        *,
        sleep_duration_ms: Optional[int] = None,
        suspect_signal: Optional[bool] = None,
    ) -> StateUpdate:
        required_sleep_ms = self.sleep_duration_ms if sleep_duration_ms is None else sleep_duration_ms
        if required_sleep_ms <= 0:
            raise ValueError("sleep duration must be positive")
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
        candidate_signal = low_head if suspect_signal is None else suspect_signal

        if runtime.state == SleepState.NORMAL:
            if candidate_signal:
                runtime.low_since_ms = timestamp_ms if low_head else None
                changed = self._set_state(runtime, SleepState.SUSPECTED, timestamp_ms)

        elif runtime.state == SleepState.SUSPECTED:
            if not candidate_signal:
                changed = self._reset(runtime, timestamp_ms)
            elif low_head:
                if runtime.low_since_ms is None:
                    runtime.low_since_ms = timestamp_ms
                elif timestamp_ms - runtime.low_since_ms >= required_sleep_ms:
                    changed = self._set_state(runtime, SleepState.SLEEPING, timestamp_ms)
                    sleep_event = changed
                    runtime.normal_since_ms = None
            else:
                runtime.low_since_ms = None

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
