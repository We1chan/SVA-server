import unittest

import numpy as np

from sleep_logic import (
    EyeEvidence,
    EyeClosureTracker,
    EyeInferenceScheduler,
    HybridEvidenceTracker,
    PoseActivityTracker,
    SleepState,
    SleepStateMachine,
    estimate_head_pose,
)


def make_pose(*, nose_y: float, confidence: float = 0.95) -> np.ndarray:
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 2] = 0.01
    points[0] = (100, nose_y, confidence)
    points[1] = (92, 100, confidence)
    points[2] = (108, 100, confidence)
    points[5] = (60, 200, confidence)
    points[6] = (140, 200, confidence)
    return points


def make_upper_body_pose(*, wrist_x: float = 80.0) -> np.ndarray:
    points = make_pose(nose_y=145)
    points[7] = (75, 240, 0.95)
    points[8] = (125, 240, 0.95)
    points[9] = (wrist_x, 280, 0.95)
    points[10] = (120, 280, 0.95)
    return points


def make_desk_rest_pose(*, wrist_near_head: bool = True) -> np.ndarray:
    points = make_upper_body_pose()
    points[0] = (100, 225, 0.95)
    points[1] = (92, 212, 0.95)
    points[2] = (108, 212, 0.95)
    points[9] = (90 if wrist_near_head else 20, 230 if wrist_near_head else 300, 0.95)
    points[10] = (115 if wrist_near_head else 180, 230 if wrist_near_head else 300, 0.95)
    return points


class HeadPoseTests(unittest.TestCase):
    def test_upright_pose_is_not_low_head(self) -> None:
        evidence = estimate_head_pose(make_pose(nose_y=108), pitch_threshold_deg=18)
        self.assertTrue(evidence.valid)
        self.assertFalse(evidence.low_head)
        self.assertLess(evidence.pitch_proxy_deg, 10)

    def test_dropped_nose_is_low_head(self) -> None:
        evidence = estimate_head_pose(make_pose(nose_y=150), pitch_threshold_deg=18)
        self.assertTrue(evidence.valid)
        self.assertTrue(evidence.low_head)
        self.assertGreater(evidence.pitch_proxy_deg, 25)

    def test_low_confidence_pose_is_invalid(self) -> None:
        evidence = estimate_head_pose(make_pose(nose_y=150, confidence=0.1))
        self.assertFalse(evidence.valid)
        self.assertFalse(evidence.low_head)

    def test_large_side_offset_is_not_low_head(self) -> None:
        pose = make_pose(nose_y=150)
        pose[0, 0] = 220
        evidence = estimate_head_pose(pose, pitch_threshold_deg=18, max_head_offset_deg=50)
        self.assertTrue(evidence.valid)
        self.assertFalse(evidence.low_head)

    def test_face_below_shoulders_and_near_wrist_is_desk_rest(self) -> None:
        evidence = estimate_head_pose(make_desk_rest_pose())
        self.assertTrue(evidence.valid)
        self.assertTrue(evidence.low_head)
        self.assertEqual(evidence.posture_mode, "desk_rest")

    def test_face_below_shoulders_without_nearby_wrist_is_invalid(self) -> None:
        evidence = estimate_head_pose(make_desk_rest_pose(wrist_near_head=False))
        self.assertFalse(evidence.valid)
        self.assertFalse(evidence.low_head)


class ActivityTests(unittest.TestCase):
    def test_stable_upper_body_becomes_inactive(self) -> None:
        tracker = PoseActivityTracker(window_ms=1000, min_history_ms=500, inactivity_threshold=0.18)
        self.assertFalse(tracker.update(1, 0, make_upper_body_pose()).valid)
        tracker.update(1, 250, make_upper_body_pose())
        evidence = tracker.update(1, 500, make_upper_body_pose())
        self.assertTrue(evidence.valid)
        self.assertTrue(evidence.inactive)
        self.assertEqual(evidence.score, 0.0)

    def test_wrist_motion_is_active(self) -> None:
        tracker = PoseActivityTracker(window_ms=1000, min_history_ms=500, inactivity_threshold=0.18)
        tracker.update(2, 0, make_upper_body_pose(wrist_x=70))
        tracker.update(2, 250, make_upper_body_pose(wrist_x=100))
        evidence = tracker.update(2, 500, make_upper_body_pose(wrist_x=130))
        self.assertTrue(evidence.valid)
        self.assertFalse(evidence.inactive)
        self.assertGreater(evidence.score, 0.18)


class StateMachineTests(unittest.TestCase):
    def test_transient_low_head_returns_to_normal(self) -> None:
        machine = SleepStateMachine(sleep_duration_ms=1000, recovery_duration_ms=500)
        self.assertEqual(machine.update(1, 0, True).state, SleepState.SUSPECTED)
        update = machine.update(1, 500, False)
        self.assertEqual(update.state, SleepState.NORMAL)
        self.assertFalse(update.sleep_event)

    def test_sustained_low_head_emits_one_sleep_event(self) -> None:
        machine = SleepStateMachine(sleep_duration_ms=1000, recovery_duration_ms=500)
        machine.update(7, 0, True)
        self.assertFalse(machine.update(7, 999, True).sleep_event)
        update = machine.update(7, 1000, True)
        self.assertEqual(update.state, SleepState.SLEEPING)
        self.assertTrue(update.sleep_event)
        self.assertFalse(machine.update(7, 1200, True).sleep_event)

    def test_recovery_requires_stable_normal_pose(self) -> None:
        machine = SleepStateMachine(sleep_duration_ms=1000, recovery_duration_ms=500)
        machine.update(2, 0, True)
        machine.update(2, 1000, True)
        self.assertEqual(machine.update(2, 1100, False).state, SleepState.RECOVERING)
        self.assertEqual(machine.update(2, 1500, False).state, SleepState.RECOVERING)
        self.assertEqual(machine.update(2, 1600, False).state, SleepState.NORMAL)

    def test_invalid_pose_eventually_resets_state(self) -> None:
        machine = SleepStateMachine(
            sleep_duration_ms=1000,
            recovery_duration_ms=500,
            invalid_reset_ms=200,
        )
        machine.update(3, 0, True)
        self.assertEqual(machine.update(3, 100, None).state, SleepState.SUSPECTED)
        self.assertEqual(machine.update(3, 300, None).state, SleepState.NORMAL)

    def test_per_observation_sleep_duration_can_be_shorter(self) -> None:
        machine = SleepStateMachine(sleep_duration_ms=1000, recovery_duration_ms=500)
        machine.update(4, 0, True, sleep_duration_ms=400)
        update = machine.update(4, 400, True, sleep_duration_ms=400)
        self.assertEqual(update.state, SleepState.SLEEPING)
        self.assertTrue(update.sleep_event)

    def test_candidate_can_remain_suspected_without_confirmation(self) -> None:
        machine = SleepStateMachine(sleep_duration_ms=500, recovery_duration_ms=200)
        self.assertEqual(machine.update(5, 0, False, suspect_signal=True).state, SleepState.SUSPECTED)
        self.assertEqual(machine.update(5, 1000, False, suspect_signal=True).state, SleepState.SUSPECTED)
        update = machine.update(5, 1100, False, suspect_signal=False)
        self.assertEqual(update.state, SleepState.NORMAL)


class EyePipelineTests(unittest.TestCase):
    def test_scheduler_uses_sparse_probe_then_full_rate_candidate(self) -> None:
        scheduler = EyeInferenceScheduler(probe_interval_ms=200, candidate_hold_ms=500)
        self.assertTrue(scheduler.should_infer(1, 0, False))
        self.assertFalse(scheduler.should_infer(1, 100, False))
        self.assertTrue(scheduler.should_infer(1, 200, False))
        self.assertTrue(scheduler.should_infer(1, 250, True))
        self.assertTrue(scheduler.should_infer(1, 300, False))

    def test_closed_probe_activates_full_rate_inference(self) -> None:
        scheduler = EyeInferenceScheduler(probe_interval_ms=200, candidate_hold_ms=500)
        scheduler.should_infer(2, 0, False)
        scheduler.observe(2, 0, EyeEvidence(True, True, 0.9))
        self.assertTrue(scheduler.should_infer(2, 50, False))

    def test_perclos_tolerates_occasional_open_classification(self) -> None:
        tracker = EyeClosureTracker(window_ms=1000, min_history_ms=400, closed_ratio_threshold=0.60)
        tracker.update(3, 0, EyeEvidence(True, True, 0.9))
        tracker.update(3, 200, EyeEvidence(True, False, 0.1))
        tracker.update(3, 400, EyeEvidence(True, True, 0.9))
        evidence = tracker.update(3, 600, EyeEvidence(True, True, 0.9))
        self.assertTrue(evidence.valid)
        self.assertTrue(evidence.closed)
        self.assertIn("perclos=", evidence.reason)


class HybridEvidenceTests(unittest.TestCase):
    def test_eye_evidence_takes_priority_over_pose(self) -> None:
        tracker = HybridEvidenceTracker(eye_sleep_duration_ms=500, pose_sleep_duration_ms=1000)
        evidence = tracker.update(1, 0, EyeEvidence(True, False, 0.1), True)
        self.assertTrue(evidence.valid)
        self.assertFalse(evidence.sleep_signal)
        self.assertEqual(evidence.source, "eye")
        self.assertEqual(evidence.sleep_duration_ms, 500)

    def test_missing_eye_uses_grace_before_pose_fallback(self) -> None:
        tracker = HybridEvidenceTracker(
            eye_sleep_duration_ms=500,
            pose_sleep_duration_ms=1000,
            eye_grace_ms=200,
        )
        tracker.update(2, 0, EyeEvidence(True, True, 0.9), False)
        grace = tracker.update(2, 100, EyeEvidence(False, False, None, "missing"), False)
        self.assertTrue(grace.sleep_signal)
        self.assertEqual(grace.source, "eye_grace")
        fallback = tracker.update(2, 201, EyeEvidence(False, False, None, "missing"), False)
        self.assertFalse(fallback.sleep_signal)
        self.assertEqual(fallback.source, "pose")

    def test_invalid_eye_falls_back_to_pose(self) -> None:
        tracker = HybridEvidenceTracker(eye_sleep_duration_ms=500, pose_sleep_duration_ms=1000)
        evidence = tracker.update(3, 0, EyeEvidence(False, False, None, "too small"), True)
        self.assertTrue(evidence.valid)
        self.assertTrue(evidence.sleep_signal)
        self.assertEqual(evidence.source, "pose")
        self.assertEqual(evidence.sleep_duration_ms, 1000)


if __name__ == "__main__":
    unittest.main()
