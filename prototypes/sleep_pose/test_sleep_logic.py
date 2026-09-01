import unittest

import numpy as np

from sleep_logic import PoseActivityTracker, SleepState, SleepStateMachine, estimate_head_pose


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


if __name__ == "__main__":
    unittest.main()
