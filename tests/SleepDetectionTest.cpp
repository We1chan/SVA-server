#include "SleepDetection.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace SVAAnalyzer;

namespace
{
    std::array<PoseKeypoint, 17> pose(float noseY, float confidence = 0.95f)
    {
        std::array<PoseKeypoint, 17> points{};
        for (PoseKeypoint &point : points)
        {
            point.confidence = 0.01f;
        }
        points[0] = {100.0f, noseY, confidence};
        points[1] = {92.0f, 100.0f, confidence};
        points[2] = {108.0f, 100.0f, confidence};
        points[5] = {60.0f, 200.0f, confidence};
        points[6] = {140.0f, 200.0f, confidence};
        return points;
    }

    std::array<PoseKeypoint, 17> upperBodyPose(float wristX = 80.0f)
    {
        auto points = pose(145.0f);
        points[7] = {75.0f, 240.0f, 0.95f};
        points[8] = {125.0f, 240.0f, 0.95f};
        points[9] = {wristX, 280.0f, 0.95f};
        points[10] = {120.0f, 280.0f, 0.95f};
        return points;
    }

    void testHeadPose()
    {
        PoseEvidence result = estimateHeadPose(pose(108.0f), 0.35f, 18.0f);
        assert(result.valid && !result.lowHead && *result.pitchProxyDeg < 10.0f);
        result = estimateHeadPose(pose(150.0f), 0.35f, 18.0f);
        assert(result.valid && result.lowHead && *result.pitchProxyDeg > 25.0f);
        result = estimateHeadPose(pose(150.0f, 0.1f));
        assert(!result.valid && !result.lowHead);
        auto sidePose = pose(150.0f);
        sidePose[0].x = 220.0f;
        result = estimateHeadPose(sidePose, 0.35f, 18.0f, 50.0f);
        assert(result.valid && !result.lowHead);

        auto deskPose = upperBodyPose();
        deskPose[0] = {100.0f, 225.0f, 0.95f};
        deskPose[1] = {92.0f, 212.0f, 0.95f};
        deskPose[2] = {108.0f, 212.0f, 0.95f};
        deskPose[9] = {90.0f, 230.0f, 0.95f};
        deskPose[10] = {115.0f, 230.0f, 0.95f};
        result = estimateHeadPose(deskPose);
        assert(result.valid && result.lowHead && result.postureMode == "desk_rest");
    }

    void testActivity()
    {
        PoseActivityTracker stable(1000, 500, 0.35f, 0.18f);
        assert(!stable.update(1, 0, upperBodyPose()).valid);
        stable.update(1, 250, upperBodyPose());
        ActivityEvidence result = stable.update(1, 500, upperBodyPose());
        assert(result.valid && result.inactive && *result.score == 0.0f);

        PoseActivityTracker moving(1000, 500, 0.35f, 0.18f);
        moving.update(2, 0, upperBodyPose(70.0f));
        moving.update(2, 250, upperBodyPose(100.0f));
        result = moving.update(2, 500, upperBodyPose(130.0f));
        assert(result.valid && !result.inactive && *result.score > 0.18f);
    }

    EyeEvidence eye(bool valid, bool closed, float probability, const char *reason = "")
    {
        return {valid, closed, valid ? std::optional<float>(probability) : std::nullopt, reason};
    }

    void testEyeSchedule()
    {
        EyeInferenceScheduler scheduler(200, 1500, 3000);
        assert(scheduler.shouldInfer(1, 0, false));
        assert(!scheduler.shouldInfer(1, 100, false));
        assert(scheduler.shouldInfer(1, 200, false));
        assert(scheduler.shouldInfer(1, 250, true));
        assert(scheduler.shouldInfer(1, 300, false));
        scheduler.observe(1, 300, eye(true, true, 0.9f));
        assert(scheduler.shouldInfer(1, 1700, false));
        assert(!scheduler.shouldInfer(1, 1801, false));
    }

    void testPerclos()
    {
        EyeClosureTracker tracker(2000, 800, 0.60f, 3000);
        EyeEvidence result;
        result = tracker.update(4, 0, eye(true, true, 0.9f));
        assert(result.valid && !result.closed);
        result = tracker.update(4, 400, eye(true, true, 0.8f));
        assert(!result.closed);
        result = tracker.update(4, 800, eye(true, false, 0.2f));
        assert(result.closed);
        result = tracker.update(4, 1200, eye(true, true, 0.9f));
        assert(result.closed);
        assert(result.reason == "perclos=0.750");
    }

    void testPoseFallbackTemporalRatios()
    {
        PoseFallbackTracker tracker(8000, 3000, 0.35f, 0.15f, 3000);
        PoseTemporalEvidence evidence;
        for (int index = 0; index <= 16; ++index)
        {
            const bool candidate = index % 2 == 0;
            const bool strict = index % 5 == 0;
            evidence = tracker.update(7, index * 500, candidate, strict);
        }
        assert(evidence.valid);
        assert(evidence.candidate);
        assert(evidence.strict);

        PoseFallbackTracker normal(8000, 3000, 0.35f, 0.15f, 3000);
        for (int index = 0; index <= 16; ++index)
        {
            evidence = normal.update(8, index * 500, index == 0, false);
        }
        assert(evidence.valid);
        assert(!evidence.candidate);
        assert(!evidence.strict);
    }

    void testAdaptiveActivityThreshold()
    {
        assert(std::abs(sleepActivityThreshold(std::nullopt) - 0.35f) < 0.0001f);
        assert(std::abs(sleepActivityThreshold(79.9f) - 0.35f) < 0.0001f);
        assert(std::abs(sleepActivityThreshold(80.0f) - 0.18f) < 0.0001f);
    }

    void testStrictPoseGeometryAndActivity()
    {
        ActivityEvidence inactive{true, 0.10f, true, ""};

        PoseEvidence writing;
        writing.valid = true;
        writing.lowHead = true;
        writing.pitchProxyDeg = 52.0f;
        writing.headOffsetDeg = 0.0f;
        writing.shoulderWidthPx = 120.0f;
        writing.faceBelowShoulderRatio = -0.30f;
        writing.postureMode = "head_pitch";
        assert(strictSleepPoseSignal(writing, inactive));
        assert(!strictSleepPoseSignal(writing, inactive, true));

        PoseEvidence collapsed = writing;
        collapsed.faceBelowShoulderRatio = -0.02f;
        assert(strictSleepPoseSignal(collapsed, inactive, true));

        ActivityEvidence moving{true, 0.30f, true, ""};
        assert(!strictSleepPoseSignal(writing, moving));

        PoseEvidence deskRest = writing;
        deskRest.postureMode = "desk_rest";
        deskRest.faceBelowShoulderRatio = 0.10f;
        deskRest.headToWristRatio = 0.20f;
        assert(strictSleepPoseSignal(deskRest, inactive, true));
    }

    void testHybridPriorityAndGrace()
    {
        HybridEvidenceTracker tracker(3000, 15000, 1500, 3000);
        HybridEvidence result = tracker.update(2, 0, eye(true, false, 0.1f), true);
        assert(result.valid && !result.sleepSignal && result.source == "eye");

        result = tracker.update(2, 1000, eye(false, false, 0.0f, "occluded"), true);
        assert(result.valid && !result.sleepSignal && result.source == "eye_grace");

        result = tracker.update(2, 1600, eye(false, false, 0.0f, "occluded"), true);
        assert(result.valid && result.sleepSignal && result.source == "pose");
        assert(result.sleepDurationMs == 15000);
    }

    void testSuspectRequiresConfirmation()
    {
        SleepStateMachine machine(5000, 2000, 2000, 3000);
        SleepStateUpdate result = machine.update(1, 0, false, 3000, true);
        assert(result.state == SleepState::Suspected);
        result = machine.update(1, 5000, false, 3000, true);
        assert(result.state == SleepState::Suspected && !result.sleepEvent);
        result = machine.update(1, 5100, true, 3000, true);
        assert(result.state == SleepState::Suspected);
        result = machine.update(1, 8100, true, 3000, true);
        assert(result.state == SleepState::Sleeping && result.sleepEvent);
    }

    void testRecoveryAndSingleEvent()
    {
        SleepStateMachine machine(1000, 2000, 2000, 3000);
        assert(machine.update(7, 0, true).state == SleepState::Suspected);
        SleepStateUpdate result = machine.update(7, 1000, true);
        assert(result.state == SleepState::Sleeping && result.sleepEvent);
        result = machine.update(7, 1500, true);
        assert(result.state == SleepState::Sleeping && !result.sleepEvent);
        assert(machine.update(7, 2000, false).state == SleepState::Recovering);
        assert(machine.update(7, 3500, false).state == SleepState::Recovering);
        assert(machine.update(7, 4000, false).state == SleepState::Normal);
    }

    void testInvalidResetAndPrune()
    {
        SleepStateMachine machine(1000, 2000, 2000, 3000);
        assert(machine.update(8, 0, true).state == SleepState::Suspected);
        assert(machine.update(8, 1000, true).state == SleepState::Sleeping);
        assert(machine.update(8, 1500, std::nullopt).state == SleepState::Sleeping);
        assert(machine.update(8, 3500, std::nullopt).state == SleepState::Normal);
        const std::vector<int> expired = machine.prune(6500);
        assert(expired.size() == 1 && expired.front() == 8);
    }
}

int main()
{
    testHeadPose();
    testActivity();
    testEyeSchedule();
    testPerclos();
    testPoseFallbackTemporalRatios();
    testAdaptiveActivityThreshold();
    testStrictPoseGeometryAndActivity();
    testHybridPriorityAndGrace();
    testSuspectRequiresConfirmation();
    testRecoveryAndSingleEvent();
    testInvalidResetAndPrune();
    std::cout << "SleepDetectionTest passed" << std::endl;
    return 0;
}
