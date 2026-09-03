#ifndef ANALYZER_SLEEP_DETECTION_H
#define ANALYZER_SLEEP_DETECTION_H

#include <cstdint>
#include <array>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace SVAAnalyzer
{
    struct PoseKeypoint
    {
        float x = 0.0f;
        float y = 0.0f;
        float confidence = 0.0f;
    };

    struct PoseEvidence
    {
        bool valid = false;
        bool lowHead = false;
        std::optional<float> pitchProxyDeg;
        std::optional<float> headOffsetDeg;
        std::optional<float> shoulderWidthPx;
        float confidence = 0.0f;
        std::string reason;
        std::string postureMode = "unknown";
        std::optional<float> faceBelowShoulderRatio;
        std::optional<float> headToWristRatio;
    };

    struct ActivityEvidence
    {
        bool valid = false;
        std::optional<float> score;
        bool inactive = false;
        std::string reason;
    };

    PoseEvidence estimateHeadPose(const std::array<PoseKeypoint, 17> &keypoints,
                                  float minConfidence = 0.35f,
                                  float pitchThresholdDeg = 28.0f,
                                  float maxHeadOffsetDeg = 50.0f,
                                  float deskRestFaceRatio = 0.04f,
                                  float deskRestWristRatio = 0.35f);

    float sleepActivityThreshold(std::optional<float> shoulderWidthPx,
                                 float distantShoulderWidthPx = 80.0f,
                                 float nearFieldThreshold = 0.18f,
                                 float distantThreshold = 0.35f);

    /** Conservative instantaneous pose evidence used by the timed fallback. */
    bool strictSleepPoseSignal(const PoseEvidence &pose,
                               const ActivityEvidence &activity,
                               bool requireCollapsedPose = false);

    class PoseActivityTracker
    {
    public:
        PoseActivityTracker(int64_t windowMs = 3000,
                            int64_t minHistoryMs = 1500,
                            float minConfidence = 0.35f,
                            float inactivityThreshold = 0.18f,
                            int64_t trackTimeoutMs = 3000);

        ActivityEvidence update(int trackId,
                                int64_t timestampMs,
                                const std::array<PoseKeypoint, 17> &keypoints);
        std::vector<int> prune(int64_t timestampMs);

    private:
        struct Sample
        {
            int64_t timestampMs = 0;
            std::array<PoseKeypoint, 17> points{};
        };

        int64_t mWindowMs;
        int64_t mMinHistoryMs;
        float mMinConfidence;
        float mInactivityThreshold;
        int64_t mTrackTimeoutMs;
        std::unordered_map<int, std::deque<Sample>> mSamples;
        std::unordered_map<int, int64_t> mLastUpdateMs;
    };

    enum class SleepState
    {
        Normal,
        Suspected,
        Sleeping,
        Recovering
    };

    const char *sleepStateName(SleepState state);

    struct EyeEvidence
    {
        bool valid = false;
        bool closed = false;
        std::optional<float> closedProbability;
        std::string reason;
    };

    struct HybridEvidence
    {
        bool valid = false;
        bool sleepSignal = false;
        std::string source = "none";
        int64_t sleepDurationMs = 0;
        std::string reason;
    };

    struct SleepStateUpdate
    {
        SleepState state = SleepState::Normal;
        bool changed = false;
        bool sleepEvent = false;
        int64_t stateDurationMs = 0;
    };

    class EyeInferenceScheduler
    {
    public:
        EyeInferenceScheduler(int64_t probeIntervalMs = 200,
                              int64_t candidateHoldMs = 1500,
                              int64_t trackTimeoutMs = 3000);

        bool shouldInfer(int trackId, int64_t timestampMs, bool postureCandidate);
        void observe(int trackId, int64_t timestampMs, const EyeEvidence &eye);
        std::vector<int> prune(int64_t timestampMs);

    private:
        struct Runtime
        {
            std::optional<int64_t> lastProbeMs;
            int64_t activeUntilMs = -1;
            int64_t lastUpdateMs = 0;
        };

        int64_t mProbeIntervalMs;
        int64_t mCandidateHoldMs;
        int64_t mTrackTimeoutMs;
        std::unordered_map<int, Runtime> mTracks;
    };

    class EyeClosureTracker
    {
    public:
        EyeClosureTracker(int64_t windowMs = 2000,
                          int64_t minHistoryMs = 800,
                          float closedRatioThreshold = 0.60f,
                          int64_t trackTimeoutMs = 3000);

        EyeEvidence update(int trackId, int64_t timestampMs, const EyeEvidence &eye);
        std::vector<int> prune(int64_t timestampMs);

    private:
        struct Sample
        {
            int64_t timestampMs = 0;
            bool closed = false;
            float probability = 0.0f;
        };

        int64_t mWindowMs;
        int64_t mMinHistoryMs;
        float mClosedRatioThreshold;
        int64_t mTrackTimeoutMs;
        std::unordered_map<int, std::deque<Sample>> mSamples;
        std::unordered_map<int, int64_t> mLastUpdateMs;
    };

    struct PoseTemporalEvidence
    {
        bool valid = false;
        bool candidate = false;
        bool strict = false;
        float candidateRatio = 0.0f;
        float strictRatio = 0.0f;
    };

    /** Smooths noisy distant-camera pose evidence over a time window. */
    class PoseFallbackTracker
    {
    public:
        PoseFallbackTracker(int64_t windowMs = 8000,
                            int64_t minHistoryMs = 3000,
                            float candidateRatioThreshold = 0.35f,
                            float strictRatioThreshold = 0.15f,
                            int64_t trackTimeoutMs = 3000);

        PoseTemporalEvidence update(int trackId,
                                    int64_t timestampMs,
                                    bool candidate,
                                    bool strict);
        std::vector<int> prune(int64_t timestampMs);

    private:
        struct Sample
        {
            int64_t timestampMs = 0;
            bool candidate = false;
            bool strict = false;
        };

        int64_t mWindowMs;
        int64_t mMinHistoryMs;
        float mCandidateRatioThreshold;
        float mStrictRatioThreshold;
        int64_t mTrackTimeoutMs;
        std::unordered_map<int, std::deque<Sample>> mSamples;
        std::unordered_map<int, int64_t> mLastUpdateMs;
    };

    class HybridEvidenceTracker
    {
    public:
        HybridEvidenceTracker(int64_t eyeSleepDurationMs = 3000,
                              int64_t poseSleepDurationMs = 15000,
                              int64_t eyeGraceMs = 1500,
                              int64_t trackTimeoutMs = 3000);

        HybridEvidence update(int trackId,
                              int64_t timestampMs,
                              const EyeEvidence &eye,
                              std::optional<bool> poseSignal);
        std::vector<int> prune(int64_t timestampMs);

    private:
        struct Runtime
        {
            int64_t lastEyeMs = 0;
            bool lastEyeClosed = false;
            int64_t lastUpdateMs = 0;
        };

        int64_t mEyeSleepDurationMs;
        int64_t mPoseSleepDurationMs;
        int64_t mEyeGraceMs;
        int64_t mTrackTimeoutMs;
        std::unordered_map<int, Runtime> mTracks;
    };

    class SleepStateMachine
    {
    public:
        SleepStateMachine(int64_t sleepDurationMs = 5000,
                          int64_t recoveryDurationMs = 2000,
                          int64_t invalidResetMs = 2000,
                          int64_t trackTimeoutMs = 3000);

        SleepStateUpdate update(int trackId,
                                int64_t timestampMs,
                                std::optional<bool> sleepSignal,
                                std::optional<int64_t> sleepDurationMs = std::nullopt,
                                std::optional<bool> suspectSignal = std::nullopt);
        std::vector<int> prune(int64_t timestampMs);

    private:
        struct Runtime
        {
            SleepState state = SleepState::Normal;
            int64_t stateSinceMs = 0;
            int64_t lastUpdateMs = 0;
            std::optional<int64_t> lowSinceMs;
            std::optional<int64_t> normalSinceMs;
            std::optional<int64_t> invalidSinceMs;
        };

        static bool setState(Runtime &runtime, SleepState state, int64_t timestampMs);
        static bool reset(Runtime &runtime, int64_t timestampMs);

        int64_t mSleepDurationMs;
        int64_t mRecoveryDurationMs;
        int64_t mInvalidResetMs;
        int64_t mTrackTimeoutMs;
        std::unordered_map<int, Runtime> mTracks;
    };
}

#endif
