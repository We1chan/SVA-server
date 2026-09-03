#include "SleepDetection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace SVAAnalyzer
{
    namespace
    {
        constexpr int Nose = 0;
        constexpr int LeftEye = 1;
        constexpr int RightEye = 2;
        constexpr int LeftEar = 3;
        constexpr int RightEar = 4;
        constexpr int LeftShoulder = 5;
        constexpr int RightShoulder = 6;
        constexpr std::array<int, 4> MotionPoints = {7, 8, 9, 10};

        struct Point2
        {
            float x = 0.0f;
            float y = 0.0f;
        };

        void requirePositive(int64_t value, const char *name)
        {
            if (value <= 0)
            {
                throw std::invalid_argument(std::string(name) + " must be positive");
            }
        }

        float distance(const Point2 &first, const Point2 &second)
        {
            return std::hypot(first.x - second.x, first.y - second.y);
        }

        std::optional<std::pair<Point2, float>> visibleMean(
            const std::array<PoseKeypoint, 17> &keypoints,
            const std::vector<int> &indices,
            float minConfidence)
        {
            Point2 total;
            float confidence = 0.0f;
            int count = 0;
            for (int index : indices)
            {
                const PoseKeypoint &point = keypoints[index];
                if (point.confidence >= minConfidence)
                {
                    total.x += point.x;
                    total.y += point.y;
                    confidence += point.confidence;
                    ++count;
                }
            }
            if (count == 0)
            {
                return std::nullopt;
            }
            return std::make_pair(Point2{total.x / count, total.y / count}, confidence / count);
        }

        float percentile(std::vector<float> values, float quantile)
        {
            std::sort(values.begin(), values.end());
            if (values.size() == 1)
            {
                return values.front();
            }
            const float position = quantile * static_cast<float>(values.size() - 1);
            const size_t lower = static_cast<size_t>(std::floor(position));
            const size_t upper = static_cast<size_t>(std::ceil(position));
            const float fraction = position - static_cast<float>(lower);
            return values[lower] * (1.0f - fraction) + values[upper] * fraction;
        }
    }

    PoseEvidence estimateHeadPose(const std::array<PoseKeypoint, 17> &keypoints,
                                  float minConfidence,
                                  float pitchThresholdDeg,
                                  float maxHeadOffsetDeg,
                                  float deskRestFaceRatio,
                                  float deskRestWristRatio)
    {
        const PoseKeypoint &nose = keypoints[Nose];
        const PoseKeypoint &leftShoulder = keypoints[LeftShoulder];
        const PoseKeypoint &rightShoulder = keypoints[RightShoulder];
        const float requiredConfidence = std::min({nose.confidence,
                                                   leftShoulder.confidence,
                                                   rightShoulder.confidence});
        if (requiredConfidence < minConfidence)
        {
            PoseEvidence result;
            result.confidence = requiredConfidence;
            result.reason = "nose or shoulder confidence too low";
            return result;
        }

        auto face = visibleMean(keypoints, {LeftEye, RightEye}, minConfidence);
        if (!face.has_value())
        {
            face = visibleMean(keypoints, {LeftEar, RightEar}, minConfidence);
        }
        if (!face.has_value())
        {
            PoseEvidence result;
            result.reason = "eye and ear confidence too low";
            return result;
        }

        const Point2 leftShoulderPoint{leftShoulder.x, leftShoulder.y};
        const Point2 rightShoulderPoint{rightShoulder.x, rightShoulder.y};
        const Point2 shoulderMid{(leftShoulder.x + rightShoulder.x) * 0.5f,
                                 (leftShoulder.y + rightShoulder.y) * 0.5f};
        const float shoulderWidth = distance(leftShoulderPoint, rightShoulderPoint);
        if (shoulderWidth < 8.0f)
        {
            PoseEvidence result;
            result.shoulderWidthPx = shoulderWidth;
            result.reason = "pose geometry is too small or inverted";
            return result;
        }

        const float verticalSpan = shoulderMid.y - face->first.y;
        const float faceBelowShoulderRatio = (face->first.y - shoulderMid.y) / shoulderWidth;
        std::optional<float> headToWristRatio;
        for (int index : {9, 10})
        {
            const PoseKeypoint &wrist = keypoints[index];
            if (wrist.confidence < minConfidence)
            {
                continue;
            }
            const float ratio = distance({nose.x, nose.y}, {wrist.x, wrist.y}) / shoulderWidth;
            headToWristRatio = headToWristRatio.has_value() ? std::min(*headToWristRatio, ratio) : ratio;
        }

        constexpr float radiansToDegrees = 57.29577951308232f;
        const float headOffsetDeg = std::atan2(nose.x - shoulderMid.x,
                                               std::max(1.0f, shoulderMid.y - nose.y)) *
                                    radiansToDegrees;
        const float confidence = std::min(requiredConfidence, face->second);
        const bool deskRest = faceBelowShoulderRatio >= deskRestFaceRatio &&
                              headToWristRatio.has_value() &&
                              *headToWristRatio <= deskRestWristRatio;
        if (deskRest)
        {
            return {true,
                    true,
                    90.0f,
                    headOffsetDeg,
                    shoulderWidth,
                    confidence,
                    "",
                    "desk_rest",
                    faceBelowShoulderRatio,
                    headToWristRatio};
        }
        if (verticalSpan < 8.0f)
        {
            return {false,
                    false,
                    std::nullopt,
                    headOffsetDeg,
                    shoulderWidth,
                    confidence,
                    "pose geometry is too small or inverted",
                    "unknown",
                    faceBelowShoulderRatio,
                    headToWristRatio};
        }

        const float pitchProxyDeg = std::atan2(nose.y - face->first.y, verticalSpan) * radiansToDegrees;
        const bool lowHead = pitchProxyDeg >= pitchThresholdDeg &&
                             std::abs(headOffsetDeg) <= maxHeadOffsetDeg;
        return {true,
                lowHead,
                pitchProxyDeg,
                headOffsetDeg,
                shoulderWidth,
                confidence,
                "",
                "head_pitch",
                faceBelowShoulderRatio,
                headToWristRatio};
    }

    float sleepActivityThreshold(std::optional<float> shoulderWidthPx,
                                 float distantShoulderWidthPx,
                                 float nearFieldThreshold,
                                 float distantThreshold)
    {
        return shoulderWidthPx.has_value() && *shoulderWidthPx >= distantShoulderWidthPx
                   ? nearFieldThreshold
                   : distantThreshold;
    }

    bool strictSleepPoseSignal(const PoseEvidence &pose,
                               const ActivityEvidence &activity,
                               bool requireCollapsedPose)
    {
        if (!pose.valid || !activity.valid || !activity.inactive)
        {
            return false;
        }

        const float activityThreshold = sleepActivityThreshold(pose.shoulderWidthPx);
        if (!activity.score.has_value() || *activity.score > activityThreshold)
        {
            return false;
        }
        if (pose.postureMode == "desk_rest")
        {
            return pose.faceBelowShoulderRatio.has_value() && *pose.faceBelowShoulderRatio >= 0.08f &&
                   pose.headToWristRatio.has_value() && *pose.headToWristRatio <= 0.25f;
        }

        if (requireCollapsedPose)
        {
            // Long-running production scenes contain people who can read or
            // write almost motionlessly for many seconds.  Without usable eye
            // crops, a long-duration rule therefore also requires the face to
            // have collapsed to the shoulder line instead of ordinary gaze
            // down toward a book.
            return pose.pitchProxyDeg.has_value() && *pose.pitchProxyDeg >= 45.0f &&
                   pose.faceBelowShoulderRatio.has_value() && *pose.faceBelowShoulderRatio >= -0.05f &&
                   pose.headOffsetDeg.has_value() && std::abs(*pose.headOffsetDeg) <= 50.0f;
        }

        return pose.pitchProxyDeg.has_value() && *pose.pitchProxyDeg >= 28.0f &&
               pose.headOffsetDeg.has_value() && std::abs(*pose.headOffsetDeg) <= 50.0f;
    }

    PoseActivityTracker::PoseActivityTracker(int64_t windowMs,
                                             int64_t minHistoryMs,
                                             float minConfidence,
                                             float inactivityThreshold,
                                             int64_t trackTimeoutMs)
        : mWindowMs(windowMs),
          mMinHistoryMs(minHistoryMs),
          mMinConfidence(minConfidence),
          mInactivityThreshold(inactivityThreshold),
          mTrackTimeoutMs(trackTimeoutMs)
    {
        requirePositive(mWindowMs, "activity window");
        requirePositive(mMinHistoryMs, "minimum activity history");
        requirePositive(mTrackTimeoutMs, "track timeout");
        if (mMinHistoryMs > mWindowMs)
        {
            throw std::invalid_argument("minimum activity history must not exceed the window");
        }
        if (mInactivityThreshold < 0.0f)
        {
            throw std::invalid_argument("activity threshold must not be negative");
        }
    }

    ActivityEvidence PoseActivityTracker::update(int trackId,
                                                 int64_t timestampMs,
                                                 const std::array<PoseKeypoint, 17> &keypoints)
    {
        const PoseKeypoint &leftShoulder = keypoints[LeftShoulder];
        const PoseKeypoint &rightShoulder = keypoints[RightShoulder];
        if (std::min(leftShoulder.confidence, rightShoulder.confidence) < mMinConfidence)
        {
            return {false, std::nullopt, false, "shoulder confidence too low"};
        }
        const Point2 shoulderMid{(leftShoulder.x + rightShoulder.x) * 0.5f,
                                 (leftShoulder.y + rightShoulder.y) * 0.5f};
        const float shoulderWidth = distance({leftShoulder.x, leftShoulder.y},
                                             {rightShoulder.x, rightShoulder.y});
        if (shoulderWidth < 8.0f)
        {
            return {false, std::nullopt, false, "shoulder width too small"};
        }

        Sample sample;
        sample.timestampMs = timestampMs;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (size_t index = 0; index < keypoints.size(); ++index)
        {
            if (keypoints[index].confidence >= mMinConfidence)
            {
                sample.points[index] = {(keypoints[index].x - shoulderMid.x) / shoulderWidth,
                                        (keypoints[index].y - shoulderMid.y) / shoulderWidth,
                                        keypoints[index].confidence};
            }
            else
            {
                sample.points[index] = {nan, nan, 0.0f};
            }
        }

        std::deque<Sample> &samples = mSamples[trackId];
        samples.push_back(sample);
        mLastUpdateMs[trackId] = timestampMs;
        const int64_t cutoff = timestampMs - mWindowMs;
        while (samples.size() > 1 && samples.front().timestampMs < cutoff)
        {
            samples.pop_front();
        }
        if (timestampMs - samples.front().timestampMs < mMinHistoryMs)
        {
            return {false, std::nullopt, false, "activity window warming up"};
        }

        std::vector<float> spans;
        for (int index : MotionPoints)
        {
            std::vector<float> xs;
            std::vector<float> ys;
            for (const Sample &item : samples)
            {
                if (std::isfinite(item.points[index].x))
                {
                    xs.push_back(item.points[index].x);
                    ys.push_back(item.points[index].y);
                }
            }
            if (xs.size() < 3)
            {
                continue;
            }
            const float spanX = percentile(xs, 0.90f) - percentile(xs, 0.10f);
            const float spanY = percentile(ys, 0.90f) - percentile(ys, 0.10f);
            spans.push_back(std::hypot(spanX, spanY));
        }
        if (spans.empty())
        {
            return {false, std::nullopt, false, "elbow and wrist confidence too low"};
        }
        const float score = percentile(spans, 0.90f);
        return {true, score, score <= mInactivityThreshold, ""};
    }

    std::vector<int> PoseActivityTracker::prune(int64_t timestampMs)
    {
        std::vector<int> expired;
        for (auto iterator = mLastUpdateMs.begin(); iterator != mLastUpdateMs.end();)
        {
            if (timestampMs - iterator->second >= mTrackTimeoutMs)
            {
                expired.push_back(iterator->first);
                mSamples.erase(iterator->first);
                iterator = mLastUpdateMs.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
        return expired;
    }

    const char *sleepStateName(SleepState state)
    {
        switch (state)
        {
        case SleepState::Normal:
            return "NORMAL";
        case SleepState::Suspected:
            return "SUSPECTED";
        case SleepState::Sleeping:
            return "SLEEPING";
        case SleepState::Recovering:
            return "RECOVERING";
        }
        return "UNKNOWN";
    }

    EyeInferenceScheduler::EyeInferenceScheduler(int64_t probeIntervalMs,
                                                 int64_t candidateHoldMs,
                                                 int64_t trackTimeoutMs)
        : mProbeIntervalMs(probeIntervalMs),
          mCandidateHoldMs(candidateHoldMs),
          mTrackTimeoutMs(trackTimeoutMs)
    {
        requirePositive(mProbeIntervalMs, "probe interval");
        requirePositive(mCandidateHoldMs, "candidate hold");
        requirePositive(mTrackTimeoutMs, "track timeout");
    }

    bool EyeInferenceScheduler::shouldInfer(int trackId, int64_t timestampMs, bool postureCandidate)
    {
        Runtime &runtime = mTracks[trackId];
        runtime.lastUpdateMs = timestampMs;
        if (postureCandidate)
        {
            runtime.activeUntilMs = std::max(runtime.activeUntilMs, timestampMs + mCandidateHoldMs);
        }
        const bool active = timestampMs <= runtime.activeUntilMs;
        const bool probeDue = !runtime.lastProbeMs.has_value() ||
                              timestampMs - *runtime.lastProbeMs >= mProbeIntervalMs;
        if (active || probeDue)
        {
            runtime.lastProbeMs = timestampMs;
            return true;
        }
        return false;
    }

    void EyeInferenceScheduler::observe(int trackId, int64_t timestampMs, const EyeEvidence &eye)
    {
        Runtime &runtime = mTracks[trackId];
        runtime.lastUpdateMs = timestampMs;
        if (eye.valid && eye.closed)
        {
            runtime.activeUntilMs = std::max(runtime.activeUntilMs, timestampMs + mCandidateHoldMs);
        }
    }

    std::vector<int> EyeInferenceScheduler::prune(int64_t timestampMs)
    {
        std::vector<int> expired;
        for (auto iterator = mTracks.begin(); iterator != mTracks.end();)
        {
            if (timestampMs - iterator->second.lastUpdateMs >= mTrackTimeoutMs)
            {
                expired.push_back(iterator->first);
                iterator = mTracks.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
        return expired;
    }

    EyeClosureTracker::EyeClosureTracker(int64_t windowMs,
                                         int64_t minHistoryMs,
                                         float closedRatioThreshold,
                                         int64_t trackTimeoutMs)
        : mWindowMs(windowMs),
          mMinHistoryMs(minHistoryMs),
          mClosedRatioThreshold(closedRatioThreshold),
          mTrackTimeoutMs(trackTimeoutMs)
    {
        requirePositive(mWindowMs, "eye window");
        requirePositive(mMinHistoryMs, "minimum eye history");
        requirePositive(mTrackTimeoutMs, "track timeout");
        if (mMinHistoryMs > mWindowMs)
        {
            throw std::invalid_argument("minimum eye history must not exceed the window");
        }
        if (mClosedRatioThreshold <= 0.0f || mClosedRatioThreshold > 1.0f)
        {
            throw std::invalid_argument("closed-eye ratio threshold must be in (0, 1]");
        }
    }

    EyeEvidence EyeClosureTracker::update(int trackId, int64_t timestampMs, const EyeEvidence &eye)
    {
        mLastUpdateMs[trackId] = timestampMs;
        if (!eye.valid || !eye.closedProbability.has_value())
        {
            return eye;
        }

        std::deque<Sample> &samples = mSamples[trackId];
        samples.push_back({timestampMs, eye.closed, *eye.closedProbability});
        const int64_t cutoff = timestampMs - mWindowMs;
        while (samples.size() > 1 && samples.front().timestampMs < cutoff)
        {
            samples.pop_front();
        }

        if (timestampMs - samples.front().timestampMs < mMinHistoryMs)
        {
            return {true, false, eye.closedProbability, "eye closure window warming up"};
        }

        const int closedCount = std::accumulate(samples.begin(), samples.end(), 0,
                                                [](int count, const Sample &sample)
                                                { return count + (sample.closed ? 1 : 0); });
        const float probabilityTotal = std::accumulate(samples.begin(), samples.end(), 0.0f,
                                                       [](float total, const Sample &sample)
                                                       { return total + sample.probability; });
        const float closedRatio = static_cast<float>(closedCount) / static_cast<float>(samples.size());
        const float meanProbability = probabilityTotal / static_cast<float>(samples.size());
        std::ostringstream reason;
        reason.setf(std::ios::fixed);
        reason.precision(3);
        reason << "perclos=" << closedRatio;
        return {true, closedRatio >= mClosedRatioThreshold, meanProbability, reason.str()};
    }

    std::vector<int> EyeClosureTracker::prune(int64_t timestampMs)
    {
        std::vector<int> expired;
        for (auto iterator = mLastUpdateMs.begin(); iterator != mLastUpdateMs.end();)
        {
            if (timestampMs - iterator->second >= mTrackTimeoutMs)
            {
                expired.push_back(iterator->first);
                mSamples.erase(iterator->first);
                iterator = mLastUpdateMs.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
        return expired;
    }

    PoseFallbackTracker::PoseFallbackTracker(int64_t windowMs,
                                             int64_t minHistoryMs,
                                             float candidateRatioThreshold,
                                             float strictRatioThreshold,
                                             int64_t trackTimeoutMs)
        : mWindowMs(windowMs),
          mMinHistoryMs(minHistoryMs),
          mCandidateRatioThreshold(candidateRatioThreshold),
          mStrictRatioThreshold(strictRatioThreshold),
          mTrackTimeoutMs(trackTimeoutMs)
    {
        requirePositive(mWindowMs, "pose fallback window");
        requirePositive(mMinHistoryMs, "pose fallback minimum history");
        requirePositive(mTrackTimeoutMs, "pose fallback track timeout");
        if (mMinHistoryMs > mWindowMs)
        {
            throw std::invalid_argument("pose fallback minimum history must not exceed the window");
        }
        if (mCandidateRatioThreshold <= 0.0f || mCandidateRatioThreshold > 1.0f ||
            mStrictRatioThreshold <= 0.0f || mStrictRatioThreshold > 1.0f)
        {
            throw std::invalid_argument("pose fallback ratio thresholds must be in (0, 1]");
        }
    }

    PoseTemporalEvidence PoseFallbackTracker::update(int trackId,
                                                     int64_t timestampMs,
                                                     bool candidate,
                                                     bool strict)
    {
        std::deque<Sample> &samples = mSamples[trackId];
        samples.push_back({timestampMs, candidate, strict});
        mLastUpdateMs[trackId] = timestampMs;
        const int64_t cutoff = timestampMs - mWindowMs;
        while (samples.size() > 1 && samples.front().timestampMs < cutoff)
        {
            samples.pop_front();
        }

        PoseTemporalEvidence result;
        if (timestampMs - samples.front().timestampMs < mMinHistoryMs)
        {
            return result;
        }

        size_t candidateCount = 0;
        size_t strictCount = 0;
        for (const Sample &sample : samples)
        {
            candidateCount += sample.candidate ? 1 : 0;
            strictCount += sample.strict ? 1 : 0;
        }
        result.valid = true;
        result.candidateRatio = static_cast<float>(candidateCount) / static_cast<float>(samples.size());
        result.strictRatio = static_cast<float>(strictCount) / static_cast<float>(samples.size());
        result.candidate = result.candidateRatio >= mCandidateRatioThreshold;
        result.strict = result.strictRatio >= mStrictRatioThreshold;
        return result;
    }

    std::vector<int> PoseFallbackTracker::prune(int64_t timestampMs)
    {
        std::vector<int> expired;
        for (auto iterator = mLastUpdateMs.begin(); iterator != mLastUpdateMs.end();)
        {
            if (timestampMs - iterator->second >= mTrackTimeoutMs)
            {
                expired.push_back(iterator->first);
                mSamples.erase(iterator->first);
                iterator = mLastUpdateMs.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
        return expired;
    }

    HybridEvidenceTracker::HybridEvidenceTracker(int64_t eyeSleepDurationMs,
                                                 int64_t poseSleepDurationMs,
                                                 int64_t eyeGraceMs,
                                                 int64_t trackTimeoutMs)
        : mEyeSleepDurationMs(eyeSleepDurationMs),
          mPoseSleepDurationMs(poseSleepDurationMs),
          mEyeGraceMs(eyeGraceMs),
          mTrackTimeoutMs(trackTimeoutMs)
    {
        requirePositive(mEyeSleepDurationMs, "eye sleep duration");
        requirePositive(mPoseSleepDurationMs, "pose sleep duration");
        requirePositive(mEyeGraceMs, "eye grace");
        requirePositive(mTrackTimeoutMs, "track timeout");
    }

    HybridEvidence HybridEvidenceTracker::update(int trackId,
                                                 int64_t timestampMs,
                                                 const EyeEvidence &eye,
                                                 std::optional<bool> poseSignal)
    {
        auto iterator = mTracks.find(trackId);
        if (eye.valid)
        {
            mTracks[trackId] = {timestampMs, eye.closed, timestampMs};
            return {true, eye.closed, "eye", mEyeSleepDurationMs, eye.reason};
        }

        if (iterator != mTracks.end())
        {
            iterator->second.lastUpdateMs = timestampMs;
            if (timestampMs - iterator->second.lastEyeMs <= mEyeGraceMs)
            {
                return {true,
                        iterator->second.lastEyeClosed,
                        "eye_grace",
                        mEyeSleepDurationMs,
                        eye.reason.empty() ? "eye observation temporarily unavailable" : eye.reason};
            }
        }

        if (poseSignal.has_value())
        {
            return {true, *poseSignal, "pose", mPoseSleepDurationMs, eye.reason};
        }
        return {false, false, "none", mPoseSleepDurationMs, eye.reason};
    }

    std::vector<int> HybridEvidenceTracker::prune(int64_t timestampMs)
    {
        std::vector<int> expired;
        for (auto iterator = mTracks.begin(); iterator != mTracks.end();)
        {
            if (timestampMs - iterator->second.lastUpdateMs >= mTrackTimeoutMs)
            {
                expired.push_back(iterator->first);
                iterator = mTracks.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
        return expired;
    }

    SleepStateMachine::SleepStateMachine(int64_t sleepDurationMs,
                                         int64_t recoveryDurationMs,
                                         int64_t invalidResetMs,
                                         int64_t trackTimeoutMs)
        : mSleepDurationMs(sleepDurationMs),
          mRecoveryDurationMs(recoveryDurationMs),
          mInvalidResetMs(invalidResetMs),
          mTrackTimeoutMs(trackTimeoutMs)
    {
        requirePositive(mSleepDurationMs, "sleep duration");
        requirePositive(mRecoveryDurationMs, "recovery duration");
        requirePositive(mInvalidResetMs, "invalid reset");
        requirePositive(mTrackTimeoutMs, "track timeout");
    }

    bool SleepStateMachine::setState(Runtime &runtime, SleepState state, int64_t timestampMs)
    {
        if (runtime.state == state)
        {
            return false;
        }
        runtime.state = state;
        runtime.stateSinceMs = timestampMs;
        return true;
    }

    bool SleepStateMachine::reset(Runtime &runtime, int64_t timestampMs)
    {
        const bool changed = setState(runtime, SleepState::Normal, timestampMs);
        runtime.lowSinceMs.reset();
        runtime.normalSinceMs.reset();
        runtime.invalidSinceMs.reset();
        return changed;
    }

    SleepStateUpdate SleepStateMachine::update(int trackId,
                                               int64_t timestampMs,
                                               std::optional<bool> sleepSignal,
                                               std::optional<int64_t> sleepDurationMs,
                                               std::optional<bool> suspectSignal)
    {
        const int64_t requiredSleepMs = sleepDurationMs.value_or(mSleepDurationMs);
        requirePositive(requiredSleepMs, "sleep duration");
        Runtime initialRuntime;
        initialRuntime.state = SleepState::Normal;
        initialRuntime.stateSinceMs = timestampMs;
        initialRuntime.lastUpdateMs = timestampMs;
        auto insertion = mTracks.emplace(trackId, initialRuntime);
        Runtime &runtime = insertion.first->second;
        runtime.lastUpdateMs = timestampMs;
        bool changed = false;
        bool sleepEvent = false;

        if (!sleepSignal.has_value())
        {
            if (!runtime.invalidSinceMs.has_value())
            {
                runtime.invalidSinceMs = timestampMs;
            }
            else if (timestampMs - *runtime.invalidSinceMs >= mInvalidResetMs)
            {
                changed = reset(runtime, timestampMs);
            }
            return {runtime.state, changed, false, timestampMs - runtime.stateSinceMs};
        }

        runtime.invalidSinceMs.reset();
        const bool candidateSignal = suspectSignal.value_or(*sleepSignal);
        switch (runtime.state)
        {
        case SleepState::Normal:
            if (candidateSignal)
            {
                runtime.lowSinceMs = *sleepSignal ? std::optional<int64_t>(timestampMs) : std::nullopt;
                changed = setState(runtime, SleepState::Suspected, timestampMs);
            }
            break;
        case SleepState::Suspected:
            if (!candidateSignal)
            {
                changed = reset(runtime, timestampMs);
            }
            else if (*sleepSignal)
            {
                if (!runtime.lowSinceMs.has_value())
                {
                    runtime.lowSinceMs = timestampMs;
                }
                else if (timestampMs - *runtime.lowSinceMs >= requiredSleepMs)
                {
                    changed = setState(runtime, SleepState::Sleeping, timestampMs);
                    sleepEvent = changed;
                    runtime.normalSinceMs.reset();
                }
            }
            else
            {
                runtime.lowSinceMs.reset();
            }
            break;
        case SleepState::Sleeping:
            if (!*sleepSignal)
            {
                runtime.normalSinceMs = timestampMs;
                changed = setState(runtime, SleepState::Recovering, timestampMs);
            }
            break;
        case SleepState::Recovering:
            if (*sleepSignal)
            {
                runtime.normalSinceMs.reset();
                changed = setState(runtime, SleepState::Sleeping, timestampMs);
            }
            else if (runtime.normalSinceMs.has_value() &&
                     timestampMs - *runtime.normalSinceMs >= mRecoveryDurationMs)
            {
                changed = reset(runtime, timestampMs);
            }
            break;
        }
        return {runtime.state, changed, sleepEvent, timestampMs - runtime.stateSinceMs};
    }

    std::vector<int> SleepStateMachine::prune(int64_t timestampMs)
    {
        std::vector<int> expired;
        for (auto iterator = mTracks.begin(); iterator != mTracks.end();)
        {
            if (timestampMs - iterator->second.lastUpdateMs >= mTrackTimeoutMs)
            {
                expired.push_back(iterator->first);
                iterator = mTracks.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
        return expired;
    }
}
