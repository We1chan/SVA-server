#include "SleepAnalyzer.h"

#include "Algorithm.h"
#include "Config.h"
#include "SleepDetection.h"
#include "Utils/Log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>

namespace SVAAnalyzer
{
    namespace
    {
        constexpr float MinKeypointConfidence = 0.35f;
        constexpr float MinInterEyeDistancePx = 40.0f;
        constexpr float EyeCropScale = 0.72f;
        constexpr float EyeClosedThreshold = 0.60f;

        struct RawEyeResult
        {
            EyeEvidence evidence;
            std::optional<float> leftProbability;
            std::optional<float> rightProbability;
            std::optional<float> interEyeDistancePx;
        };

        class EyeStateEngine
        {
        public:
            explicit EyeStateEngine(const std::string &modelPath)
                : mEnv(ORT_LOGGING_LEVEL_WARNING, "SVA-eye"), mOptions(), mSession(nullptr)
            {
                mOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
#if SVA_ONNXRUNTIME_GPU
                const std::vector<std::string> providers = Ort::GetAvailableProviders();
                if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end())
                {
                    throw std::runtime_error("eye classifier requires CUDAExecutionProvider");
                }
                OrtCUDAProviderOptions cudaOptions{};
                cudaOptions.device_id = 0;
                mOptions.AppendExecutionProvider_CUDA(cudaOptions);
                mOptions.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
                mProvider = "CUDA";
#else
                mProvider = "CPU (explicit SVA_ONNXRUNTIME_GPU=OFF build)";
#endif
#ifdef WIN32
                const std::wstring widePath(modelPath.begin(), modelPath.end());
                mSession = Ort::Session(mEnv, widePath.c_str(), mOptions);
#else
                mSession = Ort::Session(mEnv, modelPath.c_str(), mOptions);
#endif
                Ort::AllocatorWithDefaultOptions allocator;
                auto inputName = mSession.GetInputNameAllocated(0, allocator);
                auto outputName = mSession.GetOutputNameAllocated(0, allocator);
                mInputName = inputName.get();
                mOutputName = outputName.get();
                const std::vector<int64_t> inputShape = mSession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
                const std::vector<int64_t> outputShape = mSession.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
                if (inputShape != std::vector<int64_t>({1, 3, 32, 32}) ||
                    outputShape != std::vector<int64_t>({1, 2, 1, 1}))
                {
                    throw std::runtime_error("unexpected open-closed-eye-0001 tensor shapes");
                }
                LOGI("sleep eye classifier loaded provider=%s path=%s", mProvider.c_str(), modelPath.c_str());
            }

            float closedProbability(const cv::Mat &crop)
            {
                cv::Mat resized;
                cv::resize(crop, resized, cv::Size(32, 32), 0.0, 0.0, cv::INTER_LINEAR);
                cv::Mat blob = cv::dnn::blobFromImage(resized,
                                                      1.0 / 255.0,
                                                      cv::Size(32, 32),
                                                      cv::Scalar(127.0, 127.0, 127.0),
                                                      false,
                                                      false,
                                                      CV_32F);
                std::array<int64_t, 4> inputShape{1, 3, 32, 32};
                Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                Ort::Value tensor = Ort::Value::CreateTensor<float>(memory,
                                                                    blob.ptr<float>(),
                                                                    3 * 32 * 32,
                                                                    inputShape.data(),
                                                                    inputShape.size());
                const std::array<const char *, 1> inputs{mInputName.c_str()};
                const std::array<const char *, 1> outputs{mOutputName.c_str()};
                std::vector<Ort::Value> result = mSession.Run(Ort::RunOptions{nullptr},
                                                              inputs.data(),
                                                              &tensor,
                                                              1,
                                                              outputs.data(),
                                                              outputs.size());
                if (result.empty())
                {
                    throw std::runtime_error("eye classifier returned no output");
                }
                const float *probabilities = result.front().GetTensorData<float>();
                return probabilities[1];
            }

        private:
            Ort::Env mEnv;
            Ort::SessionOptions mOptions;
            Ort::Session mSession;
            std::string mInputName;
            std::string mOutputName;
            std::string mProvider;
        };

        std::optional<cv::Rect> eyeCropRect(const cv::Mat &image, const PoseKeypoint &eye, int side)
        {
            const int left = static_cast<int>(std::floor(eye.x - side * 0.5f));
            const int top = static_cast<int>(std::floor(eye.y - side * 0.5f));
            const cv::Rect rect(left, top, side, side);
            if (left < 0 || top < 0 || rect.br().x > image.cols || rect.br().y > image.rows)
            {
                return std::nullopt;
            }
            return rect;
        }

        bool strictPoseSignal(const PoseEvidence &pose, const ActivityEvidence &activity)
        {
            if (!pose.valid || !activity.valid || !activity.inactive)
            {
                return false;
            }
            if (pose.postureMode == "desk_rest")
            {
                return pose.faceBelowShoulderRatio.has_value() && *pose.faceBelowShoulderRatio >= 0.08f &&
                       pose.headToWristRatio.has_value() && *pose.headToWristRatio <= 0.25f;
            }
            return pose.pitchProxyDeg.has_value() && *pose.pitchProxyDeg >= 35.0f &&
                   pose.headOffsetDeg.has_value() && std::abs(*pose.headOffsetDeg) <= 50.0f;
        }
    }

    class SleepAnalyzer::Impl
    {
    public:
        explicit Impl(const std::string &eyeModelPath)
        {
            std::ifstream model(eyeModelPath, std::ios::binary);
            if (model.good())
            {
                mEyeEngine = std::make_unique<EyeStateEngine>(eyeModelPath);
            }
            else
            {
                LOGI("sleep eye classifier unavailable: model not found at %s; strict pose fallback remains active",
                     eyeModelPath.c_str());
            }
        }

        struct StreamRuntime
        {
            PoseActivityTracker activity;
            EyeInferenceScheduler eyeScheduler;
            EyeClosureTracker eyeClosure;
            HybridEvidenceTracker hybrid;
            SleepStateMachine state;
        };

        RawEyeResult inferEyes(cv::Mat &image, const std::array<PoseKeypoint, 17> &keypoints)
        {
            if (!mEyeEngine)
            {
                return {{false, false, std::nullopt, "eye model unavailable"}};
            }
            const PoseKeypoint &left = keypoints[1];
            const PoseKeypoint &right = keypoints[2];
            if (std::min(left.confidence, right.confidence) < MinKeypointConfidence)
            {
                return {{false, false, std::nullopt, "eye keypoint confidence too low"}};
            }
            const float distance = std::hypot(left.x - right.x, left.y - right.y);
            if (distance < MinInterEyeDistancePx)
            {
                return {{false, false, std::nullopt, "inter-eye distance too small"},
                        std::nullopt,
                        std::nullopt,
                        distance};
            }
            const int side = std::max(8, static_cast<int>(std::round(distance * EyeCropScale)));
            const std::optional<cv::Rect> leftRect = eyeCropRect(image, left, side);
            const std::optional<cv::Rect> rightRect = eyeCropRect(image, right, side);
            if (!leftRect.has_value() || !rightRect.has_value())
            {
                return {{false, false, std::nullopt, "eye crop crosses the frame boundary"},
                        std::nullopt,
                        std::nullopt,
                        distance};
            }
            const float leftProbability = mEyeEngine->closedProbability(image(*leftRect));
            const float rightProbability = mEyeEngine->closedProbability(image(*rightRect));
            const float combined = (leftProbability + rightProbability) * 0.5f;
            const bool closed = std::min(leftProbability, rightProbability) >= EyeClosedThreshold;
            return {{true, closed, combined, ""}, leftProbability, rightProbability, distance};
        }

        void process(const std::string &streamCode,
                     cv::Mat &image,
                     const std::vector<DetectObject *> &detects,
                     int64_t timestampMs)
        {
            std::lock_guard<std::mutex> lock(mMutex);
            StreamRuntime &runtime = mStreams[streamCode];
            for (DetectObject *detect : detects)
            {
                if (!detect || !detect->poseValid || detect->trackId < 0)
                {
                    continue;
                }

                const int trackId = detect->trackId;
                const PoseEvidence pose = estimateHeadPose(detect->poseKeypoints);
                const ActivityEvidence activity = runtime.activity.update(trackId, timestampMs, detect->poseKeypoints);
                const bool postureCandidate = pose.valid && pose.lowHead;
                const bool strictPose = strictPoseSignal(pose, activity);

                EyeEvidence eye{false, false, std::nullopt, "eye probe not scheduled"};
                RawEyeResult raw;
                if (runtime.eyeScheduler.shouldInfer(trackId, timestampMs, postureCandidate))
                {
                    raw = inferEyes(image, detect->poseKeypoints);
                    runtime.eyeScheduler.observe(trackId, timestampMs, raw.evidence);
                    eye = runtime.eyeClosure.update(trackId, timestampMs, raw.evidence);
                }
                const HybridEvidence hybrid = runtime.hybrid.update(trackId,
                                                                     timestampMs,
                                                                     eye,
                                                                     pose.valid ? std::optional<bool>(strictPose) : std::nullopt);
                const bool suspect = hybrid.source == "eye" || hybrid.source == "eye_grace"
                                         ? hybrid.sleepSignal
                                         : postureCandidate;
                const SleepStateUpdate state = runtime.state.update(trackId,
                                                                     timestampMs,
                                                                     hybrid.valid ? std::optional<bool>(hybrid.sleepSignal) : std::nullopt,
                                                                     hybrid.sleepDurationMs,
                                                                     suspect);

                detect->sleepEvidenceEvaluated = true;
                detect->postureCandidate = postureCandidate;
                detect->strictPoseSignal = strictPose;
                detect->pitchProxyDeg = pose.pitchProxyDeg;
                detect->activityScore = activity.score;
                detect->eyeEvidenceValid = eye.valid;
                detect->eyesClosed = eye.closed;
                detect->eyeClosedProbability = eye.closedProbability;
                detect->sleepState = sleepStateName(state.state);
                detect->sleepEvidenceSource = hybrid.source;
                detect->sleepEvent = state.sleepEvent;
            }

            runtime.activity.prune(timestampMs);
            runtime.eyeScheduler.prune(timestampMs);
            runtime.eyeClosure.prune(timestampMs);
            runtime.hybrid.prune(timestampMs);
            runtime.state.prune(timestampMs);
        }

        std::mutex mMutex;
        std::unique_ptr<EyeStateEngine> mEyeEngine;
        std::unordered_map<std::string, StreamRuntime> mStreams;
    };

    SleepAnalyzer::SleepAnalyzer(Config *, const std::string &eyeModelPath)
        : mImpl(std::make_unique<Impl>(eyeModelPath))
    {
    }

    SleepAnalyzer::~SleepAnalyzer() = default;

    void SleepAnalyzer::process(const std::string &streamCode,
                                cv::Mat &image,
                                const std::vector<DetectObject *> &detects,
                                int64_t timestampMs)
    {
        mImpl->process(streamCode, image, detects, timestampMs);
    }

    void SleepAnalyzer::clearStream(const std::string &streamCode)
    {
        std::lock_guard<std::mutex> lock(mImpl->mMutex);
        mImpl->mStreams.erase(streamCode);
    }

    bool SleepAnalyzer::eyeModelAvailable() const
    {
        return static_cast<bool>(mImpl->mEyeEngine);
    }
}
