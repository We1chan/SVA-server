#include "AlgorithmOnYolo.h"
#include "Config.h"
#include "Utils/Log.h"
#include "Utils/Common.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <opencv2/dnn.hpp>

namespace SVAAnalyzer
{
    OnnxRuntimeEngine::OnnxRuntimeEngine(Config *config, std::string &modelPath, std::vector<std::string> &classNames, const std::string &algorithmCode) : mConfig(config), mClassNames(classNames), mAlgorithmCode(algorithmCode)
    {
        LOGI("modelPath=%s", modelPath.data());
        initPostprocessProfile(algorithmCode);

        mEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "YOLO");
        mSessionOptions = Ort::SessionOptions();
        mSessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        // std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
        // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

        // log available providers for diagnostics
        std::vector<std::string> providers = Ort::GetAvailableProviders();

        LOGI("supported onnxruntime providers");
        for (size_t i = 0; i < providers.size(); i++)
        {
            LOGI("%zu,%s", i, providers[i].data());
        }

    #if SVA_ONNXRUNTIME_GPU
        bool gpuAssigned = false;
        /**
         * GPU provider selection strategy (teaching note):
         * 1. TensorRT: fastest, requires engine build/cache.
         * 2. CUDA: direct GPU execution.
         * CPU execution is deliberately disabled in GPU builds. A missing or broken
         * GPU provider is a deployment error and must be visible at startup.
         *
         * Build with -DSVA_ONNXRUNTIME_GPU=OFF when using CPU-only ONNX Runtime headers/libs.
         */
        auto trt_itr = std::find(providers.begin(), providers.end(), "TensorrtExecutionProvider");
        if (trt_itr != providers.end())
        {
            try
            {
                // 关键：正确初始化TensorRT配置结构体（避免空指针崩溃）
                OrtTensorRTProviderOptions trt_options = OrtTensorRTProviderOptions();

                // 基础GPU配置
                trt_options.device_id = 0;                    // 指定GPU 0
                trt_options.trt_max_workspace_size = 1 << 30; // 1GB工作空间
                trt_options.trt_fp16_enable = 1;              // 启用FP16加速

                // 修复警告：补充缺失的必填参数
                trt_options.trt_max_partition_iterations = 1000; // 日志提示的默认值
                trt_options.trt_min_subgraph_size = 1;           // 日志提示的默认值

                // 引擎缓存配置（可选，加速后续推理）
                trt_options.trt_engine_cache_path = "/opt/SVA/tmp/trt_cache";
                trt_options.trt_engine_cache_enable = 1;

                // 添加TensorRT执行提供者
                mSessionOptions.AppendExecutionProvider_TensorRT(trt_options);
                LOGI("appended TensorrtExecutionProvider (device %d)", trt_options.device_id);
                gpuAssigned = true;
                mGpuEnabled = true;
                mActiveProvider = "TensorRT";
            }
            catch (const Ort::Exception &e)
            {
                LOGI("failed to append TensorrtExecutionProvider: %s", e.what());
                gpuAssigned = false;
            }
        }

        auto itr = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");
        if (itr != providers.end())
        {
            try
            {
                OrtCUDAProviderOptions cuda_opts;
                cuda_opts.device_id = 0; // GPU 0
                mSessionOptions.AppendExecutionProvider_CUDA(cuda_opts);
                LOGI("appended CUDAExecutionProvider (device %d)", cuda_opts.device_id);
                gpuAssigned = true;
                if (mActiveProvider == "TensorRT")
                {
                    mActiveProvider = "TensorRT/CUDA";
                }
                else
                {
                    mGpuEnabled = true;
                    mActiveProvider = "CUDA";
                }
            }
            catch (const Ort::Exception &e)
            {
                LOGI("failed to append CUDAExecutionProvider: %s", e.what());
            }
        }
        if (!gpuAssigned)
        {
            throw std::runtime_error("SVA_ONNXRUNTIME_GPU=ON but no TensorRT/CUDA provider is available");
        }
        mSessionOptions.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
#else
        LOGI("SVA_ONNXRUNTIME_GPU=OFF, use CPUExecutionProvider only");
        mGpuEnabled = false;
        mActiveProvider = "CPU";
#endif
        auto createSession = [this, &modelPath]() {
#ifdef WIN32
            std::wstring modelPath_ws = std::wstring(modelPath.begin(), modelPath.end());
            return Ort::Session(mEnv, modelPath_ws.c_str(), mSessionOptions);
#else
            return Ort::Session(mEnv, modelPath.c_str(), mSessionOptions);
#endif
        };

        mSession = createSession();

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = mSession.GetInputNameAllocated(0, allocator);
    mInputNodeName = input_name.get();
    auto input_type_info = mSession.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_dims = input_tensor_info.GetShape();
    mInputWidth = static_cast<int>(input_dims[3]);
    mInputHeight = static_cast<int>(input_dims[2]);

    auto output_name = mSession.GetOutputNameAllocated(0, allocator);
    mOutputNodeName = output_name.get();
    auto output_type_info = mSession.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    if (output_dims.size() >= 3)
    {
        mOutputDims = output_dims;
        mOutputDim = static_cast<int>(output_dims[1]);
        mOutputRow = static_cast<int>(output_dims[2]);
        if (mDecoder == YoloOutputDecoder::PoseDenseWithNms)
        {
            LOGI("AlgorithmOnYolo output dims=[%lld,%lld,%lld] decoder=pose_dense_with_nms provider=%s",
                 static_cast<long long>(mOutputDims[0]),
                 static_cast<long long>(mOutputDims[1]),
                 static_cast<long long>(mOutputDims[2]),
                 mActiveProvider.c_str());
        }
        else if (mOutputDims.back() == 6)
        {
            mDecoder = YoloOutputDecoder::DirectDetections;
            LOGI("AlgorithmOnYolo output dims=[%lld,%lld,%lld] decoder=direct_detections provider=%s",
                 static_cast<long long>(mOutputDims[0]),
                 static_cast<long long>(mOutputDims[1]),
                 static_cast<long long>(mOutputDims[2]),
                 mActiveProvider.c_str());
        }
        else
        {
            mDecoder = YoloOutputDecoder::DenseWithNms;
            LOGI("AlgorithmOnYolo output dims=[%lld,%lld,%lld] decoder=dense_with_nms provider=%s",
                 static_cast<long long>(mOutputDims[0]),
                 static_cast<long long>(mOutputDims[1]),
                 static_cast<long long>(mOutputDims[2]),
                 mActiveProvider.c_str());
        }
    }
    else
    {
        mOutputDims.clear();
        mOutputDim = 0;
        mOutputRow = 0;
        LOGE("AlgorithmOnYolo invalid output tensor dims size=%zu", output_dims.size());
    }
    }

    OnnxRuntimeEngine::~OnnxRuntimeEngine()
    {

        mSessionOptions.release();
        mSession.release();
        mEnv.release();
    }

    void OnnxRuntimeEngine::initPostprocessProfile(const std::string &algorithmCode)
    {
        mDecoder = YoloOutputDecoder::DenseWithNms;
        if (algorithmCode == "on_yolo11n_pose_sleep")
        {
            mDecoder = YoloOutputDecoder::PoseDenseWithNms;
            LOGI("AlgorithmOnYolo profile=%s decoder=pose_dense_with_nms preprocess=center_letterbox_rgb candidate_score=0.10 nms=0.70", algorithmCode.data());
            return;
        }
        if (algorithmCode == "on_yolo26n_80" || algorithmCode == "ov_yolo26n_80")
        {
            mDecoder = YoloOutputDecoder::DirectDetections;
            LOGI("AlgorithmOnYolo profile=%s decoder=direct_detections preprocess=direct_resize_bgr score=0.25", algorithmCode.data());
            return;
        }
        LOGI("AlgorithmOnYolo profile=%s decoder=dense_with_nms preprocess=square_rgb score=0.50 nms=0.50", algorithmCode.data());
    }

    bool OnnxRuntimeEngine::decodePoseOutputWithNms(const float *pdata,
                                                    int imageWidth,
                                                    int imageHeight,
                                                    float scale,
                                                    float padX,
                                                    float padY,
                                                    std::vector<DetectObject> &detects)
    {
        constexpr int poseChannels = 56;
        constexpr int keypointStart = 5;
        constexpr int keypointCount = 17;
        // Keep a low, model-wide candidate floor here.  The same ONNX engine is
        // shared by multiple controls, so the user-facing per-task threshold is
        // applied later in Analyzer::runAlgorithmTask.  Using the old 0.35 value
        // here permanently discarded distant and desk-occluded people before a
        // task could apply its configured threshold.
        constexpr float scoreThreshold = 0.10f;
        constexpr float nmsThreshold = 0.70f;
        if (mOutputDim != poseChannels || scale <= 0.0f)
        {
            LOGE("AlgorithmOnYolo invalid pose output channels=%d, expect %d", mOutputDim, poseChannels);
            return false;
        }

        cv::Mat outputChannels(mOutputDim, mOutputRow, CV_32F, const_cast<float *>(pdata));
        cv::Mat rows = outputChannels.t();
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<std::array<PoseKeypoint, keypointCount>> keypoints;
        boxes.reserve(rows.rows);
        confidences.reserve(rows.rows);
        keypoints.reserve(rows.rows);

        const auto mapX = [scale, padX, imageWidth](float value)
        {
            return std::max(0.0f, std::min((value - padX) / scale, static_cast<float>(imageWidth - 1)));
        };
        const auto mapY = [scale, padY, imageHeight](float value)
        {
            return std::max(0.0f, std::min((value - padY) / scale, static_cast<float>(imageHeight - 1)));
        };

        for (int row = 0; row < rows.rows; ++row)
        {
            const float score = rows.at<float>(row, 4);
            if (!std::isfinite(score) || score <= scoreThreshold)
            {
                continue;
            }
            const float centerX = rows.at<float>(row, 0);
            const float centerY = rows.at<float>(row, 1);
            const float width = rows.at<float>(row, 2);
            const float height = rows.at<float>(row, 3);
            if (!std::isfinite(centerX) || !std::isfinite(centerY) ||
                !std::isfinite(width) || !std::isfinite(height) || width <= 0.0f || height <= 0.0f)
            {
                continue;
            }

            const int left = static_cast<int>(mapX(centerX - width * 0.5f));
            const int top = static_cast<int>(mapY(centerY - height * 0.5f));
            const int right = static_cast<int>(mapX(centerX + width * 0.5f));
            const int bottom = static_cast<int>(mapY(centerY + height * 0.5f));
            if (right <= left || bottom <= top)
            {
                continue;
            }

            std::array<PoseKeypoint, keypointCount> personKeypoints{};
            for (int index = 0; index < keypointCount; ++index)
            {
                const int offset = keypointStart + index * 3;
                const float x = rows.at<float>(row, offset);
                const float y = rows.at<float>(row, offset + 1);
                const float confidence = rows.at<float>(row, offset + 2);
                if (std::isfinite(x) && std::isfinite(y) && std::isfinite(confidence))
                {
                    personKeypoints[index] = {mapX(x), mapY(y), confidence};
                }
            }
            boxes.emplace_back(left, top, right - left, bottom - top);
            confidences.push_back(score);
            keypoints.push_back(personKeypoints);
        }

        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indexes);
        for (int index : indexes)
        {
            const cv::Rect &box = boxes[index];
            DetectObject detect;
            detect.x1 = box.x;
            detect.y1 = box.y;
            detect.x2 = box.x + box.width;
            detect.y2 = box.y + box.height;
            detect.class_id = 0;
            detect.class_name = mClassNames.empty() ? "person" : mClassNames.front();
            detect.class_score = confidences[index];
            detect.poseValid = true;
            detect.poseKeypoints = keypoints[index];
            detects.push_back(detect);
        }
        return true;
    }

    bool OnnxRuntimeEngine::decodeDenseOutputWithNms(const float *pdata, int imageWidth, int imageHeight, int paddedImageSize, std::vector<DetectObject> &detects)
    {
        float score_threshold = 0.5;
        float nms_threshold = 0.5;
        float x_factor = static_cast<float>(paddedImageSize) / static_cast<float>(mInputWidth);
        float y_factor = static_cast<float>(paddedImageSize) / static_cast<float>(mInputHeight);
        cv::Mat dout(mOutputDim, mOutputRow, CV_32F, (float *)pdata);
        cv::Mat det_output = dout.t();
        if (det_output.cols <= 4)
        {
            LOGE("AlgorithmOnYolo invalid dense det_output cols=%d", det_output.cols);
            return false;
        }

        int class_end = det_output.cols;
        if (!mClassNames.empty() && 4 + static_cast<int>(mClassNames.size()) < class_end)
        {
            class_end = 4 + static_cast<int>(mClassNames.size());
        }

        // post-process
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        for (int i = 0; i < det_output.rows; i++)
        {
            cv::Mat classes_scores = det_output.row(i).colRange(4, class_end);
            cv::Point classIdPoint;
            double score;
            minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

            if (score > score_threshold)
            {
                float cx = det_output.at<float>(i, 0);
                float cy = det_output.at<float>(i, 1);
                float ow = det_output.at<float>(i, 2);
                float oh = det_output.at<float>(i, 3);
                if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(ow) || !std::isfinite(oh) || ow <= 0.0f || oh <= 0.0f)
                {
                    continue;
                }
                int left = static_cast<int>((cx - 0.5f * ow) * x_factor);
                int top = static_cast<int>((cy - 0.5f * oh) * y_factor);
                int right = static_cast<int>((cx + 0.5f * ow) * x_factor);
                int bottom = static_cast<int>((cy + 0.5f * oh) * y_factor);

                left = std::max(0, std::min(left, imageWidth - 1));
                top = std::max(0, std::min(top, imageHeight - 1));
                right = std::max(0, std::min(right, imageWidth));
                bottom = std::max(0, std::min(bottom, imageHeight));
                if (right <= left || bottom <= top)
                {
                    continue;
                }

                cv::Rect box;
                box.x = left;
                box.y = top;
                box.width = right - left;
                box.height = bottom - top;

                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }

        // NMS
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indexes);
        for (size_t i = 0; i < indexes.size(); i++)
        {

            int index = indexes[i];
            int class_id = classIds[index];
            if (class_id < 0 || class_id >= static_cast<int>(mClassNames.size()))
            {
                continue;
            }
            float class_score = confidences[index];
            cv::Rect box = boxes[index];

            DetectObject detect;
            detect.x1 = box.x;
            detect.y1 = box.y;
            detect.x2 = box.x + box.width;
            detect.y2 = box.y + box.height;
            detect.class_id = class_id;
            detect.class_name = mClassNames[class_id];
            detect.class_score = class_score;

            detects.push_back(detect);
        }

        return true;
    }

    bool OnnxRuntimeEngine::decodeDirectDetections(const float *pdata, int imageWidth, int imageHeight, std::vector<DetectObject> &detects)
    {
        if (mOutputDims.size() < 3)
        {
            LOGE("AlgorithmOnYolo invalid direct output dims size=%zu", mOutputDims.size());
            return false;
        }
        if (mOutputDims.back() != 6)
        {
            LOGE("AlgorithmOnYolo invalid direct output last dim=%lld, expect 6", static_cast<long long>(mOutputDims.back()));
            return false;
        }

        const int64_t detectionCount64 = mOutputDims[mOutputDims.size() - 2];
        if (detectionCount64 <= 0)
        {
            return true;
        }

        const int numDetections = static_cast<int>(detectionCount64);
        const float scaleX = static_cast<float>(imageWidth) / static_cast<float>(mInputWidth);
        const float scaleY = static_cast<float>(imageHeight) / static_cast<float>(mInputHeight);
        const float scoreThreshold = 0.25f;

        for (int i = 0; i < numDetections; ++i)
        {
            const int base = i * 6;
            const float x1 = pdata[base + 0];
            const float y1 = pdata[base + 1];
            const float x2 = pdata[base + 2];
            const float y2 = pdata[base + 3];
            const float score = pdata[base + 4];
            const int class_id = static_cast<int>(pdata[base + 5]);

            if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2) || !std::isfinite(score))
            {
                continue;
            }
            if (score < scoreThreshold || class_id < 0 || class_id >= static_cast<int>(mClassNames.size()))
            {
                continue;
            }

            int left = static_cast<int>(x1 * scaleX);
            int top = static_cast<int>(y1 * scaleY);
            int right = static_cast<int>(x2 * scaleX);
            int bottom = static_cast<int>(y2 * scaleY);

            left = std::max(0, std::min(left, imageWidth - 1));
            top = std::max(0, std::min(top, imageHeight - 1));
            right = std::max(0, std::min(right, imageWidth));
            bottom = std::max(0, std::min(bottom, imageHeight));
            if (right <= left || bottom <= top)
            {
                continue;
            }

            DetectObject detect;
            detect.x1 = left;
            detect.y1 = top;
            detect.x2 = right;
            detect.y2 = bottom;
            detect.class_id = class_id;
            detect.class_name = mClassNames[class_id];
            detect.class_score = score;
            detects.push_back(detect);
        }

        return true;
    }

    bool OnnxRuntimeEngine::runInference(cv::Mat &image, std::vector<DetectObject> &detects)
    {
        detects.clear();
        int image_w = image.cols;
        int image_h = image.rows;
        if (image_w <= 0 || image_h <= 0 || mInputWidth <= 0 || mInputHeight <= 0 || mOutputDim <= 0 || mOutputRow <= 0)
        {
            LOGE("AlgorithmOnYolo invalid inference dims image=%dx%d input=%dx%d output=%dx%d", image_w, image_h, mInputWidth, mInputHeight, mOutputDim, mOutputRow);
            return false;
        }

        cv::Mat inputImage;
        int paddedImageSize = std::max(image_h, image_w);
        float poseScale = 1.0f;
        float posePadX = 0.0f;
        float posePadY = 0.0f;
        if (mDecoder == YoloOutputDecoder::DirectDetections)
        {
            inputImage = image;
        }
        else if (mDecoder == YoloOutputDecoder::PoseDenseWithNms)
        {
            poseScale = std::min(static_cast<float>(mInputWidth) / static_cast<float>(image_w),
                                 static_cast<float>(mInputHeight) / static_cast<float>(image_h));
            const int resizedWidth = static_cast<int>(std::round(image_w * poseScale));
            const int resizedHeight = static_cast<int>(std::round(image_h * poseScale));
            const float halfPadX = (mInputWidth - resizedWidth) * 0.5f;
            const float halfPadY = (mInputHeight - resizedHeight) * 0.5f;
            const int left = static_cast<int>(std::round(halfPadX - 0.1f));
            const int right = static_cast<int>(std::round(halfPadX + 0.1f));
            const int top = static_cast<int>(std::round(halfPadY - 0.1f));
            const int bottom = static_cast<int>(std::round(halfPadY + 0.1f));
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0, cv::INTER_LINEAR);
            cv::copyMakeBorder(resized, inputImage, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            posePadX = static_cast<float>(left);
            posePadY = static_cast<float>(top);
        }
        else
        {
            inputImage = cv::Mat::zeros(cv::Size(paddedImageSize, paddedImageSize), CV_8UC3);
            cv::Rect roi(0, 0, image_w, image_h);
            image.copyTo(inputImage(roi));
        }

        const bool swapRB = (mDecoder != YoloOutputDecoder::DirectDetections);
        cv::Mat blob = cv::dnn::blobFromImage(inputImage, 1 / 255.0, cv::Size(mInputWidth, mInputHeight), cv::Scalar(0, 0, 0), swapRB, false);
        size_t tpixels = static_cast<size_t>(mInputHeight * mInputWidth * 3);
        std::array<int64_t, 4> input_shape_info{1, 3, mInputHeight, mInputWidth};

        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
        const std::array<const char *, 1> inputNames = {mInputNodeName.c_str()};
        const std::array<const char *, 1> outNames = {mOutputNodeName.c_str()};

        std::vector<Ort::Value> ort_outputs = mSession.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
        if (ort_outputs.empty())
        {
            LOGE("AlgorithmOnYolo empty ort outputs");
            return false;
        }

        const float *pdata = ort_outputs[0].GetTensorMutableData<float>();
        if (!pdata)
        {
            LOGE("AlgorithmOnYolo null output tensor data");
            return false;
        }

        if (mDecoder == YoloOutputDecoder::DirectDetections)
        {
            return decodeDirectDetections(pdata, image_w, image_h, detects);
        }
        if (mDecoder == YoloOutputDecoder::PoseDenseWithNms)
        {
            return decodePoseOutputWithNms(pdata, image_w, image_h, poseScale, posePadX, posePadY, detects);
        }
        return decodeDenseOutputWithNms(pdata, image_w, image_h, paddedImageSize, detects);
    }

    AlgorithmOnYolo::AlgorithmOnYolo(Config *config, std::string &modelPath, std::vector<std::string> &classNames, const std::string &algorithmCode) : Algorithm(config),
                                                                                                                     mClassNames(classNames)
    {
        mEngine = new OnnxRuntimeEngine(config, modelPath, classNames, algorithmCode);
    }

    AlgorithmOnYolo::~AlgorithmOnYolo()
    {
        LOGI("");
        delete mEngine;
        mEngine = nullptr;
    }

    bool AlgorithmOnYolo::objectDetect(cv::Mat &image, std::vector<DetectObject> &detects)
    {
        return mEngine->runInference(image, detects);
    }

}
