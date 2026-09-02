#ifndef ANALYZER_SLEEP_ANALYZER_H
#define ANALYZER_SLEEP_ANALYZER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cv
{
    class Mat;
}

namespace SVAAnalyzer
{
    class Config;
    struct DetectObject;

    /**
     * Per-stream sleep cascade. Pose is evaluated first, reliable eyes confirm
     * the candidate, and conservative pose evidence is used when eyes are not
     * observable. All ONNX inference in a GPU build is strict GPU-only.
     */
    class SleepAnalyzer
    {
    public:
        SleepAnalyzer(Config *config, const std::string &eyeModelPath);
        ~SleepAnalyzer();

        SleepAnalyzer(const SleepAnalyzer &) = delete;
        SleepAnalyzer &operator=(const SleepAnalyzer &) = delete;

        void process(const std::string &streamCode,
                     cv::Mat &image,
                     const std::vector<DetectObject *> &detects,
                     int64_t timestampMs);
        void clearStream(const std::string &streamCode);
        bool eyeModelAvailable() const;

    private:
        class Impl;
        std::unique_ptr<Impl> mImpl;
    };
}

#endif
