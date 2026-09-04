#ifndef ANALYZER_LIVE_OUTPUT_H
#define ANALYZER_LIVE_OUTPUT_H

#include <json/json.h>
#include <string>

namespace SVAAnalyzer
{
    struct LiveOutputRequest
    {
        std::string controlCode;
        bool videoEnabled = false;
        bool liveEventEnabled = false;
        float wsEventFps = 8.0f;
        std::string pushStreamUrl;
    };

    inline bool parseLiveOutputRequest(const Json::Value &root,
                                       LiveOutputRequest &request,
                                       std::string &error)
    {
        request = LiveOutputRequest();
        error.clear();
        if (!root.isObject())
        {
            error = "invalid request parameter";
            return false;
        }
        if (!root["controlCode"].isString() || root["controlCode"].asString().empty())
        {
            error = "controlCode is required";
            return false;
        }
        request.controlCode = root["controlCode"].asString();
        if (root.isMember("videoEnabled") && !root["videoEnabled"].isBool())
        {
            error = "videoEnabled must be boolean";
            return false;
        }
        if (root["videoEnabled"].isBool())
        {
            request.videoEnabled = root["videoEnabled"].asBool();
        }
        if (root.isMember("liveEventEnabled") && !root["liveEventEnabled"].isBool())
        {
            error = "liveEventEnabled must be boolean";
            return false;
        }
        if (root["liveEventEnabled"].isBool())
        {
            request.liveEventEnabled = root["liveEventEnabled"].asBool();
        }
        if (root.isMember("wsEventFps") && !root["wsEventFps"].isNumeric())
        {
            error = "wsEventFps must be numeric";
            return false;
        }
        if (root["wsEventFps"].isNumeric())
        {
            request.wsEventFps = root["wsEventFps"].asFloat();
        }
        if (request.wsEventFps <= 0.0f || request.wsEventFps > 30.0f)
        {
            error = "wsEventFps must be greater than 0 and no more than 30";
            return false;
        }
        if (root["pushStreamUrl"].isString())
        {
            request.pushStreamUrl = root["pushStreamUrl"].asString();
        }
        if (request.videoEnabled && request.pushStreamUrl.empty())
        {
            error = "pushStreamUrl is required when videoEnabled is true";
            return false;
        }
        return true;
    }
}

#endif
