#include <cassert>
#include <json/json.h>
#include "../Analyzer/Core/LiveOutput.h"

int main()
{
    SVAAnalyzer::LiveOutputRequest request;
    std::string error;

    Json::Value valid;
    valid["controlCode"] = "deployment-1";
    valid["videoEnabled"] = true;
    valid["liveEventEnabled"] = true;
    valid["wsEventFps"] = 8.0;
    valid["pushStreamUrl"] = "rtsp://127.0.0.1/live/deployment-1";
    assert(SVAAnalyzer::parseLiveOutputRequest(valid, request, error));
    assert(request.controlCode == "deployment-1");
    assert(request.videoEnabled);
    assert(request.liveEventEnabled);
    assert(request.wsEventFps == 8.0f);

    Json::Value missingCode = valid;
    missingCode.removeMember("controlCode");
    assert(!SVAAnalyzer::parseLiveOutputRequest(missingCode, request, error));

    Json::Value missingPushUrl = valid;
    missingPushUrl.removeMember("pushStreamUrl");
    assert(!SVAAnalyzer::parseLiveOutputRequest(missingPushUrl, request, error));

    Json::Value malformed = valid;
    malformed["videoEnabled"] = "false";
    assert(!SVAAnalyzer::parseLiveOutputRequest(malformed, request, error));

    Json::Value disabled;
    disabled["controlCode"] = "deployment-1";
    disabled["videoEnabled"] = false;
    assert(SVAAnalyzer::parseLiveOutputRequest(disabled, request, error));
    assert(!request.videoEnabled);

    return 0;
}
