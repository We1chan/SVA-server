#include "Control.h"

#include <cassert>

int main()
{
	assert(SVAAnalyzer::Control::normalizeBehaviorTypeValue("sleep_duty") == "sleep");
	assert(SVAAnalyzer::Control::normalizeBehaviorTypeValue("SLEEP_DUTY") == "sleep");

	SVAAnalyzer::Control control;
	control.code = "legacy-dwell-test";
	control.streamUrl = "rtsp://127.0.0.1:9994/live/test";
	control.algorithmCode = "yolo11n_80";
	control.objectCode = "person";
	control.recognitionRegion = "0,0,1,0,1,1,0,1";
	control.dwellEnabled = true;
	control.dwellThresholdMs = 5000;

	std::string result;
	assert(control.validateAdd(result));
	assert(control.behaviorRules.size() == 1);
	const SVAAnalyzer::BehaviorRuleConfig &rule = control.behaviorRules.front();
	assert(rule.enabled);
	assert(rule.behaviorType == "dwell");
	assert(rule.geometryId == "region_primary");
	assert(rule.ruleObjectCode == "person");
	assert(rule.thresholdMs == 5000);
	return 0;
}
