#include "Algorithm.h"
#include "BehaviorEvaluator.h"
#include "Control.h"

#include <cassert>

using namespace SVAAnalyzer;

namespace
{
    Control sleepControl()
    {
        Control control;
        RegionConfig region;
        region.id = "region_primary";
        region.name = "full frame";
        region.primary = true;
        region.normalizedPoints = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
        control.regions.push_back(region);

        BehaviorRuleConfig rule;
        rule.id = "sleep-rule";
        rule.behaviorType = "sleep";
        rule.geometryId = "region_primary";
        rule.ruleObjectCode = "person";
        rule.enabled = true;
        control.behaviorRules.push_back(rule);
        return control;
    }

    DetectObject trackedPerson()
    {
        DetectObject detect;
        detect.trackId = 1;
        detect.class_name = "person";
        RegionTemporalState region;
        region.inRegion = true;
        detect.regionStates["region_primary"] = region;
        return detect;
    }
}

int main()
{
    // The business API sends sleep_duty + durationMs. It must survive config
    // normalization and honor the requested duration instead of being discarded.
    Control businessControl = sleepControl();
    Json::Value rules(Json::arrayValue);
    Json::Value rule;
    rule["id"] = "business-sleep";
    rule["behaviorType"] = "sleep_duty";
    rule["geometryId"] = "region_primary";
    rule["ruleObjectCode"] = "person";
    rule["durationMs"] = 45000;
    rules.append(rule);
    assert(businessControl.loadBehaviorRulesConfig(rules));
    assert(businessControl.behaviorRules.size() == 1);
    assert(businessControl.behaviorRules.front().behaviorType == "sleep");
    assert(businessControl.behaviorRules.front().thresholdMs == 45000);

    rule["durationMs"] = "60000";
    rule["thresholdMs"] = 30000;
    rules[0] = rule;
    assert(businessControl.loadBehaviorRulesConfig(rules));
    assert(businessControl.behaviorRules.front().thresholdMs == 30000);

    const Control control = sleepControl();
    DetectObject detect = trackedPerson();
    detect.sleepEvidenceEvaluated = true;
    detect.sleepEvent = true;
    BehaviorDecision decision = evaluateAtomicBehavior(control, detect);
    assert(decision.matched);
    assert(decision.behaviorType == "sleep");
    assert(decision.ruleId == "sleep-rule");

    detect.sleepEvent = false;
    decision = evaluateAtomicBehavior(control, detect);
    assert(!decision.matched);

    // Non-pose algorithms retain the legacy stationary/aspect-ratio behavior.
    detect.sleepEvidenceEvaluated = false;
    detect.dwellMs = 15000;
    detect.regionStates["region_primary"].inRegionDurationMs = 15000;
    detect.x1 = 0;
    detect.y1 = 0;
    detect.x2 = 200;
    detect.y2 = 100;
    decision = evaluateAtomicBehavior(control, detect);
    assert(decision.matched);
    return 0;
}
