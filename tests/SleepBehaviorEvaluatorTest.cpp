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
