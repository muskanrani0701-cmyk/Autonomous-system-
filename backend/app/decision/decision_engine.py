def make_decision(
    detections: list[dict],
    lane_detected: bool,
    object_detection_available: bool = True,
) -> dict:
    # Conservative demo advisory only. No calibrated distance or vehicle control.
    relevant = {"car", "truck", "bus", "motorcycle", "motorbike", "bicycle", "person"}
    nearby = [d for d in detections if str(d.get("label", "")).lower() in relevant]
    if not object_detection_available:
        decision = "CAUTION"
        reason = "Object detection is unavailable; scene uncertainty is high."
        if not lane_detected:
            reason += " Lane-like markings were also not confidently detected."
    elif not lane_detected:
        decision, reason = "CAUTION", "Lane-like markings were not confidently detected; scene uncertainty is high."
    elif nearby:
        decision, reason = "CAUTION", f"{len(nearby)} road-user/vehicle detection(s) require review; distance is not calibrated."
    else:
        decision, reason = "MAINTAIN CAUTION", "No relevant object was detected. This does not prove the road is clear."
    return {"decision": decision, "reason": reason, "closest_obstacle_m": None,
            "lane_status": "Candidate markings detected" if lane_detected else "No candidate markings detected"}
