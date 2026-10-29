import cv2
import numpy as np

def detect_lanes(image: np.ndarray):
    # Classical CV returns candidate segments, not validated lane boundaries.
    annotated = image.copy()
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    mask = np.zeros_like(edges)
    polygon = np.array([[(int(w*.08), h-1), (int(w*.42), int(h*.55)),
                         (int(w*.58), int(h*.55)), (int(w*.92), h-1)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=35,
                            minLineLength=max(25, int(w*.08)), maxLineGap=35)
    candidates = []
    if lines is not None:
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            if np.hypot(x2-x1, y2-y1) < max(25, w*.08) or abs(y2-y1) < 12:
                continue
            slope = (y2-y1) / ((x2-x1) if abs(x2-x1) > 1 else 1)
            if abs(slope) < .25:
                continue
            candidates.append([x1, y1, x2, y2])
            cv2.line(annotated, (x1,y1), (x2,y2), (255,180,0), 3)
    detected = len(candidates) >= 1
    return detected, annotated, {
        "candidate_segment_count": len(candidates),
        "status": "Candidate lane-like markings found" if detected else "No convincing lane-like segments found",
        "warning": "Classical CV lane candidates are not a validated lane-boundary estimate."
    }
