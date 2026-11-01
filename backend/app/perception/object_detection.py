
import cv2
import numpy as np


def model_status():
    available = hasattr(cv2, "HOGDescriptor")

    return {
        "available": available,
        "model": "OpenCV HOG" if available else None,
        "message": (
            "Lightweight pedestrian detection available"
            if available
            else "OpenCV build does not support HOG detection"
        )
    }


def detect_objects(image: np.ndarray, confidence: float = 0.35):
    annotated = image.copy()

    if not hasattr(cv2, "HOGDescriptor"):
        return [], annotated, model_status()

    try:
        detector = cv2.HOGDescriptor()
        detector.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )

        boxes, weights = detector.detectMultiScale(
            image,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        detections = []

        for (x, y, w, h), score in zip(boxes, weights):
            score = float(score)

            if score < confidence:
                continue

            detections.append({
                "label": "person",
                "confidence": round(score, 3),
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "distance_m": None
            })

            cv2.rectangle(
                annotated, (x, y), (x + w, y + h),
                (20, 220, 100), 2
            )

        return detections, annotated, model_status()

    except Exception as exc:
        return [], annotated, {
            "available": False,
            "model": "OpenCV HOG",
            "message": f"Detection failed: {exc}"
        }
