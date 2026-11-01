
import cv2
import numpy as np

_hog = None

def _load_model():
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )
    return _hog


def model_status():
    return {
        "available": True,
        "model": "OpenCV HOG",
        "message": "Lightweight pedestrian detection enabled; people only"
    }


def detect_objects(image: np.ndarray, confidence: float = 0.35):
    annotated = image.copy()
    detector = _load_model()

    try:
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
                annotated,
                (x, y),
                (x + w, y + h),
                (20, 220, 100),
                2
            )

            cv2.putText(
                annotated,
                f"person {score:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 220, 100),
                2
            )

        return detections, annotated, model_status()

    except Exception as exc:
        return [], annotated, {
            "available": False,
            "model": "OpenCV HOG",
            "message": f"Detection failed: {exc}"
        }
