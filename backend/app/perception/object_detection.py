
import os
import cv2
import numpy as np

_model = None
_model_error = None


def _yolo_enabled():
    return os.getenv("ADPDS_ENABLE_YOLO", "0").lower() in (
        "1", "true", "yes", "on"
    )


def _load_model():
    global _model, _model_error

    if not _yolo_enabled():
        return None

    if _model is not None:
        return _model

    if _model_error is not None:
        return None

    try:
        from ultralytics import YOLO

        model_path = os.getenv("ADPDS_MODEL_PATH", "yolov8n.pt")
        _model = YOLO(model_path)
        return _model

    except Exception as exc:
        _model_error = str(exc)
        return None


def model_status():
    if not _yolo_enabled():
        return {
            "available": False,
            "model": None,
            "message": "YOLO disabled to conserve Render memory"
        }

    if _model is not None:
        return {
            "available": True,
            "model": os.getenv("ADPDS_MODEL_PATH", "yolov8n.pt"),
            "message": "Object detector loaded"
        }

    if _model_error is not None:
        return {
            "available": False,
            "model": None,
            "message": f"Object detection unavailable: {_model_error}"
        }

    return {
        "available": False,
        "model": os.getenv("ADPDS_MODEL_PATH", "yolov8n.pt"),
        "message": "YOLO enabled but model is not loaded yet"
    }


def detect_objects(image: np.ndarray, confidence: float = 0.35):
    annotated = image.copy()

    if not _yolo_enabled():
        return [], annotated, model_status()

    model = _load_model()

    if model is None:
        return [], annotated, model_status()

    try:
        results = model.predict(
            source=image,
            conf=confidence,
            imgsz=416,
            device="cpu",
            max_det=30,
            verbose=False
        )

        result = results[0]
        detections = []
        names = result.names

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().numpy().astype(int).tolist()
                )

                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                label = str(names.get(cls_id, cls_id))

                detections.append({
                    "label": label,
                    "confidence": round(score, 3),
                    "bbox": [x1, y1, x2, y2],
                    "distance_m": None
                })

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (20, 220, 100),
                    2
                )

                cv2.putText(
                    annotated,
                    f"{label} {score:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (20, 220, 100),
                    2
                )

        return detections, annotated, model_status()

    except Exception as exc:
        return [], annotated, {
            "available": False,
            "model": os.getenv("ADPDS_MODEL_PATH", "yolov8n.pt"),
            "message": f"Inference failed: {exc}"
        }
