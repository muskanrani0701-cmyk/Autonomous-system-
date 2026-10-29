import os
import cv2
import numpy as np

_model = None
_model_error = None

def _load_model():
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None:
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(os.getenv("ADPDS_MODEL_PATH", "yolov8n.pt"))
        return _model
    except Exception as exc:
        _model_error = str(exc)
        return None

def model_status():
    if _load_model() is not None:
        return {"available": True, "model": os.getenv("ADPDS_MODEL_PATH", "yolov8n.pt"), "message": "Object detector loaded"}
    return {"available": False, "model": None, "message": f"Object detection unavailable: {_model_error or 'model could not be loaded'}"}

def detect_objects(image: np.ndarray, confidence: float = 0.35):
    # Actual model inference only; no contour-based fake object detections.
    annotated = image.copy()
    model = _load_model()
    if model is None:
        return [], annotated, model_status()
    try:
        result = model.predict(source=image, conf=confidence, verbose=False)[0]
        detections = []
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                label = str(names.get(cls_id, cls_id))
                detections.append({"label": label, "confidence": round(score, 3),
                                   "bbox": [x1, y1, x2, y2], "distance_m": None})
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 220, 100), 2)
                cv2.putText(annotated, f"{label} {score:.2f}", (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 100), 2)
        return detections, annotated, model_status()
    except Exception as exc:
        return [], annotated, {"available": False, "model": os.getenv("ADPDS_MODEL_PATH", "yolov8n.pt"),
                               "message": f"Inference failed: {exc}"}
