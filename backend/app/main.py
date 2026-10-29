import base64, json
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from backend.app.database import get_db, AnalysisRecord
from backend.app.perception.object_detection import detect_objects, model_status
from backend.app.perception.lane_detection import detect_lanes
from backend.app.decision.decision_engine import make_decision

BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "frontend"
MAX_BYTES = 12 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
app = FastAPI(title="ADPDS", version="1.0.0", description="Educational software-in-the-loop perception prototype")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ADPDS", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/status")
def status():
    return {"service": "ADPDS", "object_detection": model_status(),
            "lane_detection": "OpenCV classical candidate detector",
            "distance_estimation": "disabled: no camera calibration",
            "safety_mode": "advisory only; not for vehicle control"}

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image.")
    content = await file.read(MAX_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 12 MB limit.")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file could not be decoded as an image.")
    detections, annotated, model_info = detect_objects(image)
    lane_detected, annotated, lane_info = detect_lanes(annotated)
    decision = make_decision(detections, lane_detected, model_info["available"])
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image.")
    result = {"filename": Path(file.filename or "upload").name,
        "image_width": int(image.shape[1]), "image_height": int(image.shape[0]),
        "object_count": len(detections), "detections": detections, "model": model_info,
        "lane_detected": lane_detected, "lane": lane_info, "decision": decision,
        "annotated_image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        "disclaimer": "Educational advisory only. Distances are not calibrated; never use for real driving."}
    record = AnalysisRecord(filename=result["filename"], decision=decision["decision"],
        lane_status=decision["lane_status"], object_count=len(detections),
        summary_json=json.dumps({k:v for k,v in result.items() if k != "annotated_image"}))
    db.add(record); db.commit(); db.refresh(record)
    result["history_id"] = record.id
    return result

@app.get("/api/history")
def history(db: Session = Depends(get_db)):
    rows = db.query(AnalysisRecord).order_by(AnalysisRecord.id.desc()).limit(50).all()
    return [{"id":r.id, "created_at":r.created_at.isoformat(), "filename":r.filename,
             "decision":r.decision, "lane_status":r.lane_status, "object_count":r.object_count} for r in rows]

@app.get("/api/history/{record_id}")
def history_detail(record_id: int, db: Session = Depends(get_db)):
    row = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if row is None: raise HTTPException(status_code=404, detail="History record not found.")
    return {"id":row.id, "created_at":row.created_at.isoformat(), **json.loads(row.summary_json)}
