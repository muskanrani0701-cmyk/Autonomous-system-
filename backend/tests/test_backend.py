from fastapi.testclient import TestClient
import numpy as np
import cv2
from backend.app.main import app, MAX_BYTES
from backend.app.decision.decision_engine import make_decision
from backend.app.perception.lane_detection import detect_lanes

client = TestClient(app)

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_status(monkeypatch):
    monkeypatch.setattr(
        "backend.app.main.model_status",
        lambda: {"available": False, "model": None, "message": "test"},
    )
    assert client.get("/api/status").status_code == 200

def test_reject_wrong_content_type():
    assert client.post("/api/analyze/image", files={"file":("x.txt",b"abc","text/plain")}).status_code == 415

def test_reject_corrupt_image():
    assert client.post("/api/analyze/image", files={"file":("x.jpg",b"not-image","image/jpeg")}).status_code == 400

def test_reject_oversized_image():
    response = client.post(
        "/api/analyze/image",
        files={"file": ("large.jpg", b"x" * (MAX_BYTES + 1), "image/jpeg")},
    )
    assert response.status_code == 413

def test_analyze_image_and_history(monkeypatch):
    monkeypatch.setattr(
        "backend.app.main.detect_objects",
        lambda image: ([], image.copy(), {"available": False, "message": "test"}),
    )
    image=np.zeros((240,320,3),dtype=np.uint8)
    cv2.line(image,(40,239),(130,130),(255,255,255),5)
    ok, encoded=cv2.imencode(".jpg",image)
    assert ok
    r=client.post("/api/analyze/image",files={"file":("test.jpg",encoded.tobytes(),"image/jpeg")})
    assert r.status_code == 200
    data=r.json()
    assert "decision" in data and "annotated_image" in data
    assert data["decision"]["closest_obstacle_m"] is None
    assert data["decision"]["decision"] == "CAUTION"
    assert any(x["id"]==data["history_id"] for x in client.get("/api/history").json())

def test_decision_is_conservative():
    assert make_decision([],False)["decision"] == "CAUTION"
    assert make_decision([],False)["closest_obstacle_m"] is None
    assert make_decision([],True,object_detection_available=False)["decision"] == "CAUTION"

def test_blank_image_has_no_lane_candidates():
    detected, annotated, info=detect_lanes(np.zeros((240,320,3),dtype=np.uint8))
    assert detected is False and annotated.shape==(240,320,3)
    assert info["candidate_segment_count"]==0

def test_lane_detection_accepts_two_dimensional_hough_lines(monkeypatch):
    lines = np.array([[40, 239, 130, 130]], dtype=np.int32)
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: lines)

    detected, annotated, info = detect_lanes(np.zeros((240,320,3),dtype=np.uint8))

    assert detected is True
    assert annotated.shape == (240,320,3)
    assert info["candidate_segment_count"] == 1
