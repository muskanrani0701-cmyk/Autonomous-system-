# ADPDS — Autonomous Driving Perception & Decision System

Educational software-in-the-loop prototype: road-image upload, object detection when a real model is available, OpenCV lane-marking candidates, conservative advisory, and SQLite history.

**Safety:** Not a real self-driving system. Do not connect it to vehicle controls. No calibrated distance estimation is claimed. The default YOLO model is general pretrained weights, not trained on your selected Kaggle dataset.

## Windows setup (PowerShell)
```powershell
cd ADPDS_working_project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_server.py
```
If activation is blocked in this terminal: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

Open http://127.0.0.1:8000 ; API docs: http://127.0.0.1:8000/docs

## Test
In a second terminal, with the environment activated:
`python -m pytest -q`

## Model behavior
The app attempts to load `yolov8n.pt` using Ultralytics. If unavailable, it clearly reports lane-only mode and does not fabricate detections. Set `ADPDS_MODEL_PATH` to use custom weights. To claim dataset-specific performance, separately train/evaluate on your real-world Kaggle data with a held-out test set.

## Issues corrected from the supplied Colab export
The export repeatedly rewrote modules, included Colab-only shell commands, and used fallback logic that could invent object/lane detections. This starter has one canonical copy of each module and avoids fabricated detections.
