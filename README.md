# ADPDS — Autonomous Driving Perception & Decision System

Educational software-in-the-loop prototype for road-image upload, object detection (when YOLO is available), OpenCV lane-marking candidates, conservative advisory output, and SQLite history.

**Safety:** This is not a real self-driving system. Do not connect it to vehicle controls. It does not estimate calibrated distances. The included `yolov8n.pt` is general pretrained model weights, not a model trained on a selected Kaggle dataset.

## Fix for the Vercel 404 errors

The frontend files are stored directly in `frontend/`:

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`

The HTML now loads `/styles.css` and `/app.js` (not `/static/...`).

### Deploy the frontend to Vercel

1. Import this GitHub repository into Vercel.
2. In **Project Settings → Build and Deployment**, set **Root Directory** to `frontend`.
3. Leave the Framework Preset as **Other**. No build command is needed; the frontend is static.
4. Redeploy.

This fixes the CSS and JavaScript 404s. The frontend alone does not run the Python perception API.

## Deploy the backend separately

The FastAPI backend needs Python, OpenCV, SQLite, and Ultralytics. A `render.yaml` blueprint is included as a starting point for deploying it to Render:

1. Create a new Blueprint in Render and select this repository.
2. Deploy the `adpds-backend` web service.
3. Copy its public HTTPS URL, for example `https://adpds-backend.onrender.com`.

Then connect Vercel to the backend:

1. Open `frontend/index.html`.
2. Find `<meta name="adpds-api-base" content="">`.
3. Put your backend URL inside `content`, with no trailing slash. Example:
   `<meta name="adpds-api-base" content="https://adpds-backend.onrender.com">`
4. Commit/push that change and redeploy Vercel.

The backend CORS setting is controlled by `ADPDS_CORS_ORIGINS` in `render.yaml`. For a public demo it is currently `*`. For a restricted deployment, replace it with your exact Vercel origin, such as `https://your-project.vercel.app`.

**Important deployment limitations:** Free/low-memory Python hosts may not have enough RAM or build time for PyTorch/Ultralytics. If the backend fails to build or the model cannot load, the dashboard should report API/model unavailable rather than invent detections. SQLite on an ephemeral host may reset when the service restarts; use persistent storage or a managed database if history must survive restarts.

## Run locally on Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_server.py
```

Open http://127.0.0.1:8000 and API docs at http://127.0.0.1:8000/docs.

## Test

```powershell
python -m pytest -q
```

## Model behavior

The app attempts to load `yolov8n.pt` using Ultralytics. If unavailable, it clearly reports lane-only mode and does not fabricate detections. Set `ADPDS_MODEL_PATH` to use custom weights. To claim dataset-specific performance, separately train and evaluate on real-world data with a held-out test set.


## Render free-tier memory settings

This version keeps one Uvicorn worker, delays importing/loading YOLO until the first image analysis, serializes inference, uses CPU inference at 512 pixels, limits uploads to 6 MB, resizes large images to at most 960 pixels on the long side, and returns a smaller JPEG. The `/api/status` check no longer loads PyTorch just to open the home page.

If the 512 MB free instance still restarts during the first YOLO inference, set the Render environment variable `ADPDS_ENABLE_YOLO` to `0`. This keeps the website and OpenCV lane-candidate analysis online but disables YOLO object detection. Set it back to `1` to retry YOLO. Free-tier RAM may still be insufficient for PyTorch/Ultralytics; do not treat the demo as vehicle-control or safety-certified software.
