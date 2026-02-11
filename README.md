# Smart Parking System

Real-time parking occupancy detection and monitoring. The backend runs YOLOv8 + ByteTrack to detect vehicles and exposes a Flask API for status, video stream, and predictions. The frontend (Next.js) shows live stats, camera feed, and estimated wait time when the lot is full.

## Features
- Real-time vehicle detection and tracking (YOLOv8 + ByteTrack)
- Parking occupancy and available slots calculation
- MJPEG video stream + snapshot endpoint
- Full-lot alarm from backend (Windows system beep)
- Estimated time-to-free based on CSV history
- Dashboard UI (Next.js + Tailwind)

## Repository Layout
```
AI_Journey/
  VechileParkingSystem/  # Flask API + CV inference
  parking_ui/            # Next.js dashboard
  BackEnd/               # Legacy folder (not used by current app)
```

## Requirements
- Python 3.8+
- Node.js 18+
- pnpm (recommended)
- CUDA optional (for GPU inference)

## Backend Setup (Flask API)
```bash
cd VechileParkingSystem
py -m pip install -r requirements.txt
py -m pip install lap
```

Download YOLO weights:
- Get `yolov8n.pt` from Ultralytics releases and place it in `VechileParkingSystem/`.

Run the API:
```bash
py api.py
```

API endpoints:
- `GET /api/health`
- `GET /api/parking-status`
- `GET /api/parking-details`
- `GET /api/video-stream`
- `GET /api/snapshot`
- `GET /api/events`
- `GET /api/statistics`
- `POST /api/alarm/stop`

## Frontend Setup (Next.js)
```bash
cd parking_ui
pnpm install
pnpm dev
```

Configure API base URL (optional):
- Create `parking_ui/.env.local`
```
NEXT_PUBLIC_API_BASE=http://localhost:5000
```

## Configuration
Backend settings in `VechileParkingSystem/api.py`:
- `TOTAL_PARKING_SLOTS` (set to 10)
- `PROCESS_IMG_WIDTH`, `INFERENCE_IMGSZ`, `FRAME_SKIP`, `STREAM_FPS`
- `MIN_PREDICTION_MINUTES`

Input video:
- Place the video at `VechileParkingSystem/inputs/entrance_video.mp4`

History CSVs used for prediction:
- `VechileParkingSystem/outputs/parking_history.csv`
- `VechileParkingSystem/outputs/parking_events.csv`

## Troubleshooting
- `Cannot find module ... next ...`: run `pnpm install` inside `parking_ui/` (this creates `node_modules`).
- `pnpm dev` was run in `FrontEnd/`: the correct folder is `parking_ui/`.
- If the API fails to start, verify `yolov8n.pt` is in `VechileParkingSystem/`.
- If MJPEG stream is slow, reduce `STREAM_FPS` or increase `FRAME_SKIP`.
- If prediction seems off, check the CSV history format in `outputs/`.

## License
MIT (add your preferred license if different).
