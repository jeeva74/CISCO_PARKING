# -*- coding: utf-8 -*-
"""
Flask API Server for Smart Parking System.
Provides REST endpoints for parking status, video stream, and real-time data.
"""

import os
import json
import threading
import time
import csv
from datetime import datetime, timedelta
from collections import deque
from flask import Flask, jsonify, Response, stream_with_context
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
try:
    import torch
except Exception:
    torch = None
try:
    import winsound
except Exception:
    winsound = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ======================
# Global State Variables
# ======================
TOTAL_PARKING_SLOTS = 10
MIN_PREDICTION_MINUTES = 5
MAX_PREDICTION_MINUTES = 20
# Performance tuning
PROCESS_IMG_WIDTH = 960
INFERENCE_IMGSZ = 640
FRAME_SKIP = 1
STREAM_FPS = 15
JPEG_QUALITY = 80
WRITE_LATEST_FRAME = False
DISK_FRAME_INTERVAL = 30
DEBUG_IO = False
class ParkingState:
    def __init__(self):
        self.two_wheeler_count = 0
        self.four_wheeler_count = 0
        self.is_running = False
        self.lock = threading.Lock()
        self.alarm_suppressed = False
        self.recent_events = deque(maxlen=100)
        self.status_history = deque(maxlen=100)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.prediction = {
            "status": "available",
            "occupancy_trend": "stable",
            "free_in_minutes": None,
            "free_at": None
        }

parking_state = ParkingState()

# Alarm controls
alarm_thread = None
alarm_stop_event = threading.Event()

# YOLO Model
model = None

# ======================
# Helper Functions
# ======================

def get_parking_status():
    """Calculate parking status from current counts."""
    with parking_state.lock:
        occupied = parking_state.two_wheeler_count + parking_state.four_wheeler_count
    
    occupied = max(0, min(occupied, TOTAL_PARKING_SLOTS))
    available = TOTAL_PARKING_SLOTS - occupied
    occupancy_percentage = (occupied / TOTAL_PARKING_SLOTS) * 100
    
    if occupied >= TOTAL_PARKING_SLOTS:
        status = "full"
    elif occupancy_percentage >= 70:
        status = "partial"
    else:
        status = "available"
    
    return {
        "occupied": occupied,
        "available": available,
        "total": TOTAL_PARKING_SLOTS,
        "occupancy_percentage": occupancy_percentage,
        "status": status
    }

def compute_prediction():
    """Simple prediction based on historical data."""
    status = get_parking_status()
    occupied = status["occupied"]
    
    prediction = {
        "status": status["status"],
        "occupancy_trend": "stable",
        "free_in_minutes": None,
        "free_at": None
    }
    
    if occupied >= TOTAL_PARKING_SLOTS:
        est_seconds = estimate_time_to_free_from_history(required_slots=1, lookback_hours=72.0)
        if est_seconds is None:
            est_minutes = 15
        else:
            est_minutes = int(round(est_seconds / 60.0))
        est_minutes = max(MIN_PREDICTION_MINUTES, min(MAX_PREDICTION_MINUTES, est_minutes))
        prediction["free_in_minutes"] = est_minutes
        prediction["free_at"] = (datetime.now() + timedelta(minutes=est_minutes)).strftime("%H:%M")
    
    return prediction

def estimate_time_to_free_from_history(
    required_slots: int = 1,
    lookback_hours: float | None = 72.0,
    max_samples: int = 10,
):
    """Estimate time until at least `required_slots` are free using CSV history."""
    rows = []

    def _read_rows_from(path: str):
        out = []
        if not os.path.isfile(path):
            return out
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ts = r.get("Timestamp") or r.get("timestamp")
                    avail_s = r.get("Available") or r.get("available")
                    if not ts or avail_s is None:
                        continue
                    try:
                        t = datetime.fromisoformat(ts)
                        avail = int(avail_s)
                    except Exception:
                        continue
                    out.append((t, avail))
        except Exception:
            return []
        return out

    history_path = os.path.join("outputs", "parking_history.csv")
    events_path = os.path.join("outputs", "parking_events.csv")
    rows.extend(_read_rows_from(history_path))
    rows.extend(_read_rows_from(events_path))

    if not rows:
        return None

    if lookback_hours is not None:
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        rows = [r for r in rows if r[0] >= cutoff]
        if not rows:
            return None

    rows.sort(key=lambda x: x[0])

    deltas = []
    n = len(rows)
    for i, (t_i, avail_i) in enumerate(rows):
        if avail_i <= 0:
            for j in range(i + 1, n):
                t_j, avail_j = rows[j]
                if avail_j >= required_slots:
                    deltas.append((t_j - t_i).total_seconds())
                    break

    if not deltas:
        return None

    if max_samples is not None and len(deltas) > max_samples:
        deltas = deltas[-max_samples:]

    deltas.sort()
    mid = len(deltas) // 2
    if len(deltas) % 2 == 1:
        return deltas[mid]
    return 0.5 * (deltas[mid - 1] + deltas[mid])

def log_event(event_type, label, occupied_2w, occupied_4w):
    """Log parking events."""
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "vehicle_type": label,
        "two_wheeler_occupied": occupied_2w,
        "four_wheeler_occupied": occupied_4w
    }
    parking_state.recent_events.append(event)

def _alarm_worker(stop_event: threading.Event):
    """Background worker that beeps until stop_event is set."""
    freq = 2000
    dur = 800
    while not stop_event.is_set():
        if winsound is not None:
            try:
                winsound.Beep(freq, dur)
            except RuntimeError:
                pass
        else:
            print("[ALARM] Parking full!")
        stop_event.wait(0.4)

def start_alarm():
    """Start the alarm thread if not already running."""
    global alarm_thread, alarm_stop_event
    if alarm_thread is not None and alarm_thread.is_alive():
        return
    alarm_stop_event.clear()
    alarm_thread = threading.Thread(target=_alarm_worker, args=(alarm_stop_event,), daemon=True)
    alarm_thread.start()

def stop_alarm():
    """Signal the alarm thread to stop and wait briefly."""
    global alarm_thread, alarm_stop_event
    alarm_stop_event.set()
    if alarm_thread is not None:
        alarm_thread.join(timeout=1.0)
    alarm_thread = None


# ======================
# API Endpoints
# ======================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    is_running_value = parking_state.is_running
    print(f"[HEALTH] parking_state.is_running = {is_running_value}")
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "is_running": is_running_value
    })

@app.route('/api/parking-status', methods=['GET'])
def parking_status():
    """Get current parking status."""
    status = get_parking_status()
    with parking_state.lock:
        status.update({
            "two_wheeler_occupied": parking_state.two_wheeler_count,
            "four_wheeler_occupied": parking_state.four_wheeler_count,
            "timestamp": datetime.now().isoformat()
        })
    
    prediction = compute_prediction()
    status["prediction"] = prediction
    
    return jsonify(status)

@app.route('/api/parking-details', methods=['GET'])
def parking_details():
    """Get detailed parking information."""
    status = get_parking_status()
    
    with parking_state.lock:
        two_wheeler = {
            "occupied": parking_state.two_wheeler_count,
            "available": max(0, TOTAL_PARKING_SLOTS - parking_state.two_wheeler_count),
            "total": TOTAL_PARKING_SLOTS
        }
        
        four_wheeler = {
            "occupied": parking_state.four_wheeler_count,
            "available": max(0, TOTAL_PARKING_SLOTS - parking_state.four_wheeler_count),
            "total": TOTAL_PARKING_SLOTS
        }
    
    prediction = compute_prediction()
    parking_state.prediction = prediction

    return jsonify({
        "twoWheeler": two_wheeler,
        "fourWheeler": four_wheeler,
        "predictions": prediction,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/events', methods=['GET'])
def get_events():
    """Get recent parking events."""
    return jsonify({
        "events": list(parking_state.recent_events),
        "total_events": len(parking_state.recent_events),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/alarm/stop', methods=['POST'])
def stop_alarm_endpoint():
    """Stop the backend alarm sound."""
    parking_state.alarm_suppressed = True
    stop_alarm()
    return jsonify({"status": "stopped", "timestamp": datetime.now().isoformat()})

@app.route('/api/video-stream', methods=['GET'])
def video_stream():
    """Stream video feed with detection overlays."""
    return Response(
        stream_with_context(generate_frames()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/snapshot', methods=['GET'])
def snapshot():
    """Return the latest processed frame as a single JPEG image.

    This is useful for frontends that cannot reliably render MJPEG streams.
    The endpoint returns the most recent frame stored in `parking_state.current_frame`.
    """
    # Optional disk fallback for process isolation
    if WRITE_LATEST_FRAME:
        latest_frame_path = r"E:\Project\AI_Journey\VechileParkingSystem\_latest_frame.jpg"
        if os.path.exists(latest_frame_path):
            try:
                with open(latest_frame_path, 'rb') as f:
                    data = f.read()
                if data:
                    return Response(data, mimetype='image/jpeg')
            except Exception:
                if DEBUG_IO:
                    with open('snapshot_log.txt', 'a') as log:
                        log.write(f"[{datetime.now()}] snapshot disk read failed\n")
                        log.flush()

    with parking_state.frame_lock:
        frame = parking_state.current_frame

    if frame is None:
        # Try to open a video file and grab a single frame as fallback
        video_path = os.path.join("inputs", "entrance_video.mp4")
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                success, fallback_frame = cap.read()
                cap.release()
                if success and fallback_frame is not None:
                    ret, buffer = cv2.imencode(
                        '.jpg',
                        fallback_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )
                    if ret:
                        return Response(buffer.tobytes(), mimetype='image/jpeg')
        return jsonify({"error": "no_frame_available"}), 503

    ret, buffer = cv2.imencode(
        '.jpg',
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ret:
        return jsonify({"error": "encode_failed"}), 500

    return Response(buffer.tobytes(), mimetype='image/jpeg')

def generate_frames():
    """Generator function to stream video frames."""
    frame_delay = 1.0 / max(1, STREAM_FPS)
    while True:
        with parking_state.frame_lock:
            frame = None if parking_state.current_frame is None else parking_state.current_frame.copy()
        if frame is None:
            time.sleep(frame_delay)
            continue

        ret, buffer = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

        time.sleep(frame_delay)

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get parking statistics."""
    status = get_parking_status()
    recent_events = list(parking_state.recent_events)
    
    entry_count = sum(1 for e in recent_events if e.get("event_type") == "entry")
    exit_count = sum(1 for e in recent_events if e.get("event_type") == "exit")
    
    return jsonify({
        "current_status": status,
        "recent_events_count": len(recent_events),
        "total_entries": entry_count,
        "total_exits": exit_count,
        "timestamp": datetime.now().isoformat()
    })


# ======================
# Background Frame Processing
# ======================

def process_video_frames():
    """Background thread to continuously read and process video frames."""
    global model
    parking_state.is_running = True
    print("🎥Frame processor thread started - is_running set to True")
    
    try:
        video_path = os.path.join("inputs", "entrance_video.mp4")
        print(f"📁 Looking for video at: {os.path.abspath(video_path)}")
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            parking_state.is_running = False
            return
        
        print(f"✅ Video file found, attempting to open...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            parking_state.is_running = False
            return
        
        print(f"✅ Video opened successfully, starting frame processing...")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / fps
        if model is None:
            initialize_model()
        device = 0 if (torch is not None and torch.cuda.is_available()) else "cpu"
        COUNT_LINE_Y = None
        frame_count = 0
        vehicle_history = {}
        FOUR_WHEELERS = ["car"]
        TWO_WHEELERS = ["motorcycle"]
        while parking_state.is_running:
            success, frame = cap.read()
            if not success:
                print(f"⏸  Video ended, looping back. Frame count: {frame_count}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue

            frame_count += 1
            if PROCESS_IMG_WIDTH and frame.shape[1] != PROCESS_IMG_WIDTH:
                scale = PROCESS_IMG_WIDTH / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (PROCESS_IMG_WIDTH, int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            if COUNT_LINE_Y is None:
                COUNT_LINE_Y = int(frame.shape[0] * 0.6)

            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                with parking_state.frame_lock:
                    parking_state.current_frame = frame.copy()
                time.sleep(frame_delay)
                continue

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                imgsz=INFERENCE_IMGSZ,
                device=device,
            )[0]

            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else None
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cy = int((y1 + y2) / 2)
                    label = model.names[cls_id]

                    if label not in FOUR_WHEELERS + TWO_WHEELERS or track_id is None:
                        continue

                    if track_id not in vehicle_history:
                        initial_side = "above" if cy < COUNT_LINE_Y else "below"
                        vehicle_history[track_id] = {
                            "label": label,
                            "last_cy": cy,
                            "last_side": initial_side,
                        }

                    last_side = vehicle_history[track_id]["last_side"]
                    current_side = "above" if cy < COUNT_LINE_Y else "below"

                    if last_side == "above" and current_side == "below":
                        with parking_state.lock:
                            current_occupied = parking_state.two_wheeler_count + parking_state.four_wheeler_count
                            if current_occupied < TOTAL_PARKING_SLOTS:
                                if label in FOUR_WHEELERS:
                                    parking_state.four_wheeler_count += 1
                                elif label in TWO_WHEELERS:
                                    parking_state.two_wheeler_count += 1
                                log_event(
                                    "entry",
                                    label,
                                    parking_state.two_wheeler_count,
                                    parking_state.four_wheeler_count,
                                )

                    elif last_side == "below" and current_side == "above":
                        with parking_state.lock:
                            if label in FOUR_WHEELERS and parking_state.four_wheeler_count > 0:
                                parking_state.four_wheeler_count -= 1
                            elif label in TWO_WHEELERS and parking_state.two_wheeler_count > 0:
                                parking_state.two_wheeler_count -= 1
                            log_event(
                                "exit",
                                label,
                                parking_state.two_wheeler_count,
                                parking_state.four_wheeler_count,
                            )

                    vehicle_history[track_id]["last_cy"] = cy
                    vehicle_history[track_id]["last_side"] = current_side

                    color = (0, 255, 0) if label in FOUR_WHEELERS else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

            cv2.line(frame, (0, COUNT_LINE_Y), (frame.shape[1], COUNT_LINE_Y), (0, 255, 255), 2)
            with parking_state.lock:
                occupied = parking_state.two_wheeler_count + parking_state.four_wheeler_count
            if occupied >= TOTAL_PARKING_SLOTS:
                if not parking_state.alarm_suppressed:
                    start_alarm()
            else:
                parking_state.alarm_suppressed = False
                stop_alarm()
            cv2.putText(
                frame,
                f"Occupied: {occupied}/{TOTAL_PARKING_SLOTS}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            with parking_state.frame_lock:
                parking_state.current_frame = frame.copy()

            if WRITE_LATEST_FRAME and frame_count % DISK_FRAME_INTERVAL == 0:
                try:
                    ret, buffer = cv2.imencode(
                        '.jpg',
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )
                    if ret:
                        latest_frame_path = os.path.join(os.getcwd(), '_latest_frame.jpg')
                        with open(latest_frame_path, 'wb') as f:
                            f.write(buffer.tobytes())
                except Exception:
                    pass

            if frame_count % 120 == 0:
                print(f"[FRAME] Processed {frame_count}, is_running={parking_state.is_running}")

            time.sleep(frame_delay)
    except Exception as e:
        print(f"❌ Error in frame processor: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals():
            cap.release()
        parking_state.is_running = False
        print("🛑 Frame processor thread stopped")


# ======================
# Server Initialization
# ======================

def initialize_model():
    """Initialize YOLO model."""
    global model
    model_path = os.path.join(".", "yolov8n.pt")
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}. Downloading...")
    model = YOLO(model_path)
    print("✅ YOLO model loaded successfully")

@app.before_request
def before_request():
    """Initialize on first request."""
    if model is None:
        initialize_model()

if __name__ == '__main__':
    # Initialize model before starting server
    initialize_model()
    
    # Start background frame processor thread
    frame_processor_thread = threading.Thread(target=process_video_frames, daemon=True)
    frame_processor_thread.start()
    print("✅ Background frame processor thread started")
    
    # Run Flask app with threading support
    print("🚀 Starting Smart Parking API Server...")
    print("📊 API Documentation:")
    print("   - Health: GET /api/health")
    print("   - Status: GET /api/parking-status")
    print("   - Details: GET /api/parking-details")
    print("   - Video: GET /api/video-stream")
    print("   - Events: GET /api/events")
    print("   - Stats: GET /api/statistics")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
