# -*- coding: utf-8 -*-
import csv
import os
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time as _time
try:
    import winsound  # Windows-only simple beep API
except Exception:
    winsound = None
try:
    import torch
except Exception:
    torch = None


# -----------------------------
# Step 3: Parking Capacity Definition
# -----------------------------
TOTAL_PARKING_SLOTS = 10
LOG_INTERVAL_SECONDS = 300  # 5 minutes for historical storage (step 5)
HISTORY_CSV_PATH = os.path.join("outputs", "parking_history.csv")
EVENTS_CSV_PATH = os.path.join("outputs", "parking_events.csv")


COUNT_LINE_Y = None  # will be set dynamically based on frame height

# Real-time counters (vehicles that have entered = occupied slots)
two_wheeler_count = 0
four_wheeler_count = 0

# Track vehicles that attempted entry while lot was full so we can resume counting
# when slots free up. Maps track_id -> label
blocked_entries: dict[int, str] = {}

# Alarm controls
alarm_thread: threading.Thread | None = None
alarm_stop_event = threading.Event()

# Performance tuning: process smaller frames and use GPU if available
PROCESS_IMG_WIDTH = 960  # width to resize frames for processing/display
INFERENCE_IMGSZ = 640  # pass to model to reduce compute

# CSV loggers (open once to reduce I/O overhead)
HISTORY_FH = None
EVENTS_FH = None
HISTORY_WRITER = None
EVENTS_WRITER = None
CSV_LOCK = threading.Lock()
LOGGER_INITIALIZED = False

def init_loggers():
    """Open CSV files once and write headers if needed."""
    global HISTORY_FH, EVENTS_FH, HISTORY_WRITER, EVENTS_WRITER, LOGGER_INITIALIZED
    if LOGGER_INITIALIZED:
        return
    os.makedirs(os.path.dirname(HISTORY_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EVENTS_CSV_PATH), exist_ok=True)

    # History file
    history_exists = os.path.isfile(HISTORY_CSV_PATH) and os.path.getsize(HISTORY_CSV_PATH) > 0
    HISTORY_FH = open(HISTORY_CSV_PATH, mode="a", newline="", encoding="utf-8")
    HISTORY_WRITER = csv.writer(HISTORY_FH)
    if not history_exists:
        HISTORY_WRITER.writerow([
            "Timestamp",
            "Hour",
            "Weekday",
            "TotalSlots",
            "TwoWheelers",
            "FourWheelers",
            "Occupied",
            "Available",
            "Status",
            "Pred_Time",
            "Pred_Level",
        ])

    # Events file
    events_exists = os.path.isfile(EVENTS_CSV_PATH) and os.path.getsize(EVENTS_CSV_PATH) > 0
    EVENTS_FH = open(EVENTS_CSV_PATH, mode="a", newline="", encoding="utf-8")
    EVENTS_WRITER = csv.writer(EVENTS_FH)
    if not events_exists:
        EVENTS_WRITER.writerow([
            "Timestamp",
            "TrackID",
            "Label",
            "Event",
            "TwoWheelers",
            "FourWheelers",
            "Occupied",
            "Available",
        ])

    LOGGER_INITIALIZED = True


def close_loggers():
    global HISTORY_FH, EVENTS_FH, LOGGER_INITIALIZED
    try:
        if HISTORY_FH is not None:
            HISTORY_FH.flush()
            HISTORY_FH.close()
    except Exception:
        pass
    try:
        if EVENTS_FH is not None:
            EVENTS_FH.flush()
            EVENTS_FH.close()
    except Exception:
        pass
    LOGGER_INITIALIZED = False


def _alarm_worker(stop_event: threading.Event):
    """Background worker that beeps until stop_event is set.

    Uses winsound.Beep on Windows. If winsound is not available, falls back to
    printing a message every second.
    """
    freq = 1200  # Hz
    dur = 400  # ms per beep
    while not stop_event.is_set():
        if winsound is not None:
            try:
                winsound.Beep(freq, dur)
            except RuntimeError:
                # sometimes Beep can raise if the system can't play sound
                pass
        else:
            print("[ALARM] Parking full!")
        # sleep a bit but wake early if stop_event is set
        stop_event.wait(0.8)


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
# Dictionary to hold tracking history per vehicle ID.
# For each track_id we store:
# - label: class name (car / motorcycle)
# - last_cy: last vertical center position
# - last_side: "above" or "below" the counting line to detect crossings
vehicle_history = {}

# Classes to detect - simplified to just car and motorcycle
FOUR_WHEELERS = ["car"]
TWO_WHEELERS = ["motorcycle"]


def compute_parking_status(occupied: int):
    """
    Step 3 & 4:
    - Calculate available slots
    - Derive qualitative status and color code.
    """
    occupied = max(0, min(occupied, TOTAL_PARKING_SLOTS))
    available = TOTAL_PARKING_SLOTS - occupied

    if TOTAL_PARKING_SLOTS == 0:
        ratio = 1.0
    else:
        ratio = occupied / TOTAL_PARKING_SLOTS

    # Map to UI status for Step 9
    if ratio < 0.6:
        status = "Available"
        color = (0, 255, 0)  # Green
    elif ratio < 0.9:
        status = "Likely available"
        color = (0, 255, 255)  # Yellow
    else:
        status = "Full"
        color = (0, 0, 255)  # Red

    return occupied, available, status, color


def predict_future_availability(
    occupied: int, available: int, now: datetime | None = None
):
    """
    Step 7: Simple rule-based parking availability prediction.
    
    Very simple logic based on available slots:
    - If available > 20: High availability
    - If available > 0: Likely available
    - If available = 0: Low availability (Full)
    """
    if now is None:
        now = datetime.now()

    if available > 20:
        level = "High availability"
        indicator = "🟢"
    elif available > 0:
        level = "Likely available"
        indicator = "🟡"
    else:
        level = "Low availability"
        indicator = "🔴"

    return {
        "time": now.strftime("%H:%M"),
        "level": level,
        "indicator": indicator,
    }


def recommend_parking_slot(prediction: dict, available: int):
    """
    Step 8: Decision making based on predicted level and current availability.
    """
    level = prediction.get("level", "")

    if available <= 0 or "Low" in level:
        return "Recommendation: Use alternate parking – current lot is or will be full."
    if "High" in level:
        return "Recommendation: Use main parking – high chance of free slots."
    return "Recommendation: Parking may be partially available – plan a short wait."


def estimate_time_to_free(required_slots: int = 1, lookback_hours: float | None = 72.0):
    """
    Estimate time until at least `required_slots` are free based on historical
    `HISTORY_CSV_PATH` data.

    Method:
    - Read historical rows (Timestamp, Available).
    - For each moment where Available == 0, find the next timestamp where
      Available >= required_slots and compute the delta.
    - Return the median delta as the expected wait. If no examples are found,
      return None.

    lookback_hours: if set, only consider rows within this many hours from now.
    """
    rows = []

    # Helper to read a CSV file and extract (timestamp, available)
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

    # Read both history and event-level CSVs (events often have higher resolution)
    rows.extend(_read_rows_from(HISTORY_CSV_PATH))
    rows.extend(_read_rows_from(EVENTS_CSV_PATH))

    if not rows:
        return None

    if not rows:
        return None

    # Optionally filter by lookback window
    if lookback_hours is not None:
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        rows = [r for r in rows if r[0] >= cutoff]
        if not rows:
            return None

    # Sort by timestamp
    rows.sort(key=lambda x: x[0])

    deltas = []
    n = len(rows)
    for i, (t_i, avail_i) in enumerate(rows):
        if avail_i <= 0:
            # find next j where Available >= required_slots
            for j in range(i + 1, n):
                t_j, avail_j = rows[j]
                if avail_j >= required_slots:
                    deltas.append((t_j - t_i).total_seconds())
                    break

    if not deltas:
        return None

    # median
    deltas.sort()
    mid = len(deltas) // 2
    if len(deltas) % 2 == 1:
        median_s = deltas[mid]
    else:
        median_s = 0.5 * (deltas[mid - 1] + deltas[mid])

    return median_s


def log_parking_snapshot(two_wheels: int, four_wheels: int, status: str, prediction: dict):
    """
    Improved historical logging:
    - Recompute occupied = two_wheels + four_wheels to avoid stale/incorrect values.
    - Record total slots and per-class counters so each row is self-describing.
    - Add prediction/time summary for easier downstream analysis.
    """
    # Use persistent file handle to reduce I/O overhead
    try:
        init_loggers()
        now = datetime.now()
        timestamp = now.isoformat(timespec="seconds")
        hour = now.hour
        weekday = now.strftime("%A")

        occupied = two_wheels + four_wheels
        occupied = max(0, min(occupied, TOTAL_PARKING_SLOTS))
        available = max(0, TOTAL_PARKING_SLOTS - occupied)

        with CSV_LOCK:
            HISTORY_WRITER.writerow([
                timestamp,
                hour,
                weekday,
                TOTAL_PARKING_SLOTS,
                two_wheels,
                four_wheels,
                occupied,
                available,
                status,
                prediction.get("time", ""),
                prediction.get("level", ""),
            ])
            HISTORY_FH.flush()
    except Exception:
        # Best-effort: don't raise to avoid breaking main loop
        pass


def log_parking_event(track_id: int | None, label: str, event: str, two_wheels: int, four_wheels: int):
    """
    Log every entry/exit event with per-class counters and recomputed occupancy.
    Columns: Timestamp,TrackID,Label,Event,TwoWheelers,FourWheelers,Occupied,Available
    """
    try:
        init_loggers()
        now = datetime.now()
        timestamp = now.isoformat(timespec="seconds")

        occupied = two_wheels + four_wheels
        occupied = max(0, min(occupied, TOTAL_PARKING_SLOTS))
        available = max(0, TOTAL_PARKING_SLOTS - occupied)

        with CSV_LOCK:
            EVENTS_WRITER.writerow([
                timestamp,
                track_id if track_id is not None else "",
                label,
                event,
                two_wheels,
                four_wheels,
                occupied,
                available,
            ])
            EVENTS_FH.flush()
    except Exception:
        pass


def resume_blocked_entries():
    """
    Attempt to resume counting of previously-blocked entries when slots become available.

    Rules:
    - Iterate blocked_entries in insertion order and assign available slots to those
      whose track is still present and still on the 'below' side (i.e., waiting inside).
    - For each resumed entry, increment appropriate counter, log 'entry_resumed', and remove
      it from blocked_entries.
    - If a blocked track is no longer present (not in vehicle_history) or isn't below the line,
      drop it from blocked_entries.
    """
    global two_wheeler_count, four_wheeler_count
    available_now = TOTAL_PARKING_SLOTS - (two_wheeler_count + four_wheeler_count)
    if available_now <= 0 or not blocked_entries:
        return

    # Iterate over a copy since we'll mutate the dict
    for tid in list(blocked_entries.keys()):
        if available_now <= 0:
            break
        label = blocked_entries.get(tid)
        # If vehicle is still tracked and below the line, resume
        vh = vehicle_history.get(tid)
        if vh is None or vh.get("last_side") != "below":
            # vehicle left or moved away; drop from blocked list
            blocked_entries.pop(tid, None)
            continue

        # Resume counting
        if label in FOUR_WHEELERS:
            four_wheeler_count += 1
        elif label in TWO_WHEELERS:
            two_wheeler_count += 1
        else:
            # Unknown class — drop and continue
            blocked_entries.pop(tid, None)
            continue

        # Log resumed entry and remove from blocked list
        log_parking_event(tid, label, "entry_resumed", two_wheeler_count, four_wheeler_count)
        blocked_entries.pop(tid, None)
        available_now -= 1


def draw_parking_dashboard(
    frame,
    two_w_count: int,
    four_w_count: int,
    occupied: int,
    available: int,
    status: str,
    status_color,
    prediction: dict,
    recommendation: str,
):
    # More attractive dashboard with circular occupancy indicator,
    # progress bar, icons and a pulsing full badge for attention.
    h, w, _ = frame.shape

    # Panel
    pad = 14
    panel_w = min(680, w - 2 * pad)
    panel_h = 300
    panel_x0 = (w - panel_w) - pad
    panel_y0 = pad
    panel_x1 = panel_x0 + panel_w
    panel_y1 = panel_y0 + panel_h

    # Dark translucent panel background
    panel = frame.copy()
    cv2.rectangle(panel, (panel_x0, panel_y0), (panel_x1, panel_y1), (30, 34, 40), -1)
    cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)

    # Decorative top bar
    top_bar_h = 44
    cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x1, panel_y0 + top_bar_h), (24, 130, 196), -1)
    title = "SMART PARKING — LIVE"
    cv2.putText(frame, title, (panel_x0 + 18, panel_y0 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Circular occupancy indicator (left side of panel)
    circle_cx = panel_x0 + 120
    circle_cy = panel_y0 + panel_h // 2
    radius = 70
    thickness = 14
    ratio = 0.0 if TOTAL_PARKING_SLOTS == 0 else float(occupied) / TOTAL_PARKING_SLOTS
    ratio = max(0.0, min(1.0, ratio))

    # background ring
    cv2.ellipse(frame, (circle_cx, circle_cy), (radius, radius), -90, 0, 360, (50, 50, 60), thickness)
    # foreground arc representing occupancy
    sweep = int(360 * ratio)
    # color gradient (green -> yellow -> red)
    rcol = int(200 * ratio + 55)
    gcol = int(200 * (1 - ratio) + 55)
    arc_color = (0 if ratio < 0.6 else (0 if ratio < 0.9 else 30), gcol if ratio < 0.6 else 200, rcol)
    # draw arc from top (-90 degrees)
    if sweep > 0:
        cv2.ellipse(frame, (circle_cx, circle_cy), (radius, radius), -90, -90, -90 + sweep, arc_color, thickness)

    # Center percent
    percent = int(round(ratio * 100))
    cv2.putText(frame, f"{percent}%", (circle_cx - 36, circle_cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, "OCCUPANCY", (circle_cx - 64, circle_cy + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Right side details
    rx = circle_cx + radius + 30
    ry = panel_y0 + 64

    # Vehicle counts with icons
    # Car icon (simple rectangle + wheels)
    cv2.putText(frame, "Cars", (rx, ry), cv2.FONT_HERSHEY_DUPLEX, 0.5, (220, 220, 220), 1)
    cv2.rectangle(frame, (rx + 70, ry - 12), (rx + 120, ry + 6), (255, 200, 60), -1)
    cv2.circle(frame, (rx + 82, ry + 10), 5, (30, 30, 30), -1)
    cv2.circle(frame, (rx + 108, ry + 10), 5, (30, 30, 30), -1)
    cv2.putText(frame, f"{four_w_count}", (rx + 132, ry), cv2.FONT_HERSHEY_DUPLEX, 0.6, (230, 230, 230), 1)
    ry += 34

    # Bike icon (triangle + wheel)
    cv2.putText(frame, "Bikes", (rx, ry), cv2.FONT_HERSHEY_DUPLEX, 0.5, (220, 220, 220), 1)
    pts = np.array([[rx + 78, ry - 10], [rx + 98, ry - 10], [rx + 88, ry + 6]], np.int32)
    cv2.fillConvexPoly(frame, pts, (120, 200, 255))
    cv2.circle(frame, (rx + 82, ry + 10), 4, (30, 30, 30), -1)
    cv2.circle(frame, (rx + 98, ry + 10), 4, (30, 30, 30), -1)
    cv2.putText(frame, f"{two_w_count}", (rx + 132, ry), cv2.FONT_HERSHEY_DUPLEX, 0.6, (230, 230, 230), 1)
    ry += 36

    # Available big number (occupied removed per request)
    cv2.putText(frame, "Available:", (rx, ry), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"{available}", (rx + 120, ry), cv2.FONT_HERSHEY_DUPLEX, 0.9, (120, 220, 140) if available > 0 else (120, 120, 200), 2)
    ry += 36

    # Progress bar for quick visual cue
    bar_x0 = rx
    bar_x1 = panel_x1 - 20
    bar_y = panel_y1 - 48
    cv2.rectangle(frame, (bar_x0, bar_y - 12), (bar_x1, bar_y + 12), (50, 50, 60), -1)
    fill_w = int((bar_x1 - bar_x0) * ratio)
    cv2.rectangle(frame, (bar_x0, bar_y - 12), (bar_x0 + fill_w, bar_y + 12), arc_color, -1)
    cv2.putText(frame, f"{occupied}/{TOTAL_PARKING_SLOTS} slots used", (bar_x0, bar_y - 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200,200,200), 1)

    # Left-side stats: Two/Four wheelers
    ly = panel_y0 + 66
    cv2.putText(frame, "Two wheelers", (panel_x0 + 18, ly), cv2.FONT_HERSHEY_DUPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(frame, f"{two_w_count}", (panel_x0 + 170, ly), cv2.FONT_HERSHEY_DUPLEX, 0.7, (120, 200, 255), 2)
    ly += 26
    cv2.putText(frame, "Four wheelers", (panel_x0 + 18, ly), cv2.FONT_HERSHEY_DUPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(frame, f"{four_w_count}", (panel_x0 + 170, ly), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 200, 60), 2)
    ly += 30

    # Prediction panel small
    pry = ly + 6
    cv2.putText(frame, "Prediction", (panel_x0 + 18, pry), cv2.FONT_HERSHEY_DUPLEX, 0.5, (230, 230, 230), 1)
    pry += 22
    pred_level = prediction.get("level", "")
    cv2.putText(frame, f"{pred_level}", (panel_x0 + 18, pry), cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 255), 1)
    pry += 22
    free5_at = prediction.get("free_5_at")
    free5_in = prediction.get("free_5_in_minutes")
    if free5_at:
        cv2.putText(frame, f"Est. free for 5 cars: {free5_at} ({free5_in}m)", (panel_x0 + 18, pry), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1)
        pry += 20

    # Pulsing full badge when full
    if available <= 0:
        pulse = (int(time.time() * 2) % 2)  # 0/1 toggle
        badge_col = (0, 40, 200) if pulse else (20, 20, 160)
        cv2.rectangle(frame, (panel_x0 + 18, panel_y0 + 6), (panel_x0 + 220, panel_y0 + 36), badge_col, -1)
        cv2.putText(frame, "LOT FULL", (panel_x0 + 28, panel_y0 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 2)

    # Recommendation at bottom right of panel
    rec_x = rx
    rec_y = panel_y1 - 24
    cv2.putText(frame, recommendation.replace("Recommendation:", ""), (rec_x, rec_y), cv2.FONT_HERSHEY_DUPLEX, 0.45, (220,220,220), 1)


def send_to_cisco_smart_city(payload: dict):
    """
    Step 10: Cisco Smart City Integration (conceptual).

    This is a placeholder to show how the system could push data to a Cisco IoT /
    smart city platform over MQTT, HTTP, or other protocols.

    For now, we don't perform actual network operations here; this function can
    be extended with Cisco-specific SDKs or APIs.
    """
    # Example (conceptual only):
    # cisco_iot_client.publish("smart-city/parking", json.dumps(payload))
    _ = payload  # avoid unused variable warning


def main():
    global two_wheeler_count, four_wheeler_count
    # Detect GPU availability (if torch is available)
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        device = "cuda:0"
        print("⚡ Using GPU for inference (cuda)")
    else:
        device = "cpu"
        print("ℹ️  Using CPU for inference; performance may be slower")

    model = YOLO("yolov8n.pt", device=device)

    # Initialize CSV loggers once to avoid repeated file opens
    init_loggers()

    # Use a relative path to the input video inside the project.
    # Change the filename here if you want to use a different input video.
    video_path = os.path.join("inputs", "entrance_video.mp4")
    print(f"🎥 Using video source: {os.path.abspath(video_path)}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Could not open video source. Please check the path in main.py.")
        return

    print("✅ Video source opened successfully.")
    print("▶ Smart Parking Dashboard is starting...")
    print(f"   Total parking slots configured: {TOTAL_PARKING_SLOTS}")
    print("   A GUI window will appear. Press 'q' in the window to stop.\n")

    last_log_time = 0.0
    first_frame_printed = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("✅ Video complete.")
            break

        # Resize frame for processing/display to reduce inference and drawing cost
        try:
            h0, w0 = frame.shape[:2]
            if w0 != PROCESS_IMG_WIDTH:
                scale = PROCESS_IMG_WIDTH / float(w0)
                new_w = PROCESS_IMG_WIDTH
                new_h = max(64, int(h0 * scale))
                frame = cv2.resize(frame, (new_w, new_h))
        except Exception:
            # If resize fails, continue with original frame
            pass

        # Set counting line position dynamically on first frame (around 60% height)
        global COUNT_LINE_Y
        if COUNT_LINE_Y is None:
            frame_height = frame.shape[0]
            COUNT_LINE_Y = int(frame_height * 0.6)
            print(f"📏 COUNT_LINE_Y set to {COUNT_LINE_Y} (60% of frame height).")

        if not first_frame_printed:
            print("✅ First frame processed. Real-time detection and parking status are running.")
            first_frame_printed = True

        # Run tracking with a smaller model image size to reduce compute
        try:
            results = model.track(
                source=frame,
                imgsz=INFERENCE_IMGSZ,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                device=device,
            )[0]
        except TypeError:
            # If model.track doesn't accept imgsz/device here, call without them
            results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else None
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cy = int((y1 + y2) / 2)

                label = model.names[cls_id]

                if label not in FOUR_WHEELERS + TWO_WHEELERS or track_id is None:
                    continue

                # Initialize if new track_id
                if track_id not in vehicle_history:
                    initial_side = "above" if cy < COUNT_LINE_Y else "below"
                    vehicle_history[track_id] = {
                        "label": label,
                        "last_cy": cy,
                        "last_side": initial_side,
                    }

                # Check if it crossed the line in either direction
                last_cy = vehicle_history[track_id]["last_cy"]
                last_side = vehicle_history[track_id]["last_side"]
                current_side = "above" if cy < COUNT_LINE_Y else "below"

                # Entry: moved from above -> below the line
                if last_side == "above" and current_side == "below":
                    # Before counting an entry, check if parking is already full
                    current_occupied = two_wheeler_count + four_wheeler_count
                    if current_occupied >= TOTAL_PARKING_SLOTS:
                        # Lot full: do not increment counters. Log a blocked entry event.
                        log_parking_event(track_id, label, "entry_blocked", two_wheeler_count, four_wheeler_count)
                        # keep track so we can resume counting when slots free up
                        if track_id is not None:
                            blocked_entries[track_id] = label
                    else:
                        if label in FOUR_WHEELERS:
                            four_wheeler_count += 1
                            # Log event: four-wheeler entered
                            log_parking_event(track_id, label, "entry", two_wheeler_count, four_wheeler_count)
                        elif label in TWO_WHEELERS:
                            two_wheeler_count += 1
                            # Log event: two-wheeler entered
                            log_parking_event(track_id, label, "entry", two_wheeler_count, four_wheeler_count)

                # Exit: moved from below -> above the line
                elif last_side == "below" and current_side == "above":
                    if label in FOUR_WHEELERS and four_wheeler_count > 0:
                        four_wheeler_count -= 1
                        # Log event: four-wheeler exited
                        log_parking_event(track_id, label, "exit", two_wheeler_count, four_wheeler_count)
                    elif label in TWO_WHEELERS and two_wheeler_count > 0:
                        two_wheeler_count -= 1
                        # Log event: two-wheeler exited
                        log_parking_event(track_id, label, "exit", two_wheeler_count, four_wheeler_count)

                # Update last y position and side
                vehicle_history[track_id]["last_cy"] = cy
                vehicle_history[track_id]["last_side"] = current_side

                # Draw box and label
                color = (0, 255, 0) if label in FOUR_WHEELERS else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        # After processing detections, attempt to resume any previously-blocked entries
        resume_blocked_entries()

        # Compute real-time parking status
        occupied = two_wheeler_count + four_wheeler_count
        occupied, available, status, status_color = compute_parking_status(occupied)

        # Simple rule-based prediction (now depends on current occupancy)
        prediction = predict_future_availability(occupied, available)
        recommendation = recommend_parking_slot(prediction, available)

        # Alarm: start when full, stop when slots available
        if occupied >= TOTAL_PARKING_SLOTS:
            start_alarm()
            # estimate time until next free slot using history (1 slot)
            est_s_1 = estimate_time_to_free(required_slots=1)
            if est_s_1 is not None:
                minutes = int(round(est_s_1 / 60.0))
                free_at = (datetime.now() + timedelta(seconds=est_s_1)).strftime("%H:%M")
                prediction["free_in_minutes"] = minutes
                prediction["free_at"] = free_at
            else:
                prediction["free_in_minutes"] = None
                prediction["free_at"] = None

            # estimate time until 5 slots are free using history
            est_s_5 = estimate_time_to_free(required_slots=5)
            if est_s_5 is not None:
                minutes5 = int(round(est_s_5 / 60.0))
                free5_at = (datetime.now() + timedelta(seconds=est_s_5)).strftime("%H:%M")
                prediction["free_5_in_minutes"] = minutes5
                prediction["free_5_at"] = free5_at
            else:
                prediction["free_5_in_minutes"] = None
                prediction["free_5_at"] = None
        else:
            stop_alarm()
            prediction.pop("free_in_minutes", None)
            prediction.pop("free_at", None)
            prediction.pop("free_5_in_minutes", None)
            prediction.pop("free_5_at", None)

        # Step 5: Log historical data at fixed intervals
        now_ts = time.time()
        if now_ts - last_log_time >= LOG_INTERVAL_SECONDS:
            # Pass per-class counters; logger will recompute occupied/available
            log_parking_snapshot(two_wheeler_count, four_wheeler_count, status, prediction)
            last_log_time = now_ts

            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{log_time}] 📊 Logged snapshot to '{HISTORY_CSV_PATH}': "
                f"occupied={occupied}, available={available}, status={status}"
            )

            # Conceptual Cisco integration: send a summarized payload
            payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "total_slots": TOTAL_PARKING_SLOTS,
                "occupied": occupied,
                "available": available,
                "status": status,
                "prediction": prediction,
            }
            send_to_cisco_smart_city(payload)

        # Draw the counting line so you can see where vehicles are counted
        cv2.line(
            frame,
            (0, COUNT_LINE_Y),
            (frame.shape[1], COUNT_LINE_Y),
            (0, 255, 255),
            2,
        )

        # Step 9: UI overlay – smart parking dashboard
        draw_parking_dashboard(
            frame,
            two_wheeler_count,
            four_wheeler_count,
            occupied,
            available,
            status,
            status_color,
            prediction,
            recommendation,
        )

        cv2.imshow("Smart Parking Dashboard", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    # Close persistent file handles
    close_loggers()


if __name__ == "__main__":
    main()
