import cv2
import sqlite3
import os
import asyncio
import io
import time
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager


import RPi.GPIO as GPIO

LOCK_PIN = 17      # Solenoid/main LED
BUTTON_PIN = 27    # Physical button
BUTTON_LED = 22    # Button LED

GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)
GPIO.setup(BUTTON_LED, GPIO.OUT)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.output(LOCK_PIN, GPIO.LOW)
GPIO.output(BUTTON_LED, GPIO.LOW)

# Directory where this script lives — HTML must be here too
BASE_DIR = Path(__file__).parent

import face_recognition
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
from anti_spoof import (
    AntiSpoofClassifier,
    DEFAULT_MODEL_PATH as DEFAULT_ANTI_SPOOF_MODEL_PATH,
    DEFAULT_THRESHOLD as DEFAULT_ANTI_SPOOF_THRESHOLD,
    evaluate_liveness,
    parse_bool_env,
)
from arclight_face_recognition import (
    DEFAULT_CV_SCALER,
    DEFAULT_ANTI_SPOOF_BBOX_EXPANSION,
    DEFAULT_DETECTION_MODEL,
    DEFAULT_ENCODING_MODEL,
    DEFAULT_TOLERANCE,
    delete_person as delete_person_encodings,
    detect_and_encode_faces,
    expand_bbox_to_frame,
    list_people,
    load_encodings,
    match_face,
    match_faces,
    save_encodings,
    scaled_face_location_to_bbox,
    upsert_person,
)


# ── CONFIG ───────────────────────────────────────────────────────────────
ENCODINGS_FILE = Path(os.getenv("ARCLIGHT_ENCODINGS_FILE", str(BASE_DIR / "encodings.pickle")))
DB_FILE      = Path(os.getenv("ARCLIGHT_ATTENDANCE_DB", str(BASE_DIR / "attendance.db")))
DATASET      = Path(os.getenv("ARCLIGHT_DATASET_DIR", str(BASE_DIR / "Faces4Arclight")))
FACE_RECOGNITION_TOLERANCE = float(os.getenv("ARCLIGHT_RECOGNITION_TOLERANCE", str(DEFAULT_TOLERANCE)))
FACE_CV_SCALER = int(os.getenv("ARCLIGHT_FACE_CV_SCALER", str(DEFAULT_CV_SCALER)))
FACE_DETECTION_MODEL = os.getenv("ARCLIGHT_FACE_DETECTION_MODEL", DEFAULT_DETECTION_MODEL)
FACE_ENCODING_MODEL = os.getenv("ARCLIGHT_FACE_ENCODING_MODEL", DEFAULT_ENCODING_MODEL)
COOLDOWN_SEC = 5
ENROLL_FRAMES = 30
ANTI_SPOOF_ENABLED = parse_bool_env("ARCLIGHT_ANTI_SPOOF_ENABLED", True)
ANTI_SPOOF_FAIL_CLOSED = parse_bool_env("ARCLIGHT_ANTI_SPOOF_FAIL_CLOSED", True)
ANTI_SPOOF_MODEL = os.getenv("ARCLIGHT_ANTI_SPOOF_MODEL", str(DEFAULT_ANTI_SPOOF_MODEL_PATH))
ANTI_SPOOF_THRESHOLD = float(os.getenv("ARCLIGHT_ANTI_SPOOF_THRESHOLD", str(DEFAULT_ANTI_SPOOF_THRESHOLD)))
ANTI_SPOOF_SHOW_SCORE = parse_bool_env("ARCLIGHT_ANTI_SPOOF_SHOW_SCORE", False)
ANTI_SPOOF_THREADS = int(os.getenv("ARCLIGHT_ANTI_SPOOF_THREADS", "2"))
ANTI_SPOOF_BBOX_EXPANSION = float(
    os.getenv("ARCLIGHT_ANTI_SPOOF_BBOX_EXPANSION", str(DEFAULT_ANTI_SPOOF_BBOX_EXPANSION))
)
ANTI_SPOOF_CROP_MARGIN = float(os.getenv("ARCLIGHT_ANTI_SPOOF_CROP_MARGIN", "0.2"))
# ─────────────────────────────────────────────────────────────────────────

# ── GLOBALS ───────────────────────────────────────────────────────────────
known_face_encodings = []
known_face_names = []
cap         = None
is_running  = False
last_seen   = {}
anti_spoof_classifier = None
anti_spoof_error = None
enroll_state = {
    "active": False,
    "name": None,
    "mode": None,        # 'add' or 'update'
    "embeddings": [],
    "count": 0,
    "done": False,
    "message": ""
}

def trigger_unlock():
    GPIO.output(LOCK_PIN, GPIO.HIGH)  # LED ON / Solenoid unlock
    time.sleep(1.5)                      # Stay on for 2 seconds
    GPIO.output(LOCK_PIN, GPIO.LOW)   # LED OFF / Solenoid lock
    

def monitor_button():
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:  # Button pressed
            GPIO.output(BUTTON_LED, GPIO.HIGH)   # LED on
            time.sleep(1.5)                         # ← CHANGE THIS NUMBER FOR TIME
            GPIO.output(BUTTON_LED, GPIO.LOW)    # Lock again
            time.sleep(0.3) 
        else:
            GPIO.output(BUTTON_LED, GPIO.LOW)    # LED off
        time.sleep(0.05)
# ── WebSocket connection manager ──────────────────────────────────────────
class ConnectionManager:

    
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

ws_manager = ConnectionManager()

# ── STARTUP / SHUTDOWN ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global known_face_encodings, known_face_names, anti_spoof_classifier, anti_spoof_error
    print("Loading face_recognition encodings...")
    known_face_encodings, known_face_names = load_encodings(ENCODINGS_FILE, missing_ok=True)
    people_count = len(set(known_face_names))
    print(f"  {people_count} people loaded from {len(known_face_names)} samples.")
    init_db()

    if ANTI_SPOOF_ENABLED:
        print("Loading anti-spoofing...")
        try:
            anti_spoof_classifier = AntiSpoofClassifier(
                model_path=ANTI_SPOOF_MODEL,
                threshold=ANTI_SPOOF_THRESHOLD,
                num_threads=ANTI_SPOOF_THREADS,
                crop_margin=ANTI_SPOOF_CROP_MARGIN,
            )
            anti_spoof_error = None
            print(
                "  Anti-spoofing loaded at "
                f"threshold {ANTI_SPOOF_THRESHOLD:.2f}, "
                f"bbox expansion {ANTI_SPOOF_BBOX_EXPANSION:.2f}, "
                f"crop margin {ANTI_SPOOF_CROP_MARGIN:.2f}."
            )
        except Exception as exc:
            anti_spoof_classifier = None
            anti_spoof_error = str(exc)
            mode = "fail-closed" if ANTI_SPOOF_FAIL_CLOSED else "fail-open"
            print(f"  Anti-spoofing unavailable ({mode}): {anti_spoof_error}")
    else:
        print("Anti-spoofing disabled by ARCLIGHT_ANTI_SPOOF_ENABLED.")
    
    # Start button monitor in background
    import threading
    threading.Thread(target=monitor_button, daemon=True).start()
    
    yield
    if cap:
        cap.release()
    GPIO.cleanup()

app = FastAPI(lifespan=lifespan)

# ── DATABASE HELPERS ──────────────────────────────────────────────────────
def save_known_faces():
    save_encodings(ENCODINGS_FILE, known_face_encodings, known_face_names)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT NOT NULL,
        timestamp  TEXT NOT NULL,
        date       TEXT NOT NULL,
        confidence REAL
    )''')
    conn.commit()
    conn.close()

# ── RECOGNITION ───────────────────────────────────────────────────────────
def recognize(embedding):
    return match_face(
        embedding,
        known_face_encodings,
        known_face_names,
        tolerance=FACE_RECOGNITION_TOLERANCE,
    )

def log_entry_sync(name, confidence):
    if name == "Unknown":
        return None

    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    current_ts = now.strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Get the most recent entry for this person today
    c.execute("""
        SELECT id, timestamp, confidence 
        FROM logs 
        WHERE name = ? AND date = ? 
        ORDER BY timestamp DESC LIMIT 1
    """, (name, today))
    row = c.fetchone()


    if not row:
        # First entry today
        c.execute(
            'INSERT INTO logs (name, timestamp, date, confidence) VALUES (?, ?, ?, ?)',
            (name, current_ts, today, float(confidence))
        )
        conn.commit()
        rowid = c.lastrowid
        conn.close()
        trigger_unlock()
        return {"event": "new", "name": name, "confidence": round(float(confidence), 3),
                "time": now.strftime('%H:%M:%S')}

    log_id, last_ts, last_conf = row
    last_time = datetime.strptime(last_ts, '%Y-%m-%d %H:%M:%S')

    if (now - last_time).total_seconds() > COOLDOWN_SEC:
        # Cooldown passed → new entry
        c.execute(
            'INSERT INTO logs (name, timestamp, date, confidence) VALUES (?, ?, ?, ?)',
            (name, current_ts, today, float(confidence))
        )
        conn.commit()
        conn.close()
        return {"event": "new", "name": name, "confidence": round(float(confidence), 3),
                "time": now.strftime('%H:%M:%S')}
    elif float(confidence) > last_conf:
        # Higher confidence → update existing entry
        c.execute(
            'UPDATE logs SET confidence = ?, timestamp = ? WHERE id = ?',
            (float(confidence), current_ts, log_id)
        )
        conn.commit()
        conn.close()
        return {"event": "update", "name": name, "confidence": round(float(confidence), 3),
                "time": now.strftime('%H:%M:%S')}

    conn.close()
    return None

# ── MJPEG STREAM ──────────────────────────────────────────────────────────
async def generate_frames():
    global cap, is_running, last_seen
    while is_running and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.03)
            continue

        face_locations, face_encodings = detect_and_encode_faces(
            frame,
            cv2,
            face_recognition,
            cv_scaler=FACE_CV_SCALER,
            detection_model=FACE_DETECTION_MODEL,
            encoding_model=FACE_ENCODING_MODEL,
        )
        matches = match_faces(
            face_encodings,
            known_face_encodings,
            known_face_names,
            tolerance=FACE_RECOGNITION_TOLERANCE,
        )

        if face_locations:
            known_detected = False
            for face_location, (name, score) in zip(face_locations, matches):
                x1, y1, x2, y2 = scaled_face_location_to_bbox(face_location, FACE_CV_SCALER)
                liveness = None
                display_name = name
                display_text = f"{display_name}  {score:.2f}"
                accepted_known = False

                if name != "Unknown":
                    anti_spoof_bbox = expand_bbox_to_frame(
                        [x1, y1, x2, y2],
                        frame.shape,
                        ANTI_SPOOF_BBOX_EXPANSION,
                    )
                    liveness = evaluate_liveness(
                        frame,
                        anti_spoof_bbox,
                        anti_spoof_classifier,
                        enabled=ANTI_SPOOF_ENABLED,
                        fail_closed=ANTI_SPOOF_FAIL_CLOSED,
                        unavailable_reason=anti_spoof_error,
                        show_score=ANTI_SPOOF_SHOW_SCORE,
                    )
                    if liveness.is_live:
                        accepted_known = True
                        if ANTI_SPOOF_SHOW_SCORE and liveness.spoof_score is not None:
                            display_name = f"{name} S:{liveness.spoof_score:.2f}"
                            display_text = f"{display_name}  {score:.2f}"
                        log_data = log_entry_sync(name, score)
                        if log_data:
                            asyncio.create_task(ws_manager.broadcast(log_data))
                    else:
                        display_name = (
                            "Anti-spoof unavailable"
                            if liveness.reason == "anti_spoof_unavailable"
                            else liveness.label
                        )
                        display_text = display_name

                if accepted_known:
                    known_detected = True
                    color = (0, 255, 136)
                elif liveness is not None and not liveness.is_live:
                    color = (0, 0, 255)
                else:
                    color = (0, 80, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, display_text,
                            (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if known_detected:
                GPIO.output(LOCK_PIN, GPIO.HIGH)  # Unlock
            else:
                GPIO.output(LOCK_PIN, GPIO.LOW)   # Lock

        else:
            GPIO.output(LOCK_PIN, GPIO.LOW)  # No face — Lock

        # Also handle enrollment capture
        if enroll_state["active"] and not enroll_state["done"]:
            _enroll_tick(frame)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        await asyncio.sleep(0.03)

def _enroll_tick(frame):
    """Accumulate embeddings during enrollment (called from frame loop)."""
    if enroll_state["count"] >= ENROLL_FRAMES:
        return
    face_locations, face_encodings = detect_and_encode_faces(
        frame,
        cv2,
        face_recognition,
        cv_scaler=FACE_CV_SCALER,
        detection_model=FACE_DETECTION_MODEL,
        encoding_model=FACE_ENCODING_MODEL,
    )
    if face_locations and face_encodings:
        bbox = scaled_face_location_to_bbox(face_locations[0], FACE_CV_SCALER)
        bbox = expand_bbox_to_frame(bbox, frame.shape, ANTI_SPOOF_BBOX_EXPANSION)
        liveness = evaluate_liveness(
            frame,
            bbox,
            anti_spoof_classifier,
            enabled=ANTI_SPOOF_ENABLED,
            fail_closed=ANTI_SPOOF_FAIL_CLOSED,
            unavailable_reason=anti_spoof_error,
            show_score=ANTI_SPOOF_SHOW_SCORE,
        )
        if not liveness.is_live:
            enroll_state["message"] = liveness.label
            return

        enroll_state["embeddings"].append(face_encodings[0])
        enroll_state["count"] += 1
        enroll_state["message"] = f"Captured {enroll_state['count']}/{ENROLL_FRAMES}"
        if enroll_state["count"] >= ENROLL_FRAMES:
            _finish_enrollment()

def _finish_enrollment():
    global known_face_encodings, known_face_names
    embs = enroll_state["embeddings"]
    if len(embs) >= 10:
        known_face_encodings, known_face_names = upsert_person(
            known_face_encodings,
            known_face_names,
            enroll_state["name"],
            embs,
        )
        save_known_faces()
        enroll_state["done"] = True          # ← moved up, before folder ops
        action = "updated" if enroll_state["mode"] == "update" else "added"
        enroll_state["message"] = f"✓ '{enroll_state['name']}' {action} successfully!"
        folder = DATASET / enroll_state["name"]
        folder.mkdir(parents=True, exist_ok=True)
    else:
        enroll_state["done"] = True
        enroll_state["message"] = "✗ Not enough faces detected. Try again."

# ══════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "arclight_ui.html"
    if not html_path.exists():
        return HTMLResponse(
            f"<pre>ERROR: arclight_ui.html not found at {html_path}\n"
            "Put both files in the same folder.</pre>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

# ── Video stream ──────────────────────────────────────────────────────────
@app.get("/video_feed")
async def video_feed():
    if not is_running:
        raise HTTPException(status_code=503, detail="Camera not running")
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ── Camera control ────────────────────────────────────────────────────────
@app.post("/api/camera/start")
async def start_camera():
    global cap, is_running, last_seen
    if is_running:
        return {"status": "already running"}
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open camera")
    is_running = True
    last_seen  = {}
    return {"status": "started"}

@app.post("/api/camera/stop")
async def stop_camera():
    global cap, is_running
    is_running = False
    if cap:
        cap.release()
        cap = None
    return {"status": "stopped"}

@app.get("/api/camera/status")
async def camera_status():
    return {
        "running": is_running,
        "anti_spoof_enabled": ANTI_SPOOF_ENABLED,
        "anti_spoof_ready": anti_spoof_classifier is not None,
        "anti_spoof_fail_closed": ANTI_SPOOF_FAIL_CLOSED,
        "anti_spoof_error": anti_spoof_error,
        "anti_spoof_bbox_expansion": ANTI_SPOOF_BBOX_EXPANSION,
        "anti_spoof_crop_margin": ANTI_SPOOF_CROP_MARGIN,
    }

# ── Faces CRUD ────────────────────────────────────────────────────────────
@app.get("/api/faces")
async def list_faces():
    return list_people(known_face_encodings, known_face_names)

class EnrollRequest(BaseModel):
    name: str
    mode: str  # 'add' or 'update'

@app.post("/api/faces/enroll/start")
async def enroll_start(req: EnrollRequest):
    global cap, is_running
    name = req.name.strip()
    if req.mode == "add" and name in set(known_face_names):
        raise HTTPException(400, f"'{name}' already exists. Use update mode.")
    if not is_running:
        cap = cv2.VideoCapture(0)
        is_running = True
    enroll_state.update({
        "active": True, "name": name, "mode": req.mode,
        "embeddings": [], "count": 0, "done": False,
        "message": f"Ready — position face for '{name}'"
    })
    return {"status": "started", "name": name}

@app.get("/api/faces/enroll/status")
async def enroll_status():
    return {
        "active":   enroll_state["active"],
        "count":    enroll_state["count"],
        "total":    ENROLL_FRAMES,
        "done":     enroll_state["done"],
        "message":  enroll_state["message"],
    }

@app.post("/api/faces/enroll/cancel")
async def enroll_cancel():
    enroll_state.update({"active": False, "done": False, "count": 0,
                          "embeddings": [], "message": ""})
    return {"status": "cancelled"}

@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names, removed = delete_person_encodings(
        known_face_encodings,
        known_face_names,
        name,
    )
    if not removed:
        raise HTTPException(404, f"'{name}' not found")
    save_known_faces()
    folder = DATASET / name
    if folder.exists():
        import shutil
        shutil.rmtree(folder)
    return {"status": "deleted", "name": name}

# ── Logs ──────────────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_logs():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, timestamp, confidence FROM logs ORDER BY timestamp DESC LIMIT 200")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "time": r[2], "confidence": round(r[3], 3)}
            for r in rows]


@app.get("/api/logs/recent")
async def get_recent_logs():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT id, name, timestamp, confidence 
        FROM logs 
        ORDER BY timestamp DESC LIMIT 100
    """)
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "name": r[1],
            "time": r[2].split()[1][:8] if isinstance(r[2], str) and ' ' in r[2] else str(r[2]),
            "confidence": round(float(r[3]), 3)
        }
        for r in rows
    ]


@app.get("/api/logs/export")
async def export_logs():
    """Export logs to Excel and save it to savedlogs folder"""
    import os
    from pathlib import Path

    
    # Define the save directory
    SAVE_DIR = BASE_DIR / "savedlogs" 
    SAVE_DIR.mkdir(parents=True, exist_ok=True)   # Create folder if it doesn't exist

    # Filename with date
    filename = f"attendance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    save_path = SAVE_DIR / filename

    # Get data from database
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM logs ORDER BY timestamp DESC", conn)
    conn.close()

    # Save to the specific folder
    df.to_excel(save_path, index=False)

    print(f"✅ Logs exported to: {save_path}")   # For debugging in terminal

    # Return for browser download
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

@app.delete("/api/logs")
async def clear_logs():
    """Clear all attendance logs"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM logs")
        conn.commit()
        conn.close()

        # Also clear the in-memory last_seen to be safe
        global last_seen
        last_seen = {}

        print("✅ All logs cleared successfully")
        return {"status": "cleared", "message": "All attendance logs have been deleted."}

    except Exception as e:
        print(f"❌ Error clearing logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")

# ── WebSocket ─────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            try:
                # Wait for ping with timeout; disconnect if client gone
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # No ping received — send a keepalive pong back
                try:
                    await ws.send_json({"event": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(ws)

# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("arclight_server:app", host="0.0.0.0", port=8000, reload=False)
