# app.py
# Flask server running YOLO/YOLOE and Monocular Depth estimation on Mac

import os
import sys
import time
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template

# Resolve paths
DEMO_DIR = Path(__file__).resolve().parent
ROOT_DIR = DEMO_DIR.parent
MODEL_DIR = ROOT_DIR / "models"
SAVE_DIR = ROOT_DIR / "_outputs" / "demo_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Add demo directory to path
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

# Import local depth estimator
from depth_estimator import DepthEstimator

# Import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics package not found. Run setup_env.sh first.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__, 
            template_folder=str(DEMO_DIR / "templates"),
            static_folder=str(DEMO_DIR / "static"))

# ------------------------------------------------------------
# Shared State & Configuration
# ------------------------------------------------------------
state_lock = threading.Lock()
current_raw_frame = None
current_processed_frame = None
current_depth_frame = None
current_seg_frame = None

TARGET_FPS = 12
DEPTH_THRESHOLD = 0.5   # Meters

active_zones = {}       # zone_id -> zone dict
tracker_states = {}     # zone_id -> {track_id -> {enter_time, notified}}
alert_events = []       # List of triggered alerts

# --- Scanning State ---
is_scanning = False
scan_end_time = 0
auto_zones_buffer = {} # class_name -> list of polygons

# Load Target Objects for scanning
TARGET_OBJECTS = set()
TARGET_LIST_PATH = DEMO_DIR / "indoor_large_objects.txt"
if TARGET_LIST_PATH.exists():
    try:
        with open(TARGET_LIST_PATH, "r", encoding="utf-8") as f:
            for line in f:
                obj = line.strip().lower()
                if obj:
                    TARGET_OBJECTS.add(obj)
    except Exception as e:
        print(f"❌ Error loading target objects: {e}")

# Telemetry data for SSE
latest_telemetry = {
    "fps": 0.0,
    "active_zones_count": 0,
    "current_tracks_count": 0,
    "is_scanning": False,
    "tracks": []
}

sse_queues = []

# Load local zones config
ZONES_CONFIG_PATH = DEMO_DIR / "zones_config.json"
if ZONES_CONFIG_PATH.exists():
    try:
        with open(ZONES_CONFIG_PATH, "r", encoding="utf-8") as f:
            active_zones = json.load(f)
    except Exception: pass
else:
    active_zones = {
        "Zone_A1": {
            "polygon": [[100, 100], [540, 100], [540, 380], [100, 380]],
            "enter_threshold_sec": 2.0,
            "min_people": 1,
            "is_active": True,
            "class_name": "default"
        }
    }

# ------------------------------------------------------------
# Camera Thread (Robust)
# ------------------------------------------------------------
class Camera:
    def __init__(self, src=0):
        print(f"📷 [Camera] Initializing source {src}...")
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"⚠️ [Camera] Failed to open source {src}. Trying source 1...")
            self.cap = cv2.VideoCapture(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimal buffer
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

# ------------------------------------------------------------
# Core Processing Thread
# ------------------------------------------------------------
stop_processing = threading.Event()

def processing_worker():
    global current_raw_frame, current_processed_frame, current_depth_frame, current_seg_frame
    global latest_telemetry, active_zones, tracker_states, alert_events
    global is_scanning, scan_end_time, auto_zones_buffer

    print("🚀 [Processing Worker] Starting STABLE spatial AI pipeline...")
    
    # 1. Initialize Camera (Threaded)
    cam = Camera(src=0).start()
    if not cam.grabbed:
        print("❌ [Processing Worker] FATAL: Could not initialize any camera.")
        return

    # 2. Load Depth Estimator
    try:
        print("📦 [Processing Worker] Loading Depth Estimator...")
        depth_estimator = DepthEstimator()
        print("✅ [Processing Worker] Depth Estimator Ready.")
    except Exception as e:
        print(f"❌ [Processing Worker] Depth estimator initialization failed: {e}")
        cam.stop()
        return

    # 3. Load YOLO models
    try:
        print("📦 [Processing Worker] Loading YOLO Detection Model...")
        yolo_model = YOLO(str(MODEL_DIR / "yolo26n_ncnn_model") if (MODEL_DIR / "yolo26n_ncnn_model").exists() else "yolov8n.pt", task="detect")
        print("📦 [Processing Worker] Loading YOLOE Segmentation Model...")
        yoloe_model = YOLO(str(MODEL_DIR / "yoloe-26n-seg-pf.pt") if (MODEL_DIR / "yoloe-26n-seg-pf.pt").exists() else "yolov8n-seg.pt", task="segment")
        print("✅ [Processing Worker] All Models Ready.")
    except Exception as e:
        print(f"❌ [Processing Worker] Model loading failed: {e}")
        cam.stop()
        return

    fps_start_time = time.time()
    frame_count = 0
    last_process_time = 0

    print("🏃 [Processing Worker] Entering main loop...")

    while not stop_processing.is_set():
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
            
        now = time.time()
        target_interval = 1.0 / TARGET_FPS
        
        # Rate limit AI processing to TARGET_FPS, but keep loop spinning
        if now - last_process_time < target_interval:
            # We still update the raw frame for 'idle' viewing if needed
            with state_lock:
                current_raw_frame = frame.copy()
            time.sleep(0.005)
            continue
            
        last_process_time = now
        frame = cv2.resize(frame, (640, 480))
        
        # 1. YOLO Tracking
        frame_320 = cv2.resize(frame, (320, 320))
        tracks = []
        try:
            results = yolo_model.track(frame_320, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                for box, track_id in zip(boxes, track_ids):
                    # Correct scale 320->640
                    x1, y1, x2, y2 = map(int, [box[0]*2, box[1]*1.5, box[2]*2, box[3]*1.5])
                    tracks.append({"id": int(track_id), "bbox": [x1, y1, x2, y2], "center": [(x1+x2)//2, (y1+y2)//2]})
        except Exception: pass

        # 2. Depth Estimation
        depth_map_m = None
        color_depth = None
        try:
            depth_map_m, color_depth = depth_estimator.estimate(frame)
        except Exception: pass
        
        if depth_map_m is None:
            depth_map_m = np.ones((480, 640), dtype=np.float32) * 5.0
            color_depth = np.zeros((480, 640, 3), dtype=np.uint8)

        # 3. YOLOE Scan Logic
        if is_scanning and now < scan_end_time:
            try:
                seg_results = yoloe_model.predict(frame, verbose=False)
                if seg_results and seg_results[0].boxes is not None:
                    names = yoloe_model.names
                    for box in seg_results[0].boxes:
                        class_name = names[int(box.cls[0].item())].lower()
                        if class_name in TARGET_OBJECTS:
                            b = box.xyxy[0].cpu().numpy().astype(int)
                            if class_name not in auto_zones_buffer: auto_zones_buffer[class_name] = []
                            auto_zones_buffer[class_name].append([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
            except Exception: pass
        elif is_scanning and now >= scan_end_time:
            with state_lock:
                is_scanning = False
                active_zones = {k: v for k, v in active_zones.items() if v.get("class_name") != "auto"}
                for cn, polys in auto_zones_buffer.items():
                    for i, p in enumerate(polys):
                        active_zones[f"Auto_{cn}_{i+1}"] = {"polygon": p, "enter_threshold_sec": 2.0, "is_active": True, "class_name": "auto"}
            auto_zones_buffer.clear()

        # 4. Spatial Verification
        processed_frame = frame.copy()
        telemetry_tracks = []
        with state_lock:
            zones_to_check = {k: v for k, v in active_zones.items() if v.get("is_active", True)}
            
        for zone_id, zone_data in zones_to_check.items():
            poly = np.array(zone_data["polygon"], dtype=np.int32)
            cv2.polylines(processed_frame, [poly], True, (0, 255, 0), 2)
            cv2.putText(processed_frame, zone_id, (poly[0][0], poly[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for track in tracks:
            t_id, (cx, cy), [x1, y1, x2, y2] = track["id"], track["center"], track["bbox"]
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
            
            matched_zone_id, status, p_depth, z_depth, diff = None, "SAFE", 0.0, 0.0, 0.0
            for zone_id, zone_data in zones_to_check.items():
                poly = np.array(zone_data["polygon"], dtype=np.int32)
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    matched_zone_id = zone_id
                    # Person Depth
                    pad_w, pad_h = int((x2-x1)*0.25), int((y2-y1)*0.25)
                    roi_p = depth_map_m[max(0,y1+pad_h):min(480,y2-pad_h), max(0,x1+pad_w):min(640,x2-pad_w)]
                    p_depth = float(np.mean(roi_p[roi_p > 0.1])) if roi_p.size > 0 else 0.0
                    
                    # Zone Floor Depth
                    mask = np.zeros((480, 640), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 255)
                    roi_z = depth_map_m[mask > 0]
                    z_depth = float(np.mean(roi_z[roi_z > 0.1])) if roi_z.size > 0 else 0.0
                    
                    diff = abs(p_depth - z_depth)
                    if zone_id not in tracker_states: tracker_states[zone_id] = {}
                    if t_id not in tracker_states[zone_id]: tracker_states[zone_id][t_id] = {"enter_time": now, "notified": False}
                    state = tracker_states[zone_id][t_id]; elapsed = now - state["enter_time"]
                    
                    if elapsed >= zone_data.get("enter_threshold_sec", 2.0):
                        if diff <= DEPTH_THRESHOLD:
                            status = "INTRUSION"; cv2.polylines(processed_frame, [poly], True, (0, 0, 255), 4)
                            if not state["notified"]:
                                state["notified"] = True
                                broadcast_sse("alert", {"id": f"evt_{int(now*1000)}", "timestamp": datetime.now().strftime("%H:%M:%S"), "zone_id": zone_id, "track_id": t_id, "diff": diff})
                        else: status = "VERIFYING"
                    else: status = "PENDING"
                    
                    cv2.putText(processed_frame, f"ID:{t_id} ({p_depth:.1f}m)", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2)
                    telemetry_tracks.append({"id": t_id, "zone_id": zone_id, "p_depth": p_depth, "z_depth": z_depth, "diff": diff, "status": status, "elapsed": elapsed})
                    break
            
            if not matched_zone_id:
                cv2.putText(processed_frame, f"ID:{t_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
                telemetry_tracks.append({"id": t_id, "zone_id": "None", "status": "SAFE"})

        with state_lock:
            current_raw_frame = frame
            current_processed_frame, current_depth_frame = processed_frame, color_depth
            if current_seg_frame is None or frame_count % 30 == 0: current_seg_frame = frame.copy()
            
            frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                latest_telemetry["fps"] = round(frame_count / (time.time() - fps_start_time), 1)
                frame_count = 0; fps_start_time = time.time()
            
            latest_telemetry.update({
                "active_zones_count": len(zones_to_check), 
                "current_tracks_count": len(tracks), 
                "is_scanning": is_scanning, 
                "tracks": telemetry_tracks
            })
        broadcast_sse("telemetry", latest_telemetry)

    cam.stop()
    print("🔴 [Processing Worker] Stopped.")

def broadcast_sse(event_type, data):
    global sse_queues
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    for q in sse_queues:
        try: q.put_nowait(payload)
        except Exception: pass

@app.route("/")
def index(): return render_template("index.html")

def gen_video(feed_type="processed"):
    while True:
        with state_lock:
            if feed_type == "processed": f = current_processed_frame
            elif feed_type == "depth": f = current_depth_frame
            else: f = current_seg_frame
        if f is None:
            time.sleep(0.1); continue
        ret, jpeg = cv2.imencode(".jpg", f)
        if not ret: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.08)

@app.route("/video_feed")
def video_feed(): return Response(gen_video("processed"), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/depth_feed")
def depth_feed(): return Response(gen_video("depth"), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/seg_feed")
def seg_feed(): return Response(gen_video("segmentation"), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/api/events")
def events():
    q = queue.Queue(maxsize=10); sse_queues.append(q)
    def sse_stream():
        try:
            while True: yield q.get()
        except GeneratorExit: pass
    return Response(sse_stream(), mimetype="text/event-stream")

@app.route("/api/zones", methods=["GET", "POST"])
def manage_zones():
    global active_zones
    if request.method == "POST":
        data = request.json
        with state_lock:
            active_zones[data["zone_id"]] = {"polygon": data["polygon"], "enter_threshold_sec": float(data.get("enter_threshold_sec", 2.0)), "is_active": True, "class_name": "custom"}
            try:
                with open(ZONES_CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(active_zones, f, indent=4)
            except Exception: pass
        return jsonify({"success": True})
    return jsonify(active_zones)

@app.route("/api/zones/<zone_id>", methods=["DELETE"])
def delete_zone(zone_id):
    global active_zones
    with state_lock:
        if zone_id in active_zones: del active_zones[zone_id]; return jsonify({"success": True})
    return jsonify({"success": False}), 404

@app.route("/api/scan", methods=["POST"])
def trigger_scan():
    global is_scanning, scan_end_time; duration = float(request.json.get("duration", 3.0))
    with state_lock: is_scanning = True; scan_end_time = time.time() + duration; auto_zones_buffer.clear()
    return jsonify({"success": True})

if __name__ == "__main__":
    threading.Thread(target=processing_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
