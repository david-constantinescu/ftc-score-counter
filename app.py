import sys
import os
import warnings
import subprocess
import importlib.util
import logging
import threading
import time
import signal
import json
from collections import OrderedDict
from flask import Flask, render_template, Response, jsonify, request

os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")

# ── Auto-install dependencies ────────────────────────────────────────────────
def _install_deps():
    pkgs = {
        "opencv-python": "cv2",
        "pillow": "PIL",
        "numpy": "numpy",
        "ultralytics": "ultralytics",
        "flask": "flask"
    }
    missing = [p for p, m in pkgs.items() if importlib.util.find_spec(m) is None]
    if missing:
        print(f"Installing: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )

_install_deps()

import cv2
import numpy as np
from ultralytics import YOLO

# ── Constants ─────────────────────────────────────────────────────────────────
PURPLE_LOW  = np.array([128, 60, 60])
PURPLE_HIGH = np.array([152, 255, 255])
GREEN_LOW   = np.array([45, 80, 60])
GREEN_HIGH  = np.array([80, 255, 255])

MIN_BALL_AREA   = 300
MAX_BALL_AREA   = 40000
MIN_RADIUS      = 8
MAX_RADIUS      = 120
KERN_SIZE       = (5, 5)
MIN_CIRCULARITY = 0.20
CONFIRM_FRAMES  = 3

PROCESS_W, PROCESS_H = 640, 480

YOLO_BALL_CLASS = 32
YOLO_CONF       = 0.25

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  Centroid Tracker
# ══════════════════════════════════════════════════════════════════════════════
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_dist=250):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_dist = max_dist
        self.ever_confirmed = set()

    def reset(self):
        self.next_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.ever_confirmed.clear()

    @property
    def confirmed_count(self):
        return len(self.ever_confirmed)

    def update(self, dets):
        if not dets:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)
            return self.objects

        if not self.objects:
            for d in dets:
                self._register(d)
            return self.objects

        ids = list(self.objects.keys())
        opos = np.array([[self.objects[i]["x"], self.objects[i]["y"]] for i in ids])
        dpos = np.array([[d["x"], d["y"]] for d in dets])
        D = np.linalg.norm(opos[:, None] - dpos[None, :], axis=2)

        used_r, used_c = set(), set()
        for flat_idx in np.argsort(D, axis=None):
            r, c = divmod(int(flat_idx), D.shape[1])
            if r in used_r or c in used_c:
                continue
            if D[r, c] > self.max_dist:
                break
            oid = ids[r]
            self.objects[oid].update(dets[c])
            self.objects[oid]["age"] = self.objects[oid].get("age", 0) + 1
            self.disappeared[oid] = 0
            if self.objects[oid]["age"] >= CONFIRM_FRAMES:
                self.ever_confirmed.add(oid)
            used_r.add(r)
            used_c.add(c)

        for r_idx, oid in enumerate(ids):
            if r_idx not in used_r:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)

        for c_idx, d in enumerate(dets):
            if c_idx not in used_c:
                self._register(d)

        return self.objects

    def _register(self, d):
        d = dict(d)
        d["age"] = 1
        self.objects[self.next_id] = d
        self.disappeared[self.next_id] = 0
        self.next_id += 1


# ══════════════════════════════════════════════════════════════════════════════
#  Threaded Camera Capture
# ══════════════════════════════════════════════════════════════════════════════
class CameraThread:
    def __init__(self):
        self.cap = None
        self.src = None
        self.frame = None
        self.lock = threading.Lock()
        self._running = False
        self._thread = None
        self.is_ready = False
        self.is_loading = False

    def open(self, src):
        self.stop()
        self.src = src
        self._running = True
        self.is_ready = False
        self.is_loading = True
        self.frame = None
        self._thread = threading.Thread(target=self._loop, args=(src,), daemon=True)
        self._thread.start()
        return True

    def _loop(self, src):
        s = int(src) if isinstance(src, str) and src.isdigit() else src
        
        if sys.platform == "darwin" and isinstance(s, int):
            cap = cv2.VideoCapture(s, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(s)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_H)
            self.cap = cap
            self.is_ready = True
            self.is_loading = False
        else:
            self.is_ready = False
            self.is_loading = False
            self._running = False
            return

        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)
                
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_ready = False

    def grab(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def is_open(self):
        return self.is_ready and self._running

    def stop(self):
        self._running = False
        with self.lock:
            self.frame = None


# ══════════════════════════════════════════════════════════════════════════════
#  Ball Detector
# ══════════════════════════════════════════════════════════════════════════════
class BallDetector:
    def __init__(self):
        self.yolo = None
        self.ready = False

    def load(self, callback=None):
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
            self.yolo = YOLO(model_path)
            self.yolo.predict(
                np.zeros((480, 640, 3), dtype=np.uint8),
                verbose=False, conf=YOLO_CONF,
            )
            self.ready = True
            logging.info("YOLOv8n loaded and warmed up")
            if callback:
                callback(True, "YOLOv8n ready")
        except Exception as e:
            logging.error(f"YOLOv8 load error: {e}")
            self.ready = False
            if callback:
                callback(False, f"YOLO failed — HSV only ({e})")

    def _yolo_detect(self, frame):
        if not self.ready:
            return []
        results = self.yolo.predict(
            frame, verbose=False, conf=YOLO_CONF,
            classes=[YOLO_BALL_CLASS],
            imgsz=320,
        )
        out = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                radius = int(max(x2 - x1, y2 - y1) / 2)
                out.append({"x": cx, "y": cy, "radius": radius, "src": "yolo"})
        return out

    def _hsv_detect(self, hsv, low, high):
        mask = cv2.inRange(hsv, low, high)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERN_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < MIN_CIRCULARITY:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < MIN_RADIUS or r > MAX_RADIUS:
                continue
            out.append({"x": int(cx), "y": int(cy), "radius": int(r), "src": "hsv"})
        return out

    def detect(self, frame, use_yolo=True):
        yol_dets = []
        if use_yolo and self.ready:
            yol_dets = self._yolo_detect(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_dets = []
        hsv_dets += self._hsv_detect(hsv, PURPLE_LOW, PURPLE_HIGH)
        hsv_dets += self._hsv_detect(hsv, GREEN_LOW, GREEN_HIGH)

        dets = list(yol_dets)
        for hd in hsv_dets:
            is_dup = False
            for yd in yol_dets:
                dist = ((yd["x"] - hd["x"]) ** 2 + (yd["y"] - hd["y"]) ** 2) ** 0.5
                if dist < max(yd["radius"], hd["radius"]) * 1.5:
                    is_dup = True
                    break
            if not is_dup:
                dets.append(hd)
        return dets


# ══════════════════════════════════════════════════════════════════════════════
#  Audio Player
# ══════════════════════════════════════════════════════════════════════════════
class AudioPlayer:
    def __init__(self, filename="ftctimer.mp3"):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, filename)
        self.path = path if os.path.exists(path) else None
        self.proc = None
        self.paused = False

    def play(self):
        self.stop()
        if not self.path: return
        try:
            self.proc = subprocess.Popen(["afplay", self.path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.paused = False
        except Exception: pass

    def pause(self):
        if self.proc and self.proc.poll() is None and not self.paused:
            try: os.kill(self.proc.pid, signal.SIGSTOP); self.paused = True
            except: pass

    def resume(self):
        if self.proc and self.paused:
            try: os.kill(self.proc.pid, signal.SIGCONT); self.paused = False
            except: pass

    def stop(self):
        if self.proc:
            try:
                if self.paused: os.kill(self.proc.pid, signal.SIGCONT)
                self.proc.terminate()
            except: pass
            self.proc = None
            self.paused = False


# ══════════════════════════════════════════════════════════════════════════════
#  App State & Globals
# ══════════════════════════════════════════════════════════════════════════════
state = {
    "timer_seconds": 150,
    "timer_running": False,
    "timer_started": False,
    "blue_hw": 0,
    "red_hw": 0,
    "blue_live": 0,
    "red_live": 0,
    "fps": 0.0,
    "status_msg": "Initializing ...",
    "status_ok": True,
    "cameras": ["None (disabled)"],
    "blue_cam_val": "None (disabled)",
    "red_cam_val": "None (disabled)",
}

blue_tracker = CentroidTracker()
red_tracker = CentroidTracker()
blue_cam = CameraThread()
red_cam = CameraThread()
detector = BallDetector()
audio = AudioPlayer()

latest_frames = {
    "blue": None,
    "red": None
}

def set_status(msg, ok=True):
    state["status_msg"] = msg
    state["status_ok"] = ok

def load_yolo():
    detector.load(callback=lambda ok, msg: set_status(msg, ok))

threading.Thread(target=load_yolo, daemon=True).start()

def _scan_cameras_sys():
    cams = []
    if sys.platform == "darwin":
        try:
            p = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                               capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                items = json.loads(p.stdout).get("SPCameraDataType", [])
                for i, item in enumerate(items):
                    cams.append(f"{i}: {item.get('_name', f'Camera {i}')}")
                if cams: return cams
        except Exception: pass

    for idx in range(3):
        try:
            import os as _os
            old_stderr = _os.dup(2)
            f = open(_os.devnull, 'w')
            _os.dup2(f.fileno(), 2)
            cap = cv2.VideoCapture(idx)
            if cap.isOpened() and cap.read()[0]:
                cams.append(f"{idx}: Camera {idx}")
            cap.release()
            _os.dup2(old_stderr, 2)
            _os.close(old_stderr)
            f.close()
        except: pass
    return cams or ["0: Default Camera"]

def refresh_cameras():
    set_status("Scanning cameras ...", False)
    cams = _scan_cameras_sys()
    choices = ["None (disabled)"] + cams + ["Add URL…"]
    state["cameras"] = choices
    active = sum(1 for c in [blue_cam, red_cam] if c.is_open())
    set_status(f"{len(cams)} camera(s) found, {active} active", True)

    # ── Auto-load first 2 cameras if not set ──────────────────────────────────
    valid_cams = [c for c in cams if ":" in c]
    if valid_cams:
        if state["blue_cam_val"] == "None (disabled)":
            val = valid_cams[0]
            logging.info(f"Auto-assigning BLUE -> {val}")
            state["blue_cam_val"] = val
            src = _parse_cam_val(val)
            if src is not None:
                blue_tracker.reset()
                blue_cam.open(src)
        
        if len(valid_cams) > 1 and state["red_cam_val"] == "None (disabled)":
            val = valid_cams[1]
            logging.info(f"Auto-assigning RED -> {val}")
            state["red_cam_val"] = val
            src = _parse_cam_val(val)
            if src is not None:
                red_tracker.reset()
                red_cam.open(src)

threading.Thread(target=refresh_cameras, daemon=True).start()


def timer_loop():
    while True:
        if state["timer_running"] and state["timer_started"]:
            if state["timer_seconds"] > 0:
                state["timer_seconds"] -= 1
            else:
                state["timer_running"] = False
                state["timer_started"] = False
                audio.stop()
                set_status("Match ended!", True)
        time.sleep(1)

threading.Thread(target=timer_loop, daemon=True).start()


def video_loop():
    frame_no = 0
    yolo_every = 3
    fps_val = 0.0
    while True:
        t0 = time.time()
        frame_no += 1
        use_yolo = detector.ready and (frame_no % yolo_every == 0)

        for side, cam, tracker in [("blue", blue_cam, blue_tracker), ("red", red_cam, red_tracker)]:
            if cam.is_open():
                frm = cam.grab()
                if frm is not None:
                    frm = cv2.resize(frm, (PROCESS_W, PROCESS_H))
                    dets = detector.detect(frm, use_yolo=use_yolo)
                    tracked = tracker.update(dets)

                    live = sum(1 for oid in tracked if oid in tracker.ever_confirmed)
                    hw = tracker.confirmed_count

                    if side == "blue":
                        state["blue_live"] = live
                        if hw > state["blue_hw"]: state["blue_hw"] = hw
                        hw_val = state["blue_hw"]
                    else:
                        state["red_live"] = live
                        if hw > state["red_hw"]: state["red_hw"] = hw
                        hw_val = state["red_hw"]

                    for oid, obj in tracked.items():
                        x, y = obj.get("x", 0), obj.get("y", 0)
                        r = max(obj.get("radius", 16), 10)
                        confirmed = oid in tracker.ever_confirmed
                        color = (0, 255, 0) if confirmed else (0, 180, 255)
                        thick = 3 if confirmed else 1
                        cv2.circle(frm, (x, y), r, color, thick)
                        if confirmed:
                            cv2.putText(frm, str(oid), (x - 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.putText(frm, f"Balls: {hw_val}  (live {live})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    _, buf = cv2.imencode('.jpg', frm)
                    latest_frames[side] = buf.tobytes()
            else:
                frm = np.zeros((PROCESS_H, PROCESS_W, 3), dtype=np.uint8)
                msg = "Loading..." if cam.is_loading else f"No {side.upper()} camera"
                cv2.putText(frm, msg, (PROCESS_W // 2 - 80, PROCESS_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                _, buf = cv2.imencode('.jpg', frm)
                latest_frames[side] = buf.tobytes()

        dt = time.time() - t0
        fps_val = 0.8 * fps_val + 0.2 * (1.0 / max(dt, 0.001))
        state["fps"] = fps_val

        delay = max(0.01, 0.033 - dt)
        time.sleep(delay)

threading.Thread(target=video_loop, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  Flask Routes
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

def gen_frames(side):
    while True:
        frame = latest_frames[side]
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed/<side>')
def video_feed(side):
    return Response(gen_frames(side), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    return jsonify(state)

def _parse_cam_val(val):
    if not val or val.startswith("None"): return None
    val = val.strip()
    if ":" in val and val.split(":")[0].strip().isdigit() and "http" not in val:
        return int(val.split(":")[0])
    if val.isdigit(): return int(val)
    return val

@app.route('/api/action', methods=['POST'])
def handle_action():
    data = request.json
    action = data.get("action")
    side = data.get("side")
    val = data.get("val")

    if action == "manual_add":
        if side == "blue": state["blue_hw"] += 1
        elif side == "red": state["red_hw"] += 1
    elif action == "manual_sub":
        if side == "blue": state["blue_hw"] = max(0, state["blue_hw"] - 1)
        elif side == "red": state["red_hw"] = max(0, state["red_hw"] - 1)
    elif action == "clear_side":
        if side == "blue":
            blue_tracker.reset()
            state["blue_hw"] = 0
            state["blue_live"] = 0
        elif side == "red":
            red_tracker.reset()
            state["red_hw"] = 0
            state["red_live"] = 0
    elif action == "reset_all":
        blue_tracker.reset()
        red_tracker.reset()
        state["blue_hw"] = state["red_hw"] = 0
        state["blue_live"] = state["red_live"] = 0
        state["timer_seconds"] = 150
        state["timer_running"] = False
        state["timer_started"] = False
        audio.stop()
        set_status("All scores reset", True)
    elif action == "toggle_timer":
        if not state["timer_running"] and state["timer_seconds"] == 150:
            set_status("Match audio started — timer in 3 s", True)
            audio.play()
            state["timer_running"] = True
            state["timer_started"] = False
            def begin_countdown():
                time.sleep(3)
                if state["timer_running"]:
                    state["timer_started"] = True
                    set_status("Match started!", True)
            threading.Thread(target=begin_countdown, daemon=True).start()
        elif state["timer_running"]:
            state["timer_running"] = False
            audio.pause()
        else:
            state["timer_running"] = True
            state["timer_started"] = True
            audio.resume()
    elif action == "reset_timer":
        state["timer_running"] = False
        state["timer_started"] = False
        state["timer_seconds"] = 150
        audio.stop()
    elif action == "refresh_cams":
        threading.Thread(target=refresh_cameras, daemon=True).start()
    elif action == "set_cam":
        src = _parse_cam_val(val)
        if side == "blue":
            state["blue_cam_val"] = val
            if src is None:
                blue_cam.stop()
                blue_tracker.reset()
            else:
                blue_tracker.reset()
                blue_cam.open(src)
                set_status(f"Loading BLUE camera ({src})...", True)
        elif side == "red":
            state["red_cam_val"] = val
            if src is None:
                red_cam.stop()
                red_tracker.reset()
            else:
                red_tracker.reset()
                red_cam.open(src)
                set_status(f"Loading RED camera ({src})...", True)

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True, debug=False)
