"""
DECODE Goal Scorer — Web Server
════════════════════════════════════════════════════════════════════
Flask-based dual-camera ball counting system.
Streams MJPEG video, sends real-time scores via SSE, and exposes
REST API for match control. Optimised for Intel CPU (Dell Optiplex).
Timer audio plays on the CLIENT browser, not the server.
"""

import sys
import os
import warnings
import subprocess
import importlib.util
import logging
import threading
import multiprocessing
import time
import json
import queue
import re
import urllib.request
import urllib.error
from collections import OrderedDict

os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["USE_NNPACK"] = "0"
# Let OpenCV auto-select the best video backend per platform
if sys.platform == "darwin":
    os.environ["OPENCV_VIDEOIO_PRIORITY_LIST"] = "AVFOUNDATION,FFMPEG"

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)

# ── Auto-install dependencies ────────────────────────────────────────────────
def _install_deps():
    pkgs = {
        "opencv-python": "cv2",
        "numpy": "numpy",
        "ultralytics": "ultralytics",
        "flask": "flask",
    }
    missing = [p for p, m in pkgs.items() if importlib.util.find_spec(m) is None]
    if missing:
        logging.info(f"Installing: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )

_install_deps()

import cv2
import numpy as np
from ultralytics import YOLO
import torch

torch.set_num_threads(multiprocessing.cpu_count())

from flask import (Flask, Response, jsonify, request,
                   render_template, send_from_directory)

# Maximise CPU threading for Intel
torch.set_num_threads(multiprocessing.cpu_count())
cv2.setUseOptimized(True)
cv2.setNumThreads(multiprocessing.cpu_count())

# ── Device detection ──────────────────────────────────────────────────────────
def _detect_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logging.info(f"CUDA GPU detected: {name}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logging.info("Apple Silicon GPU detected (MPS)")
        return "mps"
    logging.info("No GPU — using CPU")
    return "cpu"

DEVICE = _detect_device()
USE_HALF = DEVICE in ("cuda",)  # MPS half is flaky on some models; CUDA only
logging.info(f"Inference device: {DEVICE}  |  FP16: {USE_HALF}")

# ── Constants ─────────────────────────────────────────────────────────────────
PURPLE_LOW  = np.array([115, 90, 90])
PURPLE_HIGH = np.array([155, 255, 255])
# Lower saturation & value for green to catch extreme motion blur fading
GREEN_LOW   = np.array([33,  30, 30])
GREEN_HIGH  = np.array([85,  255, 255])

PROCESS_W, PROCESS_H = 384, 288   # multiple of 32 for YOLO

YOLO_BALL_CLASS = 32
YOLO_CONF       = 0.35

MIN_BALL_AREA   = 300
MAX_BALL_AREA   = 80000
MIN_RADIUS      = 12
MAX_RADIUS      = 150
KERN_SIZE       = (7, 7)
CONFIRM_FRAMES  = 2

STREAM_QUALITY  = 70   # JPEG quality for MJPEG stream


# ══════════════════════════════════════════════════════════════════════════════
#  Centroid Tracker
# ══════════════════════════════════════════════════════════════════════════════
class CentroidTracker:
    def __init__(self, max_disappeared=15, max_dist=200):
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


# Global lock to prevent multiple cameras from opening simultaneously and 
# exhausting USB bandwidth before MJPEG format is set.
camera_connect_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
#  Camera Thread  (always grabs the freshest frame)
# ══════════════════════════════════════════════════════════════════════════════
class CameraThread:
    def __init__(self):
        self.cap = None
        self.src = None
        self.frame = None
        self.frame_queue = queue.Queue(maxsize=300)
        self.lock = threading.Lock()
        self._running = False
        self._thread = None
        self.is_ready = False

    def open(self, src):
        self.stop()
        self.src = src
        self._running = True
        self.is_ready = False
        self.frame = None
        self._thread = threading.Thread(target=self._loop, args=(src,), daemon=True)
        self._thread.start()
        return True

    def _loop(self, src):

        s = int(src) if isinstance(src, str) and str(src).isdigit() else src
        
        def _configure(c):
            """Set MJPEG format + resolution to avoid USB bandwidth issues."""
            if sys.platform == "linux":
                c.set(cv2.CAP_PROP_FOURCC,
                      cv2.VideoWriter.fourcc(*"MJPG"))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        def _try_open(target, api=None):
            """Try to open a single target; return configured cap or None."""
            try:
                c = cv2.VideoCapture(target, api) if api is not None else cv2.VideoCapture(target)
            except Exception:
                return None
            if not c.isOpened():
                c.release()
                return None
            _configure(c)
            for _ in range(8):
                ret, _ = c.read()
                if ret:
                    return c
                time.sleep(0.15)
            c.release()
            return None

        def connect():
            with camera_connect_lock:
                if sys.platform == "darwin" and isinstance(s, int):
                    return _try_open(s, cv2.CAP_AVFOUNDATION)

                # On Linux, scan_cameras returns pre-verified integer indices.
                # Just open directly. Also try the original src as fallback.
                logging.info(f"Opening camera: {src} (s={s})")
                c = _try_open(s)
                if c is not None:
                    logging.info(f"Camera {src} opened, backend={c.getBackendName()}")
                    return c
                # Fallback: if s != src (e.g. path string), try that too
                if s != src:
                    c = _try_open(src)
                    if c is not None:
                        logging.info(f"Camera {src} opened via fallback, backend={c.getBackendName()}")
                        return c
                return None

        self.cap = connect()
        if self.cap:
            self.is_ready = True
            logging.info(f"Camera {src} opened successfully")

        fails = 0
        while self._running:
            if not self.cap or not self.cap.isOpened():
                self.is_ready = False
                time.sleep(1.0)
                self.cap = connect()
                if self.cap:
                    self.is_ready = True
                    fails = 0
                continue

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                fails = 0
            else:
                fails += 1
                if fails > 30:
                    logging.warning(f"Camera {src}: 30 consecutive fails, reconnecting")
                    self.cap.release()
                    self.cap = None
                    self.is_ready = False
                    time.sleep(2.0)  # Give device time to fully release
                time.sleep(0.01)

        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_ready = False

    def grab(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            with self.lock:
                return self.frame.copy() if self.frame is not None else None

    def _loop_realsense(self, src):
        parts = src.split(":")
        serial = parts[1]
        stream_idx = int(parts[2]) if len(parts) > 2 else None

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)

        if stream_idx is not None:
            # T265 Fisheye
            config.enable_stream(rs.stream.fisheye, stream_idx, rs.format.y8, 30)
            config.enable_stream(rs.stream.pose)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            pipeline.start(config)
            self.is_ready = True
            logging.info(f"RealSense camera {src} opened successfully")
            while self._running:
                frames = pipeline.wait_for_frames()
                _frame = None

                if stream_idx is not None:
                    f = frames.get_fisheye_frame(stream_idx)
                    pose = frames.get_pose_frame()
                    if f:
                        img = np.asanyarray(f.get_data())
                        # Convert 8-bit grayscale to BGR so YOLO can process it
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        if pose:
                            data = pose.get_pose_data()
                            tx, ty, tz = data.translation.x, data.translation.y, data.translation.z
                            vx, vy, vz = data.velocity.x, data.velocity.y, data.velocity.z
                            # Add small text to show we are using all T265 features
                            cv2.putText(img_bgr, f"Pose X: {tx:.2f} m/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(img_bgr, f"Vel: {vx:.2f}, {vy:.2f}, {vz:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        _frame = img_bgr
                else:
                    c_frame = frames.get_color_frame()
                    if c_frame:
                        _frame = np.asanyarray(c_frame.get_data())

                if _frame is not None:
                    with self.lock:
                        self.frame = _frame
                    try: self.frame_queue.put_nowait(_frame.copy())
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(_frame.copy())
                        except queue.Empty: pass
        except Exception as e:
            logging.error(f"RealSense error for {src}: {e}")
            self.is_ready = False
        finally:
            try: pipeline.stop()
            except: pass

    def is_open(self):
        return self.is_ready and self._running

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.1)
            self._thread = None
        with self.lock:
            self.frame = None
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except queue.Empty: pass


# ══════════════════════════════════════════════════════════════════════════════
#  Ball Detector  (YOLOv8 + HSV hybrid)
# ══════════════════════════════════════════════════════════════════════════════
class BallDetector:
    def __init__(self):
        self.yolo = None
        self.ready = False

    def load(self, callback=None):
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
            if not os.path.exists(model_path):
                model_path = "yolov8n.pt"  # ultralytics will auto-download
            self.yolo = YOLO(model_path)
            self.yolo.to(DEVICE)
            self.yolo.predict(
                np.zeros((PROCESS_H, PROCESS_W, 3), dtype=np.uint8),
                verbose=False, conf=YOLO_CONF,
                device=DEVICE, half=USE_HALF,
            )
            self.ready = True
            msg = f"YOLOv8n ready [{DEVICE.upper()}]"
            logging.info(msg)
            if callback:
                callback(True, msg)
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
            classes=[YOLO_BALL_CLASS], iou=0.45,
            imgsz=384, device=DEVICE, half=USE_HALF,
        )
        out = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                if w < MIN_RADIUS * 2 or h < MIN_RADIUS * 2:
                    continue
                if w > MAX_RADIUS * 2 or h > MAX_RADIUS * 2:
                    continue
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect > 4.5:  # Tolerate extreme motion blur elongation
                    continue
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                radius = int(max(w, h) / 2)
                out.append({"x": cx, "y": cy, "radius": radius, "src": "yolo"})
        return out

    def _hsv_detect(self, hsv, low, high, erode_iter=1):
        mask = cv2.inRange(hsv, low, high)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERN_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=1)
        if erode_iter > 0:
            mask = cv2.erode(mask, kern, iterations=erode_iter)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            out.append({"x": int(cx), "y": int(cy), "radius": int(r), "src": "hsv"})
        return out

    def detect(self, frame, use_yolo=True, skip_hsv=False):
        yol_dets = []
        if use_yolo and self.ready:
            yol_dets = self._yolo_detect(frame)
        hsv_dets = []
        if not skip_hsv:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_dets += self._hsv_detect(hsv, PURPLE_LOW, PURPLE_HIGH, erode_iter=1)
            hsv_dets += self._hsv_detect(hsv, GREEN_LOW, GREEN_HIGH, erode_iter=0)
        # Merge all detections, then do full pairwise dedup
        all_raw = yol_dets + hsv_dets
        all_raw.sort(key=lambda d: d["radius"], reverse=True)
        dets = []
        for d in all_raw:
            is_dup = False
            for fd in dets:
                dist = ((d["x"] - fd["x"]) ** 2 +
                        (d["y"] - fd["y"]) ** 2) ** 0.5
                if dist < max(d["radius"], fd["radius"]) * 1.85:
                    is_dup = True
                    break
            if not is_dup:
                dets.append(d)
        return dets


# ══════════════════════════════════════════════════════════════════════════════
#  Flask Application
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            static_folder=os.path.join(BASE_DIR, "static"),
            template_folder=os.path.join(BASE_DIR, "templates"))

# ── Global objects ────────────────────────────────────────────────────────────
blue_cam = CameraThread()
red_cam  = CameraThread()
blue_tracker = CentroidTracker()
red_tracker  = CentroidTracker()
detector = BallDetector()

state_lock = threading.Lock()
state = {
    "blue_hw": 0, "red_hw": 0,
    "blue_live": 0, "red_live": 0,
    "timer_seconds": 150,
    "timer_running": False,
    "timer_started": False,
    "status": "Initializing…",
    "status_ok": False,
    "device": DEVICE,
    "infer_fps": 0.0,
    "audio_cmd": None,
    "audio_seq": 0,
}

# Latest processed frames (pre-encoded JPEG bytes)
frame_lock = threading.Lock()
latest_blue_jpg = None
latest_red_jpg  = None

_infer_running = True
_timer_tick_time = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def _set_status(msg, ok=True):
    with state_lock:
        state["status"] = msg
        state["status_ok"] = ok

def _audio_cmd(cmd):
    with state_lock:
        state["audio_cmd"] = cmd
        state["audio_seq"] += 1


# ── IP / DroidCam camera helpers ──────────────────────────────────────────────
_ip_cameras = []  # user-added IP cameras: [{"id": "ip:...", "name": "..."}]
_ip_cameras_lock = threading.Lock()

def _check_droidcam(ip_port):
    """Check if a DroidCam (or compatible) IP camera is reachable.
    Tries http first, then https.  Returns the working video URL or None."""
    # Sanitise: strip protocol if user included it
    ip_port = re.sub(r'^https?://', '', ip_port).strip().rstrip('/')
    for scheme in ('http', 'https'):
        url = f"{scheme}://{ip_port}/video"
        try:
            req = urllib.request.Request(url, method='GET')
            resp = urllib.request.urlopen(req, timeout=3)
            # DroidCam returns multipart MJPEG — any 2xx is good
            if resp.status < 400:
                return url
        except Exception:
            continue
    return None


def _add_ip_camera(ip_port, name=None):
    """Register an IP camera so it shows in the camera list.
    Returns the camera dict or None if unreachable."""
    video_url = _check_droidcam(ip_port)
    if video_url is None:
        return None
    cam_id = f"ip:{ip_port}"
    display = name or f"IP Camera ({ip_port})"
    entry = {"id": cam_id, "name": display}
    with _ip_cameras_lock:
        # Avoid duplicates
        if not any(c["id"] == cam_id for c in _ip_cameras):
            _ip_cameras.append(entry)
    return entry


def _resolve_camera_src(src):
    """Translate a camera id to the value CameraThread.open() needs.
    Regular integer ids pass through; ip:host:port ids become an MJPEG URL."""
    src_str = str(src)
    if src_str.startswith("ip:"):
        ip_port = src_str[3:]
        video_url = _check_droidcam(ip_port)
        return video_url if video_url else src_str
    try:
        return int(src_str)
    except (ValueError, TypeError):
        return src_str


# ── Camera scanning ──────────────────────────────────────────────────────────
import glob as _glob

def _linux_v4l2_cameras():
    """Enumerate V4L2 capture devices on Linux with real names.
    Uses v4l2-ctl --list-devices to get grouped output, falling back to sysfs.
    Only returns the primary capture node per camera (not metadata nodes).
    After listing, probes each device to verify it can actually be opened.
    """
    named_cams = []

    # Method 1: v4l2-ctl --list-devices (most reliable for grouping)
    try:
        p = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=2)
        if p.returncode == 0 and p.stdout.strip():
            current_name = None
            for line in p.stdout.splitlines():
                line_s = line.strip()
                if not line_s:
                    continue
                if not line.startswith("\t") and not line.startswith(" "):
                    current_name = line_s.rstrip(":")
                elif current_name and line_s.startswith("/dev/video"):
                    named_cams.append({"path": line_s, "name": current_name})
                    current_name = None  # skip subsequent nodes (metadata)
    except Exception:
        pass

    # Method 2 fallback: sysfs
    if not named_cams:
        devs = sorted(_glob.glob("/dev/video*"))
        seen = set()
        for dev in devs:
            idx_str = dev.replace("/dev/video", "")
            if not idx_str.isdigit():
                continue
            name_path = f"/sys/class/video4linux/video{idx_str}/name"
            try:
                with open(name_path) as f:
                    hw_name = f.read().strip()
            except Exception:
                hw_name = f"Camera {idx_str}"
            if hw_name in seen:
                hw_name = f"{hw_name} (Node {idx_str})"
            else:
                seen.add(hw_name)
            named_cams.append({"path": dev, "name": f"{hw_name} ({dev})"})

    if not named_cams:
        return []

    # Probe to find which devices can actually open and read frames.
    # Build a name map from /dev/videoN paths to camera names.
    path_to_name = {c["path"]: c["name"] for c in named_cams}
    working = []
    seen_fds = set()  # track which /dev/video* fds are actually opened

    for idx in range(max(int(c["path"].replace("/dev/video", "")) for c in named_cams) + 2):
        try:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            ret, _ = cap.read()
            if not ret:
                cap.release()
                continue
            # Check which physical device this index opened via /proc/self/fd
            fd_check = subprocess.run(
                f"ls -la /proc/{os.getpid()}/fd/ 2>/dev/null | grep video",
                shell=True, capture_output=True, text=True)
            dev_path = None
            for fd_line in fd_check.stdout.strip().splitlines():
                if "/dev/video" in fd_line:
                    dev_path = "/dev/video" + fd_line.rsplit("/dev/video", 1)[-1].strip()
                    break
            cap.release()

            if dev_path and dev_path in seen_fds:
                continue  # already found a working index for this physical camera
            if dev_path:
                seen_fds.add(dev_path)

            name = path_to_name.get(dev_path, f"Camera (index {idx})")
            working.append({"id": idx, "name": name})
            logging.info(f"Camera probe: idx={idx} -> {dev_path} ({name}) works")
        except Exception:
            pass

    return working

_cam_cache = {"ts": 0, "data": []}
_CAM_CACHE_TTL = 300  # seconds – camera list changes rarely; re-probe is expensive

def scan_cameras():
    now = time.monotonic()
    if _cam_cache["data"] and (now - _cam_cache["ts"]) < _CAM_CACHE_TTL:
        return _cam_cache["data"]
    cams = []

    if sys.platform == "darwin":
        try:
            p = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                items = json.loads(p.stdout).get("SPCameraDataType", [])
                for i, item in enumerate(items):
                    cams.append({"id": i, "name": f"{item.get('_name', f'Camera {i}')} (index {i})"})
        except Exception:
            pass
    elif sys.platform == "linux":
        cams.extend(_linux_v4l2_cameras())
    
    if not cams:
        # Fallback: probe by index (Windows or if above failed)
        for idx in range(8):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened() and cap.read()[0]:
                    cams.append({"id": idx, "name": f"Camera {idx} (index {idx})"})
                cap.release()
            except Exception:
                pass
    
    result = cams or [{"id": 0, "name": "Default Camera (index 0)"}]
    # Append any user-added IP cameras
    with _ip_cameras_lock:
        for ipc in _ip_cameras:
            if not any(c["id"] == ipc["id"] for c in result):
                result.append(ipc)
    _cam_cache["ts"] = time.monotonic()
    _cam_cache["data"] = result
    return result


# ── Placeholder frame ────────────────────────────────────────────────────────
def _placeholder_jpg(text):
    img = np.zeros((PROCESS_H, PROCESS_W, 3), dtype=np.uint8)
    cv2.putText(img, text, (30, PROCESS_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return buf.tobytes()

_no_blue_jpg = _placeholder_jpg("No BLUE camera")
_no_red_jpg  = _placeholder_jpg("No RED camera")


# ══════════════════════════════════════════════════════════════════════════════
#  Background Threads
# ══════════════════════════════════════════════════════════════════════════════
def inference_loop():
    """Continuously process frames from both cameras (runs on server)."""
    global latest_blue_jpg, latest_red_jpg
    infer_fps = 0.0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY]

    while _infer_running:
        t0 = time.time()
        processed = False

        for side, cam, tracker in [
            ("blue", blue_cam, blue_tracker),
            ("red",  red_cam,  red_tracker),
        ]:
            if not cam.is_open():
                continue
            frame = cam.grab()
            if frame is None:
                continue

            proc = cv2.resize(frame, (PROCESS_W, PROCESS_H))
            is_rs = str(cam.src).startswith("rs:")
            try:
                dets = detector.detect(proc, use_yolo=detector.ready, skip_hsv=is_rs)
            except Exception:
                dets = detector.detect(proc, use_yolo=False, skip_hsv=True)

            tracked = tracker.update(dets)
            live = sum(1 for oid in tracked if oid in tracker.ever_confirmed)
            hw = tracker.confirmed_count

            # Annotate frame with ball count
            disp = proc.copy()
            for d in dets:
                r = int(d.get("radius", 10))
                color = (0, 255, 0) if d.get("src") == "hsv" else (255, 0, 255)
                cv2.circle(disp, (int(d["x"]), int(d["y"])), r, color, 2)

            # Pre-encode JPEG once (all streaming clients get same bytes)
            _, buf = cv2.imencode('.jpg', disp, encode_params)
            jpg_bytes = buf.tobytes()

            with state_lock:
                if side == "blue":
                    state["blue_hw"] = max(state["blue_hw"], hw)
                    state["blue_live"] = live
                else:
                    state["red_hw"] = max(state["red_hw"], hw)
                    state["red_live"] = live

            with frame_lock:
                if side == "blue":
                    latest_blue_jpg = jpg_bytes
                else:
                    latest_red_jpg = jpg_bytes

            processed = True

        if processed:
            dt = time.time() - t0
            infer_fps = 0.8 * infer_fps + 0.2 * (1.0 / max(dt, 0.001))
            with state_lock:
                state["infer_fps"] = round(infer_fps, 1)
                # Update status to show active cameras
                cams_active = []
                if blue_cam.is_open(): cams_active.append("Blue")
                if red_cam.is_open(): cams_active.append("Red")
                if cams_active:
                    state["status"] = f"Running — {', '.join(cams_active)} cam{'s' if len(cams_active)>1 else ''} active"
                    state["status_ok"] = True
        else:
            time.sleep(0.02)


def timer_loop():
    """Background thread: decrement timer every second when running."""
    global _timer_tick_time
    while True:
        with state_lock:
            running = state["timer_running"]
            started = state["timer_started"]
            secs = state["timer_seconds"]

        if running and started and secs > 0:
            now = time.time()
            if now - _timer_tick_time >= 1.0:
                _timer_tick_time = now
                with state_lock:
                    state["timer_seconds"] = max(0, state["timer_seconds"] - 1)
                    if state["timer_seconds"] <= 0:
                        state["timer_running"] = False
                        state["status"] = "Match ended! Scores are final."
                        state["status_ok"] = True
                _audio_cmd("stop")
                logging.info(f"Timer: {state['timer_seconds']}s remaining")
        time.sleep(0.05)


def _begin_countdown():
    """Called 3 s after Start to begin the actual countdown."""
    global _timer_tick_time
    with state_lock:
        if not state["timer_running"]:
            return  # user paused/reset during the 3 s lead-in
        state["timer_started"] = True
        state["status"] = "Match started! LIVE counting…"
        state["status_ok"] = True
        # Reset trackers and scores for the new match
        blue_tracker.reset()
        red_tracker.reset()
        state["blue_hw"] = 0
        state["red_hw"] = 0
        state["blue_live"] = 0
        state["red_live"] = 0
        _timer_tick_time = time.time()
    logging.info("Match countdown started!")


def load_yolo():
    def cb(ok, msg):
        _set_status(msg, ok)
    detector.load(callback=cb)


# ══════════════════════════════════════════════════════════════════════════════
#  Flask Routes
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream/<side>")
def video_feed(side):
    if side not in ("blue", "red"):
        return "Invalid side", 400

    def gen():
        while True:
            with frame_lock:
                jpg = latest_blue_jpg if side == "blue" else latest_red_jpg
            if jpg is None:
                jpg = _no_blue_jpg if side == "blue" else _no_red_jpg
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.033)  # cap at ~30 fps for streaming

    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/events")
def sse():
    def gen():
        while True:
            with state_lock:
                data = dict(state)
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.1)  # 10 updates/sec

    return Response(gen(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/cameras")
def api_cameras():
    if request.args.get("refresh"):
        _cam_cache["ts"] = 0
        _cam_cache["data"] = []
    cams = scan_cameras()
    return jsonify({
        "cameras": cams,
        "blue_src": blue_cam.src if blue_cam._running else "none",
        "red_src": red_cam.src if red_cam._running else "none"
    })


@app.route("/api/camera/<side>", methods=["POST"])
def api_set_camera(side):
    if side not in ("blue", "red"):
        return jsonify({"error": "Invalid side"}), 400
    data = request.get_json(force=True) or {}
    src = data.get("src")
    if src is None:
        return jsonify({"error": "Missing src"}), 400

    cam = blue_cam if side == "blue" else red_cam
    tracker = blue_tracker if side == "blue" else red_tracker

    if str(src) == "none":
        cam.stop()
        tracker.reset()
        _set_status(f"{side.upper()} camera disabled", True)
    else:
        resolved = _resolve_camera_src(src)
        tracker.reset()
        cam.open(resolved)
        _set_status(f"Opening {side.upper()} camera\u2026", True)

    return jsonify({"ok": True})


@app.route("/api/timer/start", methods=["POST"])
def api_timer_start():
    global _timer_tick_time
    with state_lock:
        if state["timer_running"]:
            return jsonify({"ok": True, "msg": "already running"})
        if state["timer_seconds"] <= 0:
            return jsonify({"ok": False, "msg": "timer ended — reset first"})
        fresh = not state["timer_started"]
        state["timer_running"] = True
        if fresh:
            state["status"] = "Audio started — match in 3 s"
        else:
            state["timer_started"] = True
            state["status"] = "Match resumed"
            _timer_tick_time = time.time()
        state["status_ok"] = True

    if fresh:
        _audio_cmd("play")
        threading.Timer(3.0, _begin_countdown).start()
    else:
        _audio_cmd("resume")

    return jsonify({"ok": True})


@app.route("/api/timer/pause", methods=["POST"])
def api_timer_pause():
    with state_lock:
        if state["timer_running"]:
            state["timer_running"] = False
            state["status"] = "Paused — live counting continues"
            state["status_ok"] = True
    _audio_cmd("pause")
    return jsonify({"ok": True})


@app.route("/api/timer/reset", methods=["POST"])
def api_timer_reset():
    with state_lock:
        state["timer_running"] = False
        state["timer_started"] = False
        state["timer_seconds"] = 150
        state["status"] = "Timer reset"
        state["status_ok"] = True
    _audio_cmd("stop")
    return jsonify({"ok": True})


@app.route("/api/score/adjust", methods=["POST"])
def api_score_adjust():
    data = request.get_json(force=True) or {}
    side = data.get("side")
    action = data.get("action")
    if side not in ("blue", "red") or action not in ("plus", "minus", "clear"):
        return jsonify({"error": "Invalid params"}), 400

    key_hw = f"{side}_hw"
    key_live = f"{side}_live"
    with state_lock:
        if action == "plus":
            state[key_hw] += 1
        elif action == "minus":
            state[key_hw] = max(0, state[key_hw] - 1)
        elif action == "clear":
            state[key_hw] = 0
            state[key_live] = 0
    if action == "clear":
        tracker = blue_tracker if side == "blue" else red_tracker
        tracker.reset()

    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with state_lock:
        state["blue_hw"] = 0
        state["red_hw"] = 0
        state["blue_live"] = 0
        state["red_live"] = 0
        state["timer_running"] = False
        state["timer_started"] = False
        state["timer_seconds"] = 150
        state["status"] = "All reset"
        state["status_ok"] = True
    blue_tracker.reset()
    red_tracker.reset()
    _audio_cmd("stop")
    return jsonify({"ok": True})


@app.route("/api/camera/test_ip", methods=["POST"])
def api_test_ip():
    """Test if an IP camera (DroidCam) is reachable and optionally add it."""
    data = request.get_json(force=True) or {}
    ip_port = data.get("ip", "").strip()
    if not ip_port:
        return jsonify({"ok": False, "error": "Missing ip"}), 400
    # Basic validation
    ip_port = re.sub(r'^https?://', '', ip_port).strip().rstrip('/')
    entry = _add_ip_camera(ip_port)
    if entry is None:
        return jsonify({"ok": False, "error": f"Cannot reach camera at {ip_port}"}), 200
    # Invalidate camera cache so the new IP cam shows up
    _cam_cache["ts"] = 0
    _cam_cache["data"] = []
    return jsonify({"ok": True, "camera": entry})


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(dict(state))


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.info("Starting DECODE Goal Scorer server on :2016 …")

    # Background threads
    threading.Thread(target=load_yolo, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    threading.Thread(target=timer_loop, daemon=True).start()

    # Auto-detect cameras in the background and open the first two that work
    def auto_cameras():
        # Try DroidCam at 192.168.0.197:4747 for the red goal first
        droidcam_ip = "192.168.0.197:4747"
        droidcam_url = None
        logging.info(f"Checking for DroidCam at {droidcam_ip}…")
        try:
            droidcam_url = _check_droidcam(droidcam_ip)
        except Exception:
            pass
        if droidcam_url:
            logging.info(f"DroidCam found at {droidcam_ip}")
            _add_ip_camera(droidcam_ip, f"DroidCam ({droidcam_ip})")
        else:
            logging.info(f"DroidCam not found at {droidcam_ip}, skipping")

        cams = scan_cameras()
        logging.info(f"Cameras found: {cams}")
        _set_status(f"Cameras ready: {len(cams)} found", True)

        # Assign cameras: DroidCam goes to red, local cams to blue (then red fallback)
        local_cams = [c for c in cams if not str(c["id"]).startswith("ip:")]
        red_assigned = False

        if droidcam_url:
            red_cam.open(droidcam_url)
            red_assigned = True
            logging.info(f"Red goal: DroidCam {droidcam_ip}")

        if local_cams:
            blue_cam.open(local_cams[0]["id"])
            logging.info(f"Blue goal: {local_cams[0]['name']}")
            if not red_assigned and len(local_cams) >= 2:
                red_cam.open(local_cams[1]["id"])
                red_assigned = True
        elif not droidcam_url and cams:
            # Only IP cameras available; use first for blue
            resolved = _resolve_camera_src(cams[0]["id"])
            blue_cam.open(resolved)

        active = sum(1 for c in (blue_cam, red_cam) if c.is_open() or c._running)
        _set_status(f"{len(cams)} cameras found, {active} active", True)

    threading.Thread(target=auto_cameras, daemon=True).start()

    app.run(host="0.0.0.0", port=2016, threaded=True, debug=False)
