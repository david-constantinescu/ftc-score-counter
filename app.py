"""
DECODE Goal Scorer — Dual Camera Ball Counter (Tkinter + YOLOv8)
════════════════════════════════════════════════════════════════════
Two cameras placed above the goals (one RED, one BLUE).
Counts balls going into each goal regardless of color (green/purple).
Uses YOLOv8 for ball detection + HSV colour backup.
Maximum-performance tkinter UI with threaded capture & inference.
"""

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
import tkinter as tk
from tkinter import ttk, simpledialog
from collections import OrderedDict

os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)

# ── Auto-install dependencies ────────────────────────────────────────────────
def _install_deps():
    pkgs = {
        "opencv-python": "cv2",
        "pillow": "PIL",
        "numpy": "numpy",
        "ultralytics": "ultralytics",
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
from PIL import Image, ImageTk
from ultralytics import YOLO

# ── Constants ─────────────────────────────────────────────────────────────────
# HSV colour ranges for purple and green foam balls
# Broadened the ranges again to make sure both Green and Purple balls are easily picked up in any lighting
PURPLE_LOW  = np.array([125, 60, 50])
PURPLE_HIGH = np.array([160, 255, 255])
GREEN_LOW   = np.array([35,  60, 50])
GREEN_HIGH  = np.array([85,  255, 255])

# "Placed just above the corner of the goal" means balls fall through fast, 
# but making minimum size much bigger to avoid tiny false positive blurs.
MIN_BALL_AREA   = 6000     # Quite a lot bigger minimum size
MAX_BALL_AREA   = 40000    # Lowered so it doesn't count massive merged blobs
MIN_RADIUS      = 15
MAX_RADIUS      = 115      # Lowered appropriately with MAX_BALL_AREA
KERN_SIZE       = (7, 7)
MIN_CIRCULARITY = 0.0     # Ignored
CONFIRM_FRAMES  = 1        # Instant trigger

PROCESS_W, PROCESS_H = 640, 480   # processing resolution

# YOLO sports-ball class in COCO = 32
YOLO_BALL_CLASS = 32
YOLO_CONF       = 0.30


# ══════════════════════════════════════════════════════════════════════════════
#  Centroid Tracker
# ══════════════════════════════════════════════════════════════════════════════
class CentroidTracker:
    """Lightweight multi-object tracker by centroid distance."""

    # Balls pass very quickly so max_dist must be huge to prevent getting double-counted 
    # when the ball jumps a large portion of the frame in one tick
    def __init__(self, max_disappeared=45, max_dist=400):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_dist = max_dist
        self.ever_confirmed = set()   # IDs that passed CONFIRM_FRAMES

    def reset(self):
        self.next_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.ever_confirmed.clear()

    @property
    def confirmed_count(self):
        """How many unique balls have been confirmed (high-water)."""
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
#  Threaded Camera Capture  (always grabs the freshest frame)
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
        s = int(src) if isinstance(src, str) and src.isdigit() else src
        
        def connect():
            # On macOS, use AVFoundation directly to avoid out-of-bound or V4L warnings
            if sys.platform == "darwin" and isinstance(s, int):
                c = cv2.VideoCapture(s, cv2.CAP_AVFOUNDATION)
            else:
                c = cv2.VideoCapture(s)
            
            if c.isOpened():
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                c.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_W)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_H)
                return c
            return None

        self.cap = connect()
        if self.cap:
            self.is_ready = True
        else:
            self.is_ready = False
            # We don't abort instantly; we will retry in the loop if disconnected
            
        fails = 0
        while self._running:
            if not self.cap or not self.cap.isOpened():
                self.is_ready = False
                time.sleep(1.0)  # Wait before retrying connection
                self.cap = connect()
                if self.cap:
                    self.is_ready = True
                    fails = 0
                continue

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                fails = 0
            else:
                fails += 1
                if fails > 30:
                    # If we fail for ~30 consecutive attempts, assume disconnected and restart
                    self.cap.release()
                    self.cap = None
                    self.is_ready = False
                time.sleep(0.01)
                
        # Graceful background cleanup (prevents blocking the main GUI thread)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_ready = False

    def grab(self):
        """Return the latest frame (or None)."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def is_open(self):
        return self.is_ready and self._running

    def stop(self):
        # Set running false to break loop. 
        # Do NOT release the camera here, let the background thread do it to prevent GUI freezes.
        self._running = False
        with self.lock:
            self.frame = None


# ══════════════════════════════════════════════════════════════════════════════
#  Ball Detector  (YOLOv8 + HSV hybrid)
# ══════════════════════════════════════════════════════════════════════════════
class BallDetector:
    def __init__(self):
        self.yolo = None
        self.ready = False

    def load(self, callback=None):
        """Load YOLOv8n model (blocking — call from background thread)."""
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
            self.yolo = YOLO(model_path)
            # warm-up run
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

    # ── YOLO detection ───────────────────────────────────────────────────
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
                w, h = x2 - x1, y2 - y1
                # Reject boxes that are too small or too large (likely merged balls)
                if w < MIN_RADIUS * 2 or h < MIN_RADIUS * 2:
                    continue
                if w > MAX_RADIUS * 2 or h > MAX_RADIUS * 2:
                    continue
                # Reject non-square-ish boxes (aspect ratio filter)
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect > 2.0:
                    continue
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                radius = int(max(w, h) / 2)
                out.append({"x": cx, "y": cy, "radius": radius, "src": "yolo"})
        return out

    # ── HSV detection ────────────────────────────────────────────────────
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
            
            # Since we just want color and NOT shape, skip the circularity entirely
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            out.append({"x": int(cx), "y": int(cy), "radius": int(r), "src": "hsv"})
        return out

    # ── Combined detection ───────────────────────────────────────────────
    def detect(self, frame, use_yolo=True):
        """Detect balls. Returns list of {x, y, radius}."""
        yol_dets = []
        if use_yolo and self.ready:
            yol_dets = self._yolo_detect(frame)

        # Always run HSV (fast, catches coloured foam balls reliably)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_dets = []
        hsv_dets += self._hsv_detect(hsv, PURPLE_LOW, PURPLE_HIGH)
        hsv_dets += self._hsv_detect(hsv, GREEN_LOW, GREEN_HIGH)

        # Merge them: Prefer YOLO. Add HSV only if it doesn't overlap YOLO.
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
#  Audio Player (macOS afplay)
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
        if not self.path:
            return
        try:
            self.proc = subprocess.Popen(
                ["afplay", self.path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.paused = False
        except Exception as e:
            logging.error(f"Audio play: {e}")

    def pause(self):
        if self.proc and self.proc.poll() is None and not self.paused:
            try:
                os.kill(self.proc.pid, signal.SIGSTOP)
                self.paused = True
            except Exception:
                pass

    def resume(self):
        if self.proc and self.paused:
            try:
                os.kill(self.proc.pid, signal.SIGCONT)
                self.paused = False
            except Exception:
                pass

    def stop(self):
        if self.proc:
            try:
                if self.paused:
                    os.kill(self.proc.pid, signal.SIGCONT)
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
            self.paused = False


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════
class GoalScorerApp:
    # Colours
    BG      = "#0f0f23"
    FG      = "#e0e0e0"
    BLUE_C  = "#42a5f5"
    RED_C   = "#ef5350"
    ACCENT  = "#ffd740"
    DARK    = "#1a1a2e"

    def __init__(self, root):
        self.root = root
        self.root.title("DECODE Goal Scorer")
        self.root.geometry("1340x760")
        self.root.configure(bg=self.BG)
        self.root.minsize(1000, 600)

        self.detector = BallDetector()

        self.blue_cam = CameraThread()
        self.red_cam  = CameraThread()
        self.blue_tracker = CentroidTracker()
        self.red_tracker  = CentroidTracker()

        # High-water ball counts (never decrease until reset)
        self.blue_hw = 0
        self.red_hw  = 0

        # Live count (current visible confirmed balls)
        self.blue_live = 0
        self.red_live  = 0

        # Timer
        self.timer_seconds = 150
        self.timer_running = False
        self._timer_started = False

        self.audio = AudioPlayer()

        # Frame counter for YOLO scheduling
        self._frame_no = 0
        self._yolo_every = 3    # run YOLO every N frames

        # FPS
        self._fps = 0.0

        # Tk image references (prevent GC)
        self._blue_imgtk = None
        self._red_imgtk  = None

        self._build_ui()

        # Load YOLO in background
        self._set_status("Loading YOLOv8 …", False)
        threading.Thread(target=self._load_yolo, daemon=True).start()

        # Scan cameras after UI is shown
        self.root.after(400, self._refresh_cameras)

    # ── YOLO load ─────────────────────────────────────────────────────────
    def _load_yolo(self):
        self.detector.load(
            callback=lambda ok, msg: self.root.after(
                0, lambda: self._set_status(msg, ok))
        )

    # ══════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure(".", background=self.BG, foreground=self.FG)
        sty.configure("TFrame", background=self.BG)
        sty.configure("TLabel", background=self.BG, foreground=self.FG)
        sty.configure("TButton", padding=4)
        sty.configure("TLabelframe", background=self.BG, foreground=self.FG)
        sty.configure("TLabelframe.Label", background=self.BG,
                       foreground=self.ACCENT, font=("Helvetica", 11, "bold"))

        # ── Top bar: camera selectors + timer ─────────────────────────────
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)

        # BLUE camera
        ttk.Label(top, text="BLUE Goal Cam:",
                  foreground=self.BLUE_C,
                  font=("Helvetica", 11, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.blue_cam_cb = ttk.Combobox(top, width=26)
        self.blue_cam_cb.pack(side=tk.LEFT, padx=2)
        self.blue_cam_cb.bind("<<ComboboxSelected>>", self._on_blue_cam)
        self.blue_cam_cb.bind("<Return>", self._on_blue_cam)

        ttk.Label(top, text="  ").pack(side=tk.LEFT)

        # RED camera
        ttk.Label(top, text="RED Goal Cam:",
                  foreground=self.RED_C,
                  font=("Helvetica", 11, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.red_cam_cb = ttk.Combobox(top, width=26)
        self.red_cam_cb.pack(side=tk.LEFT, padx=2)
        self.red_cam_cb.bind("<<ComboboxSelected>>", self._on_red_cam)
        self.red_cam_cb.bind("<Return>", self._on_red_cam)

        ttk.Button(top, text="Refresh", width=8,
                   command=self._refresh_cameras).pack(side=tk.LEFT, padx=8)

        # Timer
        ttk.Label(top, text=" | ", foreground="gray").pack(side=tk.LEFT, padx=2)

        self.lbl_timer = tk.Label(
            top, text="2:30", font=("Helvetica", 28, "bold"),
            fg=self.ACCENT, bg=self.DARK, width=6, anchor="center",
            relief="groove", bd=2)
        self.lbl_timer.pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Start / Pause", width=12,
                   command=self._toggle_timer).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Reset Timer", width=10,
                   command=self._reset_timer).pack(side=tk.LEFT, padx=3)

        ttk.Label(top, text=" | ", foreground="gray").pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Reset All", width=9,
                   command=self._reset_all).pack(side=tk.LEFT, padx=3)

        # ── Main area: two panels side-by-side ────────────────────────────
        main = ttk.Frame(self.root, padding=4)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # ── BLUE panel ────────────────────────────────────────────────────
        blue_panel = ttk.LabelFrame(main, text="  BLUE GOAL  ", padding=4)
        blue_panel.grid(row=0, column=0, sticky="nsew", padx=(2, 4), pady=2)
        blue_panel.rowconfigure(0, weight=1)
        blue_panel.columnconfigure(0, weight=1)

        self.blue_canvas = tk.Canvas(blue_panel, bg="black",
                                      highlightthickness=0)
        self.blue_canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self._blue_img_id = None

        blue_score_frame = tk.Frame(blue_panel, bg=self.DARK)
        blue_score_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.blue_score_lbl = tk.Label(
            blue_score_frame, text="0", font=("Helvetica", 72, "bold"),
            fg=self.BLUE_C, bg=self.DARK, anchor="center")
        self.blue_score_lbl.pack(fill=tk.X)

        self.blue_info_lbl = tk.Label(
            blue_score_frame, text="balls scored  |  live: 0",
            font=("Helvetica", 13), fg="#90caf9", bg=self.DARK)
        self.blue_info_lbl.pack()

        blue_btn_row = tk.Frame(blue_panel, bg=self.BG)
        blue_btn_row.grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(blue_btn_row, text="+1", width=4,
                   command=lambda: self._manual_adj("blue", 1)).pack(
                       side=tk.LEFT, padx=4)
        ttk.Button(blue_btn_row, text="-1", width=4,
                   command=lambda: self._manual_adj("blue", -1)).pack(
                       side=tk.LEFT, padx=2)
        ttk.Button(blue_btn_row, text="Clear Blue", width=10,
                   command=self._clear_blue).pack(side=tk.LEFT, padx=8)

        # ── RED panel ─────────────────────────────────────────────────────
        red_panel = ttk.LabelFrame(main, text="  RED GOAL  ", padding=4)
        red_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 2), pady=2)
        red_panel.rowconfigure(0, weight=1)
        red_panel.columnconfigure(0, weight=1)

        self.red_canvas = tk.Canvas(red_panel, bg="black",
                                     highlightthickness=0)
        self.red_canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self._red_img_id = None

        red_score_frame = tk.Frame(red_panel, bg=self.DARK)
        red_score_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.red_score_lbl = tk.Label(
            red_score_frame, text="0", font=("Helvetica", 72, "bold"),
            fg=self.RED_C, bg=self.DARK, anchor="center")
        self.red_score_lbl.pack(fill=tk.X)

        self.red_info_lbl = tk.Label(
            red_score_frame, text="balls scored  |  live: 0",
            font=("Helvetica", 13), fg="#ef9a9a", bg=self.DARK)
        self.red_info_lbl.pack()

        red_btn_row = tk.Frame(red_panel, bg=self.BG)
        red_btn_row.grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(red_btn_row, text="+1", width=4,
                   command=lambda: self._manual_adj("red", 1)).pack(
                       side=tk.LEFT, padx=4)
        ttk.Button(red_btn_row, text="-1", width=4,
                   command=lambda: self._manual_adj("red", -1)).pack(
                       side=tk.LEFT, padx=2)
        ttk.Button(red_btn_row, text="Clear Red", width=10,
                   command=self._clear_red).pack(side=tk.LEFT, padx=8)

        # ── Status bar ───────────────────────────────────────────────────
        sb = tk.Frame(self.root, bg=self.DARK, height=26)
        sb.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_dot = tk.Canvas(sb, width=14, height=14,
                                     bg=self.DARK, highlightthickness=0)
        self.status_dot.pack(side=tk.LEFT, padx=6, pady=4)
        self._dot = self.status_dot.create_oval(2, 2, 12, 12, fill="gray")

        self.status_lbl = tk.Label(sb, text="Initializing …",
                                    font=("Helvetica", 10), fg="gray",
                                    bg=self.DARK)
        self.status_lbl.pack(side=tk.LEFT)

        self.fps_lbl = tk.Label(sb, text="", font=("Helvetica", 10),
                                 fg="gray", bg=self.DARK)
        self.fps_lbl.pack(side=tk.RIGHT, padx=8)

    # ══════════════════════════════════════════════════════════════════════
    #  Status helper
    # ══════════════════════════════════════════════════════════════════════
    def _set_status(self, msg, ok=True):
        self.status_dot.itemconfig(self._dot,
                                    fill="#4caf50" if ok else "#e53935")
        self.status_lbl.configure(text=msg)
    # ══════════════════════════════════════════════════════════════════════
    #  Camera management
    # ══════════════════════════════════════════════════════════════════════
    def _scan_cameras(self):
        cams = []
        # Suppress errors by querying macOS native system profiler first
        if sys.platform == "darwin":
            try:
                p = subprocess.run(
                    ["system_profiler", "SPCameraDataType", "-json"],
                    capture_output=True, text=True, timeout=5)
                if p.returncode == 0:
                    items = json.loads(p.stdout).get("SPCameraDataType", [])
                    for i, item in enumerate(items):
                        cams.append(f"{i}: {item.get('_name', f'Camera {i}')}")
                    if cams:
                        return cams
            except Exception:
                pass

        # Fallback probe - keep small to avoid spamming the console
        for idx in range(3):
            try:
                # To suppress stdout errors inside OpenCV when failing
                import os
                old_stderr = os.dup(2)
                f = open(os.devnull, 'w')
                os.dup2(f.fileno(), 2)
                
                cap = cv2.VideoCapture(idx)
                if cap.isOpened() and cap.read()[0]:
                    cams.append(f"{idx}: Camera {idx}")
                cap.release()
                
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
                f.close()
            except Exception:
                pass
        return cams or ["0: Default Camera"]

    def _refresh_cameras(self):
        self._set_status("Scanning cameras …", False)

        def scan():
            cams = self._scan_cameras()
            self.root.after(0, lambda: self._populate_cams(cams))

        threading.Thread(target=scan, daemon=True).start()

    def _populate_cams(self, cams):
        choices = ["None (disabled)"] + cams + ["Add URL…"]
        self.blue_cam_cb["values"] = choices
        self.red_cam_cb["values"] = choices
        self.blue_cam_cb.current(0)
        self.red_cam_cb.current(0)

        # Auto-assign if 2+ cams
        if len(cams) >= 2:
            self.blue_cam_cb.current(1)
            self.red_cam_cb.current(2)
            self._start_cam("blue", int(cams[0].split(":")[0]))
            self._start_cam("red",  int(cams[1].split(":")[0]))
        elif len(cams) == 1:
            self.blue_cam_cb.current(1)
            self._start_cam("blue", int(cams[0].split(":")[0]))

        active = sum(1 for c in [self.blue_cam, self.red_cam] if c.is_open())
        self._set_status(f"{len(cams)} camera(s) found, {active} active", True)

        # Start the video loop
        if not hasattr(self, "_loop_active"):
            self._loop_active = True
            self._video_loop()

    def _parse_cam_val(self, val):
        if not val or val.startswith("None"):
            return None
        val = val.strip()
        if ":" in val and val.split(":")[0].strip().isdigit() and "http" not in val:
            return int(val.split(":")[0])
        if val.isdigit():
            return int(val)
        return val

    def _on_blue_cam(self, _=None):
        val = self.blue_cam_cb.get()
        if val == "Add URL…":
            url = simpledialog.askstring("Camera URL", "Enter URL or path:")
            if url and url.strip():
                vals = list(self.blue_cam_cb["values"])
                if "Add URL…" in vals:
                    vals.remove("Add URL…")
                vals.append(url.strip())
                vals.append("Add URL…")
                self.blue_cam_cb["values"] = vals
                self.blue_cam_cb.set(url.strip())
                val = url.strip()
            else:
                self.blue_cam_cb.set("None (disabled)")
                self.blue_cam.stop()
                return
        src = self._parse_cam_val(val)
        if src is None:
            self.blue_cam.stop()
            self.blue_tracker.reset()
        else:
            self._start_cam("blue", src)

    def _on_red_cam(self, _=None):
        val = self.red_cam_cb.get()
        if val == "Add URL…":
            url = simpledialog.askstring("Camera URL", "Enter URL or path:")
            if url and url.strip():
                vals = list(self.red_cam_cb["values"])
                if "Add URL…" in vals:
                    vals.remove("Add URL…")
                vals.append(url.strip())
                vals.append("Add URL…")
                self.red_cam_cb["values"] = vals
                self.red_cam_cb.set(url.strip())
                val = url.strip()
            else:
                self.red_cam_cb.set("None (disabled)")
                self.red_cam.stop()
                return
        src = self._parse_cam_val(val)
        if src is None:
            self.red_cam.stop()
            self.red_tracker.reset()
        else:
            self._start_cam("red", src)

    def _start_cam(self, side, src):
        cam = self.blue_cam if side == "blue" else self.red_cam
        tracker = self.blue_tracker if side == "blue" else self.red_tracker
        tracker.reset()
        cam.open(src)
        self._set_status(f"Loading {side.upper()} camera ({src})...", True)

    # ══════════════════════════════════════════════════════════════════════
    #  Timer
    # ══════════════════════════════════════════════════════════════════════
    def _toggle_timer(self):
        if not self.timer_running and self.timer_seconds == 150:
            # Fresh start — play audio, 3 s lead-in
            self._set_status("Match audio started — timer in 3 s", True)
            self.audio.play()
            self.timer_running = True
            self._timer_started = False
            self.root.after(3000, self._begin_countdown)
            return

        if self.timer_running:
            self.timer_running = False
            self.audio.pause()
        else:
            self.timer_running = True
            self._timer_started = True
            self.audio.resume()
            self._tick()

    def _begin_countdown(self):
        if self.timer_running:
            self._timer_started = True
            self._set_status("Match started!", True)
            self._tick()

    def _tick(self):
        if not self.timer_running or not self._timer_started:
            return
        if self.timer_seconds > 0:
            self.timer_seconds -= 1
            m, s = divmod(self.timer_seconds, 60)
            colour = "#ff1744" if self.timer_seconds <= 30 else self.ACCENT
            self.lbl_timer.configure(text=f"{m}:{s:02d}", fg=colour)
            self.root.after(1000, self._tick)
        else:
            self.timer_running = False
            self.lbl_timer.configure(text="0:00", fg="#ff1744")
            self.audio.stop()
            self._set_status("Match ended!", True)

    def _reset_timer(self):
        self.timer_running = False
        self._timer_started = False
        self.timer_seconds = 150
        self.lbl_timer.configure(text="2:30", fg=self.ACCENT)
        self.audio.stop()

    # ══════════════════════════════════════════════════════════════════════
    #  Score management
    # ══════════════════════════════════════════════════════════════════════
    def _manual_adj(self, side, delta):
        if side == "blue":
            self.blue_hw = max(0, self.blue_hw + delta)
        else:
            self.red_hw = max(0, self.red_hw + delta)
        self._update_scores()

    def _clear_blue(self):
        self.blue_tracker.reset()
        self.blue_hw = 0
        self.blue_live = 0
        self._update_scores()
        self._set_status("Blue scores cleared", True)

    def _clear_red(self):
        self.red_tracker.reset()
        self.red_hw = 0
        self.red_live = 0
        self._update_scores()
        self._set_status("Red scores cleared", True)

    def _reset_all(self):
        self.blue_tracker.reset()
        self.red_tracker.reset()
        self.blue_hw = self.red_hw = 0
        self.blue_live = self.red_live = 0
        self._reset_timer()
        self._update_scores()
        self._set_status("All scores reset", True)

    def _update_scores(self):
        self.blue_score_lbl.configure(text=str(self.blue_hw))
        self.blue_info_lbl.configure(
            text=f"balls scored  |  live: {self.blue_live}")
        self.red_score_lbl.configure(text=str(self.red_hw))
        self.red_info_lbl.configure(
            text=f"balls scored  |  live: {self.red_live}")

    # ══════════════════════════════════════════════════════════════════════
    #  Video Loop  (processes both cameras each tick)
    # ══════════════════════════════════════════════════════════════════════
    def _video_loop(self):
        t0 = time.time()
        self._frame_no += 1
        # Set to False completely so we NEVER wait for the YOLO shape model. Just pure HSV color!
        use_yolo = False

        # Process BLUE
        if self.blue_cam.is_open():
            frame = self.blue_cam.grab()
            if frame is not None:
                frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))
                self._process_frame(frame, self.blue_tracker, "blue", use_yolo)
        else:
            msg = "Loading..." if self.blue_cam._running else "No BLUE camera"
            self._show_placeholder(self.blue_canvas, msg)

        # Process RED
        if self.red_cam.is_open():
            frame = self.red_cam.grab()
            if frame is not None:
                frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))
                self._process_frame(frame, self.red_tracker, "red", use_yolo)
        else:
            msg = "Loading..." if self.red_cam._running else "No RED camera"
            self._show_placeholder(self.red_canvas, msg)

        # FPS
        dt = time.time() - t0
        self._fps = 0.8 * self._fps + 0.2 * (1.0 / max(dt, 0.001))
        self.fps_lbl.configure(text=f"FPS: {self._fps:.0f}")

        # Schedule next tick (run as fast as possible, target ~120 fps to not miss fast balls)
        delay = max(1, 8 - int(dt * 1000))
        self.root.after(delay, self._video_loop)

    def _process_frame(self, frame, tracker, side, use_yolo):
        """Detect balls, update tracker & counts, draw overlays, display."""
        dets = self.detector.detect(frame, use_yolo=use_yolo)
        tracked = tracker.update(dets)

        # Count confirmed balls currently visible
        live_count = sum(1 for oid in tracked if oid in tracker.ever_confirmed)

        # High-water: total unique balls ever confirmed
        hw = tracker.confirmed_count

        if side == "blue":
            self.blue_live = live_count
            if hw > self.blue_hw:
                self.blue_hw = hw
        else:
            self.red_live = live_count
            if hw > self.red_hw:
                self.red_hw = hw

        self._update_scores()

        # Draw overlays
        for oid, obj in tracked.items():
            x, y = obj.get("x", 0), obj.get("y", 0)
            r = max(obj.get("radius", 16), 10)
            confirmed = oid in tracker.ever_confirmed
            colour = (0, 255, 0) if confirmed else (0, 180, 255)
            thickness = 3 if confirmed else 1
            cv2.circle(frame, (x, y), r, colour, thickness)
            if confirmed:
                cv2.putText(frame, str(oid), (x - 8, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        # Ball count overlay on frame
        hw_val = self.blue_hw if side == "blue" else self.red_hw
        label = f"Balls: {hw_val}  (live {live_count})"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        self._show_frame(frame, side)

    # ── Canvas rendering ─────────────────────────────────────────────────
    def _show_frame(self, frame, side):
        canvas = self.blue_canvas if side == "blue" else self.red_canvas
        try:
            if not canvas.winfo_exists():
                return
            canvas.delete("placeholder")
            cw = max(canvas.winfo_width(), 320)
            ch = max(canvas.winfo_height(), 240)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            scale = min(cw / img.width, ch / img.height)
            nw = max(1, int(img.width * scale))
            nh = max(1, int(img.height * scale))
            img = img.resize((nw, nh), Image.Resampling.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=img)

            if side == "blue":
                self._blue_imgtk = imgtk
                if self._blue_img_id is None:
                    self._blue_img_id = canvas.create_image(
                        cw // 2, ch // 2, anchor=tk.CENTER, image=imgtk)
                else:
                    canvas.coords(self._blue_img_id, cw // 2, ch // 2)
                    canvas.itemconfig(self._blue_img_id, image=imgtk)
            else:
                self._red_imgtk = imgtk
                if self._red_img_id is None:
                    self._red_img_id = canvas.create_image(
                        cw // 2, ch // 2, anchor=tk.CENTER, image=imgtk)
                else:
                    canvas.coords(self._red_img_id, cw // 2, ch // 2)
                    canvas.itemconfig(self._red_img_id, image=imgtk)
        except tk.TclError:
            pass

    def _show_placeholder(self, canvas, text):
        try:
            if not canvas.winfo_exists():
                return
            cw = max(canvas.winfo_width(), 320)
            ch = max(canvas.winfo_height(), 240)
            canvas.delete("placeholder")
            canvas.create_text(cw // 2, ch // 2, text=text,
                               fill="gray", font=("Helvetica", 16),
                               tags="placeholder")
        except tk.TclError:
            pass

    # ── Cleanup ──────────────────────────────────────────────────────────
    def shutdown(self):
        self.blue_cam.stop()
        self.red_cam.stop()
        self.audio.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = GoalScorerApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
    root.mainloop()
