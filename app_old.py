"""
FTC DECODE Season Field Scorer
─────────────────────────────────────────────────────────────────────────────
Uses YOLO object detection + HSV color filtering to:
  1. Count purple/green ARTIFACTS (balls) scored into each alliance GOAL
  2. Check if the RAMP pattern matches the user-selected MOTIF
  3. Score BLUE and RED alliance zones separately with live counters
  4. Separate Autonomous and TeleOp scoring periods
  5. Check robot parking via GLM-4V-Flash vision API

Game reference (FTC DECODE – presented by RTX):
  - ARTIFACTS are ~5-inch purple (P) and green (G) foam balls
  - Autonomous: CLASSIFIED -> 6 pts, OVERFLOW -> 2 pts
  - TeleOp:     CLASSIFIED -> 3 pts, OVERFLOW -> 1 pt
  - MOTIF is one of GPP / PGP / PPG, repeated x3 for 9 RAMP slots
  - PATTERN = each RAMP position matching the MOTIF index -> 2 pts each
  - PARKING = robot in observation zone -> 3 pts per robot
"""

import sys
import os
import re as _re
import warnings
import subprocess
import importlib.util
import logging
import threading
import time
import json
import base64
import signal
import tkinter as tk
from tkinter import ttk, messagebox
from collections import OrderedDict

os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ── Auto-install dependencies ────────────────────────────────────────────────
def install_deps():
    pkgs = {
        "opencv-python": "cv2",
        "pillow": "PIL",
        "numpy": "numpy",
        "ultralytics": "ultralytics",
        "zhipuai": "zhipuai",
    }
    missing = [p for p, m in pkgs.items() if importlib.util.find_spec(m) is None]
    if missing:
        print(f"Installing missing packages: {missing} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )

install_deps()

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from zhipuai import ZhipuAI

# ── Game constants ────────────────────────────────────────────────────────────
MOTIFS = ["GPP", "PGP", "PPG"]
MAX_RAMP = 9

# Autonomous period points
AUTO_PTS_CLASSIFIED = 6
AUTO_PTS_OVERFLOW   = 2

# TeleOp period points
TELEOP_PTS_CLASSIFIED = 3
TELEOP_PTS_OVERFLOW   = 1

# End Game / Pattern
PTS_PATTERN = 2               # per matching RAMP position
PTS_PARKING = 3               # per robot parked in observation zone

# HSV colour ranges  (tune via UI if lighting differs)
PURPLE_LOW  = np.array([110, 40, 40])
PURPLE_HIGH = np.array([165, 255, 255])
GREEN_LOW   = np.array([35, 40, 40])
GREEN_HIGH  = np.array([85, 255, 255])

MIN_BALL_AREA    = 600   # Increased to reduce noise
MIN_CIRCULARITY  = 0.50  # Slightly stricter
CONFIRM_FRAMES   = 10    # Increased stability (was 3)

# ZhipuAI API
ZHIPU_API_KEY = "550c508378bb491ab596b75b93ef5e9d.x6MyLwFvHgJ7aMd1"


# ══════════════════════════════════════════════════════════════════════════════
#  Ball Detector  (YOLO + HSV hybrid)
# ══════════════════════════════════════════════════════════════════════════════
class BallDetector:
    """Detects purple / green balls using YOLO object detection enhanced with
    HSV colour classification."""

    def __init__(self):
        self.model = None
        self.model_ready = False
        self.model_name = "not loaded"

    def load_model(self, callback=None):
        try:
            self.model = YOLO("yolov8n.pt")
            self.model_ready = True
            self.model_name = "YOLOv8n"
            logging.info("YOLO model loaded")
            if callback:
                callback(True, f"Model ready: {self.model_name}")
        except Exception as e:
            logging.warning(f"YOLO load failed — HSV-only mode: {e}")
            self.model_ready = False
            self.model_name = "HSV-only (YOLO unavailable)"
            if callback:
                callback(True, self.model_name)

    def detect(self, frame, run_yolo=True):
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections += self._hsv_detect(hsv, PURPLE_LOW, PURPLE_HIGH, "P")
        detections += self._hsv_detect(hsv, GREEN_LOW, GREEN_HIGH, "G")
        if run_yolo and self.model_ready:
            detections = self._yolo_enhance(frame, hsv, detections)
        return detections

    def _hsv_detect(self, hsv, low, high, label):
        mask = cv2.inRange(hsv, low, high)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BALL_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < MIN_CIRCULARITY:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            out.append({"x": int(cx), "y": int(cy), "radius": int(r), "color": label})
        return out

    def _yolo_enhance(self, frame, hsv, existing):
        try:
            results = self.model(frame, verbose=False, conf=0.30)
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) != 32:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    rad = max(x2 - x1, y2 - y1) // 2
                    if any(np.hypot(d["x"] - cx, d["y"] - cy) < rad for d in existing):
                        continue
                    roi = hsv[max(0, y1):y2, max(0, x1):x2]
                    if roi.size == 0:
                        continue
                    p = np.count_nonzero(cv2.inRange(roi, PURPLE_LOW, PURPLE_HIGH)) / roi.size
                    g = np.count_nonzero(cv2.inRange(roi, GREEN_LOW, GREEN_HIGH)) / roi.size
                    if p > 0.15 and p > g:
                        col = "P"
                    elif g > 0.15:
                        col = "G"
                    else:
                        continue
                    existing.append({"x": cx, "y": cy, "radius": rad, "color": col})
        except Exception as e:
            logging.debug(f"YOLO inference error: {e}")
        return existing


# ══════════════════════════════════════════════════════════════════════════════
#  Centroid Tracker
# ══════════════════════════════════════════════════════════════════════════════
class CentroidTracker:
    def __init__(self, max_disappeared=20, max_dist=80):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_dist = max_dist
        self.confirmed = OrderedDict()

    def reset(self):
        self.next_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.confirmed.clear()

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
        for flat in np.argsort(D, axis=None):
            r, c = divmod(int(flat), D.shape[1])
            if r in used_r or c in used_c:
                continue
            if D[r, c] > self.max_dist:
                break
            oid = ids[r]
            self.objects[oid].update(dets[c])
            self.objects[oid]["age"] = self.objects[oid].get("age", 0) + 1
            self.disappeared[oid] = 0
            if self.objects[oid]["age"] >= CONFIRM_FRAMES and oid not in self.confirmed:
                self.confirmed[oid] = self.objects[oid]["color"]
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
#  Zone Score Data
# ══════════════════════════════════════════════════════════════════════════════
class ZoneData:
    def __init__(self, name):
        self.name = name
        # Autonomous
        self.auto_classified = 0
        self.auto_overflow = 0
        self.auto_total = 0
        self.auto_ramp_colors = []
        # TeleOp
        self.teleop_classified = 0
        self.teleop_overflow = 0
        self.teleop_total = 0
        # Current (live detection)
        self.classified = 0
        self.overflow = 0
        self.total_balls = 0
        self.ramp_colors = []
        self.pattern_matches = 0
        # Parking
        self.parking_status = "Not checked"
        self.parking_pts = 0

    def reset(self):
        self.auto_classified = self.auto_overflow = self.auto_total = 0
        self.auto_ramp_colors = []
        self.teleop_classified = self.teleop_overflow = self.teleop_total = 0
        self.classified = self.overflow = self.total_balls = 0
        self.ramp_colors = []
        self.pattern_matches = 0
        self.parking_status = "Not checked"
        self.parking_pts = 0

    def lock_auto(self):
        """Snapshot current live counts as Autonomous scores."""
        self.auto_classified = self.classified
        self.auto_overflow = self.overflow
        self.auto_total = self.total_balls
        self.auto_ramp_colors = list(self.ramp_colors)

    def compute_teleop(self):
        """TeleOp = current totals minus what was scored during Auto."""
        self.teleop_total = max(0, self.total_balls - self.auto_total)
        self.teleop_classified = max(0, self.classified - self.auto_classified)
        self.teleop_overflow = max(0, self.overflow - self.auto_overflow)

    def auto_score(self):
        return self.auto_classified * AUTO_PTS_CLASSIFIED + self.auto_overflow * AUTO_PTS_OVERFLOW

    def teleop_score(self):
        return self.teleop_classified * TELEOP_PTS_CLASSIFIED + self.teleop_overflow * TELEOP_PTS_OVERFLOW

    def total_score(self):
        return (self.auto_score() + self.teleop_score()
                + self.pattern_matches * PTS_PATTERN + self.parking_pts)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════
class FTCDecodeApp:

    def __init__(self, root):
        self.root = root
        self.root.title("FTC DECODE Scorer — YOLO + GLM-4V")
        self.root.geometry("1440x920")
        self.root.minsize(1100, 750)

        self.detector = BallDetector()
        self.tracker = CentroidTracker()

        self.cap = None
        self.is_running = False
        self.camera_index = 0
        self.frame_no = 0
        self.yolo_every = 5
        self.last_frame = None          # store latest frame for parking check

        self.blue = ZoneData("Blue")
        self.red = ZoneData("Red")
        self.split_x = 0.5

        # Timer
        self.timer_seconds = 150
        self.timer_running = False

        # Audio
        self._audio_proc = None
        self._audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ftctimer.mp3")

        self._build_ui()
        self._update_motif_display()

        self.set_status("Loading YOLO model ...", False)
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(600, self._init_camera)

    def _load_model(self):
        self.detector.load_model(
            callback=lambda ok, msg: self.root.after(0, lambda: self.set_status(msg, ok))
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure("Big.TLabel",   font=("Helvetica", 30, "bold"))
        sty.configure("Med.TLabel",   font=("Helvetica", 13))
        sty.configure("Sm.TLabel",    font=("Helvetica", 10))
        sty.configure("Ramp.TLabel",  font=("Courier", 12, "bold"))

        # ── Row 1: Camera + MOTIF + Timer + Tracking ─────────────────────
        row1 = ttk.Frame(self.root, padding=4)
        row1.pack(fill=tk.X)

        ttk.Label(row1, text="Camera:").pack(side=tk.LEFT, padx=(5, 2))
        self.cam_cb = ttk.Combobox(row1, state="readonly", width=28)
        self.cam_cb.pack(side=tk.LEFT, padx=2)
        self.cam_cb.bind("<<ComboboxSelected>>", self._on_cam_change)
        ttk.Button(row1, text="Refresh", width=7,
                   command=self._refresh_cameras).pack(side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(row1, text="MOTIF:").pack(side=tk.LEFT, padx=(2, 2))
        self.motif_var = tk.StringVar(value="GPP")
        self.motif_cb = ttk.Combobox(row1, textvariable=self.motif_var,
                                      values=MOTIFS, state="readonly", width=5)
        self.motif_cb.pack(side=tk.LEFT, padx=2)
        self.motif_cb.bind("<<ComboboxSelected>>",
                           lambda _: self._update_motif_display())

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Timer
        self.lbl_timer = ttk.Label(
            row1, text="2:30", font=("Helvetica", 16, "bold"),
            width=5, anchor=tk.CENTER)
        self.lbl_timer.pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Start/Pause", width=10,
                   command=self._toggle_timer).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Reset", width=5,
                   command=self._reset_timer).pack(side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Tracking checkboxes
        self.var_score_blue = tk.BooleanVar(value=True)  # Default: True
        ttk.Checkbutton(row1, text="Track BLUE (L)",
                        variable=self.var_score_blue).pack(side=tk.LEFT, padx=3)
        self.var_score_red = tk.BooleanVar(value=True)   # Default: True
        ttk.Checkbutton(row1, text="Track RED (R)",
                        variable=self.var_score_red).pack(side=tk.LEFT, padx=3)

        # ── Row 2: Period controls + Actions ─────────────────────────────
        row2 = ttk.Frame(self.root, padding=2)
        row2.pack(fill=tk.X)

        # (Hidden) buttons that are now automated:
        # ttk.Button(row2, text="Lock AUTO Scores", command=self._lock_auto).pack(...)
        # ttk.Button(row2, text="Lock TELEOP Scores", command=self._lock_teleop).pack(...)
        # ttk.Button(row2, text="Check Parking (AI)", command=self._check_parking).pack(...)

        ttk.Button(row2, text="Snapshot Pattern",
                   command=self._snapshot_pattern).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Reset All",
                   command=self._reset_all).pack(side=tk.LEFT, padx=4)
        
        # Add a clear label explaining automation
        ttk.Label(row2, text="(Auto/TeleOp/Parking are automated via Timer)", 
                  foreground="gray").pack(side=tk.LEFT, padx=5)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        # Manual adjust
        self.adj_zone = tk.StringVar(value="blue")
        ttk.Radiobutton(row2, text="Blue", variable=self.adj_zone,
                        value="blue").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(row2, text="Red", variable=self.adj_zone,
                        value="red").pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="+1", width=3,
                   command=lambda: self._man_adj(1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(row2, text="-1", width=3,
                   command=lambda: self._man_adj(-1)).pack(side=tk.LEFT, padx=1)

        # ── Main area ────────────────────────────────────────────────────
        main = ttk.Frame(self.root, padding=4)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: video
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._canvas_img_id = None

        info = ttk.Frame(left)
        info.pack(fill=tk.X, padx=4)
        self.det_lbl = ttk.Label(info, text="Detected: --", style="Med.TLabel")
        self.det_lbl.pack(side=tk.LEFT)
        self.split_val = tk.DoubleVar(value=0.5)
        ttk.Scale(info, from_=0.1, to=0.9, variable=self.split_val,
                  command=self._on_split_change).pack(side=tk.RIGHT, padx=4)
        ttk.Label(info, text="Split:").pack(side=tk.RIGHT)

        # Right: score panels
        right = ttk.Frame(main, width=420)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=4)
        right.pack_propagate(False)

        # Motif display
        mf = ttk.LabelFrame(right, text="MOTIF Pattern (x3 = 9 slots)", padding=6)
        mf.pack(fill=tk.X, pady=(0, 3))
        self.motif_disp = ttk.Label(mf, text="", style="Ramp.TLabel")
        self.motif_disp.pack()

        # BLUE zone
        bf = ttk.LabelFrame(right, text="  BLUE Alliance", padding=8)
        bf.pack(fill=tk.X, pady=3)
        self.b_total = ttk.Label(bf, text="0", style="Big.TLabel",
                                 foreground="#1565C0")
        self.b_total.pack()
        self.b_auto = ttk.Label(
            bf, text=f"AUTO:    Cls 0x{AUTO_PTS_CLASSIFIED}  "
                     f"Ovf 0x{AUTO_PTS_OVERFLOW}  = 0", style="Sm.TLabel")
        self.b_auto.pack(anchor=tk.W)
        self.b_teleop = ttk.Label(
            bf, text=f"TELEOP:  Cls 0x{TELEOP_PTS_CLASSIFIED}  "
                     f"Ovf 0x{TELEOP_PTS_OVERFLOW}  = 0", style="Sm.TLabel")
        self.b_teleop.pack(anchor=tk.W)
        self.b_pattern = ttk.Label(
            bf, text=f"PATTERN: 0/0 x{PTS_PATTERN}  = 0", style="Sm.TLabel")
        self.b_pattern.pack(anchor=tk.W)
        self.b_parking = ttk.Label(
            bf, text="PARKING: Not checked", style="Sm.TLabel")
        self.b_parking.pack(anchor=tk.W)
        self.b_ramp = ttk.Label(bf, text="RAMP: [ ]", style="Ramp.TLabel")
        self.b_ramp.pack(anchor=tk.W, pady=(3, 0))

        # RED zone
        rf = ttk.LabelFrame(right, text="  RED Alliance", padding=8)
        rf.pack(fill=tk.X, pady=3)
        self.r_total = ttk.Label(rf, text="0", style="Big.TLabel",
                                 foreground="#C62828")
        self.r_total.pack()
        self.r_auto = ttk.Label(
            rf, text=f"AUTO:    Cls 0x{AUTO_PTS_CLASSIFIED}  "
                     f"Ovf 0x{AUTO_PTS_OVERFLOW}  = 0", style="Sm.TLabel")
        self.r_auto.pack(anchor=tk.W)
        self.r_teleop = ttk.Label(
            rf, text=f"TELEOP:  Cls 0x{TELEOP_PTS_CLASSIFIED}  "
                     f"Ovf 0x{TELEOP_PTS_OVERFLOW}  = 0", style="Sm.TLabel")
        self.r_teleop.pack(anchor=tk.W)
        self.r_pattern = ttk.Label(
            rf, text=f"PATTERN: 0/0 x{PTS_PATTERN}  = 0", style="Sm.TLabel")
        self.r_pattern.pack(anchor=tk.W)
        self.r_parking = ttk.Label(
            rf, text="PARKING: Not checked", style="Sm.TLabel")
        self.r_parking.pack(anchor=tk.W)
        self.r_ramp = ttk.Label(rf, text="RAMP: [ ]", style="Ramp.TLabel")
        self.r_ramp.pack(anchor=tk.W, pady=(3, 0))

        # HSV tuning
        tf = ttk.LabelFrame(right, text="HSV Tuning (Hue)", padding=4)
        tf.pack(fill=tk.X, pady=3)
        self.h_purple_lo = tk.IntVar(value=int(PURPLE_LOW[0]))
        self.h_purple_hi = tk.IntVar(value=int(PURPLE_HIGH[0]))
        self.h_green_lo  = tk.IntVar(value=int(GREEN_LOW[0]))
        self.h_green_hi  = tk.IntVar(value=int(GREEN_HIGH[0]))
        for label, vlo, vhi in [
            ("Purple", self.h_purple_lo, self.h_purple_hi),
            ("Green", self.h_green_lo, self.h_green_hi),
        ]:
            r = ttk.Frame(tf)
            r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=f"{label}:", width=8).pack(side=tk.LEFT)
            ttk.Spinbox(r, from_=0, to=179, width=4, textvariable=vlo,
                        command=self._apply_hsv).pack(side=tk.LEFT, padx=2)
            ttk.Label(r, text="-").pack(side=tk.LEFT)
            ttk.Spinbox(r, from_=0, to=179, width=4, textvariable=vhi,
                        command=self._apply_hsv).pack(side=tk.LEFT, padx=2)

        # Status bar
        sb = ttk.Frame(self.root, padding=2)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        self.st_cvs = tk.Canvas(sb, width=14, height=14, highlightthickness=0)
        self.st_cvs.pack(side=tk.LEFT, padx=5)
        self.st_dot = self.st_cvs.create_oval(2, 2, 12, 12, fill="gray")
        self.st_lbl = ttk.Label(sb, text="Initializing ...",
                                font=("Helvetica", 9))
        self.st_lbl.pack(side=tk.LEFT)
        self.fps_lbl = ttk.Label(sb, text="", font=("Helvetica", 9))
        self.fps_lbl.pack(side=tk.RIGHT, padx=8)

    # ══════════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════════
    def set_status(self, msg, ok=True):
        self.st_cvs.itemconfig(
            self.st_dot, fill="#4CAF50" if ok else "#E53935")
        self.st_lbl.configure(text=msg)

    def _apply_hsv(self):
        global PURPLE_LOW, PURPLE_HIGH, GREEN_LOW, GREEN_HIGH
        PURPLE_LOW[0]  = self.h_purple_lo.get()
        PURPLE_HIGH[0] = self.h_purple_hi.get()
        GREEN_LOW[0]   = self.h_green_lo.get()
        GREEN_HIGH[0]  = self.h_green_hi.get()

    def _on_split_change(self, val):
        self.split_x = float(val)

    def _update_motif_display(self):
        m = list(self.motif_var.get())
        txt = "  ".join("  ".join(m) for _ in range(3))
        self.motif_disp.configure(text=txt)
        self._check_pattern(self.blue)
        self._check_pattern(self.red)
        self._refresh_zone_ui(self.blue)
        self._refresh_zone_ui(self.red)

    def _scan_motif(self):
        """Uses ZhipuAI to detect the match motif at the start."""
        if self.last_frame is None: return

        frame = self.last_frame.copy()
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_b64 = base64.b64encode(buf).decode('utf-8')

        def do_scan():
            try:
                client = ZhipuAI(api_key=ZHIPU_API_KEY)
                response = client.chat.completions.create(
                    model="glm-4v-flash",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_b64}},
                            {"type": "text", "text": (
                                "Identify the FTC game randomization motif visible on the field. "
                                "It will be one of these three patterns: 'GPP', 'PGP', 'PPG' "
                                "(representing Green/Purple ordering). "
                                "Look for the signal sleeve or indicator. "
                                "Reply with ONLY valid JSON: {\"motif\": \"GPP\"}"
                            )}
                        ]
                    }],
                    max_tokens=100,
                    temperature=0.1,
                )
                text = response.choices[0].message.content.strip()
                json_match = _re.search(r'\{.*\}', text, _re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    motif = data.get("motif", "GPP").upper()
                    if motif in MOTIFS:
                        self.root.after(0, lambda: self.motif_var.set(motif))
                        self.root.after(0, self._update_motif_display)
                        self.root.after(0, lambda: self.set_status(f"Auto-detected Motif: {motif}", True))
                    else:
                        logging.warning(f"Invalid motif from AI: {motif}")
            except Exception as e:
                logging.error(f"Motif scan error: {e}")

        threading.Thread(target=do_scan, daemon=True).start()

    # ── Audio ─────────────────────────────────────────────────────────────
    def _start_audio(self):
        """Start playing the match timer MP3."""
        self._stop_audio()
        try:
            self._audio_proc = subprocess.Popen(
                ["afplay", self._audio_file],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info("Match audio started")
        except Exception as e:
            logging.warning(f"Could not play match audio: {e}")
            self._audio_proc = None

    def _pause_audio(self):
        """Pause the match timer MP3 (macOS SIGSTOP)."""
        if self._audio_proc and self._audio_proc.poll() is None:
            try:
                os.kill(self._audio_proc.pid, signal.SIGSTOP)
            except Exception as e:
                logging.debug(f"Audio pause error: {e}")

    def _resume_audio(self):
        """Resume the match timer MP3 (macOS SIGCONT)."""
        if self._audio_proc and self._audio_proc.poll() is None:
            try:
                os.kill(self._audio_proc.pid, signal.SIGCONT)
            except Exception as e:
                logging.debug(f"Audio resume error: {e}")

    def _stop_audio(self):
        """Stop and clean up the match timer MP3 process."""
        if self._audio_proc:
            try:
                # Resume first in case it was paused, then terminate
                if self._audio_proc.poll() is None:
                    os.kill(self._audio_proc.pid, signal.SIGCONT)
                    self._audio_proc.terminate()
                    self._audio_proc.wait(timeout=2)
            except Exception as e:
                logging.debug(f"Audio stop error: {e}")
            self._audio_proc = None

    # ── Timer ─────────────────────────────────────────────────────────────
    def _toggle_timer(self):
        # If starting fresh from 2:30, trigger Motif scan + audio
        if not self.timer_running and self.timer_seconds == 150:
            self.set_status("Match Started! Scanning field for MOTIF...", False)
            self._scan_motif()  # Auto-scan pattern
            self._start_audio()

        self.timer_running = not self.timer_running
        if self.timer_running:
            # Resume audio if we're un-pausing mid-match
            if self.timer_seconds < 150:
                self._resume_audio()
            self._timer_tick()
        else:
            # Pausing mid-match — pause audio too
            self._pause_audio()

    def _reset_timer(self):
        self.timer_running = False
        self.timer_seconds = 150
        self.lbl_timer.configure(text="2:30", foreground="black")
        self._stop_audio()

    def _timer_tick(self):
        if not self.timer_running:
            return
        if self.timer_seconds > 0:
            self.timer_seconds -= 1
            m, s = divmod(self.timer_seconds, 60)
            self.lbl_timer.configure(text=f"{m}:{s:02d}")
            
            # ── Auto / TeleOp transition ──
            if self.timer_seconds == 120:  # End of first 30s
                self._lock_auto()
                self.set_status("Autonomous ended. TeleOp started.", True)
                # Ensure TeleOp tracking is active if not already
                # (Optional: force tracking on/off based on rules, 
                # but user might want manual control. Just locking auto is key.)

            if self.timer_seconds <= 30:
                self.lbl_timer.configure(foreground="red")
            
            self.root.after(1000, self._timer_tick)
        else:
            self.timer_running = False
            self.lbl_timer.configure(text="0:00", foreground="red")
            self.set_status("Match ended. Checking parking...", True)
            self._stop_audio()
            
            # ── End Game: Lock TeleOp and Check Parking ──
            self._lock_teleop()
            self._check_parking()

    # ══════════════════════════════════════════════════════════════════════════
    #  Camera
    # ══════════════════════════════════════════════════════════════════════════
    def _available_cameras(self):
        cams = []
        if sys.platform == "darwin":
            try:
                p = subprocess.run(
                    ["system_profiler", "SPCameraDataType", "-json"],
                    capture_output=True, text=True, timeout=5)
                if p.returncode == 0:
                    items = json.loads(p.stdout).get("SPCameraDataType", [])
                    for i, item in enumerate(items):
                        cams.append(
                            f"{i}: {item.get('_name', f'Camera {i}')}")
            except Exception:
                pass
        if not cams:
            for idx in range(8):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened() and cap.read()[0]:
                    cams.append(f"{idx}: Camera {idx}")
                cap.release()
        return cams or ["0: Default Camera"]

    def _refresh_cameras(self):
        self.cam_cb.set("Scanning...")
        self.set_status("Scanning for cameras...", False)
        
        def scan():
            cams = self._available_cameras()
            # update UI on main thread
            self.root.after(0, lambda: self._update_cam_list(cams))

        threading.Thread(target=scan, daemon=True).start()

    def _update_cam_list(self, cams):
        self.cam_cb["values"] = cams
        if cams:
            # maintain current selection if possible
            current_idx = self.camera_index
            found = False
            for i, c in enumerate(cams):
                if c.startswith(f"{current_idx}:"):
                    self.cam_cb.current(i)
                    found = True
                    break
            if not found:
                self.cam_cb.current(0)
        self.set_status(f"Found {len(cams)} cameras", True)

    def _init_camera(self):
        self.set_status("Initializing camera...", False)
        threading.Thread(target=self._async_init_cam, daemon=True).start()

    def _async_init_cam(self):
        try:
            cams = self._available_cameras()
            
            def finish_init():
                self._update_cam_list(cams)
                if cams:
                    idx = int(cams[0].split(":")[0])
                    self._start_camera(idx)

            self.root.after(0, finish_init)
        except Exception as e:
            logging.error(f"Camera init: {e}")
            self.root.after(0, lambda: self.set_status(f"Camera error: {e}", False))

    def _start_camera(self, idx):
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
             # Try fallback to 0 if preferred failed
            if idx != 0:
                self.cap = cv2.VideoCapture(0)
                idx = 0
            
            if not self.cap.isOpened():
                self.set_status(f"Failed to open camera {idx}", False)
                return

        self.camera_index = idx
        self.is_running = True
        self.set_status(f"Camera {idx} active | {self.detector.model_name}", True)
        
        # Start loop if not already running
        # (check if loop call is pending? simple flag check is enough)
        if not hasattr(self, '_loop_running') or not self._loop_running:
            self._video_loop()

    def _on_cam_change(self, _=None):
        try:
            val = self.cam_cb.get()
            if not val or ":" not in val: return
            new = int(val.split(":")[0])
            if new == self.camera_index:
                return
            
            # Run switch in thread to avoid UI freeze
            self.set_status(f"Switching to camera {new}...", False)
            threading.Thread(target=lambda: self._thread_switch_cam(new), daemon=True).start()
        except Exception as e:
            logging.error(f"Camera switch: {e}")

    def _thread_switch_cam(self, new_idx):
        if self.cap:
            self.cap.release()
        cap = cv2.VideoCapture(new_idx)
        
        def update():
            if cap.isOpened():
                self.cap = cap
                self.camera_index = new_idx
                self.is_running = True
                self.set_status(f"Switched to camera {new_idx}", True)
            else:
                self.set_status(f"Failed to open camera {new_idx}", False)

        self.root.after(0, update)

    # ══════════════════════════════════════════════════════════════════════════
    #  Video loop
    # ══════════════════════════════════════════════════════════════════════════
    def _video_loop(self):
        self._loop_running = True
        if not self.is_running or not self.cap:
            self.root.after(100, self._video_loop)
            return
            
        t0 = time.time()
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._video_loop)
            return

        self.last_frame = frame.copy()
        self.frame_no += 1
        use_yolo = (self.detector.model_ready
                    and self.frame_no % self.yolo_every == 0)

        dets = self.detector.detect(frame, run_yolo=use_yolo)
        tracked = self.tracker.update(dets)

        purple_c = sum(1 for c in self.tracker.confirmed.values() if c == "P")
        green_c  = sum(1 for c in self.tracker.confirmed.values() if c == "G")

        # ── Zone processing ──
        height, width = frame.shape[:2]
        split_px = int(width * self.split_x)

        confirmed_objs = [self.tracker.objects[oid]
                          for oid in self.tracker.confirmed
                          if oid in self.tracker.objects]
        blue_balls = [o for o in confirmed_objs if o["x"] < split_px]
        red_balls  = [o for o in confirmed_objs if o["x"] >= split_px]

        def process_zone(zone, balls):
            zone.total_balls = len(balls)
            zone.classified = min(len(balls), MAX_RAMP)
            zone.overflow = max(0, len(balls) - MAX_RAMP)
            sorted_z = sorted(balls, key=lambda o: o.get("x", 0))
            zone.ramp_colors = [o["color"] for o in sorted_z][:MAX_RAMP]
            zone.compute_teleop()
            self._check_pattern(zone)
            self.root.after(0, lambda z=zone: self._refresh_zone_ui(z))

        if self.var_score_blue.get():
            process_zone(self.blue, blue_balls)
        if self.var_score_red.get():
            process_zone(self.red, red_balls)

        # ── Draw overlays ──
        vis = frame.copy()

        # Split line
        cv2.line(vis, (split_px, 0), (split_px, height), (0, 255, 255), 2)

        # Zone labels with background
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (170, 44), (180, 100, 0), -1)
        cv2.rectangle(overlay, (width - 160, 0), (width, 44), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
        cv2.putText(vis, "BLUE ZONE", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "RED ZONE", (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Live ball counts on video
        if self.var_score_blue.get():
            cv2.putText(vis,
                        f"Blue: {self.blue.total_balls} balls "
                        f"(C:{self.blue.classified} O:{self.blue.overflow})",
                        (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 200, 80), 1)
        if self.var_score_red.get():
            txt = (f"Red: {self.red.total_balls} balls "
                   f"(C:{self.red.classified} O:{self.red.overflow})")
            sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(vis, txt, (width - sz[0] - 8, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        # Draw ball annotations
        self._draw_balls(vis, tracked, split_px)

        # Status labels
        self.root.after(
            0, lambda p=purple_c, g=green_c:
            self.det_lbl.configure(
                text=f"Detected: {p}P  {g}G  ({p+g} total)"))

        elapsed = time.time() - t0
        fps = 1.0 / max(elapsed, 0.001)
        self.root.after(
            0, lambda f=fps: self.fps_lbl.configure(text=f"FPS: {f:.0f}"))

        self._show(vis)
        self.root.after(16, self._video_loop)

    # ── Ball overlay drawing ─────────────────────────────────────────────
    def _draw_balls(self, frame, tracked, split_px):
        motif = list(self.motif_var.get())
        expected = motif * 3  # 9 positions

        sorted_objs = sorted(
            tracked.items(), key=lambda item: item[1].get("x", 0))
        b_idx, r_idx = 0, 0

        for oid, obj in sorted_objs:
            x, y = obj.get("x", 0), obj.get("y", 0)
            r = max(obj.get("radius", 18), 12)
            col = obj.get("color", "?")
            is_blue = x < split_px
            idx = b_idx if is_blue else r_idx
            if is_blue:
                b_idx += 1
            else:
                r_idx += 1

            # Base fill color
            if col == "P":
                fill_bgr = (180, 50, 180)
            elif col == "G":
                fill_bgr = (40, 180, 40)
            else:
                fill_bgr = (180, 180, 180)

            confirmed = oid in self.tracker.confirmed
            zone_active = (self.var_score_blue.get() if is_blue
                           else self.var_score_red.get())

            # Border color encodes pattern match status
            border_bgr = fill_bgr
            if confirmed and zone_active and idx < len(expected):
                if col == expected[idx]:
                    border_bgr = (0, 220, 0)   # green = match
                else:
                    border_bgr = (0, 0, 240)   # red = mismatch

            # Draw the ball circle
            thickness = 3 if confirmed else 1
            cv2.circle(frame, (x, y), r, border_bgr, thickness)

            # Letter label centered inside the circle
            (tw, th), _ = cv2.getTextSize(
                col, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, col, (x - tw // 2, y + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Position label below the ball
            if zone_active:
                if idx < MAX_RAMP:
                    lbl = f"#{idx+1}"
                else:
                    lbl = "OVF"
                (lw, lh), _ = cv2.getTextSize(
                    lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.putText(frame, lbl, (x - lw // 2, y + r + lh + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (220, 220, 220), 1)

    # ── Show frame on canvas ─────────────────────────────────────────────
    def _show(self, frame):
        try:
            canvas = self.video_canvas
            if not canvas.winfo_exists():
                return
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 10 or ch < 10:
                cw, ch = 640, 480
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            scale = min(cw / img.width, ch / img.height)
            nw = max(1, int(img.width * scale))
            nh = max(1, int(img.height * scale))
            # Use BILINEAR for faster resizing (LANCZOS is too slow for live video)
            img = img.resize((nw, nh), Image.Resampling.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas._current_image = imgtk
            if self._canvas_img_id is None:
                self._canvas_img_id = canvas.create_image(
                    cw // 2, ch // 2, anchor=tk.CENTER, image=imgtk)
            else:
                canvas.coords(self._canvas_img_id, cw // 2, ch // 2)
                canvas.itemconfig(self._canvas_img_id, image=imgtk)
        except tk.TclError:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    #  Scoring controls
    # ══════════════════════════════════════════════════════════════════════════
    def _lock_auto(self):
        self.blue.lock_auto()
        self.red.lock_auto()
        self._refresh_zone_ui(self.blue)
        self._refresh_zone_ui(self.red)
        self.set_status("AUTO scores locked!", True)

    def _lock_teleop(self):
        self.blue.compute_teleop()
        self.red.compute_teleop()
        self._refresh_zone_ui(self.blue)
        self._refresh_zone_ui(self.red)
        self.set_status("TELEOP scores locked!", True)

    def _reset_all(self):
        self.var_score_blue.set(False)
        self.var_score_red.set(False)
        self.blue.reset()
        self.red.reset()
        self.tracker.reset()
        self._refresh_zone_ui(self.blue)
        self._refresh_zone_ui(self.red)
        self.set_status("All scores reset", True)

    def _man_adj(self, delta):
        z = self.blue if self.adj_zone.get() == "blue" else self.red
        z.total_balls = max(0, z.total_balls + delta)
        z.classified = min(z.total_balls, MAX_RAMP)
        z.overflow = max(0, z.total_balls - MAX_RAMP)
        self._refresh_zone_ui(z)

    def _snapshot_pattern(self):
        for z in (self.blue, self.red):
            self._check_pattern(z)
            self._refresh_zone_ui(z)
        self.set_status("Pattern snapshot taken", True)

    def _check_pattern(self, zone):
        motif = list(self.motif_var.get())
        full = motif * 3
        matches = sum(1 for i, c in enumerate(zone.ramp_colors[:MAX_RAMP])
                      if i < len(full) and c == full[i])
        zone.pattern_matches = matches

    # ── Parking check (GLM-4V-Flash) ─────────────────────────────────────
    def _check_parking(self):
        if self.last_frame is None:
            messagebox.showwarning("No Frame", "No camera frame available.")
            return

        frame = self.last_frame.copy()
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf).decode('utf-8')

        self.set_status("Sending frame to GLM-4V-Flash for parking check...",
                        False)

        def do_check():
            try:
                client = ZhipuAI(api_key=ZHIPU_API_KEY)
                response = client.chat.completions.create(
                    model="glm-4v-flash",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": img_b64}},
                            {"type": "text", "text": (
                                "This is an FTC FIRST Tech Challenge robotics "
                                "competition field photo taken from above. "
                                "Analyze this image for robot parking status "
                                "at the end of the match. Count how many "
                                "robots from each alliance (blue and red) "
                                "are parked in the observation/parking zone. "
                                "Reply with ONLY valid JSON, no markdown: "
                                '{"blue_parked": <number>, '
                                '"red_parked": <number>, '
                                '"description": "<brief description>"}'
                            )}
                        ]
                    }],
                    max_tokens=256,
                    temperature=0.1,
                )
                text = response.choices[0].message.content.strip()
                logging.info(f"Parking API response: {text}")

                # Parse JSON from response
                json_match = _re.search(r'\{.*\}', text, _re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(text)

                bp = int(data.get("blue_parked", 0))
                rp = int(data.get("red_parked", 0))
                desc = data.get("description", text)[:80]

                self.blue.parking_pts = bp * PTS_PARKING
                self.blue.parking_status = f"{bp} robot(s) parked"
                self.red.parking_pts = rp * PTS_PARKING
                self.red.parking_status = f"{rp} robot(s) parked"

                self.root.after(0, lambda: (
                    self._refresh_zone_ui(self.blue),
                    self._refresh_zone_ui(self.red),
                    self.set_status(
                        f"Parking: Blue={bp}, Red={rp} - {desc}", True),
                ))

            except Exception as e:
                logging.error(f"Parking API error: {e}")
                self.root.after(0, lambda: self.set_status(
                    f"Parking check failed: {e}", False))

        threading.Thread(target=do_check, daemon=True).start()

    # ── Zone UI refresh ──────────────────────────────────────────────────
    def _refresh_zone_ui(self, z):
        is_b = z.name == "Blue"
        lbl_t  = self.b_total   if is_b else self.r_total
        lbl_a  = self.b_auto    if is_b else self.r_auto
        lbl_tp = self.b_teleop  if is_b else self.r_teleop
        lbl_p  = self.b_pattern if is_b else self.r_pattern
        lbl_pk = self.b_parking if is_b else self.r_parking
        lbl_r  = self.b_ramp    if is_b else self.r_ramp

        lbl_t.configure(text=str(z.total_score()))
        lbl_a.configure(
            text=f"AUTO:    Cls {z.auto_classified}x{AUTO_PTS_CLASSIFIED}  "
                 f"Ovf {z.auto_overflow}x{AUTO_PTS_OVERFLOW}  "
                 f"= {z.auto_score()}")
        lbl_tp.configure(
            text=f"TELEOP:  Cls {z.teleop_classified}x{TELEOP_PTS_CLASSIFIED}  "
                 f"Ovf {z.teleop_overflow}x{TELEOP_PTS_OVERFLOW}  "
                 f"= {z.teleop_score()}")
        n = min(len(z.ramp_colors), MAX_RAMP)
        lbl_p.configure(
            text=f"PATTERN: {z.pattern_matches}/{n} x{PTS_PATTERN}  "
                 f"= {z.pattern_matches * PTS_PATTERN}")
        lbl_pk.configure(
            text=f"PARKING: {z.parking_status}  = {z.parking_pts} pts")
        seq = " ".join(z.ramp_colors[:MAX_RAMP]) if z.ramp_colors else "--"
        lbl_r.configure(text=f"RAMP: [ {seq} ]")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = FTCDecodeApp(root)
    root.mainloop()
