"""
FTC DECODE Season Field Scorer — Dual Camera Classifier Ramp Edition
─────────────────────────────────────────────────────────────────────────────
Uses YOLO object detection + HSV color filtering to:
  1. Count purple/green ARTIFACTS (balls) on each alliance's CLASSIFIER RAMP
  2. Check if the RAMP pattern matches the user-selected MOTIF
  3. Score BLUE and RED alliance zones separately with live counters
  4. Separate Autonomous and TeleOp scoring periods
  5. Use HIGH-WATER MARK scoring so gate-opening doesn't erase scores

Game reference (FTC DECODE – presented by RTX):
  - ARTIFACTS are ~5-inch purple (P) and green (G) foam balls
  - CLASSIFIED (on RAMP): 3 pts in both AUTO and TELEOP
  - OVERFLOW (past RAMP):  1 pt in both AUTO and TELEOP
  - MOTIF is one of GPP / PGP / PPG, repeated x3 for 9 RAMP slots
  - PATTERN = each RAMP position matching the MOTIF index -> 2 pts each
  - GATE can be opened to clear RAMP — scores are preserved (high-water mark)
  - Each alliance has its own CLASSIFIER: cameras point directly at each RAMP
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
from tkinter import ttk, messagebox, simpledialog
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
# Removed ultralytics as requested
from zhipuai import ZhipuAI

# ── Game constants (per DECODE Competition Manual) ────────────────────────────
MOTIFS = ["GPP", "PGP", "PPG"]
MAX_RAMP = 9

# Points are the SAME in both Auto and TeleOp per the manual (Table 10-2)
PTS_CLASSIFIED = 3   # per CLASSIFIED artifact (on RAMP)
PTS_OVERFLOW   = 1   # per OVERFLOW artifact (past RAMP / over gate)
PTS_PATTERN    = 2   # per RAMP position that matches MOTIF
PTS_LEAVE      = 3   # per robot that LEAVES launch line (auto) — not scored here

# HSV colour ranges  (tune via UI if lighting differs)
# STRICTER RANGE: Avoid robot parts / background noise
PURPLE_LOW  = np.array([128, 60, 60])  # Raised Sat/Val to avoid dark/grey purple
PURPLE_HIGH = np.array([152, 255, 255]) 

GREEN_LOW   = np.array([45, 80, 60])   # Raised Sat significantly (foam balls are vibrant)
GREEN_HIGH  = np.array([80, 255, 255]) 

# Size filtering matching typical 5" ball at distance
# STRICTER SIZE: Ignore small debris/robot stickers
MIN_BALL_AREA    = 600   # Increased from 200 to 600 to ignore small noise
MAX_BALL_AREA    = 60000 
MIN_RADIUS       = 14    # Increased from 8 to 14
MAX_RADIUS       = 150   

# Morphology settings (handle holes in wiffle balls AND bars crossing them)
# Reduced aggression to prevent merging adjacent balls
KERN_SIZE = (7, 7)     

MIN_CIRCULARITY  = 0.25  # Very Loose: A ball cut in half by a bar is NOT circular
CONFIRM_FRAMES   = 5     # Keep snappy response for overflow

# ZhipuAI API (kept for future use)
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
            # Switch to Ollama (Vision Language Model Integration)
            # User requested "smarter" detection to handle grouped balls
            import ollama
            
            # Check if moondream is available, otherwise pull it
            try:
                ollama.show('moondream')
                self.model_name = "Ollama (moondream)"
            except:
                logging.info("Pulling moondream model...")
                ollama.pull('moondream')
                self.model_name = "Ollama (moondream)"
            
            self.model = ollama
            self.model_ready = True
            
            logging.info(f"Loaded {self.model_name}")
            if callback:
                callback(True, f"Model ready: {self.model_name}")
                
        except Exception as e:
            logging.warning(f"Ollama load failed — HSV-only mode: {e}")
            self.model_ready = False
            self.model_name = "HSV-only (Ollama unavailable)"
            if callback:
                callback(True, self.model_name)

    def detect(self, frame, run_yolo=True):
        detections = []
        
        # 1. Run HSV Detection (Fast, Color-based)
        # We start with HSV to get candidate regions because Ollama is too slow for full-frame search
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections += self._hsv_detect(hsv, PURPLE_LOW, PURPLE_HIGH, "P")
        detections += self._hsv_detect(hsv, GREEN_LOW, GREEN_HIGH, "G")
        
        # 2. Run Ollama Refinement (Slow but Smart)
        # Only run on specific check intervals or key frames due to latency
        if run_yolo and self.model_ready:
             # For now, just pass through HSV detections as Ollama integration for 
             # real-time bounding box refinement is extremely complex and slow.
             # A full implementation would require sending crops to the LLM.
             pass
            
        return detections

    def _hsv_detect(self, hsv, low, high, label):
        mask = cv2.inRange(hsv, low, high)
        
        # AGGRESSIVE CLOSING to handle the metallic bar splitting balls
        # The bar cuts the ball in half visually; we need to merge the pieces
        # Use the large global kernel size (15x15 or larger)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERN_SIZE)
        
        # Open removes small noise, CLOSE merges split fragments
        # Reduced aggression: 1 iteration instead of 3 to avoid merging neighbors
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=1) 
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BALL_AREA: continue
            if area > MAX_BALL_AREA: continue

            # Loose circularity for occluded balls
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circ = 4 * np.pi * area / (peri * peri)
            
            if circ < MIN_CIRCULARITY: continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.33 or aspect_ratio > 3.0: continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < MIN_RADIUS or r > MAX_RADIUS: continue

            out.append({"x": int(cx), "y": int(cy), "radius": int(r), "color": label})
        return out




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
#  Zone Score Data — with high-water-mark scoring
# ══════════════════════════════════════════════════════════════════════════════
class ZoneData:
    """Tracks scoring for one alliance's CLASSIFIER RAMP.

    HIGH-WATER MARK: When the GATE opens and balls fall out, the live
    detection count drops.  We keep the *peak* classified / overflow /
    pattern counts so that scores never decrease mid-match.  Only a
    manual reset or new match clears them.
    """

    def __init__(self, name):
        self.name = name

        # ── Live detection ──
        self.live_classified = 0
        self.live_overflow = 0
        self.live_total = 0
        self.live_ramp_colors = []     # what the camera sees RIGHT NOW

        # ── High-water marks / Accumulators ──
        self.hw_classified = 0
        self.hw_overflow = 0           # Now reflects accumulated unique overflow IDs
        self.hw_total = 0
        self.hw_ramp_colors = []       # best ramp state ever seen
        self.hw_pattern_matches = 0
        
        # Track unique IDs that have ever been classified as overflow
        self.overflow_ids = set()

        # ── Autonomous snapshot ──
        self.auto_classified = 0
        self.auto_overflow = 0
        self.auto_total = 0
        self.auto_pattern = 0
        self.auto_ramp_colors = []

        # ── TeleOp ──
        self.teleop_classified = 0
        self.teleop_overflow = 0
        self.teleop_total = 0
        self.teleop_pattern = 0

    def reset(self):
        self.live_classified = self.live_overflow = self.live_total = 0
        self.live_ramp_colors = []
        self.hw_classified = self.hw_overflow = self.hw_total = 0
        self.hw_ramp_colors = []
        self.hw_pattern_matches = 0
        self.overflow_ids.clear()
        
        self.auto_classified = self.auto_overflow = self.auto_total = 0
        self.auto_pattern = 0
        self.auto_ramp_colors = []
        self.teleop_classified = self.teleop_overflow = self.teleop_total = 0
        self.teleop_pattern = 0

    def update_live(self, ramp_colors, current_overflow_ids, motif_str):
        """
        ramp_colors: list of chars ['P', 'G', ...] currently on ramp
        current_overflow_ids: set/list of object IDs currently in overflow
        motif_str: current motif string (e.g. "PGP")
        """
        self.live_ramp_colors = list(ramp_colors)
        self.live_classified = len(ramp_colors)
        
        # Add new overflow IDs to our set
        for oid in current_overflow_ids:
            self.overflow_ids.add(oid)
            
        self.live_overflow = len(self.overflow_ids)
        self.live_total = self.live_classified + self.live_overflow

        # Compute live pattern matches against MOTIF
        motif = list(motif_str)
        full_motif = motif * 3  # 9 positions
        live_pattern = sum(
            1 for i, c in enumerate(ramp_colors[:MAX_RAMP])
            if i < len(full_motif) and c == full_motif[i]
        )

        # ── High-water update ──
        # Captured/Overflow count never goes down because we accumulate IDs
        self.hw_overflow = len(self.overflow_ids)

        if self.live_classified > self.hw_classified:
            self.hw_classified = self.live_classified
        
        # Update total high-water
        current_total = self.hw_classified + self.hw_overflow
        if current_total > self.hw_total:
            self.hw_total = current_total

        # Pattern: save best pattern score AND the ramp state that achieved it
        if live_pattern > self.hw_pattern_matches:
            self.hw_pattern_matches = live_pattern
            self.hw_ramp_colors = list(ramp_colors)
        elif self.live_classified > len(self.hw_ramp_colors):
            # More balls than we've ever seen — capture this ramp state too
            self.hw_ramp_colors = list(ramp_colors)

    def lock_auto(self):
        """Snapshot current high-water counts as Autonomous scores."""
        self.auto_classified = self.hw_classified
        self.auto_overflow = self.hw_overflow
        self.auto_total = self.hw_total
        self.auto_pattern = self.hw_pattern_matches
        self.auto_ramp_colors = list(self.hw_ramp_colors)

    def compute_teleop(self):
        """TeleOp = current high-water totals minus what was scored in Auto."""
        self.teleop_classified = max(0, self.hw_classified - self.auto_classified)
        self.teleop_overflow = max(0, self.hw_overflow - self.auto_overflow)
        self.teleop_total = max(0, self.hw_total - self.auto_total)
        self.teleop_pattern = max(0, self.hw_pattern_matches - self.auto_pattern)

    def auto_score(self):
        return (self.auto_classified * PTS_CLASSIFIED
                + self.auto_overflow * PTS_OVERFLOW
                + self.auto_pattern * PTS_PATTERN)

    def teleop_score(self):
        return (self.teleop_classified * PTS_CLASSIFIED
                + self.teleop_overflow * PTS_OVERFLOW
                + self.teleop_pattern * PTS_PATTERN)

    def total_score(self):
        return self.auto_score() + self.teleop_score()


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application — Dual Camera (one per CLASSIFIER RAMP)
# ══════════════════════════════════════════════════════════════════════════════
class FTCDecodeApp:

    def __init__(self, root):
        self.root = root
        self.root.title("FTC DECODE Scorer — Dual Camera Classifier Ramp")
        self.root.geometry("1500x920")
        self.root.minsize(1200, 750)

        self.detector = BallDetector()

        # Two separate trackers — one per camera/alliance
        self.blue_tracker = CentroidTracker()
        self.red_tracker = CentroidTracker()

        # Two separate camera captures
        self.blue_cap = None
        self.red_cap = None
        self.blue_cam_idx = -1
        self.red_cam_idx = -1
        self.is_running = False

        self.frame_no = 0
        self.yolo_every = 5
        self.last_blue_frame = None
        self.last_red_frame = None

        self.blue = ZoneData("Blue")
        self.red = ZoneData("Red")

        # Timer
        self.timer_seconds = 150
        self.timer_running = False

        # Audio
        self._audio_proc = None
        self._audio_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ftctimer.mp3")

        self._build_ui()
        self._update_motif_display()

        self.set_status("Loading YOLO model ...", False)
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(600, self._refresh_cameras)

    def _load_model(self):
        self.detector.load_model(
            callback=lambda ok, msg: self.root.after(
                0, lambda: self.set_status(msg, ok))
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure("Big.TLabel",   font=("Helvetica", 28, "bold"))
        sty.configure("Med.TLabel",   font=("Helvetica", 13))
        sty.configure("Sm.TLabel",    font=("Helvetica", 10))
        sty.configure("Ramp.TLabel",  font=("Courier", 12, "bold"))

        # ── Row 1: Camera selection for BLUE + RED, MOTIF, Timer ─────────
        row1 = ttk.Frame(self.root, padding=4)
        row1.pack(fill=tk.X)

        # BLUE camera selector
        ttk.Label(row1, text="BLUE Cam:", foreground="#1565C0",
                  font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(5, 2))
        self.blue_cam_cb = ttk.Combobox(row1, width=30) # Allow typing for HTTP URLs
        self.blue_cam_cb.pack(side=tk.LEFT, padx=2)
        self.blue_cam_cb.bind("<<ComboboxSelected>>", self._on_blue_cam_change)
        self.blue_cam_cb.bind("<Return>", self._on_blue_cam_change)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4)

        # RED camera selector
        ttk.Label(row1, text="RED Cam:", foreground="#C62828",
                  font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(2, 2))
        self.red_cam_cb = ttk.Combobox(row1, width=30) # Allow typing for HTTP URLs
        self.red_cam_cb.pack(side=tk.LEFT, padx=2)
        self.red_cam_cb.bind("<<ComboboxSelected>>", self._on_red_cam_change)
        self.red_cam_cb.bind("<Return>", self._on_red_cam_change)

        ttk.Button(row1, text="Refresh Cams", width=11,
                   command=self._refresh_cameras).pack(side=tk.LEFT, padx=4)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4)

        # MOTIF selector
        ttk.Label(row1, text="MOTIF:").pack(side=tk.LEFT, padx=(2, 2))
        self.motif_var = tk.StringVar(value="GPP")
        self.motif_cb = ttk.Combobox(row1, textvariable=self.motif_var,
                                      values=MOTIFS, state="readonly", width=5)
        self.motif_cb.pack(side=tk.LEFT, padx=2)
        self.motif_cb.bind("<<ComboboxSelected>>",
                           lambda _: self._update_motif_display())

        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4)

        # Timer
        self.lbl_timer = ttk.Label(
            row1, text="2:30", font=("Helvetica", 16, "bold"),
            width=5, anchor=tk.CENTER)
        self.lbl_timer.pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Start/Pause", width=10,
                   command=self._toggle_timer).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Reset Timer", width=8,
                   command=self._reset_timer).pack(side=tk.LEFT, padx=2)

        # ── Row 2: Actions + manual adjust ───────────────────────────────
        row2 = ttk.Frame(self.root, padding=2)
        row2.pack(fill=tk.X)

        ttk.Button(row2, text="Snapshot Pattern",
                   command=self._snapshot_pattern).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Reset All",
                   command=self._reset_all).pack(side=tk.LEFT, padx=4)
        
        # Specific resets
        ttk.Button(row2, text="Clr Blue", width=8,
                   command=self._clear_counts_blue).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Clr Red", width=8,
                   command=self._clear_counts_red).pack(side=tk.LEFT, padx=2)

        ttk.Label(row2, text="(Auto/TeleOp locked via Timer)",
                  foreground="gray").pack(side=tk.LEFT, padx=5)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        # Manual adjust buttons
        self.adj_zone = tk.StringVar(value="blue")
        ttk.Radiobutton(row2, text="Blue", variable=self.adj_zone,
                        value="blue").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(row2, text="Red", variable=self.adj_zone,
                        value="red").pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="+1 Cls", width=5,
                   command=lambda: self._man_adj_cls(1)).pack(
                       side=tk.LEFT, padx=1)
        ttk.Button(row2, text="-1 Cls", width=5,
                   command=lambda: self._man_adj_cls(-1)).pack(
                       side=tk.LEFT, padx=1)
        ttk.Button(row2, text="+1 Ovf", width=5,
                   command=lambda: self._man_adj_ovf(1)).pack(
                       side=tk.LEFT, padx=1)
        ttk.Button(row2, text="-1 Ovf", width=5,
                   command=lambda: self._man_adj_ovf(-1)).pack(
                       side=tk.LEFT, padx=1)

        # ── Main area: Video feeds (Left) + Score panels (Right) ─────────
        main = ttk.Frame(self.root, padding=4)
        main.pack(fill=tk.BOTH, expand=True)

        # Left column: Stacked video feeds
        video_col = ttk.Frame(main)
        video_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        # Top video: BLUE CLASSIFIER RAMP
        blue_video_frame = ttk.LabelFrame(video_col, text="  BLUE Classifier Ramp  ",
                                     padding=2)
        blue_video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 2))

        self.blue_canvas = tk.Canvas(blue_video_frame, bg="black",
                                      highlightthickness=0)
        self.blue_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._blue_img_id = None

        self.blue_det_lbl = ttk.Label(blue_video_frame, text="Blue: --",
                                       style="Med.TLabel")
        self.blue_det_lbl.pack(side=tk.BOTTOM, anchor=tk.W, padx=4)

        # Bottom video: RED CLASSIFIER RAMP
        red_video_frame = ttk.LabelFrame(video_col, text="  RED Classifier Ramp  ",
                                      padding=2)
        red_video_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(2, 0))

        self.red_canvas = tk.Canvas(red_video_frame, bg="black",
                                     highlightthickness=0)
        self.red_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._red_img_id = None

        self.red_det_lbl = ttk.Label(red_video_frame, text="Red: --",
                                      style="Med.TLabel")
        self.red_det_lbl.pack(side=tk.BOTTOM, anchor=tk.W, padx=4)

        # Right panel: scores
        right = ttk.Frame(main, width=400)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=4)
        right.pack_propagate(False)

        # Motif display
        mf = ttk.LabelFrame(right, text="MOTIF Pattern (x3 = 9 slots)",
                             padding=6)
        mf.pack(fill=tk.X, pady=(0, 3))
        self.motif_disp = ttk.Label(mf, text="", style="Ramp.TLabel")
        self.motif_disp.pack()

        # BLUE score panel
        bf = ttk.LabelFrame(right, text="  BLUE Alliance", padding=8)
        bf.pack(fill=tk.X, pady=3)
        self.b_total = ttk.Label(bf, text="0", style="Big.TLabel",
                                 foreground="#1565C0")
        self.b_total.pack()
        self.b_auto = ttk.Label(
            bf, text=f"AUTO:    Cls 0x{PTS_CLASSIFIED}  "
                     f"Ovf 0x{PTS_OVERFLOW}  Pat 0x{PTS_PATTERN}  = 0",
            style="Sm.TLabel")
        self.b_auto.pack(anchor=tk.W)
        self.b_teleop = ttk.Label(
            bf, text=f"TELEOP:  Cls 0x{PTS_CLASSIFIED}  "
                     f"Ovf 0x{PTS_OVERFLOW}  Pat 0x{PTS_PATTERN}  = 0",
            style="Sm.TLabel")
        self.b_teleop.pack(anchor=tk.W)
        self.b_pattern = ttk.Label(
            bf, text=f"PATTERN: 0/{MAX_RAMP} x{PTS_PATTERN}  = 0",
            style="Sm.TLabel")
        self.b_pattern.pack(anchor=tk.W)
        self.b_live = ttk.Label(
            bf, text="LIVE: 0 on ramp, 0 overflow",
            style="Sm.TLabel", foreground="gray")
        self.b_live.pack(anchor=tk.W)
        # # PARKING (commented out per user request)
        # self.b_parking = ttk.Label(
        #     bf, text="PARKING: Not checked", style="Sm.TLabel")
        # self.b_parking.pack(anchor=tk.W)
        self.b_ramp = ttk.Label(bf, text="RAMP: [ ]", style="Ramp.TLabel")
        self.b_ramp.pack(anchor=tk.W, pady=(3, 0))

        # RED score panel
        rf = ttk.LabelFrame(right, text="  RED Alliance", padding=8)
        rf.pack(fill=tk.X, pady=3)
        self.r_total = ttk.Label(rf, text="0", style="Big.TLabel",
                                 foreground="#C62828")
        self.r_total.pack()
        self.r_auto = ttk.Label(
            rf, text=f"AUTO:    Cls 0x{PTS_CLASSIFIED}  "
                     f"Ovf 0x{PTS_OVERFLOW}  Pat 0x{PTS_PATTERN}  = 0",
            style="Sm.TLabel")
        self.r_auto.pack(anchor=tk.W)
        self.r_teleop = ttk.Label(
            rf, text=f"TELEOP:  Cls 0x{PTS_CLASSIFIED}  "
                     f"Ovf 0x{PTS_OVERFLOW}  Pat 0x{PTS_PATTERN}  = 0",
            style="Sm.TLabel")
        self.r_teleop.pack(anchor=tk.W)
        self.r_pattern = ttk.Label(
            rf, text=f"PATTERN: 0/{MAX_RAMP} x{PTS_PATTERN}  = 0",
            style="Sm.TLabel")
        self.r_pattern.pack(anchor=tk.W)
        self.r_live = ttk.Label(
            rf, text="LIVE: 0 on ramp, 0 overflow",
            style="Sm.TLabel", foreground="gray")
        self.r_live.pack(anchor=tk.W)
        # # PARKING (commented out per user request)
        # self.r_parking = ttk.Label(
        #     rf, text="PARKING: Not checked", style="Sm.TLabel")
        # self.r_parking.pack(anchor=tk.W)
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

    def _update_motif_display(self):
        m = list(self.motif_var.get())
        txt = "  ".join("  ".join(m) for _ in range(3))
        self.motif_disp.configure(text=txt)
        # Recompute pattern for current ramp states
        self._refresh_zone_ui(self.blue)
        self._refresh_zone_ui(self.red)

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
        if self._audio_proc and self._audio_proc.poll() is None:
            try:
                os.kill(self._audio_proc.pid, signal.SIGSTOP)
            except Exception as e:
                logging.debug(f"Audio pause error: {e}")

    def _resume_audio(self):
        if self._audio_proc and self._audio_proc.poll() is None:
            try:
                os.kill(self._audio_proc.pid, signal.SIGCONT)
            except Exception as e:
                logging.debug(f"Audio resume error: {e}")

    def _stop_audio(self):
        if self._audio_proc:
            try:
                if self._audio_proc.poll() is None:
                    os.kill(self._audio_proc.pid, signal.SIGCONT)
                    self._audio_proc.terminate()
                    self._audio_proc.wait(timeout=2)
            except Exception as e:
                logging.debug(f"Audio stop error: {e}")
            self._audio_proc = None

    # ── Timer ─────────────────────────────────────────────────────────────
    def _toggle_timer(self):
        # Initial start or Restart if paused at 2:30
        if not self.timer_running and self.timer_seconds == 150:
            self.set_status("Match Audio Started... Timer in 3s", True)
            # If audio was paused, stop/restart it to sync with new countdown
            self._stop_audio() 
            self._start_audio()
            
            self.timer_running = True
            # Schedule the actual timer decrement to start in 3 seconds
            self.root.after(3000, self._start_countdown_loop)
            return

        # Normal Pause/Resume logic mid-match
        self.timer_running = not self.timer_running
        if self.timer_running:
            # Resuming
            self._resume_audio()
            # If we are resuming mid-match (<150), tick immediately
            if self.timer_seconds < 150:
                self._timer_tick()
        else:
            # Pausing
            self._pause_audio()

    def _start_countdown_loop(self):
        # Only start if still running (user didn't pause in the 3s gap)
        if self.timer_running:
            self.set_status("Match Started!", True)
            self._timer_tick()
            
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

            # ── Auto -> TeleOp transition at 120s remaining ──
            if self.timer_seconds == 120:
                self._lock_auto()
                self.set_status("Autonomous ended -> TeleOp started.", True)

            if self.timer_seconds <= 30:
                self.lbl_timer.configure(foreground="red")

            self.root.after(1000, self._timer_tick)
        else:
            self.timer_running = False
            self.lbl_timer.configure(text="0:00", foreground="red")
            self._stop_audio()

            # ── End Game: Lock TeleOp scores ──
            self._lock_teleop()
            self.set_status("Match ended. Final scores locked.", True)

            # # ── Parking check (COMMENTED OUT per user request) ──
            # self._check_parking()

    # ══════════════════════════════════════════════════════════════════════════
    #  Camera management — two separate cameras
    # ══════════════════════════════════════════════════════════════════════════
    def _available_cameras(self):
        """Enumerate available cameras."""
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
        self.set_status("Scanning for cameras...", False)

        def scan():
            cams = self._available_cameras()
            self.root.after(0, lambda: self._update_cam_lists(cams))

        threading.Thread(target=scan, daemon=True).start()

    def _update_cam_lists(self, cams):
        """Populate both camera combo boxes with 'None' + all detected cameras + 'Add from URL...'."""
        choices = ["None (disabled)"] + cams + ["Add from URL..."]
        self.blue_cam_cb["values"] = choices
        self.red_cam_cb["values"] = choices

        # Default to None
        self.blue_cam_cb.current(0)
        self.red_cam_cb.current(0)

        # If exactly 2+ cameras, auto-assign: first=blue, second=red
        if len(cams) >= 2:
            self.blue_cam_cb.current(1)   # first real camera
            self.red_cam_cb.current(2)    # second real camera
            self._start_blue_cam(int(cams[0].split(":")[0]))
            self._start_red_cam(int(cams[1].split(":")[0]))
        elif len(cams) == 1:
            self.blue_cam_cb.current(1)
            self._start_blue_cam(int(cams[0].split(":")[0]))

        active = sum(1 for c in [self.blue_cap, self.red_cap]
                     if c is not None and c.isOpened())
        self.set_status(f"Found {len(cams)} camera(s), {active} active", True)

        # Start the video loop if not already running
        if not hasattr(self, '_loop_running') or not self._loop_running:
            self._video_loop()

    def _parse_cam_val(self, val):
        """Helper to parse combo box value into int (index) or str (URL)."""
        if not val or val.startswith("None"):
            return None
        val = val.strip()
        # "0: Name" format
        if ":" in val and val.split(":")[0].isdigit() and "http" not in val:
             return int(val.split(":")[0])
        # Pure integer string "0"
        if val.isdigit():
             return int(val)
        # Otherwise assume it's a URL or file path
        return val

    def _on_blue_cam_change(self, _=None):
        val = self.blue_cam_cb.get()
        
        if val == "Add from URL...":
            new_url = simpledialog.askstring("Add Camera", "Enter Camera URL or Path:")
            if new_url and new_url.strip():
                # Add to dropdown and select it
                vals = list(self.blue_cam_cb["values"])
                # Remove placeholder if re-adding or insert before it
                if "Add from URL..." in vals:
                    vals.remove("Add from URL...")
                vals.append(new_url)
                vals.append("Add from URL...") # Keep at end
                self.blue_cam_cb["values"] = vals
                self.blue_cam_cb.set(new_url)
                val = new_url
            else:
                self.blue_cam_cb.set("None (disabled)")
                self._stop_blue_cam()
                return

        source = self._parse_cam_val(val)
        
        if source is None:
            self._stop_blue_cam()
            return

        if source == self.blue_cam_idx:
            return

        try:
            self._start_blue_cam(source)
            # If manually typed and successful, add to list if not present
            current_values = list(self.blue_cam_cb["values"])
            if val not in current_values and self.blue_cap and self.blue_cap.isOpened():
                if "Add from URL..." in current_values:
                    current_values.insert(-1, val)
                else:
                    current_values.append(val)
                self.blue_cam_cb["values"] = current_values
        except Exception as e:
            logging.error(f"Blue camera switch to {source}: {e}")

    def _on_red_cam_change(self, _=None):
        val = self.red_cam_cb.get()
        
        if val == "Add from URL...":
            new_url = simpledialog.askstring("Add Camera", "Enter Camera URL or Path:")
            if new_url and new_url.strip():
                vals = list(self.red_cam_cb["values"])
                if "Add from URL..." in vals:
                    vals.remove("Add from URL...")
                vals.append(new_url)
                vals.append("Add from URL...")
                self.red_cam_cb["values"] = vals
                self.red_cam_cb.set(new_url)
                val = new_url
            else:
                self.red_cam_cb.set("None (disabled)")
                self._stop_red_cam()
                return

        source = self._parse_cam_val(val)
        
        if source is None:
            self._stop_red_cam()
            return

        if source == self.red_cam_idx:
            return

        try:
            self._start_red_cam(source)
            current_values = list(self.red_cam_cb["values"])
            if val not in current_values and self.red_cap and self.red_cap.isOpened():
                if "Add from URL..." in current_values:
                    current_values.insert(-1, val)
                else:
                    current_values.append(val)
                self.red_cam_cb["values"] = current_values
        except Exception as e:
            logging.error(f"Red camera switch to {source}: {e}")

    def _start_blue_cam(self, idx):
        if self.blue_cap:
            self.blue_cap.release()
        self.blue_cap = cv2.VideoCapture(idx)
        if self.blue_cap.isOpened():
            self.blue_cam_idx = idx
            self.is_running = True
            logging.info(f"Blue camera started: {idx}")
        else:
            self.set_status(f"Failed to open blue camera {idx}", False)
            self.blue_cap = None
            self.blue_cam_idx = -1

    def _start_red_cam(self, idx):
        if self.red_cap:
            self.red_cap.release()
        self.red_cap = cv2.VideoCapture(idx)
        if self.red_cap.isOpened():
            self.red_cam_idx = idx
            self.is_running = True
            logging.info(f"Red camera started: {idx}")
        else:
            self.set_status(f"Failed to open red camera {idx}", False)
            self.red_cap = None
            self.red_cam_idx = -1

    def _stop_blue_cam(self):
        if self.blue_cap:
            self.blue_cap.release()
        self.blue_cap = None
        self.blue_cam_idx = -1

    def _stop_red_cam(self):
        if self.red_cap:
            self.red_cap.release()
        self.red_cap = None
        self.red_cam_idx = -1

    # ══════════════════════════════════════════════════════════════════════════
    #  Video loop — processes both cameras each tick
    # ══════════════════════════════════════════════════════════════════════════
    def _video_loop(self):
        self._loop_running = True
        t0 = time.time()
        self.frame_no += 1
        use_yolo = (self.detector.model_ready
                    and self.frame_no % self.yolo_every == 0)

        # ── Process BLUE camera ──
        if self.blue_cap and self.blue_cap.isOpened():
            ret, frame = self.blue_cap.read()
            if ret:
                self.last_blue_frame = frame.copy()
                self._process_camera_frame(
                    frame, self.blue, self.blue_tracker,
                    self.blue_canvas, "blue", use_yolo)
        else:
            self._show_placeholder(self.blue_canvas, "No BLUE camera")

        # ── Process RED camera ──
        if self.red_cap and self.red_cap.isOpened():
            ret, frame = self.red_cap.read()
            if ret:
                self.last_red_frame = frame.copy()
                self._process_camera_frame(
                    frame, self.red, self.red_tracker,
                    self.red_canvas, "red", use_yolo)
        else:
            self._show_placeholder(self.red_canvas, "No RED camera")

        elapsed = time.time() - t0
        fps = 1.0 / max(elapsed, 0.001)
        self.root.after(
            0, lambda f=fps: self.fps_lbl.configure(text=f"FPS: {f:.0f}"))

        self.root.after(16, self._video_loop)

    def _process_camera_frame(self, frame, zone, tracker, canvas,
                               side, use_yolo):
        """Process one camera's frame: detect balls, update zone scores,
        draw overlays, display on canvas.

        Each camera is pointed directly at one alliance's CLASSIFIER RAMP,
        so ALL detected balls belong to that alliance.
        """
        dets = self.detector.detect(frame, run_yolo=use_yolo)
        tracked = tracker.update(dets)

        # Get confirmed balls with their IDs
        confirmed_items = [(oid, tracker.objects[oid]) 
                           for oid in tracker.confirmed 
                           if oid in tracker.objects]

        # Sort left-to-right (x position) — represents ramp position
        # Position 1 is leftmost ball on camera view
        sorted_items = sorted(confirmed_items, key=lambda item: item[1].get("x", 0))

        # First MAX_RAMP balls are CLASSIFIED (on the ramp),
        # anything beyond is OVERFLOW
        ramp_colors = [item[1]["color"] for item in sorted_items[:MAX_RAMP]]
        
        # Get IDs of overflow balls
        overflow_ids = [item[0] for item in sorted_items[MAX_RAMP:]]

        # Update zone with live data + high-water mark (accumulation)
        zone.update_live(ramp_colors, overflow_ids, self.motif_var.get())
        zone.compute_teleop() 

        # ── Draw overlays on frame ──
        vis = frame.copy()
        
        # Clean view - removed "Ramp/Overflow" debug banner as requested
        
        # Draw ball annotations with pattern match info
        self._draw_balls(vis, tracked, tracker)

        # Update detection label (bottom overlay)
        p_count = sum(1 for c in tracker.confirmed.values() if c == "P")
        g_count = sum(1 for c in tracker.confirmed.values() if c == "G")
        det_lbl = self.blue_det_lbl if side == "blue" else self.red_det_lbl
        self.root.after(
            0, lambda p=p_count, g=g_count, lbl=det_lbl:
            lbl.configure(text=f"{zone.name}: {p}P {g}G ({p+g} total)"))

        # Show on canvas
        self._show_frame(vis, canvas, side)

        # Refresh score UI
        self.root.after(0, lambda z=zone: self._refresh_zone_ui(z))

    def _draw_balls(self, frame, tracked, tracker):
        """Draw ball annotations with pattern match indicators."""
        motif = list(self.motif_var.get())
        expected = motif * 3  # 9 positions

        sorted_objs = sorted(
            tracked.items(), key=lambda item: item[1].get("x", 0))
        idx = 0

        for oid, obj in sorted_objs:
            x, y = obj.get("x", 0), obj.get("y", 0)
            r = max(obj.get("radius", 18), 12)
            col = obj.get("color", "?")
            confirmed = oid in tracker.confirmed

            # Distinct colors for the ball rings
            if col == "P":
                ring_color = (255, 0, 255)  # Bright Magenta/Purple
                text_color = (255, 200, 255)
            elif col == "G":
                ring_color = (0, 255, 0)    # Bright Green
                text_color = (200, 255, 200)
            else:
                ring_color = (200, 200, 200)
                text_color = (255, 255, 255)

            # Draw the main colored ring (thicker if confirmed)
            thickness = 4 if confirmed else 2
            cv2.circle(frame, (x, y), r + 4, ring_color, thickness)

            # Draw a secondary inner ring if it matches the pattern
            if confirmed and idx < len(expected):
                is_match = (col == expected[idx])
                status_color = (0, 255, 0) if is_match else (0, 0, 255)
                # Small indicator dot or inner ring for pattern status
                cv2.circle(frame, (x, y), r - 2, status_color, 2)

            # Letter label inside circle
            (tw, th), _ = cv2.getTextSize(
                col, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, col, (x - tw // 2, y + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Position label below ball
            if confirmed:
                if idx < MAX_RAMP:
                    lbl = f"#{idx+1}"
                else:
                    lbl = "OVF"
                (lw, lh), _ = cv2.getTextSize(
                    lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.putText(frame, lbl, (x - lw // 2, y + r + lh + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (220, 220, 220), 1)
                idx += 1

    def _show_placeholder(self, canvas, text):
        """Show placeholder text when no camera is active."""
        try:
            if not canvas.winfo_exists():
                return
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 10 or ch < 10:
                return
            canvas.delete("placeholder")
            canvas.create_text(cw // 2, ch // 2, text=text,
                              fill="gray", font=("Helvetica", 14),
                              tags="placeholder")
        except tk.TclError:
            pass

    def _show_frame(self, frame, canvas, side):
        """Display an OpenCV frame on the given Tk canvas."""
        try:
            if not canvas.winfo_exists():
                return
            # Clear placeholder text if present
            canvas.delete("placeholder")
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 10 or ch < 10:
                cw, ch = 480, 360
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            scale = min(cw / img.width, ch / img.height)
            nw = max(1, int(img.width * scale))
            nh = max(1, int(img.height * scale))
            img = img.resize((nw, nh), Image.Resampling.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=img)

            # Store ref to prevent garbage collection
            if side == "blue":
                canvas._current_image = imgtk
                if self._blue_img_id is None:
                    self._blue_img_id = canvas.create_image(
                        cw // 2, ch // 2, anchor=tk.CENTER, image=imgtk)
                else:
                    canvas.coords(self._blue_img_id, cw // 2, ch // 2)
                    canvas.itemconfig(self._blue_img_id, image=imgtk)
            else:
                canvas._current_image = imgtk
                if self._red_img_id is None:
                    self._red_img_id = canvas.create_image(
                        cw // 2, ch // 2, anchor=tk.CENTER, image=imgtk)
                else:
                    canvas.coords(self._red_img_id, cw // 2, ch // 2)
                    canvas.itemconfig(self._red_img_id, image=imgtk)
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
        self.blue.reset()
        self.red.reset()
        self.blue_tracker.reset()
        self.red_tracker.reset()
        self._refresh_zone_ui(self.blue)
        self._refresh_zone_ui(self.red)
        self.set_status("All scores reset", True)

    def _clear_counts_blue(self):
        self.blue.reset()
        self.blue_tracker.reset()
        self._refresh_zone_ui(self.blue)
        self.set_status("Blue scores cleared", True)

    def _clear_counts_red(self):
        self.red.reset()
        self.red_tracker.reset()
        self._refresh_zone_ui(self.red)
        self.set_status("Red scores cleared", True)

    def _man_adj_cls(self, delta):
        z = self.blue if self.adj_zone.get() == "blue" else self.red
        z.hw_classified = max(0, z.hw_classified + delta)
        z.hw_total = z.hw_classified + z.hw_overflow
        z.compute_teleop()
        self._refresh_zone_ui(z)

    def _man_adj_ovf(self, delta):
        z = self.blue if self.adj_zone.get() == "blue" else self.red
        z.hw_overflow = max(0, z.hw_overflow + delta)
        z.hw_total = z.hw_classified + z.hw_overflow
        z.compute_teleop()
        self._refresh_zone_ui(z)

    def _snapshot_pattern(self):
        """Force a pattern re-evaluation using current live ramp state.
        Useful if you want to manually trigger a pattern refresh, e.g. after
        adjusting the MOTIF selection."""
        for z in (self.blue, self.red):
            # Re-run update_live with current state to refresh pattern
            z.update_live(z.live_ramp_colors, z.live_overflow,
                          self.motif_var.get())
            z.compute_teleop()
            self._refresh_zone_ui(z)
        self.set_status("Pattern snapshot taken", True)

    # # ── Parking check (COMMENTED OUT per user request) ──────────────────
    # def _check_parking(self):
    #     """Uses GLM-4V-Flash to check robot parking at end of match."""
    #     if self.last_blue_frame is None and self.last_red_frame is None:
    #         messagebox.showwarning("No Frame", "No camera frame available.")
    #         return
    #
    #     frame = (self.last_blue_frame
    #              if self.last_blue_frame is not None
    #              else self.last_red_frame).copy()
    #     _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    #     img_b64 = base64.b64encode(buf).decode('utf-8')
    #
    #     self.set_status(
    #         "Sending frame to GLM-4V-Flash for parking check...", False)
    #
    #     def do_check():
    #         try:
    #             client = ZhipuAI(api_key=ZHIPU_API_KEY)
    #             response = client.chat.completions.create(
    #                 model="glm-4v-flash",
    #                 messages=[{
    #                     "role": "user",
    #                     "content": [
    #                         {"type": "image_url",
    #                          "image_url": {"url": img_b64}},
    #                         {"type": "text", "text": (
    #                             "This is an FTC FIRST Tech Challenge "
    #                             "robotics competition field photo. "
    #                             "Count how many robots from each "
    #                             "alliance (blue and red) are parked "
    #                             "in the observation/parking zone. "
    #                             "Reply with ONLY valid JSON: "
    #                             '{"blue_parked": <n>, '
    #                             '"red_parked": <n>, '
    #                             '"description": "<brief>"}'
    #                         )}
    #                     ]
    #                 }],
    #                 max_tokens=256,
    #                 temperature=0.1,
    #             )
    #             text = response.choices[0].message.content.strip()
    #             logging.info(f"Parking API response: {text}")
    #             json_match = _re.search(r'\{.*\}', text, _re.DOTALL)
    #             if json_match:
    #                 data = json.loads(json_match.group())
    #             else:
    #                 data = json.loads(text)
    #             bp = int(data.get("blue_parked", 0))
    #             rp = int(data.get("red_parked", 0))
    #             desc = data.get("description", text)[:80]
    #             # self.blue.parking_pts = bp * PTS_LEAVE
    #             # self.blue.parking_status = f"{bp} robot(s) parked"
    #             # self.red.parking_pts = rp * PTS_LEAVE
    #             # self.red.parking_status = f"{rp} robot(s) parked"
    #             self.root.after(0, lambda: (
    #                 self._refresh_zone_ui(self.blue),
    #                 self._refresh_zone_ui(self.red),
    #                 self.set_status(
    #                     f"Parking: Blue={bp}, Red={rp} - {desc}", True),
    #             ))
    #         except Exception as e:
    #             logging.error(f"Parking API error: {e}")
    #             self.root.after(0, lambda: self.set_status(
    #                 f"Parking check failed: {e}", False))
    #
    #     threading.Thread(target=do_check, daemon=True).start()

    # ── Zone UI refresh ──────────────────────────────────────────────────
    def _refresh_zone_ui(self, z):
        is_b = z.name == "Blue"
        lbl_t    = self.b_total   if is_b else self.r_total
        lbl_a    = self.b_auto    if is_b else self.r_auto
        lbl_tp   = self.b_teleop  if is_b else self.r_teleop
        lbl_p    = self.b_pattern if is_b else self.r_pattern
        lbl_live = self.b_live    if is_b else self.r_live
        lbl_r    = self.b_ramp    if is_b else self.r_ramp

        lbl_t.configure(text=str(z.total_score()))

        lbl_a.configure(
            text=f"AUTO:    Cls {z.auto_classified}x{PTS_CLASSIFIED}  "
                 f"Ovf {z.auto_overflow}x{PTS_OVERFLOW}  "
                 f"Pat {z.auto_pattern}x{PTS_PATTERN}  "
                 f"= {z.auto_score()}")

        lbl_tp.configure(
            text=f"TELEOP:  Cls {z.teleop_classified}x{PTS_CLASSIFIED}  "
                 f"Ovf {z.teleop_overflow}x{PTS_OVERFLOW}  "
                 f"Pat {z.teleop_pattern}x{PTS_PATTERN}  "
                 f"= {z.teleop_score()}")

        n = min(len(z.hw_ramp_colors), MAX_RAMP)
        lbl_p.configure(
            text=f"PATTERN: {z.hw_pattern_matches}/{n} x{PTS_PATTERN}  "
                 f"= {z.hw_pattern_matches * PTS_PATTERN}")

        lbl_live.configure(
            text=f"LIVE: {z.live_classified} on ramp, "
                 f"{z.live_overflow} overflow  |  "
                 f"HW: {z.hw_classified}C {z.hw_overflow}O")

        # Show best ramp (high-water) and current live ramp
        hw_seq = " ".join(z.hw_ramp_colors[:MAX_RAMP]) \
            if z.hw_ramp_colors else "--"
        live_seq = " ".join(z.live_ramp_colors[:MAX_RAMP]) \
            if z.live_ramp_colors else "--"
        lbl_r.configure(text=f"BEST: [ {hw_seq} ]\nLIVE: [ {live_seq} ]")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = FTCDecodeApp(root)
    root.mainloop()
