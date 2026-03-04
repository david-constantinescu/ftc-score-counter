import cv2
import numpy as np
import threading
import time
import queue
import logging
import os
import signal
import subprocess
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
from collections import OrderedDict

# ── Configuration ─────────────────────────────────────────────────────────────
USERNAME = "pulert"
PASSWORD = "softhoarderscnfbmuzeu"
SECRET_KEY = "decoder_secret_key_change_me"

# ── Game Constants (Matching app.py) ──────────────────────────────────────────
MOTIFS = ["GPP", "PGP", "PPG"]
MAX_RAMP = 9

# HSV colour ranges
PURPLE_LOW  = np.array([128, 60, 60])
PURPLE_HIGH = np.array([152, 255, 255]) 
GREEN_LOW   = np.array([45, 80, 60])
GREEN_HIGH  = np.array([80, 255, 255]) 

# Size filtering
MIN_BALL_AREA    = 600
MAX_BALL_AREA    = 60000 
MIN_RADIUS       = 14
MAX_RADIUS       = 150   

# Morphology
KERN_SIZE = (7, 7)     
MIN_CIRCULARITY  = 0.25
CONFIRM_FRAMES   = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# ── Global State ──────────────────────────────────────────────────────────────
class GameState:
    def __init__(self):
        self.timer_seconds = 150 # 2:30
        self.timer_running = False
        self.mode = "AUTO" # AUTO or TELEOP
        self.motif = "GPP"
        self.blue_cam_src = 0
        self.red_cam_src = 1
        self.lock = threading.Lock()

state = GameState()

# ── Core Logic Classes (Adapted from app.py) ──────────────────────────────────

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

        objectIDs = list(self.objects.keys())
        objectCentroids = []
        for objectID in objectIDs:
            objectCentroids.append([self.objects[objectID]["x"], self.objects[objectID]["y"]])
        objectCentroids = np.array(objectCentroids)
        
        inputCentroids = []
        for d in dets:
            inputCentroids.append([d["x"], d["y"]])
        inputCentroids = np.array(inputCentroids)

        D = np.linalg.norm(objectCentroids[:, np.newaxis] - inputCentroids[np.newaxis, :], axis=2)
        
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.max_dist:
                continue
                
            objectID = objectIDs[row]
            self.objects[objectID].update(dets[col])
            self.objects[objectID]["age"] = self.objects[objectID].get("age", 0) + 1
            self.disappeared[objectID] = 0
            
            if self.objects[objectID]["age"] >= CONFIRM_FRAMES and objectID not in self.confirmed:
                self.confirmed[objectID] = self.objects[objectID]["color"]

            usedRows.add(row)
            usedCols.add(col)

        usedRows_set = set(range(D.shape[0]))
        unusedRows = usedRows_set.difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.max_disappeared:
                self.objects.pop(objectID, None)
                self.disappeared.pop(objectID, None)

        usedCols_set = set(range(D.shape[1]))
        unusedCols = usedCols_set.difference(usedCols)
        for col in unusedCols:
            self._register(dets[col])

        return self.objects

    def _register(self, d):
        d = dict(d)
        d["age"] = 1
        self.objects[self.next_id] = d
        self.disappeared[self.next_id] = 0
        self.next_id += 1

class ZoneData:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.live_classified = 0
        self.live_overflow = 0
        self.live_total = 0
        self.live_ramp_colors = []
        
        self.hw_classified = 0
        self.hw_overflow = 0
        self.hw_total = 0
        self.hw_ramp_colors = []
        self.hw_pattern_matches = 0
        
        self.overflow_ids = set()
        
        self.auto_classified = 0
        self.auto_overflow = 0
        self.auto_total = 0
        self.auto_pattern = 0
        self.auto_ramp_colors = []

    def update(self, classified_balls, overflow_balls, motif_str):
        # Update live state
        self.live_ramp_colors = [b['color'] for b in classified_balls]
        self.live_classified = len(classified_balls)
        
        for ball in overflow_balls:
            if 'id' in ball:
                 self.overflow_ids.add(ball['id'])
        
        self.live_overflow = len(self.overflow_ids)
        self.live_total = self.live_classified + self.live_overflow # Note: Using live here might differ from HW logic slightly

        # Update High Water Mark
        if self.live_classified > self.hw_classified:
            self.hw_classified = self.live_classified
        
        # Accumulate overflow (never decreases)
        self.hw_overflow = len(self.overflow_ids)
        
        # Match Pattern
        current_motif = list(motif_str * 3) # upscale to 9
        matches = 0
        for i, color in enumerate(self.live_ramp_colors[:MAX_RAMP]):
            if i < len(current_motif) and color == current_motif[i]:
                matches += 1
        
        if matches > self.hw_pattern_matches:
            self.hw_pattern_matches = matches
            self.hw_ramp_colors = list(self.live_ramp_colors)
            
        # Calculate Total Score
        self.score_classified = self.hw_classified * 3
        self.score_overflow = self.hw_overflow * 1
        self.score_pattern = self.hw_pattern_matches * 2
        self.score_total = self.score_classified + self.score_overflow + self.score_pattern

class BallDetector:
    def _hsv_detect(self, hsv, low, high, label):
        mask = cv2.inRange(hsv, low, high)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERN_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BALL_AREA or area > MAX_BALL_AREA: continue
            
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < MIN_CIRCULARITY: continue
            
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < MIN_RADIUS or r > MAX_RADIUS: continue
            
            out.append({"x": int(cx), "y": int(cy), "radius": int(r), "color": label})
        return out

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        detections += self._hsv_detect(hsv, PURPLE_LOW, PURPLE_HIGH, "P")
        detections += self._hsv_detect(hsv, GREEN_LOW, GREEN_HIGH, "G")
        return detections


class RampDetector:
    def __init__(self):
        self.last_y = 400  # Default fallback
        self.alpha  = 0.1  # Smoothing factor (EWMA)

    def detect(self, frame):
        """Find the horizontal classifier bar (approx Y=300-480)."""
        height, width = frame.shape[:2]
        
        # Region of interest: lower half of the frame (where the ramp bar is expected)
        roi_y_start = int(height * 0.5)
        roi = frame[roi_y_start:, :]  # Lower half only

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better edge detection? Just Canny is usually good
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough Lines: focus on horizontal lines
        # Threshold: 80 votes, minLength: 1/3 of width, maxGap: 20px
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                                minLineLength=width//3, maxLineGap=20)
        
        detected_y = None
        if lines is not None:
            # Filter lines that are roughly horizontal
            candidates = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 10:  # Valid horizontal line (slope close to 0)
                    candidates.append((y1 + y2) / 2)
            
            if candidates:
                # Average all strong candidate lines
                detected_y = int(np.mean(candidates)) + roi_y_start

        # Smooth the result
        if detected_y is not None:
            # Simple exponential smoothing to prevent jitter
            self.last_y = int((1 - self.alpha) * self.last_y + self.alpha * detected_y)
        
        # Hard clamp (sanity check)
        self.last_y = max(300, min(self.last_y, 470))
        return self.last_y


# Global instances
blue_zone = ZoneData("Blue")
red_zone = ZoneData("Red")
detector = BallDetector()

# ── Video Streaming ───────────────────────────────────────────────────────────
class VideoStream:
    def __init__(self, src=0, name="Cam"):
        self.src = src
        self.name = name
        self.cap = None
        self.tracker = CentroidTracker()
        self.ramp_det = RampDetector()  # Add dynamic ramp detector
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self._pending_src = None  # set by change_source(), read by update()
        
        
        # Initialize capture
        self._start_capture()
        
        # Start background thread
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def _start_capture(self):
        """Internal helper to start/restart capture."""
        if self.cap:
            self.cap.release()
            
        # Handle source types
        source = self.src
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        logging.info(f"Starting {self.name} stream on source: {source}")
        self.cap = cv2.VideoCapture(source)
        # NOTE: do NOT sleep here — this may be called while self.lock is held

    def change_source(self, new_src):
        """Thread-safe source change — sets a flag; update() applies it."""
        logging.info(f"Changing {self.name} source to: {new_src}")
        self._pending_src = new_src  # CPython GIL makes single assignment atomic

    def update(self):
        while True:
            # Apply any pending source change requested by change_source()
            pending = self._pending_src
            if pending is not None:
                self._pending_src = None
                if self.cap:
                    self.cap.release()
                self.src = pending
                self.tracker.reset()
                src = int(pending) if isinstance(pending, str) and pending.isdigit() else pending
                self.cap = cv2.VideoCapture(src)
                continue

            if self.stopped:
                time.sleep(0.1)
                continue

            if not self.cap or not self.cap.isOpened():
                time.sleep(1)
                src = int(self.src) if isinstance(self.src, str) and self.src.isdigit() else self.src
                self.cap = cv2.VideoCapture(src)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Resize based on aspect ratio to keep consistent width (640)
            # DroidCam usually 480p or 720p. 
            # Force 640x480 for consistency with logic
            try:
                frame = cv2.resize(frame, (640, 480))
            except Exception as e:
                logging.error(f"Resize error: {e}")
                continue
            
            # Detect ramp line (dynamic Y limit)
            ramp_y = self.ramp_det.detect(frame)

            # ── Processing ──
            detections = detector.detect(frame)
            objects = self.tracker.update(detections)
            
            valid_objs = []
            for oid, data in objects.items():
                data['id'] = oid
                valid_objs.append(data)
            
            # Sort by Y (Top to Bottom) for Ramp Position
            valid_objs.sort(key=lambda b: b['y'])
            
            # Split into Classified (first 9) and Overflow (rest)
            classified = [b for b in valid_objs if b['y'] < ramp_y][:MAX_RAMP]
            overflow = [b for b in valid_objs if b['y'] >= ramp_y] + [b for b in valid_objs if b['y'] < ramp_y][MAX_RAMP:]
            
            # Update Zone Data
            target_zone = blue_zone if "Blue" in self.name else red_zone
            # We don't need `with state.lock` for reading but update modifies shared state
            target_zone.update(classified, overflow, state.motif)
            
            # Draw
            for b in valid_objs:
                color = (0, 255, 0) if b['color'] == 'G' else (255, 0, 255)
                cv2.circle(frame, (b['x'], b['y']), b['radius'], color, 2)
                cv2.putText(frame, f"{b['id']}", (b['x']-5, b['y']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Draw Divider and Ramp slots guide (Dynamic Y)
            cv2.line(frame, (0, ramp_y), (640, ramp_y), (0, 255, 255), 2)
            cv2.putText(frame, "RAMP LIMIT", (10, ramp_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw Ramp Slot count
             # Show number of balls on ramp
            cv2.putText(frame, f"Ramp Count: {len(classified)}/{MAX_RAMP}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


            with self.lock:
                self.frame = frame.copy()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                # Return a placeholder black image if no frame yet
                blank = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank, "Starting Camera...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()
                
            ret, buffer = cv2.imencode('.jpg', self.frame)
            return buffer.tobytes()

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

# ── Pre-scan cameras BEFORE opening any VideoCapture in VideoStream ─────────
# This MUST run before VideoStream.__init__ to avoid concurrent device access
_CAMERA_LIST: list = []
for _i in range(8):
    try:
        _cap = cv2.VideoCapture(_i)
        if _cap.isOpened():
            _CAMERA_LIST.append({"id": str(_i), "label": f"Camera {_i}"})
        _cap.release()
    except Exception:
        pass
logging.info(f"Pre-scanned cameras: {_CAMERA_LIST}")

# Start Streams (opens physical cameras; list_cameras() won't touch them again)
_blue_src = int(_CAMERA_LIST[0]["id"]) if _CAMERA_LIST else 0
_red_src  = int(_CAMERA_LIST[1]["id"]) if len(_CAMERA_LIST) > 1 else 0
blue_stream = VideoStream(_blue_src, "Blue")
red_stream  = VideoStream(_red_src,  "Red")

# ── Audio Player ──────────────────────────────────────────────────────────────
class AudioPlayer:
    def __init__(self, filename="ftctimer.mp3"):
        self.filename = filename if os.path.exists(filename) else None
        self.proc = None
        self.paused = False

    def play(self):
        self.stop()
        if not self.filename: return
        try:
            self.proc = subprocess.Popen(["afplay", self.filename], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
            self.paused = False
        except Exception as e:
            logging.error(f"Audio play error: {e}")

    def pause(self):
        if self.proc and self.proc.poll() is None and not self.paused:
            try:
                os.kill(self.proc.pid, signal.SIGSTOP)
                self.paused = True
            except: pass

    def resume(self):
         if self.proc and self.paused:
            try:
                os.kill(self.proc.pid, signal.SIGCONT)
                self.paused = False
            except: pass
            
    def stop(self):
        if self.proc:
            try:
                if self.paused:
                    os.kill(self.proc.pid, signal.SIGCONT)
                self.proc.terminate()
            except: pass
            self.proc = None
            self.paused = False

audio = AudioPlayer()

# ── Timer Thread ──────────────────────────────────────────────────────────────
def timer_loop():
    while True:
        time.sleep(1)
        if state.timer_running and state.timer_seconds > 0:
            state.timer_seconds -= 1
            if state.timer_seconds == 0:
                state.timer_running = False
                # Optionally play end sound here if different file existed

threading.Thread(target=timer_loop, daemon=True).start()

# ── Detect connected cameras (returns pre-scanned cached list, no new probing) ─
def list_cameras():
    return _CAMERA_LIST

# ── Routes ────────────────────────────────────────────────────────────────────

def check_auth():
    return session.get('logged_in')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == USERNAME and request.form['password'] == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if not check_auth(): return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/api/cameras')
def get_cameras():
    if not check_auth(): return jsonify([])
    return jsonify(list_cameras())

@app.route('/video_feed/<zone>')
def video_feed(zone):
    if not check_auth(): return Response("Unauthorized", 401)
    
    stream = blue_stream if zone == 'blue' else red_stream
    
    def gen():
        while True:
            frame = stream.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
                
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    if not check_auth(): return jsonify({})
    
    return jsonify({
        "timer": f"{state.timer_seconds // 60}:{state.timer_seconds % 60:02d}",
        "timer_running": state.timer_running,
        "motif": state.motif,
        "blue": {
            "p_score": blue_zone.score_pattern,
            "c_score": blue_zone.score_classified,
            "o_score": blue_zone.score_overflow,
            "total": blue_zone.score_total,
            "ramp": "".join(blue_zone.hw_ramp_colors)
        },
        "red": {
            "p_score": red_zone.score_pattern,
            "c_score": red_zone.score_classified,
            "o_score": red_zone.score_overflow,
            "total": red_zone.score_total,
            "ramp": "".join(red_zone.hw_ramp_colors)
        }
    })

@app.route('/api/control', methods=['POST'])
def control():
    if not check_auth(): return jsonify({"error": "auth"}), 401
    data = request.json
    action = data.get('action')
    
    if action == 'toggle_timer':
        # TOGGLE LOGIC:
        # If currently running -> PAUSE
        # If currently paused -> RESUME
        # If fresh start (150s) -> START WITH AUDIO
        
        if state.timer_running:
             # PAUSE
             state.timer_running = False
             audio.pause()
        else:
             # START / RESUME
             if state.timer_seconds == 150:
                 # Fresh Start
                 audio.play()
                 # 3-second delay for audio intro
                 def delayed_start():
                     time.sleep(3.0)
                     state.timer_running = True
                 threading.Thread(target=delayed_start).start()
             else:
                 # Normal Resume
                 audio.resume()
                 state.timer_running = True

    elif action == 'reset_timer':
        state.timer_seconds = 150
        state.timer_running = False
        audio.stop()
        
    elif action == 'set_motif':
        state.motif = data.get('motif', 'GPP')
        
    elif action == 'reset_all':
        blue_zone.reset()
        red_zone.reset()
        state.timer_seconds = 150
        state.timer_running = False
        blue_stream.tracker.reset()
        red_stream.tracker.reset()
        audio.stop()
        
    elif action == 'clear_blue':
        blue_zone.reset()
        blue_stream.tracker.reset()
        
    elif action == 'clear_red':
        red_zone.reset()
        red_stream.tracker.reset()
        
    elif action == 'set_cam':
        # Safely switch camera source
        zone = data.get('zone')
        src = data.get('src')
        
        # Determine strict source type (int or str)
        # If it looks like an int, cast it unless it's a URL (contains http/rtsp)
        try:
            if isinstance(src, str) and src.isdigit():
                 final_src = int(src)
            else:
                 final_src = src
        except:
            final_src = src
            
        if zone == 'blue':
            blue_stream.change_source(final_src)
        elif zone == 'red':
            red_stream.change_source(final_src)
            
    return jsonify({"success": True})

if __name__ == '__main__':
    # Run on all interfaces for access from other devices
    app.run(host='0.0.0.0', port=2000, threaded=True, debug=False)
