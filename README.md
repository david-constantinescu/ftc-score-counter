# FTC DECODE Goal Scorer

AI-powered dual-camera ball counting system for FTC (FIRST Tech Challenge) DECODE field. Uses YOLOv8 + HSV computer vision to detect and count balls in real time, served via a Flask web interface with live MJPEG streams.

## Features

- **Dual Camera Streams** — simultaneous Blue / Red goal monitoring via MJPEG
- **YOLOv8 + HSV Detection** — hybrid ball detection for reliability
- **Real-time Scoring** — SSE-powered live score updates in your browser
- **IP Camera Support** — connect DroidCam / IP cameras by address
- **Match Timer** — 2:30 countdown with audio, auto-resets trackers on match start
- **Manual Adjustments** — +1 / −1 / Clear per side
- **Auto-install** — installs Python dependencies on first run
- **GPU Acceleration** — CUDA, MPS (Apple Silicon), or CPU auto-detected

---

## Quick Install (one command)

Each script only uses tools **already on your OS** (bash/curl/winget/PowerShell) to bootstrap everything else. They:

1. Install Git & Python if missing
2. Clone the repo from GitHub (or pull updates on re-run)
3. Create a virtual environment & install dependencies
4. Start the app in the background

### macOS

```bash
curl -fsSL https://raw.githubusercontent.com/david-constantinescu/ftc-score-counter/main/install-macos.sh | bash
```

Or download and run manually:

```bash
bash install-macos.sh
```

**Installs to:** `/usr/local/ftc-score-counter`

### Linux

```bash
curl -fsSL https://raw.githubusercontent.com/david-constantinescu/ftc-score-counter/main/install-linux.sh | bash
```

Or download and run manually:

```bash
bash install-linux.sh
```

**Installs to:** `/opt/ftc-score-counter`

### Linux (Permanent — systemd service)

Installs as a system service that **starts on boot** and auto-restarts on failure. Also pulls the latest code and updates dependencies on every service start.

```bash
curl -fsSL https://raw.githubusercontent.com/david-constantinescu/ftc-score-counter/main/install-linux-service.sh | bash
```

Or download and run manually:

```bash
bash install-linux-service.sh
```

After install, manage with:

```bash
sudo systemctl status ftc-scorer     # check status
sudo journalctl -u ftc-scorer -f     # live logs
sudo systemctl restart ftc-scorer    # restart
sudo systemctl stop ftc-scorer       # stop
sudo systemctl disable ftc-scorer    # disable at boot
```

### Windows

Open **PowerShell as Administrator** and run:

```powershell
powershell -ExecutionPolicy Bypass -File install-windows.ps1
```

Or download the script and right-click → **Run with PowerShell**.

**Installs to:** `C:\ProgramData\ftc-score-counter`

---

## Manual Setup

If you prefer to set things up yourself:

```bash
# 1. Clone
git clone https://github.com/david-constantinescu/ftc-score-counter.git
cd ftc-score-counter

# 2. Create venv & install deps
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run
python app.py
```

Open **http://localhost:2016** in your browser.

---

## Usage

1. **Select Cameras** — use the Blue/Red dropdowns in the camera bar. Click **Refresh Cameras** to re-scan.
2. **IP Cameras** — click **Add IP Camera**, enter the address (e.g. `192.168.0.100:4747` for DroidCam), and it will test HTTP then HTTPS.
3. **Start Match** — click **Start**. Audio plays for 3 seconds then the 2:30 countdown and ball tracking begins.
4. **Adjust Scores** — use +1 / −1 / Clear buttons under each goal.
5. **Reset** — **Reset Timer** resets the clock; **Reset All** clears everything.

> **DroidCam Auto-connect:** On startup the app tries to connect to `192.168.0.197:4747` for the Red goal. If a DroidCam is running there it's used automatically; otherwise the next available local camera is used.

---

## Scoring Rules (FTC DECODE)

| Location | Points |
|---|---|
| Ball inside obelisk / score rail | **3 pts** |
| Ball overflowing | **1 pt** |

---

## File Structure

```
ftc-score-counter/
├── app.py                    # Main server
├── requirements.txt          # Python dependencies
├── yolov8n.pt                # YOLO model weights
├── install-macos.sh          # macOS installer
├── install-linux.sh          # Linux installer (background)
├── install-linux-service.sh  # Linux installer (systemd)
├── install-windows.ps1       # Windows installer
├── static/
│   └── ftctimer.mp3          # Match timer audio
├── templates/
│   └── index.html            # Web UI
└── README.md
```

---

## Troubleshooting

### Camera not detected
- Make sure no other app is using the camera
- Try **Refresh Cameras** in the web UI
- On Linux, check permissions: `ls -l /dev/video*`
- On macOS, allow camera access in System Settings → Privacy

### IP Camera won't connect
- Ensure DroidCam (or equivalent) is running on the phone
- Phone and computer must be on the **same network**
- Try the address in a browser: `http://<ip>:<port>/video`
- The app tests HTTP first, then HTTPS

### App won't start
- Check the log file:
  - macOS: `/usr/local/ftc-score-counter/scorer.log`
  - Linux: `/opt/ftc-score-counter/scorer.log`
  - Windows: `C:\ProgramData\ftc-score-counter\scorer.log`
  - Linux service: `sudo journalctl -u ftc-scorer --no-pager -n 50`

### Port 2016 already in use
```bash
# Linux/macOS
lsof -ti:2016 | xargs kill
# Windows (PowerShell)
netstat -ano | findstr :2016
taskkill /PID <pid> /F
```

---

## Requirements

- **Python** ≥ 3.8
- **Git**
- Python packages: `opencv-python`, `numpy`, `ultralytics`, `flask` (auto-installed)
- A webcam or IP camera (DroidCam, etc.)
