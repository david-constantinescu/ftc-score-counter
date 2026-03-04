# FTC Decode Field Scorer

An AI-powered scoring system for FTC (FIRST Tech Challenge) "Decode" field using computer vision and the Ollama qwen3-vl:2b model.

## Features

- **Live Video Feed**: Real-time camera capture from selectable cameras
- **AI Vision Analysis**: Uses Qwen 3 VL 2B model via Ollama for visual recognition
- **Automatic Scoring**: Counts balls/artifacts in the obelisk (3 pts inside, 1 pt overflow)
- **JSON Output**: Displays structured AI analysis results
- **Auto-Install**: Automatically installs missing Python dependencies
- **Debug Logging**: Comprehensive logging to `debug.log`

## Prerequisites

### 1. Python 3.8+
Ensure Python is installed. Check version:
```bash
python --version
```

### 2. Ollama
Install Ollama from [ollama.com](https://ollama.com)

**Windows**: Download and run the installer
**macOS**: `brew install ollama`
**Linux**: `curl -fsSL https://ollama.com/install.sh | sh`

### 3. Pull the Model
```bash
ollama pull qwen3-vl:2b
```

This downloads the Qwen 3 Vision-Language 2B model (~1.8GB)

### 4. Start Ollama Service
Make sure Ollama is running:
- **Windows/Mac**: The Ollama app should be running (check system tray)
- **Linux**: Run `ollama serve` in a terminal

## Installation

### Quick Start

1. Clone or download this repository
2. Run the test script to verify Ollama:
   ```bash
   python test_ollama.py
   ```
3. If the test passes, run the application:
   ```bash
   python app.py
   ```

### Manual Installation

If you prefer to install dependencies manually:

```bash
pip install -r requirements.txt
```

The requirements are:
- `opencv-python` - Video capture and processing
- `pillow` - Image handling
- `ollama` - Ollama Python client

## Usage

1. **Start the Application**:
   
   **Option 1** - Direct (may show numpy warnings):
   ```bash
   python app.py
   ```
   
   **Option 2** - Clean launcher (suppresses warnings):
   ```bash
   # Windows
   run.bat
   
   # Or use the Python launcher
   python run_clean.py
   ```

2. **Select Camera**: 
   - Use the dropdown to choose your camera (usually 0 for built-in, 1+ for external)

3. **Start Recording**: 
   - Click "Start Camera" button

4. **View Results**:
   - Live video feed shows on the left
   - Score display on the right (updated every 3 seconds)
   - JSON analysis output shows AI reasoning

## Scoring Rules (FTC Decode Field)

Based on FTC scoring specifications:

- **Balls Inside Obelisk/Score Rails**: **3 Points** each
- **Balls Overflowing**: **1 Point** each

The AI model analyzes the video frame to identify:
- Location of the obelisk structure
- Number of artifacts/balls inside
- Number of artifacts/balls overflowing

## File Structure

```
ai-score/
├── app.py              # Main application
├── test_ollama.py      # Connection test script
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── debug.log          # Generated log file
```

## Troubleshooting

### Camera Issues

**Problem**: Camera not opening
- Check if another application is using the camera
- Try selecting a different camera index (0, 1, 2, etc.)
- On Windows, close Teams/Zoom/Skype which may lock the camera
- Check camera permissions in system settings

**Problem**: "Could not open camera" error
```bash
# Test camera access with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Failed'); cap.release()"
```

### Ollama Issues

**Problem**: "Connection refused" or "Cannot connect to Ollama"
- Ensure Ollama service is running:
  - Windows/Mac: Check system tray for Ollama icon
  - Linux: Run `ollama serve` in a separate terminal
- Verify Ollama is listening:
  ```bash
  curl http://localhost:11434/api/tags
  ```

**Problem**: Model not found
```bash
# List installed models
ollama list

# Pull the required model
ollama pull qwen3-vl:2b

# Verify model works
ollama run qwen3-vl:2b "Hello"
```

**Problem**: Model pulls but app doesn't recognize it
- Check the exact model name in `ollama list`
- It should show as `qwen3-vl:2b`
- If it shows differently, update `model_name` in `app.py` line 50

### AI Response Issues

**Problem**: Invalid JSON errors
- The model occasionally returns text instead of JSON
- The app attempts to clean markdown formatting
- Check `debug.log` for the raw response
- Increase lighting/image quality for better AI performance

**Problem**: Incorrect counts
- The 2B model is small and may not be perfectly accurate
- Ensure good lighting and clear view of the field
- Consider using a larger model like `qwen3-vl:8b` (requires more RAM)

### Numpy Warnings (Python 3.14)

**Problem**: Warnings about "Numpy built with MINGW-W64 on Windows 64 bits is experimental"

**Status**: **HARMLESS** - These are just warnings, the app works perfectly fine

**Solution to hide warnings**:
```bash
# Use the clean launcher
run.bat           # Windows
python run_clean.py   # All platforms
```

Or set environment variable before running:
```powershell
$env:PYTHONWARNINGS='ignore'; python app.py
```

### Python Environment Issues

**Problem**: ModuleNotFoundError
```bash
# Install missing modules
pip install opencv-python pillow ollama setuptools

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

**Problem**: pkg_resources deprecation warning
- This is just a warning and doesn't affect functionality
- The app has been updated to use `importlib` instead

### Performance Issues

**Problem**: App is slow or laggy
- Reduce video resolution in the code (see `update_video()` method)
- Increase AI interval from 3 to 5+ seconds
- Use a GPU-enabled version of Ollama if available
- Consider a smaller model or optimize the prompt

## Debug Logs

The application creates a `debug.log` file with detailed information:

```bash
# View recent logs
tail -f debug.log  # Linux/Mac
Get-Content debug.log -Tail 50 -Wait  # PowerShell
```

Log entries include:
- Model checking and pulling status
- Camera initialization
- AI request/response details
- Error messages with stack traces

## Testing

### Test Ollama Connection
```bash
python test_ollama.py
```

This script:
1. Checks Ollama connectivity
2. Lists available models
3. Verifies `qwen3-vl:2b` exists
4. Tests a simple chat interaction

### Test Camera Access
```bash
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

This shows available camera indices.

## Advanced Configuration

### Change AI Model

Edit `app.py` line 50:
```python
self.model_name = "qwen3-vl:8b"  # Larger, more accurate model
```

Then pull the new model:
```bash
ollama pull qwen3-vl:8b
```

### Adjust Processing Interval

Edit `app.py` line 49:
```python
self.ai_interval = 5.0  # Process every 5 seconds instead of 3
```

### Customize Prompt

Edit the prompt in `process_frame_with_ai()` method (line 182+) to:
- Add more detailed instructions
- Request additional analysis
- Change output format

## System Requirements

- **OS**: Windows 10+, macOS 11+, Linux
- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk**: 3GB for model and dependencies
- **Camera**: Any USB webcam or built-in camera
- **Network**: For initial model download only

## Credits

- **Ollama**: Local LLM runtime ([ollama.com](https://ollama.com))
- **Qwen VL**: Vision-language model by Alibaba
- **OpenCV**: Computer vision library
- **FIRST**: Robotics competition organization

## License

This project is for educational and competition purposes. Check FTC rules for official competition usage.
