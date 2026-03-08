#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  FTC DECODE Goal Scorer — macOS Installer
# ═══════════════════════════════════════════════════════════════════════════════
#  Uses only pre-installed macOS tools (bash, curl) to bootstrap everything.
#  Installs to /usr/local/ftc-score-counter.
#  Run with:  bash install-macos.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -e

INSTALL_DIR="/usr/local/ftc-score-counter"
REPO_URL="https://github.com/david-constantinescu/ftc-score-counter.git"
VENV_DIR="$INSTALL_DIR/.venv"
LOG_FILE="$INSTALL_DIR/scorer.log"
PID_FILE="$INSTALL_DIR/scorer.pid"
PYTHON_MIN="3.8"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Root check ────────────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    info "Requesting admin privileges (needed for /usr/local install)…"
    exec sudo bash "$0" "$@"
fi

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   FTC DECODE Goal Scorer — macOS Installer${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo ""
info "Install directory : $INSTALL_DIR"
info "Repository        : $REPO_URL"
echo ""

# ── 1. Ensure Xcode CLI tools (provides git) ─────────────────────────────────
if ! xcode-select -p &>/dev/null; then
    info "Installing Xcode Command Line Tools (includes git)…"
    xcode-select --install 2>/dev/null || true
    # Wait for installation
    until xcode-select -p &>/dev/null; do sleep 5; done
    ok "Xcode CLI tools installed"
else
    ok "Xcode CLI tools already present"
fi

# ── 2. Ensure Homebrew ───────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    info "Installing Homebrew (package manager)…"
    # Homebrew's official install uses only curl + bash (both ship with macOS)
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/null
    # Add to PATH for this session
    if [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    ok "Homebrew installed"
else
    ok "Homebrew already installed"
fi

# ── 3. Ensure Python 3 ──────────────────────────────────────────────────────
install_python() {
    if command -v python3 &>/dev/null; then
        local ver
        ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
            ok "Python $ver already installed"
            return
        fi
        warn "Python $ver is too old (need >= $PYTHON_MIN)"
    fi
    info "Installing Python 3 via Homebrew…"
    brew install python@3.11
    ok "Python 3 installed"
}
install_python

# Resolve python3 path
PYTHON="$(command -v python3)"

# ── 4. Ensure Git ────────────────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
    info "Installing git via Homebrew…"
    brew install git
    ok "Git installed"
else
    ok "Git already available"
fi

# ── 5. Clone or update repo ─────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Repository exists — pulling latest changes…"
    cd "$INSTALL_DIR"
    git pull --ff-only origin main || git pull --ff-only origin master || warn "Pull failed — continuing with existing code"
    ok "Repository updated"
else
    info "Cloning repository…"
    mkdir -p "$INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
    ok "Repository cloned to $INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ── 6. Create / update virtual environment ───────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment…"
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi

info "Installing / updating Python dependencies…"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r requirements.txt
ok "Python dependencies installed"

# ── 7. Stop previous instance if running ─────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        info "Stopping previous instance (PID $OLD_PID)…"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PID_FILE"
fi

# ── 8. Launch app in background ──────────────────────────────────────────────
info "Starting FTC DECODE Goal Scorer…"
nohup "$VENV_DIR/bin/python" "$INSTALL_DIR/app.py" > "$LOG_FILE" 2>&1 &
APP_PID=$!
echo "$APP_PID" > "$PID_FILE"

# Make sure all users can read the install dir
chmod -R a+rX "$INSTALL_DIR"

sleep 2
if kill -0 "$APP_PID" 2>/dev/null; then
    echo ""
    ok "App is running in the background (PID $APP_PID)"
    info "Open http://localhost:2016 in your browser"
    info "Log file : $LOG_FILE"
    info "To stop  : sudo kill \$(cat $PID_FILE)"
    echo ""
else
    fail "App failed to start. Check $LOG_FILE for details."
fi
