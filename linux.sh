#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  FTC DECODE Goal Scorer — Linux Installer
# ═══════════════════════════════════════════════════════════════════════════════
#  Uses only pre-installed Linux tools (bash, apt/dnf/pacman, curl) to
#  bootstrap everything.  Installs to /opt/ftc-score-counter.
#  Run with:  bash install-linux.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -e

INSTALL_DIR="/opt/ftc-score-counter"
REPO_URL="https://github.com/david-constantinescu/ftc-score-counter.git"
VENV_DIR="$INSTALL_DIR/.venv"
LOG_FILE="$INSTALL_DIR/scorer.log"
PID_FILE="$INSTALL_DIR/scorer.pid"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Root check ────────────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    info "Requesting root privileges (needed for /opt install)…"
    exec sudo bash "$0" "$@"
fi

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   FTC DECODE Goal Scorer — Linux Installer${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo ""
info "Install directory : $INSTALL_DIR"
info "Repository        : $REPO_URL"
echo ""

# ── Detect package manager ───────────────────────────────────────────────────
PKG=""
if command -v apt-get &>/dev/null; then
    PKG="apt"
elif command -v dnf &>/dev/null; then
    PKG="dnf"
elif command -v pacman &>/dev/null; then
    PKG="pacman"
elif command -v zypper &>/dev/null; then
    PKG="zypper"
else
    fail "No supported package manager found (apt, dnf, pacman, zypper)"
fi
ok "Package manager: $PKG"

pkg_install() {
    case "$PKG" in
        apt)    apt-get update -qq && apt-get install -y -qq "$@" ;;
        dnf)    dnf install -y -q "$@" ;;
        pacman) pacman -S --noconfirm --needed "$@" ;;
        zypper) zypper install -y "$@" ;;
    esac
}

# ── 1. Ensure curl ──────────────────────────────────────────────────────────
if ! command -v curl &>/dev/null; then
    info "Installing curl…"
    pkg_install curl
    ok "curl installed"
else
    ok "curl already available"
fi

# ── 2. Ensure git ───────────────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
    info "Installing git…"
    pkg_install git
    ok "Git installed"
else
    ok "Git already available"
fi

# ── 3. Ensure Python 3 + venv ───────────────────────────────────────────────
install_python() {
    local need_python=false
    local need_venv=false

    if ! command -v python3 &>/dev/null; then
        need_python=true
    elif ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
        warn "Python too old, upgrading…"
        need_python=true
    fi

    # Check if venv module works
    if ! python3 -m venv --help &>/dev/null 2>&1; then
        need_venv=true
    fi

    if $need_python || $need_venv; then
        info "Installing Python 3 and venv…"
        case "$PKG" in
            apt)    pkg_install python3 python3-venv python3-pip python3-dev ;;
            dnf)    pkg_install python3 python3-pip python3-devel ;;
            pacman) pkg_install python python-pip ;;
            zypper) pkg_install python3 python3-pip python3-venv ;;
        esac
    fi

    if ! command -v python3 &>/dev/null; then
        fail "Python 3 installation failed"
    fi
    ok "Python $(python3 --version 2>&1 | awk '{print $2}') ready"
}
install_python

PYTHON="$(command -v python3)"

# ── 4. Clone or update repo ─────────────────────────────────────────────────
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

# ── 5. Create / update virtual environment ───────────────────────────────────
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

# ── 6. Stop previous instance if running ─────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        info "Stopping previous instance (PID $OLD_PID)…"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PID_FILE"
fi

# ── 7. Launch app in background ──────────────────────────────────────────────
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
