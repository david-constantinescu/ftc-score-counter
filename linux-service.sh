#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  FTC DECODE Goal Scorer — Linux Permanent (systemd) Installer
# ═══════════════════════════════════════════════════════════════════════════════
#  Installs the app to /opt/ftc-score-counter and registers a systemd service
#  that starts on boot and restarts on failure.
#  Run with:  bash install-linux-service.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -e

INSTALL_DIR="/opt/ftc-score-counter"
REPO_URL="https://github.com/david-constantinescu/ftc-score-counter.git"
VENV_DIR="$INSTALL_DIR/.venv"
SERVICE_NAME="ftc-scorer"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Root check ────────────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    info "Requesting root privileges…"
    exec sudo bash "$0" "$@"
fi

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   FTC DECODE Goal Scorer — Linux Service Installer${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo ""
info "Install directory : $INSTALL_DIR"
info "Service name      : $SERVICE_NAME"
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
fi
ok "curl ready"

# ── 2. Ensure git ───────────────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
    info "Installing git…"
    pkg_install git
fi
ok "Git ready"

# ── 3. Ensure Python 3 + venv ───────────────────────────────────────────────
install_python() {
    local need=false
    if ! command -v python3 &>/dev/null; then
        need=true
    elif ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
        need=true
    fi
    if ! python3 -m venv --help &>/dev/null 2>&1; then
        need=true
    fi

    if $need; then
        info "Installing Python 3 and venv…"
        case "$PKG" in
            apt)    pkg_install python3 python3-venv python3-pip python3-dev ;;
            dnf)    pkg_install python3 python3-pip python3-devel ;;
            pacman) pkg_install python python-pip ;;
            zypper) pkg_install python3 python3-pip python3-venv ;;
        esac
    fi
    command -v python3 &>/dev/null || fail "Python 3 installation failed"
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

# ── 5. Virtual environment & deps ───────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment…"
    "$PYTHON" -m venv "$VENV_DIR"
fi
ok "Virtual environment ready"

info "Installing / updating Python dependencies…"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r requirements.txt
ok "Python dependencies installed"

# Make sure all users can read the install dir
chmod -R a+rX "$INSTALL_DIR"

# ── 6. Create systemd service ───────────────────────────────────────────────
info "Creating systemd service: $SERVICE_NAME"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=FTC DECODE Goal Scorer
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStartPre=/bin/bash -c 'cd $INSTALL_DIR && git pull --ff-only origin main || git pull --ff-only origin master || true'
ExecStartPre=$VENV_DIR/bin/pip install --quiet -r $INSTALL_DIR/requirements.txt
ExecStart=$VENV_DIR/bin/python $INSTALL_DIR/app.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

ok "Service file written to $SERVICE_FILE"

# ── 7. Enable and start ─────────────────────────────────────────────────────
info "Reloading systemd and enabling service…"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# Stop if already running, then start fresh
systemctl stop "$SERVICE_NAME" 2>/dev/null || true
systemctl start "$SERVICE_NAME"

sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo ""
    ok "Service '$SERVICE_NAME' is running and enabled at boot!"
    info "Open http://localhost:2016 in your browser"
    echo ""
    info "Useful commands:"
    info "  Status  : sudo systemctl status $SERVICE_NAME"
    info "  Logs    : sudo journalctl -u $SERVICE_NAME -f"
    info "  Stop    : sudo systemctl stop $SERVICE_NAME"
    info "  Restart : sudo systemctl restart $SERVICE_NAME"
    info "  Disable : sudo systemctl disable $SERVICE_NAME"
    info "  Remove  : sudo rm $SERVICE_FILE && sudo systemctl daemon-reload"
    echo ""
else
    warn "Service may still be starting. Check status:"
    echo "  sudo systemctl status $SERVICE_NAME"
    echo "  sudo journalctl -u $SERVICE_NAME --no-pager -n 30"
fi
