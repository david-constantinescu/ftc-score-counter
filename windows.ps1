<# 
.SYNOPSIS
    FTC DECODE Goal Scorer — Windows Installer
.DESCRIPTION
    Uses only pre-installed Windows tools (PowerShell, winget) to bootstrap
    everything. Installs to C:\ProgramData\ftc-score-counter.
    Run with:  powershell -ExecutionPolicy Bypass -File install-windows.ps1
#>

$ErrorActionPreference = "Stop"

$INSTALL_DIR = "C:\ProgramData\ftc-score-counter"
$REPO_URL    = "https://github.com/david-constantinescu/ftc-score-counter.git"
$VENV_DIR    = "$INSTALL_DIR\.venv"
$LOG_FILE    = "$INSTALL_DIR\scorer.log"
$PID_FILE    = "$INSTALL_DIR\scorer.pid"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-Info  { param($m) Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Write-Ok    { param($m) Write-Host "[ OK ]  $m" -ForegroundColor Green }
function Write-Warn  { param($m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-Fail  { param($m) Write-Host "[FAIL]  $m" -ForegroundColor Red; exit 1 }

# ── Admin check ──────────────────────────────────────────────────────────────
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Info "Requesting administrator privileges…"
    Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

Write-Host ""
Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   FTC DECODE Goal Scorer — Windows Installer"               -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Info "Install directory : $INSTALL_DIR"
Write-Info "Repository        : $REPO_URL"
Write-Host ""

# ── Helper: refresh PATH for this session ────────────────────────────────────
function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

# ── 1. Ensure winget is available ────────────────────────────────────────────
$hasWinget = Get-Command winget -ErrorAction SilentlyContinue
if (-not $hasWinget) {
    Write-Warn "winget not found — attempting to register App Installer…"
    try {
        Add-AppxPackage -RegisterByFamilyName -MainPackage Microsoft.DesktopAppInstaller_8wekyb3d8bbwe -ErrorAction Stop
        Refresh-Path
    } catch {
        Write-Fail "winget is not available and could not be installed. Please install 'App Installer' from the Microsoft Store."
    }
}
Write-Ok "winget available"

# ── 2. Ensure Git ───────────────────────────────────────────────────────────
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Git via winget…"
    winget install --id Git.Git --accept-source-agreements --accept-package-agreements --silent
    Refresh-Path
    # Also check common git install path
    $gitPath = "C:\Program Files\Git\cmd"
    if (Test-Path $gitPath) { $env:Path += ";$gitPath" }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Fail "Git installation succeeded but 'git' not found in PATH. Please restart this script."
    }
    Write-Ok "Git installed"
} else {
    Write-Ok "Git already installed"
}

# ── 3. Ensure Python 3 ──────────────────────────────────────────────────────
function Get-PythonCmd {
    foreach ($cmd in @("python3", "python", "py")) {
        $c = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($c) {
            $ver = & $c --version 2>&1
            if ($ver -match "3\.\d+") { return $c.Source }
        }
    }
    return $null
}

$pythonExe = Get-PythonCmd
if (-not $pythonExe) {
    Write-Info "Installing Python 3 via winget…"
    winget install --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements --silent
    Refresh-Path
    # Common Python paths
    $pyPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311",
        "$env:LOCALAPPDATA\Programs\Python\Python311\Scripts",
        "C:\Python311",
        "C:\Python311\Scripts"
    )
    foreach ($p in $pyPaths) {
        if (Test-Path $p) { $env:Path += ";$p" }
    }
    $pythonExe = Get-PythonCmd
    if (-not $pythonExe) {
        Write-Fail "Python installation succeeded but 'python' not found in PATH. Please restart this script."
    }
    Write-Ok "Python installed"
} else {
    $pyVer = & $pythonExe --version 2>&1
    Write-Ok "Python already installed ($pyVer)"
}

# ── 4. Clone or update repo ─────────────────────────────────────────────────
if (Test-Path "$INSTALL_DIR\.git") {
    Write-Info "Repository exists — pulling latest changes…"
    Push-Location $INSTALL_DIR
    try {
        git pull --ff-only origin main 2>$null
        if ($LASTEXITCODE -ne 0) { git pull --ff-only origin master 2>$null }
    } catch {
        Write-Warn "Pull failed — continuing with existing code"
    }
    Pop-Location
    Write-Ok "Repository updated"
} else {
    Write-Info "Cloning repository…"
    if (-not (Test-Path $INSTALL_DIR)) { New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null }
    git clone $REPO_URL $INSTALL_DIR
    Write-Ok "Repository cloned to $INSTALL_DIR"
}

# ── 5. Create / update virtual environment ───────────────────────────────────
$venvPython = "$VENV_DIR\Scripts\python.exe"
$venvPip    = "$VENV_DIR\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Info "Creating Python virtual environment…"
    & $pythonExe -m venv $VENV_DIR
    Write-Ok "Virtual environment created"
} else {
    Write-Ok "Virtual environment already exists"
}

Write-Info "Installing / updating Python dependencies…"
& $venvPip install --quiet --upgrade pip
& $venvPip install --quiet -r "$INSTALL_DIR\requirements.txt"
Write-Ok "Python dependencies installed"

# ── 6. Grant all users read+execute on install dir ───────────────────────────
try {
    $acl = Get-Acl $INSTALL_DIR
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule("Users", "ReadAndExecute", "ContainerInherit,ObjectInherit", "None", "Allow")
    $acl.SetAccessRule($rule)
    Set-Acl $INSTALL_DIR $acl
} catch {
    Write-Warn "Could not set permissions — non-admin users may not be able to access $INSTALL_DIR"
}

# ── 7. Stop previous instance if running ─────────────────────────────────────
if (Test-Path $PID_FILE) {
    $oldPid = Get-Content $PID_FILE -ErrorAction SilentlyContinue
    if ($oldPid) {
        $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Info "Stopping previous instance (PID $oldPid)…"
            Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        }
    }
    Remove-Item $PID_FILE -Force -ErrorAction SilentlyContinue
}

# ── 8. Launch app in background ──────────────────────────────────────────────
Write-Info "Starting FTC DECODE Goal Scorer…"
$proc = Start-Process -FilePath $venvPython `
    -ArgumentList "$INSTALL_DIR\app.py" `
    -WorkingDirectory $INSTALL_DIR `
    -WindowStyle Hidden `
    -RedirectStandardOutput $LOG_FILE `
    -RedirectStandardError "$INSTALL_DIR\scorer-err.log" `
    -PassThru

$proc.Id | Out-File -FilePath $PID_FILE -Encoding ascii

Start-Sleep -Seconds 3
$running = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue

if ($running) {
    Write-Host ""
    Write-Ok "App is running in the background (PID $($proc.Id))"
    Write-Info "Open http://localhost:2016 in your browser"
    Write-Info "Log file : $LOG_FILE"
    Write-Info "To stop  : Stop-Process -Id (Get-Content '$PID_FILE')"
    Write-Host ""
} else {
    Write-Fail "App failed to start. Check $LOG_FILE and $INSTALL_DIR\scorer-err.log for details."
}

Write-Host "Press any key to close this window…" -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
