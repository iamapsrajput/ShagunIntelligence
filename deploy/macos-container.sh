#!/bin/bash
# Shagun Intelligence Trading Platform - macOS Container Deployment
# Optimized for Apple Silicon and Intel Macs

set -euo pipefail

# Configuration
APP_NAME="shagunintelligence"
APP_VERSION="1.0.0"
BUNDLE_ID="com.shagunintelligence.trading"
INSTALL_DIR="/Applications/${APP_NAME}.app"
DATA_DIR="$HOME/Library/Application Support/${APP_NAME}"
LOG_DIR="$HOME/Library/Logs/${APP_NAME}"
PYTHON_VERSION="3.11.9"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running on macOS
check_macos() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        error "This script is designed for macOS only"
        exit 1
    fi

    # Check architecture
    ARCH=$(uname -m)
    log "Detected architecture: $ARCH"

    if [[ "$ARCH" == "arm64" ]]; then
        log "Running on Apple Silicon"
        HOMEBREW_PREFIX="/opt/homebrew"
    else
        log "Running on Intel Mac"
        HOMEBREW_PREFIX="/usr/local"
    fi
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."

    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        error "Homebrew is required but not installed"
        log "Install Homebrew from: https://brew.sh"
        exit 1
    fi

    # Check for pyenv
    if ! command -v pyenv &> /dev/null; then
        warning "pyenv not found, installing..."
        brew install pyenv
    fi

    # Check for Poetry
    if ! command -v poetry &> /dev/null; then
        warning "Poetry not found, installing..."
        curl -sSL https://install.python-poetry.org | python3 -
    fi

    success "Dependencies check completed"
}

# Setup Python environment
setup_python() {
    log "Setting up Python environment..."

    # Install Python version if not available
    if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
        log "Installing Python $PYTHON_VERSION..."
        pyenv install "$PYTHON_VERSION"
    fi

    # Set local Python version
    pyenv local "$PYTHON_VERSION"

    success "Python environment ready"
}

# Create application bundle structure
create_app_bundle() {
    log "Creating macOS application bundle..."

    # Remove existing installation
    if [[ -d "$INSTALL_DIR" ]]; then
        warning "Removing existing installation..."
        sudo rm -rf "$INSTALL_DIR"
    fi

    # Create bundle structure
    sudo mkdir -p "$INSTALL_DIR/Contents/MacOS"
    sudo mkdir -p "$INSTALL_DIR/Contents/Resources"
    sudo mkdir -p "$INSTALL_DIR/Contents/Frameworks"

    # Create Info.plist
    sudo tee "$INSTALL_DIR/Contents/Info.plist" > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>shagunintelligence</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleVersion</key>
    <string>${APP_VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${APP_VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.finance</string>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsArbitraryLoads</key>
        <true/>
    </dict>
</dict>
</plist>
EOF

    success "Application bundle created"
}

# Install application
install_application() {
    log "Installing application..."

    # Create data directories
    mkdir -p "$DATA_DIR"
    mkdir -p "$LOG_DIR"

    # Copy application files
    sudo cp -R . "$INSTALL_DIR/Contents/Resources/"

    # Install Python dependencies
    cd "$INSTALL_DIR/Contents/Resources"
    sudo -E poetry install --only=main --no-dev

    # Create launcher script
    sudo tee "$INSTALL_DIR/Contents/MacOS/shagunintelligence" > /dev/null <<EOF
#!/bin/bash
export PYTHONPATH="\$PYTHONPATH:/Applications/${APP_NAME}.app/Contents/Resources"
export DATA_DIR="$DATA_DIR"
export LOG_DIR="$LOG_DIR"

cd "/Applications/${APP_NAME}.app/Contents/Resources"
exec poetry run uvicorn app.main:app --host 127.0.0.1 --port 8000
EOF

    sudo chmod +x "$INSTALL_DIR/Contents/MacOS/shagunintelligence"

    success "Application installed"
}

# Create launchd service
create_service() {
    log "Creating system service..."

    SERVICE_PLIST="$HOME/Library/LaunchAgents/${BUNDLE_ID}.plist"

    tee "$SERVICE_PLIST" > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${BUNDLE_ID}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Applications/${APP_NAME}.app/Contents/MacOS/shagunintelligence</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/stderr.log</string>
    <key>WorkingDirectory</key>
    <string>/Applications/${APP_NAME}.app/Contents/Resources</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${HOMEBREW_PREFIX}/bin:/usr/bin:/bin</string>
        <key>DATA_DIR</key>
        <string>${DATA_DIR}</string>
        <key>LOG_DIR</key>
        <string>${LOG_DIR}</string>
    </dict>
</dict>
</plist>
EOF

    # Load the service
    launchctl load "$SERVICE_PLIST"

    success "Service created and loaded"
}

# Main deployment function
main() {
    log "Starting macOS container deployment for Shagun Intelligence Trading Platform"

    check_macos
    check_dependencies
    setup_python
    create_app_bundle
    install_application
    create_service

    success "Deployment completed successfully!"
    log "Application is now running at: http://localhost:8000"
    log "Data directory: $DATA_DIR"
    log "Log directory: $LOG_DIR"
    log "To stop the service: launchctl unload ~/Library/LaunchAgents/${BUNDLE_ID}.plist"
    log "To start the service: launchctl load ~/Library/LaunchAgents/${BUNDLE_ID}.plist"
}

# Run main function
main "$@"
