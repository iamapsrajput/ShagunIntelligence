#!/bin/bash
# Shagun Intelligence Trading Platform - Apple Native Container Deployment
# Uses Apple's native container technology instead of Docker

set -euo pipefail

# Configuration
APP_NAME="shagunintelligence"
CONTAINER_NAME="shagun-trading-system"
IMAGE_NAME="shagun-intelligence:latest"
CONTAINER_PORT="8000"
HOST_PORT="8000"
DATA_VOLUME="shagun-data"
LOG_VOLUME="shagun-logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if running on macOS with container support
check_container_support() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        error "This script is designed for macOS only"
        exit 1
    fi

    # Check macOS version (container command requires macOS 13+)
    MACOS_VERSION=$(sw_vers -productVersion | cut -d. -f1)
    if [[ $MACOS_VERSION -lt 13 ]]; then
        error "Apple container technology requires macOS 13 (Ventura) or later"
        error "Current version: $(sw_vers -productVersion)"
        exit 1
    fi

    # Check if container command is available
    if ! command -v container &> /dev/null; then
        error "Apple container command not found"
        error "This may require Xcode Command Line Tools or specific macOS configuration"
        exit 1
    fi

    success "Apple container support detected"
}

# Create container configuration
create_container_config() {
    log "Creating Apple container configuration..."

    # Create temporary directory for container build
    BUILD_DIR=$(mktemp -d)

    # Create Containerfile (Apple's equivalent to Dockerfile)
    cat > "$BUILD_DIR/Containerfile" <<EOF
# Apple Container for Shagun Intelligence Trading System
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:\$PATH"

# Copy project files
COPY . /app/

# Install Python dependencies
RUN poetry config virtualenvs.create false \\
    && poetry install --only=main --no-dev

# Create non-root user for security
RUN useradd -m -u 1000 trader
RUN chown -R trader:trader /app
USER trader

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start command
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    # Copy application files to build directory
    cp -R . "$BUILD_DIR/"

    echo "$BUILD_DIR"
}

# Build Apple container image
build_container() {
    local build_dir=$1
    log "Building Apple container image..."

    cd "$build_dir"

    # Build the container image using Apple's container command
    if container build -t "$IMAGE_NAME" .; then
        success "Container image built successfully"
    else
        error "Failed to build container image"
        exit 1
    fi

    # Clean up build directory
    rm -rf "$build_dir"
}

# Create data volumes
create_volumes() {
    log "Creating container volumes..."

    # Create data volume
    if ! container volume ls | grep -q "$DATA_VOLUME"; then
        container volume create "$DATA_VOLUME"
        success "Data volume '$DATA_VOLUME' created"
    else
        log "Data volume '$DATA_VOLUME' already exists"
    fi

    # Create log volume
    if ! container volume ls | grep -q "$LOG_VOLUME"; then
        container volume create "$LOG_VOLUME"
        success "Log volume '$LOG_VOLUME' created"
    else
        log "Log volume '$LOG_VOLUME' already exists"
    fi
}

# Stop existing container
stop_existing_container() {
    log "Checking for existing container..."

    if container ps -a | grep -q "$CONTAINER_NAME"; then
        warning "Stopping existing container '$CONTAINER_NAME'..."
        container stop "$CONTAINER_NAME" || true
        container rm "$CONTAINER_NAME" || true
        success "Existing container removed"
    fi
}

# Start container
start_container() {
    log "Starting Apple container..."

    # Run the container with Apple's container command
    container run -d \\
        --name "$CONTAINER_NAME" \\
        --restart unless-stopped \\
        -p "${HOST_PORT}:${CONTAINER_PORT}" \\
        -v "${DATA_VOLUME}:/app/data" \\
        -v "${LOG_VOLUME}:/app/logs" \\
        -e "ENVIRONMENT=production" \\
        -e "DATABASE_URL=sqlite:///data/trading.db" \\
        --health-cmd="curl -f http://localhost:8000/api/v1/health || exit 1" \\
        --health-interval=30s \\
        --health-timeout=10s \\
        --health-retries=3 \\
        "$IMAGE_NAME"

    if [[ $? -eq 0 ]]; then
        success "Container started successfully"
    else
        error "Failed to start container"
        exit 1
    fi
}

# Wait for container to be healthy
wait_for_health() {
    log "Waiting for container to be healthy..."

    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if container inspect "$CONTAINER_NAME" --format='{{.State.Health.Status}}' | grep -q "healthy"; then
            success "Container is healthy"
            return 0
        fi

        log "Attempt $attempt/$max_attempts - waiting for health check..."
        sleep 10
        ((attempt++))
    done

    error "Container failed to become healthy within timeout"
    container logs "$CONTAINER_NAME"
    exit 1
}

# Create management scripts
create_management_scripts() {
    log "Creating container management scripts..."

    # Create start script
    cat > "start-apple-container.sh" <<EOF
#!/bin/bash
echo "Starting Shagun Intelligence Trading System (Apple Container)..."
container start "$CONTAINER_NAME"
echo "System started at: http://localhost:$HOST_PORT"
EOF

    # Create stop script
    cat > "stop-apple-container.sh" <<EOF
#!/bin/bash
echo "Stopping Shagun Intelligence Trading System (Apple Container)..."
container stop "$CONTAINER_NAME"
echo "System stopped"
EOF

    # Create logs script
    cat > "logs-apple-container.sh" <<EOF
#!/bin/bash
echo "Showing logs for Shagun Intelligence Trading System..."
container logs -f "$CONTAINER_NAME"
EOF

    # Create status script
    cat > "status-apple-container.sh" <<EOF
#!/bin/bash
echo "=== Container Status ==="
container ps -a | grep "$CONTAINER_NAME"
echo ""
echo "=== Health Status ==="
container inspect "$CONTAINER_NAME" --format='{{.State.Health.Status}}'
echo ""
echo "=== Resource Usage ==="
container stats "$CONTAINER_NAME" --no-stream
EOF

    # Make scripts executable
    chmod +x start-apple-container.sh stop-apple-container.sh logs-apple-container.sh status-apple-container.sh

    success "Management scripts created"
}

# Display deployment information
show_deployment_info() {
    success "Apple Container Deployment Completed!"
    echo ""
    echo "=== Deployment Information ==="
    echo "Container Name: $CONTAINER_NAME"
    echo "Image: $IMAGE_NAME"
    echo "Application URL: http://localhost:$HOST_PORT"
    echo "API Documentation: http://localhost:$HOST_PORT/docs"
    echo ""
    echo "=== Management Commands ==="
    echo "Start:   ./start-apple-container.sh"
    echo "Stop:    ./stop-apple-container.sh"
    echo "Logs:    ./logs-apple-container.sh"
    echo "Status:  ./status-apple-container.sh"
    echo ""
    echo "=== Direct Container Commands ==="
    echo "View logs:     container logs $CONTAINER_NAME"
    echo "Stop:          container stop $CONTAINER_NAME"
    echo "Start:         container start $CONTAINER_NAME"
    echo "Remove:        container rm $CONTAINER_NAME"
    echo "Shell access:  container exec -it $CONTAINER_NAME /bin/bash"
    echo ""
    echo "=== Data Persistence ==="
    echo "Data Volume: $DATA_VOLUME"
    echo "Log Volume:  $LOG_VOLUME"
    echo ""
    echo "=== Performance Comparison ==="
    echo "Apple Container vs Local Server:"
    echo "• Native macOS integration"
    echo "• Better resource isolation"
    echo "• Automatic restart on failure"
    echo "• Consistent environment"
    echo "• Easy backup and migration"
}

# Main deployment function
main() {
    log "Starting Apple Native Container Deployment"
    log "Shagun Intelligence Trading Platform"
    echo ""

    # Check if user wants to proceed
    warning "This will deploy using Apple's native container technology"
    warning "This is different from Docker and requires macOS 13+ with container support"
    echo ""
    read -p "Do you want to proceed? (y/N): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled by user"
        exit 0
    fi

    check_container_support

    local build_dir
    build_dir=$(create_container_config)

    create_volumes
    stop_existing_container
    build_container "$build_dir"
    start_container
    wait_for_health
    create_management_scripts
    show_deployment_info
}

# Handle script arguments
case "${1:-}" in
    "start")
        container start "$CONTAINER_NAME"
        echo "Container started"
        ;;
    "stop")
        container stop "$CONTAINER_NAME"
        echo "Container stopped"
        ;;
    "logs")
        container logs -f "$CONTAINER_NAME"
        ;;
    "status")
        container ps -a | grep "$CONTAINER_NAME"
        ;;
    "remove")
        container stop "$CONTAINER_NAME" || true
        container rm "$CONTAINER_NAME" || true
        container rmi "$IMAGE_NAME" || true
        echo "Container and image removed"
        ;;
    *)
        main "$@"
        ;;
esac
