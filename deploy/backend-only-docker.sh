#!/bin/bash
# Shagun Intelligence Trading Platform - Backend-Only Docker Deployment
# Uses OrbStack/Docker for backend, local frontend

set -euo pipefail

# Configuration
CONTAINER_NAME="shagun-backend"
IMAGE_NAME="shagun-backend:latest"
CONTAINER_PORT="8000"
HOST_PORT="8000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Create simple backend Dockerfile
create_backend_dockerfile() {
    log "Creating backend-only Dockerfile..."

    cat > Dockerfile.backend <<EOF
FROM python:3.11.9-slim

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

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry config virtualenvs.create false \\
    && poetry install --only=main --no-dev

# Copy application code
COPY . /app/

# Create non-root user
RUN useradd -m -u 1000 trader && \\
    chown -R trader:trader /app && \\
    mkdir -p /app/data /app/logs && \\
    chown -R trader:trader /app/data /app/logs

USER trader

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    success "Backend Dockerfile created"
}

# Build backend image
build_backend() {
    log "Building backend Docker image..."

    if docker build -f Dockerfile.backend -t "$IMAGE_NAME" .; then
        success "Backend image built successfully"
    else
        echo "Failed to build backend image"
        exit 1
    fi
}

# Stop existing container
stop_existing() {
    if docker ps -a | grep -q "$CONTAINER_NAME"; then
        log "Stopping existing container..."
        docker stop "$CONTAINER_NAME" || true
        docker rm "$CONTAINER_NAME" || true
    fi
}

# Start backend container
start_backend() {
    log "Starting backend container..."

    docker run -d \\
        --name "$CONTAINER_NAME" \\
        --restart unless-stopped \\
        -p "${HOST_PORT}:${CONTAINER_PORT}" \\
        -v "$(pwd)/live_trading_1000.db:/app/live_trading_1000.db" \\
        -v "$(pwd)/logs:/app/logs" \\
        -e "ENVIRONMENT=production" \\
        -e "DATABASE_URL=sqlite:///./live_trading_1000.db" \\
        -e "TRADING_MODE=live" \\
        -e "AUTOMATED_TRADING_ENABLED=true" \\
        -e "LIVE_TRADING_ENABLED=true" \\
        "$IMAGE_NAME"

    success "Backend container started"
}

# Wait for health
wait_for_health() {
    log "Waiting for backend to be healthy..."

    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:8000/api/v1/health > /dev/null; then
            success "Backend is healthy"
            return 0
        fi

        log "Attempt $attempt/$max_attempts - waiting..."
        sleep 5
        ((attempt++))
    done

    echo "Backend failed to become healthy"
    docker logs "$CONTAINER_NAME"
    exit 1
}

# Main function
main() {
    log "Starting Backend-Only Docker Deployment"

    create_backend_dockerfile
    stop_existing
    build_backend
    start_backend
    wait_for_health

    success "Backend deployment completed!"
    echo ""
    echo "=== Backend Container ==="
    echo "Backend API: http://localhost:8000"
    echo "API Docs: http://localhost:8000/docs"
    echo "Container: $CONTAINER_NAME"
    echo ""
    echo "=== Frontend (Local) ==="
    echo "Run: cd dashboard && npm run dev"
    echo "Access: http://localhost:3000"
    echo ""
    echo "=== Management ==="
    echo "Logs: docker logs -f $CONTAINER_NAME"
    echo "Stop: docker stop $CONTAINER_NAME"
    echo "Start: docker start $CONTAINER_NAME"
}

main "$@"
