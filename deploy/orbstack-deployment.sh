#!/bin/bash
# Shagun Intelligence Trading Platform - OrbStack Deployment
# Uses OrbStack for Docker-compatible container deployment on macOS

set -euo pipefail

# Configuration
APP_NAME="shagunintelligence"
CONTAINER_NAME="shagun-trading-system"
IMAGE_NAME="shagun-intelligence:latest"
CONTAINER_PORT="8000"
HOST_PORT="8000"
FRONTEND_PORT="3000"
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

# Check if OrbStack is running
check_orbstack() {
    if ! command -v docker &> /dev/null; then
        error "Docker command not found. Please ensure OrbStack is installed and running."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start OrbStack."
        exit 1
    fi

    # Check if OrbStack is specifically running
    if docker info 2>/dev/null | grep -q "orbstack"; then
        success "OrbStack detected and running"
    else
        log "Docker is running (may be OrbStack or Docker Desktop)"
    fi
}

# Create Dockerfile for the application
create_dockerfile() {
    log "Creating optimized Dockerfile for OrbStack..."

    cat > Dockerfile <<EOF
# Multi-stage build for Shagun Intelligence Trading System
FROM python:3.11.9-slim as backend-builder

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

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry config virtualenvs.create false \\
    && poetry install --only=main

# Frontend build stage
FROM node:18-alpine as frontend-builder

WORKDIR /app/dashboard

# Copy package files
COPY dashboard/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY dashboard/ ./

# Build frontend
ENV VITE_API_BASE_URL=http://localhost:8000
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npm run build

# Final runtime stage
FROM python:3.11.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app/

# Copy built frontend
COPY --from=frontend-builder /app/dashboard/dist /app/dashboard/dist

# Create non-root user for security
RUN useradd -m -u 1000 trader && \\
    chown -R trader:trader /app && \\
    mkdir -p /app/data /app/logs && \\
    chown -R trader:trader /app/data /app/logs

USER trader

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    success "Dockerfile created"
}

# Create docker-compose file for easier management
create_docker_compose() {
    log "Creating docker-compose configuration..."

    cat > docker-compose.yml <<EOF
version: '3.8'

services:
  shagun-backend:
    build: .
    container_name: ${CONTAINER_NAME}
    ports:
      - "${HOST_PORT}:${CONTAINER_PORT}"
    volumes:
      - ${DATA_VOLUME}:/app/data
      - ${LOG_VOLUME}:/app/logs
      - ./live_trading_1000.db:/app/live_trading_1000.db
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=sqlite:///./live_trading_1000.db
      - TRADING_MODE=live
      - AUTOMATED_TRADING_ENABLED=true
      - LIVE_TRADING_ENABLED=true
      - MAX_RISK_PER_TRADE=0.05
      - MAX_DAILY_LOSS=0.10
      - DEFAULT_POSITION_SIZE=1000
      - CIRCUIT_BREAKER_ENABLED=true
      - EMERGENCY_STOP_LOSS_AMOUNT=80
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  ${DATA_VOLUME}:
    driver: local
  ${LOG_VOLUME}:
    driver: local

networks:
  default:
    name: shagun-network
EOF

    success "docker-compose.yml created"
}

# Create .dockerignore file
create_dockerignore() {
    log "Creating .dockerignore file..."

    cat > .dockerignore <<EOF
# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Node.js
node_modules
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# IDEs
.vscode
.idea
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
backups/
*.db-journal
dev_trading.db
logs/*.log

# Docker
Dockerfile
docker-compose.yml
.dockerignore
EOF

    success ".dockerignore created"
}

# Build the Docker image
build_image() {
    log "Building Docker image with OrbStack..."

    if docker build -t "$IMAGE_NAME" .; then
        success "Docker image built successfully"
    else
        error "Failed to build Docker image"
        exit 1
    fi
}

# Create data volumes
create_volumes() {
    log "Creating Docker volumes..."

    # Create data volume
    if ! docker volume ls | grep -q "$DATA_VOLUME"; then
        docker volume create "$DATA_VOLUME"
        success "Data volume '$DATA_VOLUME' created"
    else
        log "Data volume '$DATA_VOLUME' already exists"
    fi

    # Create log volume
    if ! docker volume ls | grep -q "$LOG_VOLUME"; then
        docker volume create "$LOG_VOLUME"
        success "Log volume '$LOG_VOLUME' created"
    else
        log "Log volume '$LOG_VOLUME' already exists"
    fi
}

# Stop existing containers
stop_existing_containers() {
    log "Checking for existing containers..."

    if docker ps -a | grep -q "$CONTAINER_NAME"; then
        warning "Stopping existing container '$CONTAINER_NAME'..."
        docker stop "$CONTAINER_NAME" || true
        docker rm "$CONTAINER_NAME" || true
        success "Existing container removed"
    fi
}

# Deploy using docker-compose
deploy_with_compose() {
    log "Deploying with docker-compose..."

    if docker-compose up -d; then
        success "Container deployed successfully"
    else
        error "Failed to deploy container"
        exit 1
    fi
}

# Wait for container to be healthy
wait_for_health() {
    log "Waiting for container to be healthy..."

    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if docker inspect "$CONTAINER_NAME" --format='{{.State.Health.Status}}' 2>/dev/null | grep -q "healthy"; then
            success "Container is healthy"
            return 0
        fi

        log "Attempt $attempt/$max_attempts - waiting for health check..."
        sleep 10
        ((attempt++))
    done

    error "Container failed to become healthy within timeout"
    docker logs "$CONTAINER_NAME"
    exit 1
}

# Create management scripts
create_management_scripts() {
    log "Creating container management scripts..."

    # Create start script
    cat > "start-orbstack.sh" <<EOF
#!/bin/bash
echo "Starting Shagun Intelligence Trading System (OrbStack)..."
docker-compose up -d
echo "System started at: http://localhost:$HOST_PORT"
echo "Dashboard: http://localhost:$FRONTEND_PORT"
EOF

    # Create stop script
    cat > "stop-orbstack.sh" <<EOF
#!/bin/bash
echo "Stopping Shagun Intelligence Trading System (OrbStack)..."
docker-compose down
echo "System stopped"
EOF

    # Create logs script
    cat > "logs-orbstack.sh" <<EOF
#!/bin/bash
echo "Showing logs for Shagun Intelligence Trading System..."
docker-compose logs -f
EOF

    # Create status script
    cat > "status-orbstack.sh" <<EOF
#!/bin/bash
echo "=== Container Status ==="
docker-compose ps
echo ""
echo "=== Health Status ==="
docker inspect "$CONTAINER_NAME" --format='{{.State.Health.Status}}' 2>/dev/null || echo "Container not running"
echo ""
echo "=== Resource Usage ==="
docker stats "$CONTAINER_NAME" --no-stream 2>/dev/null || echo "Container not running"
EOF

    # Create rebuild script
    cat > "rebuild-orbstack.sh" <<EOF
#!/bin/bash
echo "Rebuilding Shagun Intelligence Trading System..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d
echo "System rebuilt and started"
EOF

    # Make scripts executable
    chmod +x start-orbstack.sh stop-orbstack.sh logs-orbstack.sh status-orbstack.sh rebuild-orbstack.sh

    success "Management scripts created"
}

# Display deployment information
show_deployment_info() {
    success "OrbStack Deployment Completed!"
    echo ""
    echo "=== Deployment Information ==="
    echo "Container Name: $CONTAINER_NAME"
    echo "Image: $IMAGE_NAME"
    echo "Backend API: http://localhost:$HOST_PORT"
    echo "API Documentation: http://localhost:$HOST_PORT/docs"
    echo "Dashboard: http://localhost:$FRONTEND_PORT (start separately)"
    echo ""
    echo "=== Management Commands ==="
    echo "Start:   ./start-orbstack.sh"
    echo "Stop:    ./stop-orbstack.sh"
    echo "Logs:    ./logs-orbstack.sh"
    echo "Status:  ./status-orbstack.sh"
    echo "Rebuild: ./rebuild-orbstack.sh"
    echo ""
    echo "=== Direct Docker Commands ==="
    echo "View logs:     docker-compose logs -f"
    echo "Stop:          docker-compose down"
    echo "Start:         docker-compose up -d"
    echo "Rebuild:       docker-compose build --no-cache"
    echo "Shell access:  docker exec -it $CONTAINER_NAME /bin/bash"
    echo ""
    echo "=== Data Persistence ==="
    echo "Data Volume: $DATA_VOLUME"
    echo "Log Volume:  $LOG_VOLUME"
    echo "Database: live_trading_1000.db (mounted from host)"
    echo ""
    echo "=== ₹1000 Budget Configuration ==="
    echo "• Max Risk Per Trade: ₹50 (5%)"
    echo "• Max Daily Loss: ₹100 (10%)"
    echo "• Emergency Stop: ₹80 total loss"
    echo "• Automated Trading: ENABLED"
    echo "• Live Trading: ENABLED"
    echo ""
    echo "=== Next Steps ==="
    echo "1. Start frontend dashboard: cd dashboard && npm run dev"
    echo "2. Access dashboard: http://localhost:3000"
    echo "3. Login with: live_trader_1000 / LiveTrading1000!Secure"
    echo "4. Start automated trading via dashboard or API"
}

# Main deployment function
main() {
    log "Starting OrbStack Deployment"
    log "Shagun Intelligence Trading Platform"
    echo ""

    # Check if user wants to proceed
    warning "This will deploy using OrbStack with Docker compatibility"
    warning "This will create containers for the ₹1000 live trading system"
    echo ""
    read -p "Do you want to proceed? (y/N): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled by user"
        exit 0
    fi

    check_orbstack
    create_dockerfile
    create_docker_compose
    create_dockerignore
    create_volumes
    stop_existing_containers
    build_image
    deploy_with_compose
    wait_for_health
    create_management_scripts
    show_deployment_info
}

# Handle script arguments
case "${1:-}" in
    "start")
        docker-compose up -d
        echo "Container started"
        ;;
    "stop")
        docker-compose down
        echo "Container stopped"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "rebuild")
        docker-compose down
        docker-compose build --no-cache
        docker-compose up -d
        echo "Container rebuilt and started"
        ;;
    "remove")
        docker-compose down -v
        docker rmi "$IMAGE_NAME" || true
        echo "Container and image removed"
        ;;
    *)
        main "$@"
        ;;
esac
