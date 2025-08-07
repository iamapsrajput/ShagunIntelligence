#!/bin/bash

# =============================================================================
# Shagun Intelligence Trading Platform - Production Deployment Script
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="shagun-trading"
BACKUP_DIR="./backups/pre-deployment"
LOG_FILE="./logs/deployment.log"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    mkdir -p logs data backups/postgres backups/redis monitoring/grafana/dashboards monitoring/grafana/datasources
    success "Directories created successfully"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check if .env.production exists
    if [ ! -f ".env.production" ]; then
        error ".env.production file not found. Please copy .env.production.template and configure it."
    fi

    success "Prerequisites check passed"
}

# Backup existing data
backup_existing_data() {
    log "Creating backup of existing data..."

    if [ -d "$BACKUP_DIR" ]; then
        rm -rf "$BACKUP_DIR"
    fi
    mkdir -p "$BACKUP_DIR"

    # Backup database if exists
    if docker ps | grep -q "shagun-postgres"; then
        log "Backing up PostgreSQL database..."
        docker exec shagun-postgres pg_dump -U trading_user trading_db > "$BACKUP_DIR/database_backup.sql"
    fi

    # Backup Redis if exists
    if docker ps | grep -q "shagun-redis"; then
        log "Backing up Redis data..."
        docker exec shagun-redis redis-cli BGSAVE
        docker cp shagun-redis:/data/dump.rdb "$BACKUP_DIR/redis_backup.rdb"
    fi

    # Backup application data
    if [ -d "./data" ]; then
        cp -r ./data "$BACKUP_DIR/app_data"
    fi

    success "Backup completed successfully"
}

# Build Docker images
build_images() {
    log "Building Docker images..."

    # Build main application
    docker build -f Dockerfile.prod -t shagun-trading-app:latest .

    # Build dashboard
    cd dashboard
    docker build -f Dockerfile.prod -t shagun-dashboard:latest .
    cd ..

    success "Docker images built successfully"
}

# Setup monitoring configuration
setup_monitoring() {
    log "Setting up monitoring configuration..."

    # Create Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'trading-app'
    static_configs:
      - targets: ['trading-app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF

    # Create Grafana datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    success "Monitoring configuration created"
}

# Setup Nginx configuration
setup_nginx() {
    log "Setting up Nginx configuration..."

    mkdir -p nginx

    cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream trading_app {
        server trading-app:8000;
    }

    upstream dashboard {
        server dashboard:80;
    }

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=login:10m rate=1r/s;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://trading_app;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # WebSocket endpoints
        location /ws/ {
            proxy_pass http://trading_app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }

        # Dashboard
        location / {
            proxy_pass http://dashboard;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF

    success "Nginx configuration created"
}

# Deploy application
deploy_application() {
    log "Deploying application..."

    # Stop existing containers
    docker-compose -f docker-compose.prod.yml down

    # Start new containers
    docker-compose -f docker-compose.prod.yml up -d

    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30

    # Check if services are running
    if ! docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        error "Some services failed to start. Check logs with: docker-compose -f docker-compose.prod.yml logs"
    fi

    success "Application deployed successfully"
}

# Run health checks
run_health_checks() {
    log "Running health checks..."

    # Check main application
    if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        success "Trading application health check passed"
    else
        warning "Trading application health check failed"
    fi

    # Check dashboard
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        success "Dashboard health check passed"
    else
        warning "Dashboard health check failed"
    fi

    # Check database
    if docker exec shagun-postgres pg_isready -U trading_user > /dev/null 2>&1; then
        success "Database health check passed"
    else
        warning "Database health check failed"
    fi

    # Check Redis
    if docker exec shagun-redis redis-cli ping > /dev/null 2>&1; then
        success "Redis health check passed"
    else
        warning "Redis health check failed"
    fi
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."

    cat > /etc/logrotate.d/shagun-trading << EOF
/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 trading trading
    postrotate
        docker exec shagun-trading-app kill -USR1 1
    endscript
}
EOF

    success "Log rotation configured"
}

# Main deployment process
main() {
    log "Starting production deployment of Shagun Intelligence Trading Platform"

    create_directories
    check_prerequisites
    backup_existing_data
    build_images
    setup_monitoring
    setup_nginx
    deploy_application
    run_health_checks
    setup_log_rotation

    success "Production deployment completed successfully!"

    echo ""
    echo "==============================================================================="
    echo "Deployment Summary:"
    echo "==============================================================================="
    echo "✅ Trading Application: http://localhost:8000"
    echo "✅ Dashboard: http://localhost:3000"
    echo "✅ Grafana Monitoring: http://localhost:3001"
    echo "✅ Prometheus Metrics: http://localhost:9090"
    echo ""
    echo "Next Steps:"
    echo "1. Configure SSL certificates for production domain"
    echo "2. Set up DNS records"
    echo "3. Configure firewall rules"
    echo "4. Set up monitoring alerts"
    echo "5. Test backup and recovery procedures"
    echo ""
    echo "Logs are available in: $LOG_FILE"
    echo "==============================================================================="
}

# Run main function
main "$@"
