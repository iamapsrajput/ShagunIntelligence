#!/bin/bash

# Production Deployment Script with SSL/TLS Configuration
# Shagun Intelligence Trading Platform

set -e

# Configuration
DOMAIN="${DOMAIN:-shagunintelligence.com}"
EMAIL="${EMAIL:-admin@shagunintelligence.com}"
CERT_TYPE="${CERT_TYPE:-selfsigned}"  # Use selfsigned for development, letsencrypt for production
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if required files exist
    if [[ ! -f "docker-compose.prod.yml" ]]; then
        error "docker-compose.prod.yml not found!"
        exit 1
    fi

    if [[ ! -f ".env.production" ]]; then
        warning ".env.production not found. Creating template..."
        create_env_template
    fi

    success "Prerequisites check passed"
}

# Create environment template
create_env_template() {
    cat > .env.production << EOF
# Production Environment Configuration
ENVIRONMENT=production
DOMAIN=$DOMAIN

# Database Configuration
DB_PASSWORD=your_secure_db_password_here
POSTGRES_PASSWORD=your_secure_db_password_here

# Redis Configuration
REDIS_PASSWORD=your_secure_redis_password_here

# API Keys (REQUIRED - Replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
KITE_API_KEY=your_kite_api_key_here
KITE_API_SECRET=your_kite_api_secret_here
KITE_ACCESS_TOKEN=your_kite_access_token_here

# Optional API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FINNHUB_API_KEY=your_finnhub_key_here
NEWS_API_KEY=your_news_api_key_here

# Monitoring
GRAFANA_PASSWORD=your_secure_grafana_password_here

# Backup Configuration
BACKUP_S3_BUCKET=your_backup_bucket_here
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here

# SSL Configuration
SSL_EMAIL=$EMAIL
EOF

    warning "Please edit .env.production and add your actual API keys and passwords!"
}

# Setup SSL certificates
setup_ssl() {
    log "Setting up SSL certificates..."

    # Create SSL directory
    mkdir -p nginx/ssl

    # Run SSL setup script
    if [[ -f "scripts/ssl_setup.sh" ]]; then
        ./scripts/ssl_setup.sh --domain "$DOMAIN" --email "$EMAIL" --type "$CERT_TYPE" --ssl-dir "./nginx/ssl"
    else
        error "SSL setup script not found!"
        exit 1
    fi

    success "SSL certificates configured"
}

# Validate environment configuration
validate_environment() {
    log "Validating environment configuration..."

    # Check if critical environment variables are set
    source .env.production

    critical_vars=(
        "OPENAI_API_KEY"
        "KITE_API_KEY"
        "KITE_API_SECRET"
        "DB_PASSWORD"
        "REDIS_PASSWORD"
    )

    missing_vars=()
    for var in "${critical_vars[@]}"; do
        if [[ -z "${!var}" || "${!var}" == *"your_"* ]]; then
            missing_vars+=("$var")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing or template values for critical environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        error "Please update .env.production with actual values"
        exit 1
    fi

    success "Environment configuration validated"
}

# Build and deploy containers
deploy_containers() {
    log "Building and deploying containers..."

    # Stop existing containers
    docker-compose -f docker-compose.prod.yml down --remove-orphans

    # Build images
    docker-compose -f docker-compose.prod.yml build --no-cache

    # Start services
    docker-compose -f docker-compose.prod.yml up -d

    success "Containers deployed"
}

# Wait for services to be healthy
wait_for_services() {
    log "Waiting for services to be healthy..."

    services=("shagun-trading-app" "shagun-postgres" "shagun-redis" "shagun-nginx")

    for service in "${services[@]}"; do
        log "Waiting for $service to be healthy..."
        timeout=300  # 5 minutes
        elapsed=0

        while [[ $elapsed -lt $timeout ]]; do
            if docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null | grep -q "healthy"; then
                success "$service is healthy"
                break
            elif docker inspect --format='{{.State.Status}}' "$service" 2>/dev/null | grep -q "running"; then
                log "$service is running, waiting for health check..."
            else
                warning "$service is not running yet..."
            fi

            sleep 10
            elapsed=$((elapsed + 10))
        done

        if [[ $elapsed -ge $timeout ]]; then
            error "$service failed to become healthy within $timeout seconds"
            docker logs "$service" --tail 50
            exit 1
        fi
    done

    success "All services are healthy"
}

# Run health checks
run_health_checks() {
    log "Running comprehensive health checks..."

    # Check API health
    if curl -f -s "http://localhost:8000/api/v1/health" > /dev/null; then
        success "API health check passed"
    else
        error "API health check failed"
        exit 1
    fi

    # Check HTTPS redirect (if using real certificates)
    if [[ $CERT_TYPE == "letsencrypt" ]]; then
        if curl -I -s "http://$DOMAIN" | grep -q "301"; then
            success "HTTPS redirect working"
        else
            warning "HTTPS redirect may not be working properly"
        fi
    fi

    # Check SSL certificate (if using HTTPS)
    if [[ -f "nginx/ssl/fullchain.pem" ]]; then
        if openssl x509 -in nginx/ssl/fullchain.pem -text -noout | grep -q "Subject:"; then
            success "SSL certificate is valid"
        else
            error "SSL certificate validation failed"
            exit 1
        fi
    fi

    success "Health checks completed"
}

# Setup monitoring and alerting
setup_monitoring() {
    log "Setting up monitoring and alerting..."

    # Check if Grafana is accessible
    if curl -f -s "http://localhost:3000" > /dev/null; then
        success "Grafana is accessible"
        log "Grafana dashboard: http://localhost:3000"
        log "Default credentials: admin / (check GRAFANA_PASSWORD in .env.production)"
    else
        warning "Grafana may not be accessible yet"
    fi

    # Check if Prometheus is accessible
    if curl -f -s "http://localhost:9090" > /dev/null; then
        success "Prometheus is accessible"
        log "Prometheus: http://localhost:9090"
    else
        warning "Prometheus may not be accessible yet"
    fi
}

# Display deployment summary
show_summary() {
    log "Deployment Summary"
    echo "===================="
    echo "Domain: $DOMAIN"
    echo "Environment: $ENVIRONMENT"
    echo "SSL Type: $CERT_TYPE"
    echo ""
    echo "Services:"
    echo "- Trading Platform: https://$DOMAIN (or http://localhost:8000)"
    echo "- Dashboard: https://$DOMAIN (or http://localhost:3000)"
    echo "- Grafana: http://localhost:3000"
    echo "- Prometheus: http://localhost:9090"
    echo ""
    echo "SSL Certificates: nginx/ssl/"
    echo "Logs: logs/"
    echo "Backups: backups/"
    echo ""
    success "Production deployment completed successfully!"

    if [[ $CERT_TYPE == "selfsigned" ]]; then
        warning "Using self-signed certificates. Browsers will show security warnings."
        warning "For production, use: CERT_TYPE=letsencrypt ./scripts/deploy_production_ssl.sh"
    fi
}

# Main deployment function
main() {
    log "Starting production deployment with SSL/TLS for Shagun Intelligence Trading Platform"

    check_prerequisites
    setup_ssl
    validate_environment
    deploy_containers
    wait_for_services
    run_health_checks
    setup_monitoring
    show_summary
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --domain DOMAIN     Domain name (default: shagunintelligence.com)"
    echo "  -e, --email EMAIL       Email for SSL certificates (default: admin@shagunintelligence.com)"
    echo "  -t, --cert-type TYPE    Certificate type: letsencrypt or selfsigned (default: selfsigned)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Deploy with self-signed certificates"
    echo "  $0 --cert-type letsencrypt                  # Deploy with Let's Encrypt certificates"
    echo "  $0 --domain example.com --cert-type letsencrypt  # Deploy with custom domain"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -e|--email)
            EMAIL="$2"
            shift 2
            ;;
        -t|--cert-type)
            CERT_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main
