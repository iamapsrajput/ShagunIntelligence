#!/bin/bash
set -e

# Shagun Intelligence Deployment Script
# Usage: ./deploy.sh [environment] [version]

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
CLUSTER_NAME="shagunintelligence-cluster"
NAMESPACE="shagunintelligence-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-flight checks
pre_flight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check required tools
    for tool in kubectl aws docker helm; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is not installed"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Pre-flight checks passed"
}

# Build and push Docker image
build_and_push() {
    log_info "Building Docker image..."
    
    # Get ECR registry
    ECR_REGISTRY=$(aws ecr describe-repositories --repository-names shagunintelligence --query 'repositories[0].repositoryUri' --output text | cut -d'/' -f1)
    
    # Login to ECR
    aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
    
    # Build image
    docker build -t shagunintelligence:$VERSION .
    
    # Tag and push
    docker tag shagunintelligence:$VERSION $ECR_REGISTRY/shagunintelligence:$VERSION
    docker push $ECR_REGISTRY/shagunintelligence:$VERSION
    
    if [ "$ENVIRONMENT" == "production" ]; then
        docker tag shagunintelligence:$VERSION $ECR_REGISTRY/shagunintelligence:latest
        docker push $ECR_REGISTRY/shagunintelligence:latest
    fi
    
    log_info "Docker image pushed successfully"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if not exists
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply base configurations
    kubectl apply -k k8s/base/ -n $NAMESPACE
    
    # Apply environment-specific configurations
    kubectl apply -k k8s/overlays/$ENVIRONMENT/ -n $NAMESPACE
    
    # Update image
    kubectl set image deployment/shagunintelligence-app shagunintelligence-app=$ECR_REGISTRY/shagunintelligence:$VERSION -n $NAMESPACE
    
    # Wait for rollout
    kubectl rollout status deployment/shagunintelligence-app -n $NAMESPACE --timeout=300s
    
    log_info "Kubernetes deployment completed"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    kubectl run migrations-$VERSION \
        --image=$ECR_REGISTRY/shagunintelligence:$VERSION \
        --restart=Never \
        --rm=true \
        -n $NAMESPACE \
        -- alembic upgrade head
    
    log_info "Migrations completed"
}

# Health checks
health_checks() {
    log_info "Running health checks..."
    
    # Get service endpoint
    if [ "$ENVIRONMENT" == "production" ]; then
        ENDPOINT="https://shagunintelligence.com"
    else
        ENDPOINT=$(kubectl get service nginx-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        ENDPOINT="http://$ENDPOINT"
    fi
    
    # Wait for service to be ready
    for i in {1..30}; do
        if curl -f $ENDPOINT/api/v1/health &> /dev/null; then
            log_info "Health check passed"
            return 0
        fi
        log_warn "Waiting for service to be ready... ($i/30)"
        sleep 10
    done
    
    log_error "Health check failed"
    return 1
}

# Rollback function
rollback() {
    log_error "Deployment failed, rolling back..."
    
    kubectl rollout undo deployment/shagunintelligence-app -n $NAMESPACE
    kubectl rollout status deployment/shagunintelligence-app -n $NAMESPACE
    
    exit 1
}

# Main deployment flow
main() {
    log_info "Starting deployment to $ENVIRONMENT with version $VERSION"
    
    # Set error trap
    trap rollback ERR
    
    # Run deployment steps
    pre_flight_checks
    build_and_push
    deploy_kubernetes
    run_migrations
    health_checks
    
    # Clear error trap
    trap - ERR
    
    log_info "Deployment completed successfully!"
    
    # Send notification
    if [ "$ENVIRONMENT" == "production" ]; then
        curl -X POST $SLACK_WEBHOOK_URL \
            -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ Shagun Intelligence $VERSION deployed to production successfully\"}"
    fi
}

# Run main function
main