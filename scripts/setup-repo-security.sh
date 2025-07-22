#!/bin/bash

# Repository Security Setup Script for Shagun Intelligence
# This script configures GitHub repository settings for optimal security

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

REPO="iamapsrajput/ShagunIntelligence"

# Helper functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    if ! command -v gh &> /dev/null; then
        error "GitHub CLI not found. Please install from https://cli.github.com/"
        exit 1
    fi
    
    if ! gh auth status &> /dev/null; then
        error "Not authenticated with GitHub CLI. Run 'gh auth login'"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Enable security features
enable_security_features() {
    info "Enabling security features..."
    
    # Enable vulnerability alerts
    if gh api -X PATCH "/repos/${REPO}" --field has_vulnerability_alerts=true &> /dev/null; then
        success "Enabled vulnerability alerts"
    else
        warning "Could not enable vulnerability alerts (may already be enabled)"
    fi
    
    # Enable automated security fixes
    if gh api -X PUT "/repos/${REPO}/automated-security-fixes" &> /dev/null; then
        success "Enabled automated security fixes"
    else
        warning "Could not enable automated security fixes (may already be enabled)"
    fi
    
    # Try to enable secret scanning (requires Advanced Security)
    if gh api -X PATCH "/repos/${REPO}" \
        --field security_and_analysis='{"secret_scanning":{"status":"enabled"},"secret_scanning_push_protection":{"status":"enabled"}}' &> /dev/null; then
        success "Enabled secret scanning"
    else
        warning "Could not enable secret scanning (requires GitHub Advanced Security)"
    fi
}

# Set repository topics
set_topics() {
    info "Setting repository topics..."
    
    TOPICS='["algorithmic-trading","ai-agents","fintech","fastapi","crewai","python","docker","kubernetes","zerodha-kite","trading-bot","machine-learning","quantitative-finance","real-time-data","risk-management"]'
    
    if gh api -X PUT "/repos/${REPO}/topics" --field names="${TOPICS}" &> /dev/null; then
        success "Repository topics updated"
    else
        error "Failed to set repository topics"
    fi
}

# Create labels
create_labels() {
    info "Creating repository labels..."
    
    # Priority labels
    gh label create "priority: critical" --color "B60205" --description "Critical issues requiring immediate attention" 2>/dev/null || true
    gh label create "priority: high" --color "D93F0B" --description "High priority issues" 2>/dev/null || true
    gh label create "priority: medium" --color "FBCA04" --description "Medium priority issues" 2>/dev/null || true
    gh label create "priority: low" --color "0E8A16" --description "Low priority issues" 2>/dev/null || true
    
    # Type labels
    gh label create "enhancement" --color "A2EEEF" --description "New feature or request" 2>/dev/null || true
    gh label create "performance" --color "FF6B6B" --description "Performance related issues" 2>/dev/null || true
    gh label create "security" --color "FF0000" --description "Security related issues" 2>/dev/null || true
    
    # Component labels
    gh label create "agents" --color "7057FF" --description "AI agents and CrewAI related" 2>/dev/null || true
    gh label create "api" --color "00D4AA" --description "FastAPI backend issues" 2>/dev/null || true
    gh label create "trading" --color "006B75" --description "Trading system and Kite Connect" 2>/dev/null || true
    gh label create "database" --color "8B4513" --description "Database and ORM related" 2>/dev/null || true
    gh label create "infrastructure" --color "5319E7" --description "Docker, K8s, deployment" 2>/dev/null || true
    gh label create "frontend" --color "FF69B4" --description "UI and dashboard related" 2>/dev/null || true
    gh label create "ci-cd" --color "1F77B4" --description "CI/CD pipeline issues" 2>/dev/null || true
    
    # Status labels
    gh label create "needs-triage" --color "EDEDED" --description "Needs initial review and labeling" 2>/dev/null || true
    gh label create "needs-review" --color "0052CC" --description "Needs code review" 2>/dev/null || true
    gh label create "needs-testing" --color "FF8C00" --description "Needs testing" 2>/dev/null || true
    gh label create "in-progress" --color "FEF2C0" --description "Currently being worked on" 2>/dev/null || true
    gh label create "blocked" --color "B60205" --description "Blocked by external dependency" 2>/dev/null || true
    gh label create "good first issue" --color "7057FF" --description "Good for newcomers" 2>/dev/null || true
    gh label create "help wanted" --color "008672" --description "Extra attention is needed" 2>/dev/null || true
    gh label create "dependencies" --color "0366D6" --description "Dependency updates" 2>/dev/null || true
    
    # Resolution labels  
    gh label create "duplicate" --color "CFD3D7" --description "This issue or pull request already exists" 2>/dev/null || true
    gh label create "invalid" --color "E4E669" --description "This doesn't seem right" 2>/dev/null || true
    gh label create "wontfix" --color "FFFFFF" --description "This will not be worked on" 2>/dev/null || true
    
    success "Labels created/updated"
}

# Configure repository settings
configure_repo_settings() {
    info "Configuring repository settings..."
    
    if gh api -X PATCH "/repos/${REPO}" \
        --field description="AI-Powered Algorithmic Trading Platform with Multi-Agent System for Indian Stock Market" \
        --field homepage="https://github.com/${REPO}" \
        --field has_issues=true \
        --field has_projects=true \
        --field has_wiki=false \
        --field has_discussions=true \
        --field allow_squash_merge=true \
        --field allow_merge_commit=false \
        --field allow_rebase_merge=true \
        --field delete_branch_on_merge=true \
        --field allow_auto_merge=false &> /dev/null; then
        success "Repository settings updated"
    else
        error "Failed to update repository settings"
    fi
}

# Setup branch protection (requires existing CI workflows)
setup_branch_protection() {
    info "Setting up branch protection rules..."
    
    warning "Branch protection setup requires manual configuration due to CI dependency"
    info "Please manually configure branch protection for 'main' branch with:"
    info "- Require pull request reviews (1 reviewer)"
    info "- Require code owner reviews"
    info "- Require status checks to pass"
    info "- Require up-to-date branches"
    info "- Require signed commits"
    info "- Require linear history"
    info "- Restrict pushes and deletions"
    info "- Include administrators"
}

# Create environments
create_environments() {
    info "Creating deployment environments..."
    
    if gh api -X PUT "/repos/${REPO}/environments/staging" \
        --field wait_timer=0 \
        --field reviewers=null \
        --field deployment_branch_policy=null &> /dev/null; then
        success "Created staging environment"
    else
        warning "Staging environment may already exist"
    fi
    
    if gh api -X PUT "/repos/${REPO}/environments/production" \
        --field wait_timer=5 \
        --field reviewers='[{"type":"User","id":null}]' \
        --field deployment_branch_policy='{"protected_branches":true,"custom_branches":false}' &> /dev/null; then
        success "Created production environment"
    else
        warning "Production environment may already exist"
    fi
}

# Verify setup
verify_setup() {
    info "Verifying repository setup..."
    
    # Check repository info
    echo "Repository Information:"
    gh repo view "${REPO}" --json description,hasIssues,hasDiscussions,hasWiki,topics | jq
    
    # Check labels
    LABEL_COUNT=$(gh label list | wc -l)
    info "Labels created: ${LABEL_COUNT}"
    
    # Check security features
    info "Security features status:"
    gh api "/repos/${REPO}" | jq '.security_and_analysis // "Advanced Security not available"'
    
    success "Setup verification completed"
}

# Main execution
main() {
    info "Starting Shagun Intelligence repository security setup..."
    
    check_prerequisites
    enable_security_features
    set_topics
    create_labels
    configure_repo_settings
    setup_branch_protection
    create_environments
    verify_setup
    
    success "Repository security setup completed successfully!"
    
    info "Next steps:"
    info "1. Manually configure branch protection rules via GitHub UI"
    info "2. Set up required CI/CD secrets"
    info "3. Enable GitHub Advanced Security (if available)"
    info "4. Review and test all security settings"
    info "5. Configure notification settings"
    
    info "For detailed instructions, see .github/SETUP_COMMANDS.md"
}

# Execute main function
main "$@"