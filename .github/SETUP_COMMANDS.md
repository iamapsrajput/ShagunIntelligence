# GitHub CLI Commands for Repository Configuration

This document provides GitHub CLI commands to configure the repository settings automatically.

## Prerequisites

1. Install GitHub CLI: <https://cli.github.com/>
2. Authenticate: `gh auth login`
3. Ensure you have admin access to the repository

## Repository Settings Commands

### 1. Enable Security Features

```bash
# Enable vulnerability alerts
gh api -X PATCH /repos/iamapsrajput/ShagunIntelligence \
  --field has_vulnerability_alerts=true

# Enable automated security fixes (Dependabot)
gh api -X PUT /repos/iamapsrajput/ShagunIntelligence/automated-security-fixes

# Enable secret scanning (requires GitHub Advanced Security)
gh api -X PATCH /repos/iamapsrajput/ShagunIntelligence \
  --field security_and_analysis='{"secret_scanning":{"status":"enabled"},"secret_scanning_push_protection":{"status":"enabled"}}'
```

### 2. Configure Repository Topics

```bash
# Set repository topics
gh api -X PUT /repos/iamapsrajput/ShagunIntelligence/topics \
  --field names='["algorithmic-trading","ai-agents","fintech","fastapi","crewai","python","docker","kubernetes","zerodha-kite","trading-bot","machine-learning","quantitative-finance","real-time-data","risk-management"]'
```

### 3. Create Repository Labels

```bash
# Priority labels
gh label create "priority: critical" --color "B60205" --description "Critical issues requiring immediate attention"
gh label create "priority: high" --color "D93F0B" --description "High priority issues"
gh label create "priority: medium" --color "FBCA04" --description "Medium priority issues"
gh label create "priority: low" --color "0E8A16" --description "Low priority issues"

# Type labels
gh label create "bug" --color "D73A49" --description "Something isn't working"
gh label create "enhancement" --color "A2EEEF" --description "New feature or request"
gh label create "documentation" --color "0075CA" --description "Improvements or additions to documentation"
gh label create "performance" --color "FF6B6B" --description "Performance related issues"
gh label create "security" --color "FF0000" --description "Security related issues"
gh label create "question" --color "D876E3" --description "Further information is requested"

# Component labels
gh label create "agents" --color "7057FF" --description "AI agents and CrewAI related"
gh label create "api" --color "00D4AA" --description "FastAPI backend issues"
gh label create "trading" --color "006B75" --description "Trading system and Kite Connect"
gh label create "database" --color "8B4513" --description "Database and ORM related"
gh label create "infrastructure" --color "5319E7" --description "Docker, K8s, deployment"
gh label create "frontend" --color "FF69B4" --description "UI and dashboard related"
gh label create "ci-cd" --color "1F77B4" --description "CI/CD pipeline issues"

# Status labels
gh label create "needs-triage" --color "EDEDED" --description "Needs initial review and labeling"
gh label create "needs-review" --color "0052CC" --description "Needs code review"
gh label create "needs-testing" --color "FF8C00" --description "Needs testing"
gh label create "in-progress" --color "FEF2C0" --description "Currently being worked on"
gh label create "blocked" --color "B60205" --description "Blocked by external dependency"
gh label create "good first issue" --color "7057FF" --description "Good for newcomers"
gh label create "help wanted" --color "008672" --description "Extra attention is needed"
gh label create "dependencies" --color "0366D6" --description "Dependency updates"

# Resolution labels
gh label create "duplicate" --color "CFD3D7" --description "This issue or pull request already exists"
gh label create "invalid" --color "E4E669" --description "This doesn't seem right"
gh label create "wontfix" --color "FFFFFF" --description "This will not be worked on"
```

### 4. Branch Protection Rules

```bash
# Main branch protection
gh api -X PUT /repos/iamapsrajput/ShagunIntelligence/branches/main/protection \
  --field required_status_checks='{
    "strict": true,
    "checks": [
      {"context": "CI Pipeline / lint"},
      {"context": "CI Pipeline / test"},
      {"context": "CI Pipeline / build-docker"},
      {"context": "CodeQL Security Analysis / analyze (python)"},
      {"context": "CodeQL Security Analysis / security-scan"}
    ]
  }' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "require_last_push_approval": false
  }' \
  --field restrictions=null \
  --field required_linear_history=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field required_conversation_resolution=true

# Develop branch protection
gh api -X PUT /repos/iamapsrajput/ShagunIntelligence/branches/develop/protection \
  --field required_status_checks='{
    "strict": true,
    "checks": [
      {"context": "CI Pipeline / lint"},
      {"context": "CI Pipeline / test"}
    ]
  }' \
  --field required_pull_request_reviews='{
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  }' \
  --field enforce_admins=false
```

### 5. Repository Settings

```bash
# General settings
gh api -X PATCH /repos/iamapsrajput/ShagunIntelligence \
  --field description="AI-Powered Algorithmic Trading Platform with Multi-Agent System for Indian Stock Market" \
  --field homepage="https://github.com/iamapsrajput/ShagunIntelligence" \
  --field has_issues=true \
  --field has_projects=true \
  --field has_wiki=false \
  --field has_discussions=true \
  --field allow_squash_merge=true \
  --field allow_merge_commit=false \
  --field allow_rebase_merge=true \
  --field delete_branch_on_merge=true \
  --field allow_auto_merge=false
```

### 6. Enable Discussions

```bash
# Enable discussions
gh api -X PATCH /repos/iamapsrajput/ShagunIntelligence \
  --field has_discussions=true
```

### 7. Repository Secrets (for CI/CD)

```bash
# Note: Replace with actual secret values
# These are examples - use your actual secrets

# Docker Hub credentials
gh secret set DOCKER_USERNAME --body "your-docker-username"
gh secret set DOCKER_PASSWORD --body "your-docker-password"

# AWS credentials (if using AWS)
gh secret set AWS_ACCESS_KEY_ID --body "your-aws-access-key"
gh secret set AWS_SECRET_ACCESS_KEY --body "your-aws-secret-key"

# Codecov token
gh secret set CODECOV_TOKEN --body "your-codecov-token"

# Slack webhook for notifications
gh secret set SLACK_WEBHOOK --body "your-slack-webhook-url"
```

### 8. Environment-specific Secrets

```bash
# Create environments first
gh api -X PUT /repos/iamapsrajput/ShagunIntelligence/environments/staging
gh api -X PUT /repos/iamapsrajput/ShagunIntelligence/environments/production

# Add environment secrets (staging)
gh secret set DATABASE_URL --env staging --body "postgresql://user:pass@staging-db:5432/dbname"
gh secret set REDIS_URL --env staging --body "redis://staging-redis:6379/0"

# Add environment secrets (production)
gh secret set DATABASE_URL --env production --body "postgresql://user:pass@prod-db:5432/dbname"
gh secret set REDIS_URL --env production --body "redis://prod-redis:6379/0"
gh secret set ZERODHA_API_KEY --env production --body "your-zerodha-api-key"
gh secret set ZERODHA_API_SECRET --env production --body "your-zerodha-api-secret"
```

## Verification Commands

### Check Repository Settings

```bash
# View repository information
gh repo view iamapsrajput/ShagunIntelligence

# Check branch protection
gh api /repos/iamapsrajput/ShagunIntelligence/branches/main/protection

# List labels
gh label list

# List secrets
gh secret list

# Check security features
gh api /repos/iamapsrajput/ShagunIntelligence | jq '.security_and_analysis'
```

### Validate Workflows

```bash
# List workflows
gh workflow list

# View workflow runs
gh run list

# Manually trigger workflow
gh workflow run "CodeQL Security Analysis"
```

## Batch Script

Create a script file `setup-repo-security.sh`:

```bash
#!/bin/bash

# Repository Security Setup Script
echo "Setting up Shagun Intelligence repository security..."

# Enable security features
echo "1. Enabling security features..."
gh api -X PATCH /repos/iamapsrajput/ShagunIntelligence \
  --field has_vulnerability_alerts=true

gh api -X PUT /repos/iamapsrajput/ShagunIntelligence/automated-security-fixes

# Set topics
echo "2. Setting repository topics..."
gh api -X PUT /repos/iamapsrajput/ShagunIntelligence/topics \
  --field names='["algorithmic-trading","ai-agents","fintech","fastapi","crewai","python","docker","kubernetes","zerodha-kite","trading-bot","machine-learning","quantitative-finance","real-time-data","risk-management"]'

# Create labels
echo "3. Creating labels..."
gh label create "priority: critical" --color "B60205" --description "Critical issues requiring immediate attention" || true
gh label create "security" --color "FF0000" --description "Security related issues" || true
gh label create "agents" --color "7057FF" --description "AI agents and CrewAI related" || true
gh label create "trading" --color "006B75" --description "Trading system and Kite Connect" || true
gh label create "good first issue" --color "7057FF" --description "Good for newcomers" || true

# Configure repository settings
echo "4. Configuring repository settings..."
gh api -X PATCH /repos/iamapsrajput/ShagunIntelligence \
  --field description="AI-Powered Algorithmic Trading Platform with Multi-Agent System for Indian Stock Market" \
  --field has_issues=true \
  --field has_projects=true \
  --field has_wiki=false \
  --field has_discussions=true \
  --field allow_squash_merge=true \
  --field allow_merge_commit=false \
  --field delete_branch_on_merge=true

echo "Repository security setup completed!"
echo "Remember to:"
echo "- Set up branch protection rules manually if needed"
echo "- Configure secrets for CI/CD"
echo "- Enable GitHub Advanced Security (if available)"
```

Make the script executable and run:

```bash
chmod +x setup-repo-security.sh
./setup-repo-security.sh
```

## Important Notes

1. **GitHub Advanced Security**: Some features require GitHub Advanced Security (available for public repos or paid plans)
2. **Secrets Management**: Never expose actual secrets in scripts or logs
3. **Branch Protection**: May need adjustment based on your workflow
4. **Testing**: Test settings on a fork first before applying to production repository
5. **Permissions**: Ensure you have admin access to the repository

## Troubleshooting

- If commands fail with permission errors, check your GitHub CLI authentication
- Some API endpoints may require specific GitHub plan features
- Branch protection rules may need existing CI/CD workflows to be functional first
- Secret scanning requires GitHub Advanced Security for private repositories
