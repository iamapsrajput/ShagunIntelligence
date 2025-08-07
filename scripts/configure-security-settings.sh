#!/bin/bash

# Shagun Intelligence - Security Settings Configuration Script
# This script helps configure manual security settings for the repository

set -e

echo "🔒 Shagun Intelligence Security Configuration"
echo "=============================================="

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI is not installed. Please install it first:"
    echo "   https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub. Please run:"
    echo "   gh auth login"
    exit 1
fi

# Get repository name
REPO_NAME="iamapsrajput/AlgoHive"
echo "📦 Configuring repository: $REPO_NAME"

echo ""
echo "🔧 Step 1: Enabling Security Features"
echo "-------------------------------------"

# Enable vulnerability alerts
echo "✅ Enabling vulnerability alerts..."
gh api -X PATCH /repos/$REPO_NAME \
  --field has_vulnerability_alerts=true

# Enable automated security fixes
echo "✅ Enabling automated security fixes..."
gh api -X PUT /repos/$REPO_NAME/automated-security-fixes

# Enable secret scanning (if available)
echo "✅ Enabling secret scanning..."
gh api -X PATCH /repos/$REPO_NAME \
  --field security_and_analysis='{"secret_scanning":{"status":"enabled"},"secret_scanning_push_protection":{"status":"enabled"}}' || echo "⚠️  Secret scanning may require GitHub Advanced Security"

echo ""
echo "🏷️  Step 2: Setting Repository Topics"
echo "-------------------------------------"

# Set repository topics
echo "✅ Setting repository topics..."
gh api -X PUT /repos/$REPO_NAME/topics \
  --field names='["algorithmic-trading","ai-agents","fintech","fastapi","crewai","python","docker","kubernetes","zerodha-kite","trading-bot","machine-learning","quantitative-finance","real-time-data","risk-management"]'

echo ""
echo "🏷️  Step 3: Creating Repository Labels"
echo "--------------------------------------"

# Priority labels
echo "✅ Creating priority labels..."
gh label create "priority: critical" --color "B60205" --description "Critical issues requiring immediate attention" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "priority: high" --color "D93F0B" --description "High priority issues" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "priority: medium" --color "FBCA04" --description "Medium priority issues" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "priority: low" --color "0E8A16" --description "Low priority issues" --repo $REPO_NAME || echo "⚠️  Label may already exist"

# Type labels
echo "✅ Creating type labels..."
gh label create "bug" --color "D73A49" --description "Something isn't working" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "enhancement" --color "A2EEEF" --description "New feature or request" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "documentation" --color "0075CA" --description "Improvements or additions to documentation" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "performance" --color "FF6B6B" --description "Performance related issues" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "security" --color "FF0000" --description "Security related issues" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "question" --color "D876E3" --description "Further information is requested" --repo $REPO_NAME || echo "⚠️  Label may already exist"

# Component labels
echo "✅ Creating component labels..."
gh label create "agents" --color "7057FF" --description "AI agents and CrewAI related" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "api" --color "00D4AA" --description "FastAPI backend issues" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "trading" --color "006B75" --description "Trading system and Kite Connect" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "database" --color "8B4513" --description "Database and ORM related" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "infrastructure" --color "5319E7" --description "Docker, K8s, deployment" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "frontend" --color "FF69B4" --description "UI and dashboard related" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "ci-cd" --color "1F77B4" --description "CI/CD pipeline issues" --repo $REPO_NAME || echo "⚠️  Label may already exist"

# Status labels
echo "✅ Creating status labels..."
gh label create "needs-triage" --color "EDEDED" --description "Needs initial review and labeling" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "needs-review" --color "0052CC" --description "Needs code review" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "needs-testing" --color "FF8C00" --description "Needs testing" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "in-progress" --color "FEF2C0" --description "Currently being worked on" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "blocked" --color "B60205" --description "Blocked by external dependency" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "good first issue" --color "7057FF" --description "Good for newcomers" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "help wanted" --color "008672" --description "Extra attention is needed" --repo $REPO_NAME || echo "⚠️  Label may already exist"
gh label create "dependencies" --color "0366D6" --description "Dependency updates" --repo $REPO_NAME || echo "⚠️  Label may already exist"

echo ""
echo "🔒 Step 4: Manual Settings Required"
echo "-----------------------------------"

echo "⚠️  The following settings need to be configured manually in GitHub:"
echo ""
echo "1. 📋 Branch Protection Rules:"
echo "   - Go to Settings > Branches"
echo "   - Add rule for 'main' branch"
echo "   - Enable: Require pull request reviews"
echo "   - Enable: Require status checks to pass"
echo "   - Enable: Require signed commits"
echo "   - Enable: Require linear history"
echo "   - Enable: Restrict force pushes"
echo "   - Enable: Restrict deletions"
echo ""
echo "2. 🔍 Security & Analysis:"
echo "   - Go to Settings > Security & analysis"
echo "   - Enable: Dependency graph"
echo "   - Enable: Dependabot alerts"
echo "   - Enable: Dependabot security updates"
echo "   - Enable: Code scanning"
echo "   - Enable: Secret scanning"
echo ""
echo "3. 📝 Repository Rulesets:"
echo "   - Go to Settings > Rulesets"
echo "   - Create new ruleset using the provided JSON file"
echo "   - Upload: .github/ruleset.json"
echo ""

echo "✅ Configuration script completed!"
echo ""
echo "📚 Next Steps:"
echo "1. Configure branch protection rules manually"
echo "2. Enable security features in repository settings"
echo "3. Create repository ruleset using the JSON file"
echo "4. Test the security workflows by creating a test PR"
echo ""
echo "📖 For detailed instructions, see: .github/REPOSITORY_SETTINGS.md"
