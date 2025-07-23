#!/bin/bash

# Shagun Intelligence - Repository Verification Script
# This script helps verify that the repository is properly configured

set -e

echo "🔍 Shagun Intelligence Repository Verification"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ README.md not found. Are you in the correct directory?"
    exit 1
fi

echo "✅ README.md found"

# Check repository remote
echo ""
echo "🌐 Repository Configuration:"
echo "----------------------------"

REMOTE_URL=$(git remote get-url origin)
echo "Remote URL: $REMOTE_URL"

if [[ $REMOTE_URL == *"ShagunIntelligence"* ]]; then
    echo "✅ Repository remote is correctly set to ShagunIntelligence"
else
    echo "❌ Repository remote is not set to ShagunIntelligence"
    echo "   Current: $REMOTE_URL"
    echo "   Expected: https://github.com/iamapsrajput/ShagunIntelligence.git"
fi

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" = "main" ]; then
    echo "✅ On main branch"
else
    echo "⚠️  Not on main branch. Consider switching to main."
fi

# Check if we're up to date
echo ""
echo "📦 Repository Status:"
echo "--------------------"

git fetch origin
LOCAL_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse origin/main)

if [ "$LOCAL_COMMIT" = "$REMOTE_COMMIT" ]; then
    echo "✅ Repository is up to date with origin/main"
else
    echo "⚠️  Repository is not up to date with origin/main"
    echo "   Local:  $LOCAL_COMMIT"
    echo "   Remote: $REMOTE_COMMIT"
fi

# Check README.md content
echo ""
echo "📖 README.md Verification:"
echo "-------------------------"

if grep -q "Shagun Intelligence" README.md; then
    echo "✅ README.md contains correct project name"
else
    echo "❌ README.md does not contain 'Shagun Intelligence'"
fi

if grep -q "AI-Powered Algorithmic Trading Platform" README.md; then
    echo "✅ README.md contains correct project description"
else
    echo "❌ README.md does not contain correct project description"
fi

# Check if security files are present
echo ""
echo "🔒 Security Files Verification:"
echo "------------------------------"

SECURITY_FILES=(
    "SECURITY.md"
    ".github/CODEOWNERS"
    ".github/workflows/codeql.yml"
    ".github/dependabot.yml"
    ".github/ruleset.json"
    "scripts/configure-security-settings.sh"
    "SECURITY_SETUP_GUIDE.md"
)

for file in "${SECURITY_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

# Check GitHub repository URL
echo ""
echo "🌐 GitHub Repository Access:"
echo "---------------------------"

REPO_URL="https://github.com/iamapsrajput/ShagunIntelligence"
echo "Repository URL: $REPO_URL"

# Try to check if repository is accessible (basic check)
if command -v curl &> /dev/null; then
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$REPO_URL")
    if [ "$HTTP_STATUS" = "200" ]; then
        echo "✅ Repository is accessible on GitHub"
    else
        echo "⚠️  Repository might not be accessible (HTTP $HTTP_STATUS)"
    fi
else
    echo "⚠️  curl not available, cannot verify repository accessibility"
fi

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo ""
echo "1. Visit: https://github.com/iamapsrajput/ShagunIntelligence"
echo "2. Verify that README.md is displayed correctly (not SECURITY.md)"
echo "3. If README.md is still not showing:"
echo "   - Try hard refresh (Ctrl+F5 or Cmd+Shift+R)"
echo "   - Clear browser cache"
echo "   - Try incognito/private window"
echo "4. Run the security configuration: ./scripts/configure-security-settings.sh"
echo ""
echo "📚 Documentation:"
echo "================="
echo "- Security Setup Guide: SECURITY_SETUP_GUIDE.md"
echo "- Repository Settings: .github/REPOSITORY_SETTINGS.md"
echo "- Setup Commands: .github/SETUP_COMMANDS.md"
echo ""
echo "✅ Verification completed!"