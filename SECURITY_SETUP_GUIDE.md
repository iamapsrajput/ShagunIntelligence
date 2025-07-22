# 🔒 Shagun Intelligence Security Setup Guide

This guide will help you configure the manual security settings that were added by the recent security commits.

## 📋 What Changed

The recent security commits added comprehensive repository governance including:
- **Security workflows** (CodeQL, Bandit, Safety, Semgrep)
- **Issue templates** for better bug reporting
- **PR templates** with security checklists
- **CODEOWNERS** file for review requirements
- **Security documentation** and policies

## 🚀 Quick Setup

### Step 1: Run the Configuration Script

```bash
# Make sure you have GitHub CLI installed and authenticated
gh auth login

# Run the configuration script
./scripts/configure-security-settings.sh
```

### Step 2: Manual GitHub Settings

#### A. Branch Protection Rules
1. Go to your repository on GitHub
2. Navigate to **Settings** → **Branches**
3. Click **Add rule** for the `main` branch
4. Configure the following:

```yaml
Branch name pattern: main
✅ Require a pull request before merging
  - Require approvals: 1
  - Dismiss stale PR approvals when new commits are pushed: ✅
  - Require review from code owners: ✅

✅ Require status checks to pass before merging
  - Require branches to be up to date before merging: ✅
  - Status checks (add these):
    - CI Pipeline / lint
    - CI Pipeline / test
    - CI Pipeline / build-docker
    - CodeQL Security Analysis / analyze (python)
    - CodeQL Security Analysis / security-scan

✅ Require conversation resolution before merging: ✅
✅ Require signed commits: ✅
✅ Require linear history: ✅
✅ Include administrators: ✅
✅ Restrict pushes to matching branches: ✅
❌ Allow force pushes
❌ Allow deletions
```

#### B. Security & Analysis
1. Go to **Settings** → **Security & analysis**
2. Enable the following features:

```yaml
✅ Dependency graph
✅ Dependabot alerts
✅ Dependabot security updates
✅ Code scanning
✅ Secret scanning (if available)
✅ Security advisories
```

#### C. Repository Rulesets
1. Go to **Settings** → **Rulesets**
2. Click **Create ruleset**
3. Use the provided JSON file: `.github/ruleset.json`
4. Configure:
   - **Name**: "Shagun Intelligence Security Ruleset"
   - **Target**: main and develop branches
   - **Enforcement**: Active

## 📁 Files Created

### Security Configuration Files
- `.github/ruleset.json` - Repository ruleset for security enforcement
- `scripts/configure-security-settings.sh` - Automated setup script

### Existing Security Files (from previous commits)
- `.github/CODEOWNERS` - Code ownership and review requirements
- `.github/workflows/codeql.yml` - Security scanning workflow
- `.github/dependabot.yml` - Automated dependency updates
- `SECURITY.md` - Security policy and reporting
- `CONTRIBUTING.md` - Contribution guidelines with security requirements

## 🔍 Security Features Overview

### 1. Automated Security Scanning
- **CodeQL**: Advanced static analysis for Python and JavaScript
- **Bandit**: Python-specific security linting
- **Safety**: Vulnerability scanning for Python dependencies
- **Semgrep**: Additional SAST scanning

### 2. Dependency Management
- **Dependabot**: Automated security updates
- **Grouped updates**: Related packages updated together
- **Security prioritization**: Critical security updates get highest priority

### 3. Code Review Requirements
- **CODEOWNERS**: Mandatory reviews for critical components
- **Tiered ownership**: Different review requirements based on component criticality
- **Security focus**: Extra scrutiny for authentication, trading logic, and AI agents

### 4. Issue Management
- **5 specialized templates**: Bug reports, feature requests, performance issues, etc.
- **Trading platform context**: All templates include trading-specific fields
- **Priority classification**: Critical, high, medium, low priority labels

## 🧪 Testing the Setup

### Test 1: Create a Test PR
1. Create a new branch: `git checkout -b test-security-setup`
2. Make a small change to any file
3. Push and create a PR
4. Verify that:
   - CodeQL analysis runs
   - Required status checks are enforced
   - CODEOWNERS review is required

### Test 2: Test Issue Templates
1. Go to Issues tab
2. Click "New issue"
3. Verify all templates are available:
   - Bug report
   - Feature request
   - Performance issue
   - Documentation
   - Question

### Test 3: Test Security Workflows
1. Check Actions tab
2. Verify CodeQL workflow is running
3. Check Security tab for any findings

## 🔧 Troubleshooting

### Common Issues

#### Issue: "Status checks are not running"
**Solution**: 
1. Check if workflows are enabled in Actions tab
2. Verify branch protection rules are configured correctly
3. Ensure the workflow files are in the correct location

#### Issue: "CODEOWNERS not working"
**Solution**:
1. Verify the CODEOWNERS file is in `.github/CODEOWNERS`
2. Check that the usernames in CODEOWNERS are correct
3. Ensure the user has access to the repository

#### Issue: "Dependabot not creating PRs"
**Solution**:
1. Check if Dependabot is enabled in Security & analysis
2. Verify the `.github/dependabot.yml` file is present
3. Check Dependabot logs in Actions tab

### Getting Help

If you encounter issues:
1. Check the GitHub Actions logs for detailed error messages
2. Review the `.github/REPOSITORY_SETTINGS.md` file for detailed configuration
3. Check the `.github/SETUP_COMMANDS.md` file for CLI commands
4. Create an issue using the "question" template

## 📚 Additional Resources

- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [CodeQL Documentation](https://docs.github.com/en/code-security/codeql-cli)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)
- [Repository Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets)

## ✅ Checklist

- [ ] Run `./scripts/configure-security-settings.sh`
- [ ] Configure branch protection rules for main branch
- [ ] Enable security features in Settings → Security & analysis
- [ ] Create repository ruleset using `.github/ruleset.json`
- [ ] Test by creating a sample PR
- [ ] Test issue templates
- [ ] Verify CodeQL workflow is running
- [ ] Check Security tab for any findings

---

**Note**: Some features may require GitHub Advanced Security (Enterprise plan). The script will indicate which features are available for your plan.