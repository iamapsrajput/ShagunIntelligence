# Dependabot Security Best Practices

This guide provides best practices for handling Dependabot alerts and dependency updates in the Shagun Intelligence trading platform.

## 🔍 Understanding Dependabot Alerts

### Alert Severity Levels

**Critical** 🔴
- Immediate action required
- High CVSS score (9.0-10.0)
- Remote code execution, data breaches
- **Trading Impact**: Could compromise trading system integrity

**High** 🟠  
- Action required within 7 days
- CVSS score (7.0-8.9)
- Privilege escalation, information disclosure
- **Trading Impact**: Could affect trading decisions or user data

**Medium** 🟡
- Action required within 30 days  
- CVSS score (4.0-6.9)
- Denial of service, limited data exposure
- **Trading Impact**: May cause system instability

**Low** 🟢
- Action required within 90 days
- CVSS score (0.1-3.9) 
- Minor information disclosure
- **Trading Impact**: Minimal risk to trading operations

## 🚨 Critical Trading Platform Dependencies

### High-Priority Dependencies (Immediate Updates Required)

```yaml
# Authentication & Security
- cryptography
- passlib
- python-jose
- pycryptodome

# API Framework
- fastapi
- uvicorn
- starlette

# Database & ORM
- sqlalchemy
- psycopg2-binary
- alembic

# Trading APIs
- kiteconnect
- requests
- httpx

# AI/ML Framework
- crewai
- langchain
- openai

# Data Processing
- pandas
- numpy
```

### Medium-Priority Dependencies (Weekly Updates)

```yaml
# Development Tools
- pytest
- black
- flake8
- mypy

# Utilities
- python-dateutil
- pytz
- aiofiles

# Monitoring
- prometheus-client
- loguru
```

## 🔧 Handling Dependabot PRs

### 1. Automated Security Updates (Enabled)

For **critical** and **high** severity vulnerabilities, Dependabot automatically creates PRs.

**Review Process:**
```bash
# 1. Check the vulnerability details
gh pr view <PR_NUMBER>

# 2. Review the changelog
# Check the dependency's changelog for breaking changes

# 3. Run security scan
python -m safety check

# 4. Test locally
git checkout <dependabot-branch>
pip install -r requirements.txt
pytest

# 5. Approve if tests pass
gh pr review <PR_NUMBER> --approve
```

### 2. Regular Dependency Updates

**Weekly Review Process:**

1. **Triage New PRs**
   ```bash
   # List all Dependabot PRs
   gh pr list --author "dependabot[bot]" --state open
   
   # Group by severity
   gh pr list --label "dependencies" --label "security"
   ```

2. **Prioritize Updates**
   - Security updates first
   - Core trading dependencies
   - Framework updates
   - Development tools last

3. **Testing Strategy**
   ```bash
   # Create test environment
   python -m venv test-env
   source test-env/bin/activate
   
   # Install updated dependencies
   git checkout <dependabot-branch>
   pip install -r requirements.txt
   
   # Run comprehensive tests
   pytest tests/ -v
   python run_tests.py --full
   
   # Test AI agents specifically
   pytest tests/unit/test_agents/ -v
   
   # Test trading components
   pytest tests/unit/test_trading/ -v
   ```

## 📊 Dependency Groups Management

Based on our Dependabot configuration, updates are grouped:

### FastAPI Stack
```yaml
# Monitor these together for compatibility
- fastapi
- uvicorn
- pydantic
- starlette
```
**Testing Focus**: API endpoints, WebSocket connections, request validation

### CrewAI Stack  
```yaml
# AI agent dependencies
- crewai
- langchain
- openai
```
**Testing Focus**: Agent behavior, model interactions, decision making

### Database Stack
```yaml
# Database-related updates
- sqlalchemy
- alembic
- psycopg2
- asyncpg
- redis
```
**Testing Focus**: Database connections, migrations, data integrity

### Data Science Stack
```yaml
# Data processing dependencies
- pandas
- numpy
- scipy
- scikit-learn
```
**Testing Focus**: Data analysis, model training, mathematical operations

## 🛠️ Resolving Common Dependabot Issues

### 1. Version Conflicts

**Problem**: Dependabot updates one package but creates conflicts with others.

**Solution**:
```bash
# Check for conflicts
pip-compile requirements.in --verbose

# Use pip-tools to resolve
pip install pip-tools
pip-compile --upgrade requirements.in

# Or update manually
pip install package-name==specific-version
```

### 2. Breaking Changes

**Problem**: New version introduces breaking changes.

**Solution**:
```python
# Option 1: Pin to last working version temporarily
package-name==1.2.3  # TODO: Update after fixing breaking changes

# Option 2: Create compatibility layer
# In your code, handle both old and new API
try:
    from new_package import new_function
except ImportError:
    from old_package import old_function as new_function
```

### 3. Test Failures

**Problem**: Updated dependencies cause test failures.

**Solution**:
```bash
# 1. Identify failing tests
pytest --tb=short

# 2. Check if it's a test issue or real bug
pytest -v tests/specific_test.py

# 3. Update test mocks/fixtures if needed
# 4. If real bug, report to dependency maintainer
```

## 🔒 Security Alert Response Workflow

### Critical Vulnerability Response (< 24 hours)

1. **Immediate Assessment**
   ```bash
   # Check if vulnerability affects running systems
   pip list | grep <vulnerable-package>
   
   # Review CVSS score and impact
   # Check if exploit code is available
   ```

2. **Hotfix Process**
   ```bash
   # Create hotfix branch
   git checkout -b hotfix/security-<package-name>
   
   # Update only the vulnerable package
   pip install <package-name>==<secure-version>
   pip freeze > requirements.txt
   
   # Run security tests
   python -m bandit -r app/ agents/ services/
   python -m safety check
   
   # Deploy immediately if tests pass
   ```

3. **Verification**
   ```bash
   # Verify fix in staging
   curl https://staging.shagunintelligence.com/api/v1/health
   
   # Monitor for issues
   # Check logs for errors
   ```

### High/Medium Vulnerability Response (< 7 days)

1. **Planned Update**
   - Schedule maintenance window
   - Prepare rollback plan
   - Communicate with stakeholders

2. **Comprehensive Testing**
   ```bash
   # Full test suite
   pytest tests/ --cov=app --cov=agents --cov=services
   
   # Integration tests
   pytest tests/integration/
   
   # Performance tests
   python tests/performance/load_test.py
   ```

3. **Staged Deployment**
   - Deploy to staging first
   - Monitor for 24 hours
   - Deploy to production

## 📈 Monitoring Dependency Health

### Weekly Dependency Audit

```bash
#!/bin/bash
# dependency-audit.sh

echo "=== Dependency Security Audit ==="

# Check for known vulnerabilities
echo "1. Checking for security vulnerabilities..."
python -m safety check --json > security-report.json

# Check for outdated packages
echo "2. Checking for outdated packages..."
pip list --outdated > outdated-packages.txt

# Check for conflicting dependencies
echo "3. Checking for dependency conflicts..."
pip check

# Generate dependency tree
echo "4. Generating dependency tree..."
pip-tree > dependency-tree.txt

echo "Audit complete. Review reports in current directory."
```

### Automated Monitoring

Set up GitHub Actions to monitor dependency health:

```yaml
# .github/workflows/dependency-check.yml
name: Weekly Dependency Check

on:
  schedule:
    - cron: '0 9 * * 1'  # Every Monday at 9 AM

jobs:
  dependency-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install safety pip-audit
          pip install -r requirements.txt
      
      - name: Security audit
        run: |
          safety check --json --output safety-report.json
          pip-audit --format json --output pip-audit-report.json
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: "*-report.json"
```

## 🎯 Trading Platform Specific Considerations

### Financial Data Dependencies

**Critical Packages**:
- `kiteconnect`: Trading API - Monitor for API changes
- `pandas`: Data processing - Ensure backward compatibility
- `sqlalchemy`: Database - Test all queries still work
- `cryptography`: Encryption - Verify no key format changes

**Testing Checklist**:
- [ ] Order placement functionality
- [ ] Real-time data processing
- [ ] Historical data analysis
- [ ] Risk management calculations
- [ ] Authentication and authorization
- [ ] Data encryption/decryption

### AI Agent Dependencies

**Critical Packages**:
- `crewai`: Core AI framework
- `langchain`: LLM integration
- `openai`: AI model access

**Testing Checklist**:
- [ ] Agent communication
- [ ] Decision-making processes
- [ ] Model inference
- [ ] Data preprocessing
- [ ] Error handling in AI workflows

## 📚 Documentation Requirements

When updating dependencies, update:

1. **requirements.txt**: Pin to specific versions for production
2. **docs/DEPENDENCIES.md**: Document major changes
3. **CHANGELOG.md**: Log dependency updates
4. **README.md**: Update installation instructions if needed
5. **Docker files**: Update base images if needed

## 🚨 Emergency Procedures

### Rollback Process

```bash
# 1. Identify last known good version
git log --oneline --grep="dependency"

# 2. Create rollback branch
git checkout -b rollback/dependency-fix

# 3. Revert to previous requirements
git checkout <last-good-commit> -- requirements.txt

# 4. Test and deploy
pytest
docker build -t rollback-test .
```

### Communication Template

```markdown
## Dependency Security Update - [DATE]

**Affected Component**: [Package Name]
**Severity**: [Critical/High/Medium/Low]
**Impact**: [Description of impact on trading platform]
**Downtime Required**: [Yes/No - Duration if yes]

**Action Taken**:
- [x] Updated package to secure version
- [x] Ran security tests
- [x] Verified trading functionality
- [x] Deployed to staging
- [x] Deployed to production

**Verification Steps**:
- [ ] API health check passed
- [ ] Trading simulation successful
- [ ] AI agents responding normally
- [ ] No error spikes in logs

**Rollback Plan**: Available if issues detected
**Contact**: security@shagunintelligence.com
```

## 🔗 Useful Tools and Resources

### CLI Tools
```bash
# Install helpful tools
pip install safety pip-audit pip-tools pipdeptree

# Check security
safety check
pip-audit

# Manage dependencies
pip-compile requirements.in
pip-sync requirements.txt

# Visualize dependency tree
pipdeptree
```

### Resources
- [Python Package Index (PyPI)](https://pypi.org/)
- [CVE Database](https://cve.mitre.org/)
- [GitHub Advisory Database](https://github.com/advisories)
- [Snyk Vulnerability Database](https://security.snyk.io/)
- [OWASP Dependency Check](https://owasp.org/www-project-dependency-check/)

Remember: In financial applications, security is paramount. When in doubt, err on the side of caution and prefer security over convenience.