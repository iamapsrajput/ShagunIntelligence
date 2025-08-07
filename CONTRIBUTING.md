# Contributing to Shagun Intelligence

Thank you for your interest in contributing to Shagun Intelligence! This guide will help you get started with contributing to our AI-powered algorithmic trading platform.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Security Guidelines](#security-guidelines)
- [Submission Process](#submission-process)
- [Community](#community)

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please read and follow our Code of Conduct:

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome newcomers and different perspectives
- **Be collaborative**: Work together constructively
- **Be professional**: Maintain professional standards in all interactions
- **No harassment**: Zero tolerance for harassment, discrimination, or inappropriate behavior

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.11+ installed
- Docker and Docker Compose
- Git configured with your GitHub account
- Basic understanding of:
  - FastAPI and async Python
  - AI/ML concepts (CrewAI framework)
  - Trading systems (optional but helpful)
  - Docker containerization

### First Contribution Ideas

Good first issues for newcomers:

- Documentation improvements
- Adding unit tests
- Fixing small bugs
- UI/UX enhancements
- Performance optimizations

Look for issues labeled `good first issue` or `help wanted`.

## 💻 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/shagunintelligence.git
cd shagunintelligence

# Add the upstream remote
git remote add upstream https://github.com/iamapsrajput/shagunintelligence.git
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your configuration
```

### 3. Docker Setup (Recommended)

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up --build

# Or for testing
docker-compose up --build
```

### 4. Database Setup

```bash
# Run database migrations
python -m alembic upgrade head

# Create test data (if available)
python scripts/create_test_data.py
```

### 5. Verify Installation

```bash
# Run tests
pytest

# Start the application
python -m uvicorn app.main:app --reload

# Check health endpoint
curl http://localhost:8000/api/v1/health
```

## 📏 Contributing Guidelines

### Branch Naming Convention

Use descriptive branch names following this pattern:

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Urgent fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### Commit Message Format

Follow the [Conventional Commits](https://conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `build`: Build system changes

**Examples:**

```bash
feat(agents): add new risk management agent
fix(trading): resolve order execution timeout issue
docs(api): update trading endpoints documentation
test(agents): add unit tests for market analyst agent
```

### Pull Request Guidelines

1. **One feature per PR**: Keep changes focused and atomic
2. **Link issues**: Reference related issues using keywords (fixes #123)
3. **Complete description**: Use the PR template thoroughly
4. **Up-to-date branch**: Rebase on latest main before submitting
5. **Clean history**: Squash commits if needed
6. **All checks pass**: Ensure CI/CD passes
7. **Reviews addressed**: Respond to all review comments

## 🎯 Coding Standards

### Python Code Style

We follow PEP 8 with some project-specific conventions:

```python
# Use Black for formatting
black .

# Use flake8 for linting
flake8 . --max-line-length=100

# Use mypy for type checking
mypy app/ agents/ services/
```

### Code Quality Requirements

- **Type hints**: Use type hints for all functions and methods
- **Docstrings**: Document all public functions, classes, and modules
- **Error handling**: Implement proper exception handling
- **Logging**: Use structured logging with appropriate levels
- **Constants**: Define constants in uppercase with clear names
- **Imports**: Group and sort imports (isort)

### Example Code Structure

```python
"""Module docstring describing the purpose."""

from typing import Dict, List, Optional
import logging

from fastapi import HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TradingRequest(BaseModel):
    """Request model for trading operations."""

    symbol: str
    quantity: int
    order_type: str
    price: Optional[float] = None

async def execute_trade(request: TradingRequest) -> Dict[str, str]:
    """
    Execute a trading order.

    Args:
        request: Trading request parameters

    Returns:
        Dictionary containing order status and details

    Raises:
        HTTPException: If trading execution fails
    """
    try:
        # Implementation here
        logger.info(f"Executing trade for {request.symbol}")
        return {"status": "success", "order_id": "12345"}
    except Exception as e:
        logger.error(f"Trade execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Trade execution failed")
```

## 🧪 Testing Requirements

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Critical components**: 95%+ coverage for trading logic and AI agents
- **Test types**: Unit, integration, and end-to-end tests

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from app.services.trading import TradingService

class TestTradingService:
    """Test suite for TradingService."""

    @pytest.fixture
    def trading_service(self):
        """Create a TradingService instance for testing."""
        return TradingService()

    @pytest.mark.asyncio
    async def test_execute_order_success(self, trading_service):
        """Test successful order execution."""
        # Arrange
        order_data = {"symbol": "RELIANCE", "quantity": 10}

        # Act
        result = await trading_service.execute_order(order_data)

        # Assert
        assert result["status"] == "success"
        assert "order_id" in result

    @pytest.mark.asyncio
    async def test_execute_order_failure(self, trading_service):
        """Test order execution failure handling."""
        # Test failure scenarios
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=agents --cov=services

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest -m "not slow"  # Skip slow tests

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## 🔒 Security Guidelines

### Financial Data Security

⚠️ **Critical**: This is a financial trading platform. Security is paramount.

#### Never Commit

- API keys, secrets, or credentials
- Real trading account information
- Personal financial data
- Production database URLs
- Private keys or certificates

#### Always Ensure

- Input validation for all trading parameters
- Proper authentication and authorization
- Secure handling of financial data
- SQL injection prevention
- XSS prevention in web interfaces
- Rate limiting for APIs
- Audit logging for all trading activities

#### Security Checklist

- [ ] No hardcoded secrets in code
- [ ] Environment variables used for configuration
- [ ] Input validation implemented
- [ ] Error messages don't expose sensitive data
- [ ] Database queries use parameterized statements
- [ ] Authentication/authorization properly implemented
- [ ] Logging doesn't contain sensitive information

### AI/ML Security

- **Model validation**: Validate AI agent outputs
- **Input sanitization**: Sanitize inputs to AI models
- **Adversarial protection**: Consider adversarial attacks
- **Model versioning**: Track and version AI models
- **Bias detection**: Monitor for algorithmic bias

## 📨 Submission Process

### Before Submitting

1. **Test thoroughly**: All tests pass locally
2. **Code quality**: Linting and type checking pass
3. **Documentation**: Update relevant documentation
4. **Security review**: Self-review for security issues
5. **Performance check**: No significant performance regression

### Submission Steps

1. **Create branch**: `git checkout -b feature/my-new-feature`
2. **Make changes**: Implement your feature/fix
3. **Test**: Run full test suite
4. **Commit**: Follow commit message conventions
5. **Push**: `git push origin feature/my-new-feature`
6. **PR**: Create pull request using the template
7. **Review**: Address reviewer feedback
8. **Merge**: Wait for approval and merge

### After Submission

- **Monitor**: Watch for any issues after deployment
- **Respond**: Address any post-merge feedback promptly
- **Document**: Update documentation if needed
- **Support**: Help with questions about your changes

## 👥 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check docs/ directory first
- **Code Review**: Ask for feedback on your PRs

### Communication Guidelines

- **Be clear**: Describe issues and questions clearly
- **Be patient**: Maintainers may need time to respond
- **Be helpful**: Help others when you can
- **Stay on topic**: Keep discussions relevant to the project

### Recognition

Contributors are recognized through:

- GitHub contributor listings
- Release notes acknowledgments
- Hall of fame (for significant contributions)
- Recommendations and references

## 📚 Additional Resources

### Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CrewAI Framework](https://docs.crewai.com/)
- [Zerodha Kite Connect API](https://kite.trade/docs/connect/v3/)
- [Algorithmic Trading Concepts](https://www.quantstart.com/)

### Development Tools

- **IDE**: VS Code with Python extensions
- **Database**: PostgreSQL GUI (pgAdmin, DBeaver)
- **API Testing**: Postman, HTTPie
- **Docker**: Docker Desktop
- **Git**: GitHub Desktop or command line

---

## 📞 Contact

For questions about contributing:

- Create a GitHub issue
- Start a GitHub discussion
- Contact maintainers directly for sensitive issues

Thank you for contributing to Shagun Intelligence! Your contributions help make algorithmic trading more accessible and intelligent. 🚀📈
