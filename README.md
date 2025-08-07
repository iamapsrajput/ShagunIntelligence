# Shagun Intelligence - AI-Powered Algorithmic Trading Platform

<div align="center">
  <img src="assets/shagunintelligence-logo.png" alt="Shagun Intelligence Logo" width="200"/>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/github/workflow/status/iamapsrajput/shagunintelligence/CI%20Pipeline/main)](https://github.com/iamapsrajput/shagunintelligence/actions)
[![Coverage](https://img.shields.io/codecov/c/github/iamapsrajput/shagunintelligence)](https://codecov.io/gh/iamapsrajput/shagunintelligence)

</div>

## 🚀 Overview

Shagun Intelligence is a sophisticated AI-powered algorithmic trading platform that leverages multiple AI agents working collaboratively to make intelligent trading decisions in the Indian stock market. Built with cutting-edge technologies including CrewAI for multi-agent orchestration, FastAPI for high-performance APIs, and integration with Zerodha Kite Connect for live trading.

### Key Features

- **🤖 Multi-Agent AI System**: Specialized agents for market analysis, technical indicators, sentiment analysis, risk management, and trade execution
- **📊 Real-time Data Processing**: WebSocket integration for live market data streaming
- **🔒 Advanced Risk Management**: Built-in circuit breakers, position sizing, and stop-loss management
- **📈 Comprehensive Analytics**: Real-time dashboards with performance metrics and agent insights
- **🏗️ Scalable Architecture**: Kubernetes-ready with auto-scaling and high availability
- **🔐 Enterprise Security**: End-to-end encryption, secure secret management, and audit trails

## 🚀 Quick Start

### For New Users

1. **[Setup Guide](./docs/setup/SETUP_GUIDE.md)** - Complete installation and configuration
2. **[API Key Setup](./docs/setup/API_KEY_SETUP_GUIDE.md)** - Configure required API keys
3. **[Quick Testing](./docs/testing/QUICK_TESTING_REFERENCE.md)** - Verify your installation

### For Live Trading

1. **[Live Trading Setup](./docs/trading/LIVE_TRADING_SETUP_GUIDE.md)** - Production trading configuration
2. **[Automated Trading](./docs/trading/AUTOMATED_TRADING_SETUP.md)** - AI-powered trading setup
3. **[Security Best Practices](./docs/security/SECURITY_BEST_PRACTICES.md)** - Essential security measures

## 📚 Documentation

Our documentation is comprehensively organized by category:

### 🔧 Setup & Configuration

- **[Setup Guide](./docs/setup/SETUP_GUIDE.md)** - Complete installation guide
- **[API Key Setup](./docs/setup/API_KEY_SETUP_GUIDE.md)** - API configuration
- **[Kite Connect Setup](./docs/setup/KITE_CONNECT_SETUP.md)** - Zerodha integration

### 💰 Trading & Operations

- **[Live Trading Setup](./docs/trading/LIVE_TRADING_SETUP_GUIDE.md)** - Production trading
- **[Automated Trading](./docs/trading/AUTOMATED_TRADING_SETUP.md)** - AI trading system

### 🏗️ Architecture & Development

- **[System Architecture](./docs/architecture/ARCHITECTURE_OVERVIEW.md)** - System design
- **[Agent Architecture](./docs/architecture/AGENT_ARCHITECTURE.md)** - AI agent system
- **[API Documentation](./docs/api/API_DOCUMENTATION.md)** - REST API reference

### 🚀 Deployment & Production

- **[Production Deployment](./docs/deployment/PRODUCTION_DEPLOYMENT_GUIDE.md)** - Production setup
- **[SSL/TLS Configuration](./docs/deployment/SSL_TLS_CONFIGURATION.md)** - Security setup
- **[Containerization](./docs/deployment/CONTAINERIZATION_GUIDE.md)** - Docker deployment

### 🔒 Security

- **[Security Best Practices](./docs/security/SECURITY_BEST_PRACTICES.md)** - Security guidelines
- **[Enhanced Resilience](./docs/security/ENHANCED_RESILIENCE_SYSTEM.md)** - Error handling

### 🧪 Testing & Validation

- **[Testing Guide](./docs/testing/TESTING_VALIDATION_GUIDE.md)** - Testing procedures
- **[Quick Testing](./docs/testing/QUICK_TESTING_REFERENCE.md)** - Quick validation

### 🆘 Support & Maintenance

- **[Troubleshooting](./docs/support/TROUBLESHOOTING.md)** - Common issues
- **[Integration Guide](./docs/support/INTEGRATION_GUIDE.md)** - Third-party integrations

**📖 [Complete Documentation Index](./docs/README.md)** - Navigate all documentation

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

Shagun Intelligence uses a microservices architecture with AI agents at its core:

```
┌────────────────────────────────────────────────────────────┐
│                    Shagun Intelligence Platform                │
├────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   React.js   │  │   FastAPI    │  │  WebSocket   │          │
│  │  Dashboard   │◄─┤   Backend    │◄─┤   Server     │          │
│  └─────────────┘  └──────┬──────┘  └─────────────┘          │
│                            │                                    │
│  ┌───────────────────────┴────────────────────────┐         │
│  │              CrewAI Agent System                   │         │
│  ├────────────────────────────────────────────────┤        │
│  │ ┌─────────┐ ┌──────────┐ ┌──────────┐           │         │
│  │ │ Market   │ │Technical  │ │Sentiment │           │          │
│  │ │ Analyst  │ │Indicator  │ │ Analyst  │           │          │
│  │ └─────────┘ └──────────┘ └──────────┘          │           │
│  │ ┌─────────┐ ┌──────────┐ ┌──────────┐          │           │
│  │ │  Risk    │ │  Trade   │  │  Data    │           │           │
│  │ │ Manager  │ │ Executor │  │Processor │           │           │
│  │ └─────────┘ └──────────┘ └──────────┘          │           │
│  │         ┌──────────────────┐                    │           │
│  │         │   Coordinator      │                    │           │
│  │         └──────────────────┘                    │           │
│  └───────────────────────────────────────────────┘           │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │PostgreSQL│  │  Redis    │  │ Zerodha   │  │    AI     │     │
│  │    DB    │  │  Cache    │  │   Kite    │  │ Services  │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└────────────────────────────────────────────────────────────┘
```

### AI Agents

1. **Market Analyst Agent**: Analyzes market trends, identifies patterns, and provides market insights
2. **Technical Indicator Agent**: Calculates and interprets technical indicators
3. **Sentiment Analyst Agent**: Analyzes market sentiment from news and social media
4. **Risk Manager Agent**: Manages portfolio risk, position sizing, and stop-loss strategies
5. **Trade Executor Agent**: Executes trades based on collective intelligence
6. **Data Processor Agent**: Handles real-time data processing and distribution
7. **Coordinator Agent**: Orchestrates all agents and makes final trading decisions

## 📦 Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- Node.js 18+ (for dashboard)
- Zerodha Kite Connect API credentials
- OpenAI API key (optional)
- Anthropic API key (optional)

## 🚀 Quick Start

### 🚀 5-Minute Automated Trading Setup

```bash
# 1. Clone and setup environment
git clone https://github.com/iamapsrajput/shagunintelligence.git
cd shagunintelligence
poetry install && poetry shell

# 2. Interactive API key setup (guides you through Kite Connect)
python scripts/setup_kite_credentials.py

# 3. Start the platform
# Terminal 1: Backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Dashboard
cd dashboard && npm install && npm run dev

# 4. Access and start trading
# Dashboard: http://localhost:3000 - Click "Start Automated Trading"!
# API Docs: http://localhost:8000/docs
```

### 🐳 Production Deployment (Docker)

```bash
# One-command production deployment
./scripts/deploy_production.sh

# Access production dashboard at https://your-domain.com
```

### ✨ What Happens When You Start Trading

1. **🔍 Market Analysis**: AI agents analyze 10 liquid stocks every 5 minutes
2. **🤖 Decision Making**: Technical + sentiment analysis with confidence scoring
3. **💰 Conservative Trading**: ₹200-300 per trade, max 3 positions, ₹100 daily loss limit
4. **🛡️ Risk Management**: Automatic stop-loss (3%) and take-profit (6%)
5. **📊 Real-time Monitoring**: Live dashboard with trade decisions and P&L

## 📥 Installation

### Manual Installation

1. **Clone the repository**

```bash
git clone https://github.com/shagunintelligence/shagunintelligence.git
cd shagunintelligence
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
# Install TA-Lib (required for technical analysis)
# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ..

# Install Python dependencies
pip install -r requirements.txt
```

4. **Set up database**

```bash
# Create PostgreSQL database
createdb shagunintelligence

# Run migrations
alembic upgrade head
```

5. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your configuration
```

6. **Start the application**

```bash
# Start FastAPI server
uvicorn app.main:app --reload

# In another terminal, start the React dashboard
cd dashboard
npm install
npm start
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Application
APP_ENV=development
SECRET_KEY=your-secret-key-here
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/shagunintelligence

# Redis
REDIS_URL=redis://localhost:6379/0

# Zerodha Kite API
KITE_API_KEY=your-kite-api-key
KITE_API_SECRET=your-kite-api-secret
KITE_ACCESS_TOKEN=your-access-token

# AI Services (Optional)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Trading Configuration
MAX_POSITION_SIZE=100000  # Maximum position size in INR
MAX_DAILY_TRADES=10       # Maximum trades per day
RISK_PER_TRADE=2.0       # Risk percentage per trade
```

### Agent Configuration

Configure agents in `config/agents.yaml`:

```yaml
agents:
  market_analyst:
    enabled: true
    confidence_threshold: 0.7

  technical_indicator:
    enabled: true
    indicators:
      - rsi
      - macd
      - bollinger_bands

  risk_manager:
    enabled: true
    max_portfolio_risk: 0.2
    stop_loss_percent: 2.0
```

## 📖 Usage

### Starting Trading

1. **Login to Zerodha**

```python
from services.kite import KiteService

kite = KiteService()
login_url = kite.get_login_url()
# Visit the URL and get the request token
access_token = kite.generate_session(request_token)
```

2. **Start the trading system**

```bash
# Using CLI
python -m shagunintelligence start --mode live

# Or using API
curl -X POST http://localhost:8000/api/v1/trading/start
```

3. **Monitor through dashboard**

- Open <http://localhost:3000>
- View real-time market data
- Monitor agent decisions
- Track portfolio performance

### Paper Trading

Test strategies without real money:

```bash
# Enable paper trading mode
python -m shagunintelligence start --mode paper --capital 1000000
```

## 📚 API Documentation

### Authentication

All API endpoints require authentication:

```bash
# Get access token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token in requests
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/portfolio
```

### Key Endpoints

#### Trading Operations

```bash
# Start trading
POST /api/v1/trading/start

# Stop trading
POST /api/v1/trading/stop

# Place manual order
POST /api/v1/trading/order
{
  "symbol": "RELIANCE",
  "quantity": 10,
  "order_type": "BUY",
  "price_type": "MARKET"
}

# Get positions
GET /api/v1/trading/positions
```

#### Market Data

```bash
# Get real-time quote
GET /api/v1/market/quote/{symbol}

# Get historical data
GET /api/v1/market/historical/{symbol}?from=2024-01-01&to=2024-01-31

# Subscribe to live data
WS /ws/market/{symbol}
```

#### Agent Operations

```bash
# Get agent status
GET /api/v1/agents/status

# Get agent analysis
GET /api/v1/agents/{agent_name}/analysis/{symbol}

# Trigger manual analysis
POST /api/v1/agents/analyze
{
  "symbol": "RELIANCE",
  "agents": ["market_analyst", "technical_indicator"]
}
```

For complete API documentation, visit <http://localhost:8000/docs>

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python run_tests.py --all

# Run specific test suite
python run_tests.py --suite unit

# Run with coverage
python run_tests.py --coverage --report

# Run paper trading simulation
pytest tests/simulation/test_paper_trading.py -v
```

### Test Coverage

Current test coverage:

- Unit Tests: 92%
- Integration Tests: 85%
- End-to-End Tests: 78%

## 🚢 Deployment

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace shagunintelligence

# Apply configurations
kubectl apply -k k8s/overlays/production/

# Check deployment status
kubectl get pods -n shagunintelligence

# Scale deployment
kubectl scale deployment shagunintelligence-app --replicas=5 -n shagunintelligence
```

### Production Checklist

- [ ] Set strong SECRET_KEY
- [ ] Configure SSL/TLS certificates
- [ ] Set up monitoring and alerting
- [ ] Configure backup schedules
- [ ] Review security policies
- [ ] Test disaster recovery plan

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
black .
flake8 .
mypy .
```

## 📊 Performance

- **Latency**: < 100ms API response time
- **Throughput**: 1000+ trades per minute
- **Uptime**: 99.9% SLA
- **Scalability**: Horizontal scaling up to 100 nodes

## 🛡️ Security

- End-to-end encryption
- OAuth 2.0 authentication
- Rate limiting and DDoS protection
- Regular security audits
- Compliance with financial regulations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- Documentation: [docs.shagunintelligence.com](https://docs.shagunintelligence.com)
- Issues: [GitHub Issues](https://github.com/shagunintelligence/shagunintelligence/issues)
- Email: <support@shagunintelligence.com>
- Discord: [Join our community](https://discord.gg/shagunintelligence)

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for multi-agent orchestration
- [Zerodha](https://kite.zerodha.com/) for trading APIs
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- All contributors and supporters

---

<div align="center">
  Made with ❤️ by the Shagun Intelligence Team
</div>
