# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Development with hot reload
docker-compose -f docker-compose.dev.yml up --build

# Production deployment  
docker-compose up --build -d

# Local development without Docker
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Testing and Code Quality
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_trading.py

# Code formatting
black .

# Linting
flake8

# Type checking
mypy .
```

### Database Operations
```bash
# Access PostgreSQL (development)
docker exec -it shagunintelligence-postgres-1 psql -U shagunintelligence -d shagunintelligence_dev

# View database tables
\dt
```

## Architecture Overview

This is an AI-powered intraday trading platform built with:
- **FastAPI**: REST API backend (`app/`)
- **CrewAI**: Multi-agent AI system (`agents/`)
- **Zerodha Kite Connect**: Trading API integration (`services/kite/`)
- **PostgreSQL**: Primary database
- **Redis**: Caching and session storage

### Core Components

**Multi-Agent System (`agents/`)**:
- `market_analyst/`: Technical analysis and trend identification using AI
- `risk_manager/`: Automated risk assessment and position sizing
- `trader/`: Final trading decisions and execution logic
- `data_processor/`: Market data collection and preprocessing
- `crew_manager.py`: Orchestrates agent collaboration using CrewAI framework

**API Layer (`app/`)**:
- `api/routes/`: REST endpoints for trading, market data, and health checks
- `core/config.py`: Environment-based configuration management
- `main.py`: FastAPI application with CORS, lifespan management

**External Services (`services/`)**:
- `kite/`: Complete Zerodha integration with rate limiting, error handling, WebSocket support
- `database/`: SQLAlchemy database connections
- `notifications/`: Alert and notification systems

### Key Integrations

**Zerodha Kite Connect**: Full trading API integration with authentication, order management, portfolio tracking, real-time data via WebSocket, and comprehensive error handling.

**CrewAI Framework**: Coordinates multiple AI agents for collaborative trading decisions. Each agent has specialized roles and they work together through defined tasks and workflows.

### Environment Configuration

Development and production environments are configured via:
- `.env` files for sensitive data (API keys, database URLs)
- `config/environments/` for environment-specific settings
- Docker Compose files for different deployment scenarios

### API Structure

Key endpoints:
- `/api/v1/health` - System health monitoring
- `/api/v1/trading/*` - Order placement, position management, AI analysis
- `/api/v1/market/*` - Real-time quotes, historical data

Interactive documentation available at `/docs` (Swagger) and `/redoc`.