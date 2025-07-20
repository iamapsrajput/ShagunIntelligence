# AI Trading Platform

An AI-powered intraday trading platform built with CrewAI and Zerodha Kite API. This platform uses multiple AI agents to analyze market conditions, manage risk, and execute trades automatically.

## Features

- **Multi-Agent AI System**: Uses CrewAI framework with specialized agents
  - Market Analyst: Technical analysis and trend identification
  - Risk Manager: Position sizing and risk assessment
  - Trader: Final trading decisions and execution
  - Data Processor: Market data collection and processing

- **Zerodha Integration**: Full integration with Kite Connect API
- **Real-time Market Data**: Live quotes and historical data
- **Risk Management**: Automated risk controls and position sizing
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Docker Support**: Containerized deployment for development and production

## Project Structure

```
aiTrading/
├── app/                    # FastAPI application
│   ├── api/               # API routes
│   ├── core/              # Core configuration
│   └── main.py            # Application entry point
├── agents/                # CrewAI agents
│   ├── market_analyst/    # Market analysis agent
│   ├── risk_manager/      # Risk management agent
│   ├── trader/            # Trading decision agent
│   └── data_processor/    # Data processing agent
├── services/              # External services
│   ├── kite/              # Zerodha Kite Connect
│   ├── database/          # Database connections
│   └── notifications/     # Alert services
├── data/                  # Data models and repositories
├── config/                # Environment configurations
├── scripts/               # Utility scripts
├── tests/                 # Test suite
└── docker-compose.yml     # Docker orchestration
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Zerodha Trading Account with KiteConnect API access
- OpenAI API key
- PostgreSQL (for production) or SQLite (for development)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd aiTrading
```

### 2. Environment Setup

Copy the environment template:
```bash
cp .env.example .env
```

Edit `.env` file with your configuration:
```bash
# Zerodha Kite Connect API
KITE_API_KEY=your_kite_api_key_here
KITE_API_SECRET=your_kite_api_secret_here
KITE_ACCESS_TOKEN=your_kite_access_token_here

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Other configurations...
```

### 3. Docker Deployment

#### Development Environment
```bash
# Start development environment with hot reload
docker-compose -f docker-compose.dev.yml up --build

# Access the application
http://localhost:8000
```

#### Production Environment
```bash
# Start production environment
docker-compose up --build -d

# View logs
docker-compose logs -f app
```

### 4. Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m uvicorn app.main:app --reload
```

## API Documentation

Once the application is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/trading/order` - Place trading order
- `GET /api/v1/trading/positions` - Get current positions
- `POST /api/v1/trading/analyze/{symbol}` - Get AI analysis for symbol
- `GET /api/v1/market/quote/{symbol}` - Get real-time quote

## Configuration

### Zerodha Kite Connect Setup

1. Create a KiteConnect app at https://kite.trade/connect/
2. Get your API key and secret
3. Generate access token using the login flow
4. Update the `.env` file with your credentials

### OpenAI Setup

1. Get API key from https://platform.openai.com/api-keys
2. Add to `.env` file
3. Ensure you have sufficient credits for API calls

### Trading Configuration

Adjust risk parameters in `.env`:
```bash
MAX_RISK_PER_TRADE=0.02        # 2% max risk per trade
MAX_DAILY_LOSS=0.05            # 5% max daily loss
DEFAULT_POSITION_SIZE=10000    # Default position size in INR
```

## Database

### PostgreSQL (Production)
```bash
# Access database
docker exec -it aitrading_postgres_1 psql -U aitrading -d aitrading

# View tables
\dt
```

### SQLite (Development)
Database file: `dev_trading.db`

## Monitoring and Logging

- Application logs: `logs/trading.log`
- Health check: http://localhost:8000/api/v1/health
- pgAdmin (if enabled): http://localhost:5050

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_trading.py
```

## Security Considerations

- Never commit API keys to version control
- Use environment variables for sensitive data
- Implement proper authentication for production
- Monitor API usage and rate limits
- Use HTTPS in production

## Deployment

### Production Deployment

1. Set up a production server (AWS EC2, DigitalOcean, etc.)
2. Install Docker and Docker Compose
3. Clone the repository
4. Configure production environment variables
5. Run with production compose file:
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

### Environment Variables for Production

```bash
DEBUG=False
DATABASE_URL=postgresql://user:password@localhost:5432/aitrading_prod
SECRET_KEY=your-production-secret-key
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This software is for educational and research purposes only. Trading involves risk of financial loss. Always test thoroughly with paper trading before using real money. The authors are not responsible for any financial losses incurred while using this software.

## Support

For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

## Roadmap

- [ ] Paper trading mode
- [ ] Advanced technical indicators
- [ ] ML-based price prediction
- [ ] Portfolio optimization
- [ ] Risk analytics dashboard
- [ ] Mobile notifications
- [ ] Backtesting framework
- [ ] Multi-broker support