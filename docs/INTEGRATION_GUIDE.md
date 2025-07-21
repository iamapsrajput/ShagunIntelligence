# AlgoHive Integration Guide

This guide provides comprehensive instructions for integrating all components of the AlgoHive platform and ensuring smooth operation of the complete system.

## Table of Contents

1. [System Integration Overview](#system-integration-overview)
2. [Component Dependencies](#component-dependencies)
3. [Integration Steps](#integration-steps)
4. [Configuration Management](#configuration-management)
5. [Testing Integration](#testing-integration)
6. [Deployment Integration](#deployment-integration)
7. [Monitoring Setup](#monitoring-setup)
8. [Troubleshooting Integration Issues](#troubleshooting-integration-issues)

## System Integration Overview

AlgoHive consists of multiple interconnected components that must be properly integrated:

```
Frontend (React) ←→ Backend (FastAPI) ←→ Agent System (CrewAI)
                            ↓
                    External Services
                    (Zerodha, AI APIs)
```

## Component Dependencies

### Dependency Matrix

| Component | Depends On | Required For |
|-----------|------------|--------------|
| React Dashboard | FastAPI, WebSocket | User Interface |
| FastAPI Backend | PostgreSQL, Redis, Agents | Core Functionality |
| Agent System | AI APIs, Market Data | Trading Decisions |
| PostgreSQL | - | Data Persistence |
| Redis | - | Caching, Sessions |
| Zerodha Integration | Valid API Keys | Live Trading |
| AI Services | API Keys | Agent Intelligence |

### Version Compatibility

```yaml
Python: 3.11+
Node.js: 18+
PostgreSQL: 15+
Redis: 7+
Docker: 24+
Kubernetes: 1.28+
```

## Integration Steps

### Step 1: Environment Setup

1. **Clone Repository**
```bash
git clone https://github.com/algohive/algohive.git
cd algohive
```

2. **Create Environment File**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Install Dependencies**
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd dashboard
npm install
cd ..
```

### Step 2: Database Integration

1. **Create Database**
```bash
# Using Docker
docker run -d \
  --name algohive-postgres \
  -e POSTGRES_PASSWORD=algohive \
  -e POSTGRES_USER=algohive \
  -e POSTGRES_DB=algohive \
  -p 5432:5432 \
  postgres:15-alpine

# Or native PostgreSQL
createdb algohive
```

2. **Run Migrations**
```bash
alembic upgrade head
```

3. **Verify Database**
```bash
python scripts/verify_db.py
```

### Step 3: Redis Integration

1. **Start Redis**
```bash
# Using Docker
docker run -d \
  --name algohive-redis \
  -p 6379:6379 \
  redis:7-alpine

# Or native Redis
redis-server
```

2. **Test Connection**
```bash
python scripts/test_redis.py
```

### Step 4: Zerodha Kite Integration

1. **Generate Access Token**
```python
# scripts/setup_kite.py
from services.kite import KiteService

def setup_kite():
    kite = KiteService()
    
    # Get login URL
    login_url = kite.get_login_url()
    print(f"1. Visit this URL to login: {login_url}")
    print("2. After login, copy the request_token from the URL")
    
    request_token = input("Enter request_token: ")
    
    # Generate access token
    access_token = kite.generate_session(request_token)
    print(f"\nAccess Token: {access_token}")
    print("Add this to your .env file as KITE_ACCESS_TOKEN")

if __name__ == "__main__":
    setup_kite()
```

2. **Update Configuration**
```env
# .env
KITE_API_KEY=your-api-key
KITE_API_SECRET=your-api-secret
KITE_ACCESS_TOKEN=generated-access-token
```

### Step 5: AI Service Integration

1. **Configure AI Services**
```env
# .env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Optional
GROQ_API_KEY=your-groq-key
COHERE_API_KEY=your-cohere-key
```

2. **Test AI Services**
```bash
python scripts/test_ai_services.py
```

### Step 6: Agent System Integration

1. **Initialize Agents**
```python
# scripts/init_agents.py
from agents.crew_manager import CrewManager

def initialize_agents():
    crew = CrewManager()
    
    # Test each agent
    agents = [
        "market_analyst",
        "technical_indicator",
        "sentiment_analyst",
        "risk_manager",
        "trade_executor",
        "data_processor"
    ]
    
    for agent in agents:
        result = crew.test_agent(agent)
        print(f"{agent}: {'✓' if result else '✗'}")

if __name__ == "__main__":
    initialize_agents()
```

2. **Configure Agent Parameters**
```yaml
# config/agents.yaml
agents:
  market_analyst:
    enabled: true
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2000
```

### Step 7: Backend Integration

1. **Start FastAPI Server**
```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Verify API Health**
```bash
curl http://localhost:8000/api/v1/health
```

### Step 8: Frontend Integration

1. **Configure API Endpoint**
```javascript
// dashboard/src/config.js
export const config = {
  API_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws'
};
```

2. **Start Dashboard**
```bash
cd dashboard
npm start
```

### Step 9: WebSocket Integration

1. **Test WebSocket Connection**
```javascript
// scripts/test_websocket.js
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8000/ws/market');

ws.on('open', () => {
  console.log('Connected to WebSocket');
  ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['RELIANCE', 'TCS']
  }));
});

ws.on('message', (data) => {
  console.log('Received:', JSON.parse(data));
});
```

## Configuration Management

### Centralized Configuration

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    app_name: str = "AlgoHive"
    environment: str = "development"
    
    # Database
    database_url: str
    
    # Redis
    redis_url: str
    
    # Trading
    max_position_size: int = 100000
    risk_per_trade: float = 2.0
    
    # AI Services
    openai_api_key: str = None
    anthropic_api_key: str = None
    
    class Config:
        env_file = ".env"
```

### Environment-Specific Configs

```bash
# Development
export APP_ENV=development
export LOG_LEVEL=DEBUG

# Staging
export APP_ENV=staging
export LOG_LEVEL=INFO

# Production
export APP_ENV=production
export LOG_LEVEL=WARNING
```

## Testing Integration

### Integration Test Suite

```python
# tests/integration/test_full_integration.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_full_trading_cycle():
    """Test complete trading workflow"""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        # 1. Health check
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        
        # 2. Get market data
        response = await client.get("/api/v1/market/quote/RELIANCE")
        assert response.status_code == 200
        
        # 3. Trigger analysis
        response = await client.post("/api/v1/agents/analyze", json={
            "symbol": "RELIANCE",
            "agents": ["market_analyst", "technical_indicator"]
        })
        assert response.status_code == 200
        
        # 4. Place order (paper trading)
        response = await client.post("/api/v1/trading/order", json={
            "symbol": "RELIANCE",
            "quantity": 10,
            "order_type": "BUY",
            "paper_trade": True
        })
        assert response.status_code == 200
```

### Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/integration/ --cov=app --cov-report=html

# Run specific test
pytest tests/integration/test_full_integration.py::test_full_trading_cycle
```

## Deployment Integration

### Docker Compose Integration

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: algohive
      POSTGRES_PASSWORD: algohive
      POSTGRES_DB: algohive
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  backend:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://algohive:algohive@postgres:5432/algohive
      REDIS_URL: redis://redis:6379
    ports:
      - "8000:8000"

  frontend:
    build: ./dashboard
    depends_on:
      - backend
    environment:
      REACT_APP_API_URL: http://backend:8000
    ports:
      - "3000:3000"

  nginx:
    image: nginx:alpine
    depends_on:
      - backend
      - frontend
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
      - "443:443"

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Integration

```bash
# Deploy to Kubernetes
kubectl apply -k k8s/overlays/production/

# Verify deployment
kubectl get pods -n algohive
kubectl get services -n algohive
kubectl get ingress -n algohive
```

## Monitoring Setup

### Prometheus Integration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'algohive-app'
    static_configs:
      - targets: ['app:8000']
    
  - job_name: 'algohive-agents'
    static_configs:
      - targets: ['agents:9090']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboards

```bash
# Import dashboards
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/algohive-dashboard.json
```

## Troubleshooting Integration Issues

### Common Integration Problems

1. **Database Connection Failed**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Test connection
psql -h localhost -U algohive -d algohive -c "SELECT 1;"
```

2. **Redis Connection Failed**
```bash
# Check Redis is running
redis-cli ping

# Test connection
redis-cli -h localhost -p 6379 INFO
```

3. **Agent Communication Issues**
```python
# Debug agent communication
from agents.crew_manager import CrewManager

crew = CrewManager()
crew.debug_mode = True
result = crew.analyze_market("RELIANCE")
print(crew.debug_log)
```

4. **WebSocket Connection Drops**
```javascript
// Implement reconnection logic
const connectWebSocket = () => {
  const ws = new WebSocket(WS_URL);
  
  ws.onclose = () => {
    console.log('WebSocket disconnected, reconnecting...');
    setTimeout(connectWebSocket, 5000);
  };
  
  return ws;
};
```

### Verification Checklist

- [ ] All services are running
- [ ] Database migrations completed
- [ ] Redis is accessible
- [ ] API endpoints responding
- [ ] WebSocket connections stable
- [ ] Agents initialized successfully
- [ ] External APIs authenticated
- [ ] Frontend can reach backend
- [ ] Monitoring is collecting metrics
- [ ] Logs are being aggregated

## Integration Best Practices

1. **Use Health Checks**
```python
@app.get("/health/detailed")
async def detailed_health():
    return {
        "database": check_database(),
        "redis": check_redis(),
        "agents": check_agents(),
        "external_apis": check_external_apis()
    }
```

2. **Implement Circuit Breakers**
```python
from circuit_breaker import CircuitBreaker

kite_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=KiteException
)

@kite_breaker
def call_kite_api():
    # API call logic
```

3. **Use Service Discovery**
```yaml
# For Kubernetes
apiVersion: v1
kind: Service
metadata:
  name: algohive-backend
  labels:
    app: algohive
spec:
  selector:
    app: algohive-backend
  ports:
    - port: 8000
      name: http
```

4. **Implement Retry Logic**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def resilient_api_call():
    # API call with automatic retry
```

## Next Steps

After completing integration:

1. Run full system tests
2. Configure monitoring alerts
3. Set up backup procedures
4. Document custom configurations
5. Plan for scaling
6. Schedule regular maintenance

---

*Integration Guide Version: 1.0*
*Last Updated: [Current Date]*
*Next Review: [30 days from current date]*