# AlgoHive Troubleshooting Guide & FAQ

## Table of Contents

1. [Common Issues](#common-issues)
2. [Installation Problems](#installation-problems)
3. [Trading Issues](#trading-issues)
4. [Agent-Related Issues](#agent-related-issues)
5. [Performance Issues](#performance-issues)
6. [API & Connectivity](#api--connectivity)
7. [Database Issues](#database-issues)
8. [FAQ](#frequently-asked-questions)
9. [Debug Mode](#debug-mode)
10. [Getting Help](#getting-help)

## Common Issues

### Application Won't Start

**Symptoms**: Application crashes on startup or fails to initialize

**Solutions**:

1. **Check Python Version**
   ```bash
   python --version
   # Should be 3.11 or higher
   ```

2. **Verify Dependencies**
   ```bash
   pip list | grep -E "fastapi|uvicorn|crewai"
   # Ensure all required packages are installed
   ```

3. **Check Environment Variables**
   ```bash
   # Verify .env file exists
   ls -la .env
   
   # Check required variables
   grep -E "DATABASE_URL|REDIS_URL|SECRET_KEY" .env
   ```

4. **Database Connection**
   ```bash
   # Test PostgreSQL connection
   psql $DATABASE_URL -c "SELECT 1;"
   
   # Test Redis connection
   redis-cli ping
   ```

### Docker Issues

**Container Fails to Start**

```bash
# Check container logs
docker-compose logs app

# Common fixes:
# 1. Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up

# 2. Check port conflicts
sudo lsof -i :8000
sudo lsof -i :5432
sudo lsof -i :6379

# 3. Reset Docker
docker system prune -a
```

**Permission Denied Errors**

```bash
# Fix Docker socket permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix volume permissions
sudo chown -R 1000:1000 ./data ./logs
```

## Installation Problems

### TA-Lib Installation Failed

**Error**: `error: Microsoft Visual C++ 14.0 is required` (Windows)

**Solution**:
```bash
# Windows: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Linux/Mac: Install from source
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
```

**Error**: `ImportError: libta_lib.so.0: cannot open shared object file`

**Solution**:
```bash
# Update library path
sudo ldconfig
echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### pip Install Failures

**Error**: `No matching distribution found for package-name`

**Solutions**:
```bash
# Upgrade pip
pip install --upgrade pip

# Use specific index
pip install -r requirements.txt --index-url https://pypi.org/simple/

# Install with verbose output
pip install -r requirements.txt -v
```

## Trading Issues

### Zerodha Kite Connection Failed

**Error**: `KiteException: Invalid API key or access token`

**Solutions**:

1. **Regenerate Access Token**
   ```python
   from services.kite import generate_access_token
   
   # Get login URL
   login_url = generate_login_url()
   print(f"Visit: {login_url}")
   
   # After login, get request_token from URL
   access_token = generate_access_token(request_token)
   ```

2. **Check API Credentials**
   ```bash
   # Verify in .env
   grep "KITE_" .env
   
   # Test connection
   python scripts/test_kite_connection.py
   ```

3. **Common Kite Issues**
   - API key expired: Regenerate from Kite dashboard
   - Access token expired: Re-login to generate new token
   - Rate limit exceeded: Implement delays between requests

### Orders Not Executing

**Problem**: Orders placed but not executed

**Checklist**:
1. Market hours (9:15 AM - 3:30 PM IST)
2. Trading holiday check
3. Sufficient funds/margin
4. Symbol halted/suspended
5. Order validation errors

**Debug Steps**:
```python
# Check order status
order_status = await kite_client.get_order_status(order_id)
print(f"Status: {order_status}")

# Check rejection reason
if order_status['status'] == 'REJECTED':
    print(f"Rejection reason: {order_status['status_message']}")
```

### Position Size Errors

**Error**: `Risk limit exceeded` or `Position size too large`

**Solutions**:
```bash
# Check risk parameters
grep -E "MAX_POSITION_SIZE|RISK_PER_TRADE" .env

# Adjust in config/trading.yaml
max_position_size: 50000  # Reduce position size
risk_per_trade: 1.0       # Reduce risk percentage
```

## Agent-Related Issues

### Agent Not Responding

**Symptoms**: Agent analysis timing out or returning empty results

**Debug Steps**:

1. **Check Agent Status**
   ```bash
   curl http://localhost:8000/api/v1/agents/status
   ```

2. **View Agent Logs**
   ```bash
   # Check specific agent logs
   grep "market_analyst" logs/algohive.log | tail -50
   ```

3. **Test Individual Agent**
   ```python
   from agents.market_analyst import MarketAnalystAgent
   
   agent = MarketAnalystAgent()
   result = await agent.analyze("RELIANCE")
   print(result)
   ```

### AI Service Errors

**Error**: `OpenAI API error: Rate limit exceeded`

**Solutions**:
```python
# Implement rate limiting
from services.ai_integration.rate_limiter import RateLimiter

rate_limiter = RateLimiter(
    requests_per_minute=20,
    requests_per_day=1000
)

# Use with backoff
@rate_limiter.limit
@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_ai_service():
    # Your AI call here
```

**Error**: `Anthropic API error: Invalid API key`

**Solution**:
```bash
# Verify API key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"
```

### Agent Coordination Issues

**Problem**: Agents providing conflicting signals

**Solutions**:

1. **Adjust Confidence Thresholds**
   ```yaml
   # config/agents.yaml
   coordinator:
     min_consensus_confidence: 0.6  # Lower threshold
     conflict_resolution: "weighted_average"  # or "majority_vote"
   ```

2. **Check Agent Weights**
   ```python
   # Adjust agent importance
   agent_weights = {
       "market_analyst": 0.3,
       "technical_indicator": 0.3,
       "sentiment_analyst": 0.2,
       "risk_manager": 0.2
   }
   ```

## Performance Issues

### Slow API Response

**Symptoms**: API calls taking > 1 second

**Optimization Steps**:

1. **Enable Caching**
   ```python
   from fastapi_cache.decorator import cache
   
   @cache(expire=60)  # Cache for 60 seconds
   async def get_market_data(symbol: str):
       # Expensive operation
   ```

2. **Database Query Optimization**
   ```sql
   -- Add indexes
   CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, timestamp);
   CREATE INDEX idx_orders_status ON orders(status);
   
   -- Analyze query performance
   EXPLAIN ANALYZE SELECT * FROM trades WHERE symbol = 'RELIANCE';
   ```

3. **Use Connection Pooling**
   ```python
   # In config.py
   SQLALCHEMY_POOL_SIZE = 20
   SQLALCHEMY_POOL_RECYCLE = 3600
   SQLALCHEMY_POOL_PRE_PING = True
   ```

### High Memory Usage

**Symptoms**: Application consuming > 2GB RAM

**Solutions**:

1. **Memory Profiling**
   ```python
   from memory_profiler import profile
   
   @profile
   def memory_intensive_function():
       # Your code
   ```

2. **Limit DataFrame Size**
   ```python
   # Use chunking for large datasets
   for chunk in pd.read_csv('large_file.csv', chunksize=1000):
       process_chunk(chunk)
   ```

3. **Clear Caches Periodically**
   ```python
   import gc
   
   # Force garbage collection
   gc.collect()
   
   # Clear agent memory
   await agent.clear_memory()
   ```

### WebSocket Disconnections

**Problem**: Frequent WebSocket disconnections

**Solutions**:

1. **Implement Reconnection Logic**
   ```python
   class WebSocketClient:
       async def connect_with_retry(self):
           while True:
               try:
                   await self.connect()
                   await self.subscribe(self.symbols)
                   break
               except Exception as e:
                   logger.error(f"WebSocket error: {e}")
                   await asyncio.sleep(5)
   ```

2. **Heartbeat Implementation**
   ```python
   async def heartbeat(self):
       while self.connected:
           await self.send_ping()
           await asyncio.sleep(30)
   ```

## API & Connectivity

### Rate Limiting Issues

**Error**: `429 Too Many Requests`

**Solutions**:

1. **Implement Request Queue**
   ```python
   from asyncio import Queue, sleep
   
   class RequestQueue:
       def __init__(self, rate_limit=10):
           self.queue = Queue()
           self.rate_limit = rate_limit
           
       async def process(self):
           while True:
               request = await self.queue.get()
               await request()
               await sleep(1 / self.rate_limit)
   ```

2. **Use Batch Requests**
   ```python
   # Instead of multiple calls
   symbols = ["RELIANCE", "TCS", "INFY"]
   quotes = await kite.get_quotes(symbols)  # Single batch call
   ```

### SSL Certificate Errors

**Error**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solutions**:
```bash
# Update certificates
pip install --upgrade certifi

# For development only (not recommended for production)
export PYTHONHTTPSVERIFY=0

# Or in code (not recommended)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

## Database Issues

### Migration Failures

**Error**: `alembic.util.exc.CommandError: Target database is not up to date`

**Solutions**:
```bash
# Check current revision
alembic current

# Show migration history
alembic history

# Force upgrade
alembic stamp head
alembic upgrade head

# Downgrade if needed
alembic downgrade -1
```

### Connection Pool Exhausted

**Error**: `TimeoutError: QueuePool limit of size 5 overflow 10 reached`

**Solutions**:
```python
# Increase pool size in config
SQLALCHEMY_POOL_SIZE = 20
SQLALCHEMY_MAX_OVERFLOW = 40

# Or in code
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## Frequently Asked Questions

### General Questions

**Q: Can I run AlgoHive on Windows?**
A: Yes, but we recommend using WSL2 (Windows Subsystem for Linux) for better compatibility. Native Windows is supported but may require additional setup.

**Q: What are the minimum system requirements?**
A: 
- CPU: 4 cores
- RAM: 8GB (16GB recommended)
- Storage: 50GB SSD
- Network: Stable internet connection

**Q: Is paper trading available?**
A: Yes, enable paper trading mode in configuration:
```bash
python -m algohive start --mode paper --capital 1000000
```

### Trading Questions

**Q: What's the minimum capital required?**
A: While there's no hardcoded minimum, we recommend at least ₹100,000 for effective position sizing and risk management.

**Q: Can I trade in multiple accounts?**
A: Currently, AlgoHive supports one Zerodha account per instance. For multiple accounts, run separate instances with different configurations.

**Q: How do I add custom indicators?**
A: Create a new indicator in `agents/technical_indicator/custom_indicators.py`:
```python
def custom_indicator(data: pd.DataFrame) -> pd.Series:
    # Your indicator logic
    return result
```

### Agent Questions

**Q: Can I disable specific agents?**
A: Yes, in `config/agents.yaml`:
```yaml
agents:
  sentiment_analyst:
    enabled: false
```

**Q: How do I adjust agent sensitivity?**
A: Modify confidence thresholds:
```yaml
market_analyst:
  confidence_threshold: 0.8  # More conservative
```

**Q: Can I add custom agents?**
A: Yes, create a new agent following the template:
```python
from crewai import Agent

class CustomAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Custom Analyst",
            goal="Your goal here",
            backstory="Agent description"
        )
```

### Performance Questions

**Q: How many symbols can I track simultaneously?**
A: Recommended maximum is 20 symbols for optimal performance. The system can handle more but may experience delays.

**Q: What's the expected latency?**
A: 
- API response: < 100ms
- Agent analysis: < 2 seconds
- Order execution: < 500ms

**Q: How much bandwidth does AlgoHive use?**
A: Approximately 10-50 MB/hour depending on the number of symbols and market activity.

## Debug Mode

### Enable Comprehensive Debugging

1. **Set Environment Variables**
   ```bash
   export DEBUG=True
   export LOG_LEVEL=DEBUG
   export PYTHONUNBUFFERED=1
   ```

2. **Enable SQL Query Logging**
   ```python
   # In config.py
   SQLALCHEMY_ECHO = True
   ```

3. **Enable Agent Debug Mode**
   ```yaml
   # config/agents.yaml
   debug:
     enabled: true
     save_analysis_history: true
     verbose_logging: true
   ```

4. **Use Debug Endpoints**
   ```bash
   # Get system diagnostics
   curl http://localhost:8000/api/v1/debug/diagnostics
   
   # Get memory usage
   curl http://localhost:8000/api/v1/debug/memory
   
   # Get performance metrics
   curl http://localhost:8000/api/v1/debug/performance
   ```

### Debug Scripts

```bash
# Test database connection
python scripts/debug/test_db.py

# Test Kite API
python scripts/debug/test_kite.py

# Test agents
python scripts/debug/test_agents.py

# Check system health
python scripts/debug/health_check.py
```

## Getting Help

### 1. Check Logs
```bash
# Application logs
tail -f logs/algohive.log

# Error logs
grep ERROR logs/algohive.log

# Agent-specific logs
grep "agent_name" logs/algohive.log
```

### 2. Community Support
- GitHub Issues: [github.com/algohive/algohive/issues](https://github.com/algohive/algohive/issues)
- Discord: [discord.gg/algohive](https://discord.gg/algohive)
- Email: support@algohive.com

### 3. Before Reporting an Issue

Gather the following information:
1. AlgoHive version: `python -m algohive --version`
2. Python version: `python --version`
3. Operating system: `uname -a`
4. Error messages and stack traces
5. Steps to reproduce the issue
6. Relevant configuration files (remove sensitive data)

### 4. Emergency Contacts

For critical production issues:
- Emergency Support: emergency@algohive.com
- Phone: +91-XXXXXXXXXX (Business hours only)

Remember to always test changes in a development environment before applying to production!