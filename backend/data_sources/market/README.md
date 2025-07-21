# Market Data Sources

## Overview

The market data sources module provides a robust, failover-enabled system for fetching real-time and historical market data from multiple providers. It includes intelligent source selection, cost optimization, and seamless failover mechanisms.

## Supported Data Sources

### 1. GlobalDatafeeds (NSE/BSE Focus)
- **Markets**: NSE, BSE, NFO, MCX
- **Cost**: ₹299/month (Professional plan)
- **Features**:
  - Ultra-low latency (10ms) for Indian markets
  - Real-time WebSocket streaming
  - Full market depth (order book)
  - Authorized NSE/BSE data provider
- **Best For**: Indian market trading with low-latency requirements

### 2. Alpha Vantage (Global Markets)
- **Markets**: US, NYSE, NASDAQ, FOREX, CRYPTO, TSX, LSE
- **Cost**: Free tier (500 requests/day) or $49.99/month (Premium)
- **Features**:
  - Technical indicators (SMA, EMA, RSI, MACD, etc.)
  - Forex and cryptocurrency data
  - Historical data with adjustments
  - Global market coverage
- **Best For**: Technical analysis and global market data

### 3. Finnhub (Real-time + Sentiment)
- **Markets**: US, EU, UK, JP, FOREX, CRYPTO
- **Cost**: Free tier (60 req/min) or $49-999/month
- **Features**:
  - Built-in sentiment analysis
  - Company news with sentiment scores
  - Economic calendar
  - Real-time WebSocket streaming
- **Best For**: Sentiment-driven trading strategies

### 4. Polygon.io (Institutional Grade)
- **Markets**: US, NYSE, NASDAQ, OPTIONS, FOREX, CRYPTO
- **Cost**: Free (delayed) or $199-1999/month
- **Features**:
  - Institutional-grade data quality
  - Ultra-low latency (5ms)
  - Options chain data
  - Market-wide snapshots
  - Comprehensive historical data
- **Best For**: Professional traders requiring institutional data

## Usage

### Basic Setup

```python
from backend.data_sources.integration import get_data_source_integration

# Initialize the integration
integration = get_data_source_integration()
await integration.initialize()

# Get a quote
quote = await integration.get_quote("RELIANCE.NSE")
print(f"Current Price: {quote['current_price']}")
print(f"Data Source: {quote['data_source']}")
```

### Market Source Manager

```python
from backend.data_sources.market import MarketSourceManager, SourceSelectionStrategy

# Create manager with specific strategy
manager = MarketSourceManager(SourceSelectionStrategy.BALANCED)

# Initialize with credentials
configs = {
    "global_datafeeds": DataSourceConfig(
        name="global_datafeeds",
        credentials={"api_key": "YOUR_KEY", "user_id": "YOUR_ID"}
    ),
    "polygon": DataSourceConfig(
        name="polygon",
        credentials={"api_key": "YOUR_POLYGON_KEY"}
    )
}
await manager.initialize(configs)

# Fetch quotes with automatic source selection
quotes = await manager.get_quotes(["RELIANCE.NSE", "AAPL", "EURUSD"])
```

### Source Selection Strategies

1. **COST_OPTIMIZED**: Minimizes costs by preferring free/cheaper sources
2. **QUALITY_FIRST**: Prioritizes data quality regardless of cost
3. **LATENCY_OPTIMIZED**: Selects sources with lowest latency
4. **BALANCED**: Weighs quality (40%), latency (30%), and cost (30%)

```python
# Change strategy at runtime
manager.set_strategy(SourceSelectionStrategy.LATENCY_OPTIMIZED)
```

### Streaming Real-time Data

```python
# Define callback for real-time updates
async def on_quote_update(symbol, data):
    print(f"{symbol}: {data['current_price']} from {data['data_source']}")

# Start streaming
task = await integration.stream_market_data(
    ["RELIANCE.NSE", "AAPL", "EURUSD"],
    on_quote_update
)
```

### Advanced Features

#### Technical Indicators (Alpha Vantage)
```python
# Get RSI indicator
rsi = await integration.get_technical_indicators(
    "AAPL",
    "RSI",
    time_period=14,
    interval="daily"
)
```

#### Market Sentiment (Finnhub)
```python
# Get sentiment analysis
sentiment = await integration.get_market_sentiment("AAPL")
print(f"Sentiment Score: {sentiment['score']}")
print(f"Buzz Score: {sentiment['buzz']}")
```

#### Options Chain (Polygon)
```python
# Get options chain
if "polygon" in manager.sources:
    source = manager.sources["polygon"]
    chain = await source.get_options_chain("AAPL", "2024-01-19")
```

## Configuration

Set environment variables for each data source:

```bash
# GlobalDatafeeds (Indian Markets)
GLOBAL_DATAFEEDS_API_KEY=your_key
GLOBAL_DATAFEEDS_USER_ID=your_user_id

# Alpha Vantage (Free/Premium)
ALPHA_VANTAGE_API_KEY=your_av_key

# Finnhub (Sentiment + Data)
FINNHUB_API_KEY=your_finnhub_key

# Polygon (Institutional)
POLYGON_API_KEY=your_polygon_key
```

## Failover Mechanism

The system automatically handles failures:

1. **Health Monitoring**: Continuous health checks on all sources
2. **Circuit Breaker**: Sources are temporarily disabled after 5 failures
3. **Automatic Retry**: Failed sources are retried after 5-minute cooldown
4. **Seamless Failover**: Requests automatically route to next best source

## Cost Management

Monitor and control costs:

```python
# Get current usage and costs
status = manager.get_source_status()
print(f"Total Monthly Cost: ${status['total_monthly_cost']}")
print(f"Total Requests: {status['total_requests']}")

# View per-source breakdown
for source, info in status['sources'].items():
    print(f"{source}: ${info['monthly_cost']} ({info['request_count']} requests)")
```

## Market Coverage Matrix

| Source | NSE/BSE | US Markets | Forex | Crypto | Options | EU/UK/JP |
|--------|---------|------------|-------|---------|---------|-----------|
| GlobalDatafeeds | ✅ | ❌ | ❌ | ❌ | ✅ (NFO) | ❌ |
| Alpha Vantage | ❌ | ✅ | ✅ | ✅ | ❌ | Partial |
| Finnhub | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Polygon | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |

## Feature Comparison

| Feature | GlobalDatafeeds | Alpha Vantage | Finnhub | Polygon |
|---------|----------------|---------------|---------|----------|
| Real-time Data | ✅ | ✅ | ✅ | ✅ |
| Historical Data | ✅ | ✅ | ✅ | ✅ |
| Market Depth | ✅ | ❌ | Limited | ✅ |
| WebSocket Streaming | ✅ | ❌ | ✅ | ✅ |
| Technical Indicators | ❌ | ✅ | ❌ | ❌ |
| Sentiment Analysis | ❌ | ❌ | ✅ | ❌ |
| Latency | 10ms | 1000ms | 100ms | 5ms |
| Free Tier | ❌ | ✅ | ✅ | ✅ (delayed) |

## Best Practices

1. **Use appropriate strategy**: 
   - Cost-sensitive: Use COST_OPTIMIZED
   - HFT/Scalping: Use LATENCY_OPTIMIZED
   - Long-term investing: Use BALANCED

2. **Monitor health status**:
   ```python
   health = await integration.force_health_check()
   ```

3. **Handle rate limits**: Sources automatically manage rate limits internally

4. **Cache frequently accessed data**: Reduce API calls for static data

5. **Use batch operations**: Fetch multiple symbols in one call when possible

## Error Handling

```python
try:
    quote = await manager.get_quote("INVALID_SYMBOL")
except Exception as e:
    logger.error(f"Failed to get quote: {e}")
    # System already tried failover, no data available
```

## Testing

Run the test suite:

```bash
pytest tests/test_market_sources.py -v
```

## Future Enhancements

- [ ] Add more data sources (IEX Cloud, Quandl, Yahoo Finance)
- [ ] Implement data caching layer
- [ ] Add backtesting data support
- [ ] Create data quality scoring algorithm
- [ ] Add support for futures and commodities
- [ ] Implement custom failover policies