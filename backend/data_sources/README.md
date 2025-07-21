# Multi-Source Data Manager

## Overview

The Multi-Source Data Manager is a robust, fault-tolerant system for managing multiple data sources with automatic failover capabilities. It ensures continuous data availability for the AlgoHive trading platform by intelligently switching between primary and backup data sources when failures occur.

## Features

- **Automatic Failover**: Seamlessly switches to backup sources when primary fails
- **Health Monitoring**: Continuous health checks on all configured sources
- **Rate Limiting**: Built-in rate limiting to respect API limits
- **Connection Pooling**: Efficient resource management with connection pools
- **Priority Management**: Configure source priorities for failover order
- **Performance Metrics**: Track latency, success rates, and error counts
- **Extensible Architecture**: Easy to add new data sources

## Architecture

```
MultiSourceDataManager
├── MarketDataSource (Base)
│   ├── ZerodhaMarketDataSource (Primary)
│   ├── AlphaVantageMarketDataSource (Backup)
│   └── YahooFinanceMarketDataSource (Backup)
├── SentimentDataSource (Base)
│   ├── NewsAPISource
│   └── TwitterSentimentSource
└── Health Monitoring & Failover Logic
```

## Configuration

Add the following to your `.env` file:

```env
# Multi-Source Configuration
DATA_SOURCE_FAILOVER_ENABLED=true
DATA_SOURCE_HEALTH_CHECK_INTERVAL=30
DATA_SOURCE_TIMEOUT=10
DATA_SOURCE_MAX_RETRIES=3

# Zerodha (Primary)
KITE_API_KEY=your_key
KITE_API_SECRET=your_secret
KITE_ACCESS_TOKEN=your_token
ZERODHA_RATE_LIMIT=180

# Alpha Vantage (Backup)
ALPHA_VANTAGE_API_KEY=your_key
ALPHA_VANTAGE_ENABLED=true
ALPHA_VANTAGE_PRIORITY=2
ALPHA_VANTAGE_RATE_LIMIT=5

# Yahoo Finance (Backup)
YAHOO_FINANCE_ENABLED=true
YAHOO_FINANCE_PRIORITY=3
YAHOO_FINANCE_RATE_LIMIT=100
```

## Usage

### Basic Usage

```python
from backend.data_sources.integration import get_data_source_integration

# Get the integration instance
integration = get_data_source_integration()

# Initialize
await integration.initialize()

# Get market quote with automatic failover
quote = await integration.get_quote("RELIANCE")

# Get historical data
historical = await integration.get_historical_data(
    "RELIANCE",
    "day",
    from_date,
    to_date
)
```

### Direct Manager Usage

```python
from backend.data_sources import MultiSourceDataManager
from backend.data_sources.adapters import ZerodhaMarketDataSource

# Create manager
manager = MultiSourceDataManager()

# Add sources
zerodha_config = DataSourceConfig(
    name="zerodha",
    priority=1,
    rate_limit=180
)
zerodha = ZerodhaMarketDataSource(zerodha_config)
manager.add_source(zerodha)

# Start manager
await manager.start()

# Execute with failover
data = await manager.execute_with_failover(
    DataSourceType.MARKET_DATA,
    'get_quote',
    'RELIANCE'
)
```

## Creating New Data Sources

To add a new data source, extend the appropriate base class:

```python
from backend.data_sources.base import MarketDataSource

class MyCustomDataSource(MarketDataSource):
    async def connect(self) -> bool:
        # Implementation
        pass
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        # Implementation
        pass
    
    # Implement other required methods...
```

## API Endpoints

The following endpoints are available for monitoring:

- `GET /api/v1/data-sources/health` - Get health status of all sources
- `POST /api/v1/data-sources/health-check` - Force health check
- `GET /api/v1/data-sources/active/{source_type}` - Get active source
- `GET /api/v1/data-sources/metrics` - Get performance metrics
- `POST /api/v1/data-sources/failover-test` - Test failover mechanism

## Health Monitoring

Each data source maintains health metrics:

```json
{
  "status": "healthy",
  "last_check": "2024-01-15T10:30:00Z",
  "latency_ms": 45.2,
  "success_rate": 99.5,
  "error_count": 2,
  "recent_errors": []
}
```

Status values:
- `healthy` - Source is operating normally
- `degraded` - Source is experiencing issues but still functional
- `unhealthy` - Source is not functioning properly
- `disconnected` - Source is not connected

## Failover Logic

1. Sources are prioritized (lower number = higher priority)
2. Manager always tries the highest priority healthy source first
3. On failure, automatically tries next healthy source
4. Failed sources are monitored and re-enabled when healthy
5. Configurable retry logic with exponential backoff

## Performance Considerations

- Connection pools minimize connection overhead
- Rate limiting prevents API quota exhaustion
- Caching can be implemented at the source level
- Health checks run asynchronously to avoid blocking
- Metrics help identify performance bottlenecks

## Troubleshooting

### Source Not Connecting

1. Check API credentials in configuration
2. Verify network connectivity
3. Check source-specific requirements (API limits, etc.)
4. Review logs for detailed error messages

### Failover Not Working

1. Ensure multiple sources are configured and enabled
2. Check that backup sources are healthy
3. Verify failover is enabled in configuration
4. Test with failover endpoint

### High Latency

1. Check individual source metrics
2. Consider adjusting timeout settings
3. Verify network conditions
4. Review rate limiting configuration

## Best Practices

1. **Always configure at least one backup source**
2. **Set appropriate timeouts based on source characteristics**
3. **Monitor health metrics regularly**
4. **Test failover scenarios periodically**
5. **Keep API credentials secure and rotate regularly**
6. **Implement source-specific error handling**
7. **Use appropriate rate limits for each source**

## Future Enhancements

- [ ] Implement caching layer
- [ ] Add more data sources (NSE, BSE APIs)
- [ ] WebSocket support for all sources
- [ ] Machine learning for predictive failover
- [ ] Geographic failover support
- [ ] Data quality validation
- [ ] Automatic source discovery