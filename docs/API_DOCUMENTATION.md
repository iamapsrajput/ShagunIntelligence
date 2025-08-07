# Shagun Intelligence Trading Platform - API Documentation

## Overview

The Shagun Intelligence Trading Platform provides a comprehensive REST API for algorithmic trading, market data analysis, and portfolio management. This documentation covers all available endpoints, request/response formats, and integration examples.

## Base URL

```
https://api.shagunintelligence.com/api/v1
```

## Authentication

All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## API Endpoints Overview

### 1. Market Data (Live)

#### WebSocket Connection

```
WebSocket: /market-data/ws/{client_id}
```

Real-time market data streaming with subscription management.

**Subscription Message:**

```json
{
  "action": "subscribe",
  "symbols": ["RELIANCE", "TCS", "HDFCBANK"],
  "data_types": ["quote", "depth", "trades"]
}
```

**Response:**

```json
{
  "type": "quote",
  "symbol": "RELIANCE",
  "data": {
    "last_price": 2500.50,
    "change": 25.50,
    "change_percent": 1.03,
    "volume": 1250000,
    "bid": 2500.00,
    "ask": 2501.00,
    "timestamp": "2025-08-07T22:30:00Z"
  }
}
```

#### REST Endpoints

**Get Real-time Quote**

```
GET /market-data/quote/{symbol}
```

**Get Market Depth**

```
GET /market-data/depth/{symbol}
```

**Get Historical Data**

```
POST /market-data/historical
```

Request Body:

```json
{
  "symbol": "RELIANCE",
  "timeframe": "1d",
  "start_date": "2025-01-01T00:00:00Z",
  "end_date": "2025-08-07T00:00:00Z",
  "limit": 100
}
```

### 2. Broker Integration

#### Connect to Broker

```
POST /broker/connect
```

Request Body:

```json
{
  "provider": "zerodha_kite",
  "api_key": "your_api_key",
  "access_token": "your_access_token",
  "name": "primary_broker"
}
```

#### Place Order

```
POST /broker/orders/place
```

Request Body:

```json
{
  "symbol": "RELIANCE",
  "exchange": "NSE",
  "transaction_type": "BUY",
  "order_type": "LIMIT",
  "quantity": 100,
  "price": 2500.00,
  "validity": "DAY",
  "tag": "algo_trade_001"
}
```

Response:

```json
{
  "success": true,
  "data": {
    "order_id": "240807000001",
    "status": "OPEN",
    "message": "Order placed successfully",
    "timestamp": "2025-08-07T22:30:00Z"
  }
}
```

#### Get Positions

```
GET /broker/positions?broker_name=primary_broker
```

#### Get Holdings

```
GET /broker/holdings
```

#### Get Account Margins

```
GET /broker/margins
```

### 3. Advanced Order Management

#### Create Bracket Order

```
POST /advanced-orders/bracket
```

Request Body:

```json
{
  "symbol": "TCS",
  "quantity": 50,
  "entry_price": 3600.00,
  "target_price": 3700.00,
  "stop_loss_price": 3500.00,
  "trailing_stop_percent": 2.0
}
```

#### Create TWAP Order

```
POST /advanced-orders/twap
```

Request Body:

```json
{
  "symbol": "HDFCBANK",
  "total_quantity": 200,
  "duration_minutes": 60,
  "price_limit": 1600.00
}
```

#### Create Iceberg Order

```
POST /advanced-orders/iceberg
```

Request Body:

```json
{
  "symbol": "INFY",
  "total_quantity": 500,
  "disclosed_quantity": 50,
  "price": 2500.00
}
```

### 4. Risk Management

#### Calculate Position Size

```
POST /enhanced-risk/position-size
```

Request Body:

```json
{
  "portfolio_value": 1000000.00,
  "risk_per_trade": 0.02,
  "entry_price": 2500.00,
  "stop_loss_price": 2400.00
}
```

#### Calculate VaR

```
POST /enhanced-risk/var
```

Request Body:

```json
{
  "returns_data": [0.01, -0.02, 0.015, -0.01, 0.005],
  "confidence_level": 0.95,
  "portfolio_value": 1000000.00
}
```

#### Get Risk Metrics

```
GET /enhanced-risk/metrics?portfolio_id=main_portfolio
```

### 5. Multi-Timeframe Analysis

#### Add Market Data

```
POST /multi-timeframe/data
```

Request Body:

```json
{
  "symbol": "RELIANCE",
  "timeframe": "1d",
  "data": [
    {
      "timestamp": "2025-08-07T00:00:00Z",
      "open": 2480.00,
      "high": 2520.00,
      "low": 2475.00,
      "close": 2510.00,
      "volume": 1500000
    }
  ]
}
```

#### Calculate Technical Indicators

```
GET /multi-timeframe/indicators/{symbol}?timeframe=1d
```

#### Generate Trading Signals

```
GET /multi-timeframe/signals/{symbol}
```

### 6. Database Operations

#### Store Market Data

```
POST /database/market-data/store
```

#### Get Historical Data from Database

```
GET /database/market-data/{symbol}?timeframe=1d&limit=100
```

#### Get Order Analytics

```
GET /database/orders/analytics?start_date=2025-01-01&end_date=2025-08-07
```

#### Get Portfolio Summary

```
GET /database/portfolio/summary?broker_name=primary_broker
```

## Error Handling

All API endpoints return standardized error responses:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "The specified symbol is not valid",
    "details": {
      "symbol": "INVALID_STOCK",
      "valid_symbols": ["RELIANCE", "TCS", "HDFCBANK"]
    }
  }
}
```

### Common Error Codes

- `AUTHENTICATION_FAILED`: Invalid or expired JWT token
- `INVALID_SYMBOL`: Symbol not found or not supported
- `INSUFFICIENT_FUNDS`: Not enough margin for the order
- `ORDER_REJECTED`: Order rejected by broker
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `MARKET_CLOSED`: Market is closed for trading
- `INVALID_PARAMETERS`: Request parameters are invalid

## Rate Limits

- Market Data: 100 requests per minute
- Order Placement: 50 requests per minute
- Portfolio Data: 200 requests per minute
- Historical Data: 20 requests per minute

## WebSocket Events

### Market Data Events

```json
{
  "type": "quote",
  "symbol": "RELIANCE",
  "data": { ... }
}
```

```json
{
  "type": "depth",
  "symbol": "TCS",
  "data": {
    "bids": [{"price": 3599.50, "quantity": 100}],
    "asks": [{"price": 3600.50, "quantity": 150}]
  }
}
```

### Order Events

```json
{
  "type": "order_update",
  "order_id": "240807000001",
  "status": "COMPLETE",
  "filled_quantity": 100,
  "average_price": 2501.25
}
```

## SDK Examples

### Python SDK

```python
from shagun_trading_sdk import TradingClient

# Initialize client
client = TradingClient(
    api_key="your_api_key",
    base_url="https://api.shagunintelligence.com"
)

# Place order
order_response = await client.place_order(
    symbol="RELIANCE",
    transaction_type="BUY",
    order_type="MARKET",
    quantity=100
)

# Get real-time data
async with client.market_data_stream() as stream:
    await stream.subscribe(["RELIANCE", "TCS"])
    async for quote in stream:
        print(f"{quote.symbol}: {quote.last_price}")
```

### JavaScript SDK

```javascript
import { TradingClient } from '@shagun/trading-sdk';

const client = new TradingClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.shagunintelligence.com'
});

// Place order
const orderResponse = await client.placeOrder({
  symbol: 'RELIANCE',
  transactionType: 'BUY',
  orderType: 'MARKET',
  quantity: 100
});

// WebSocket connection
const ws = client.createWebSocket();
ws.subscribe(['RELIANCE', 'TCS'], ['quote']);
ws.on('quote', (data) => {
  console.log(`${data.symbol}: ${data.last_price}`);
});
```

## Testing

### Test Environment

Base URL: `https://api-test.shagunintelligence.com/api/v1`

Test credentials and mock data are available for development and testing purposes.

### Postman Collection

Download the complete Postman collection: [Shagun Trading API.postman_collection.json](./postman/Shagun_Trading_API.postman_collection.json)

## Support

- Documentation: <https://docs.shagunintelligence.com>
- Support Email: <support@shagunintelligence.com>
- GitHub Issues: <https://github.com/shagunintelligence/trading-platform/issues>

## Changelog

### v1.0.0 (2025-08-07)

- Initial release
- Market data integration
- Broker API support
- Advanced order management
- Risk management features
- Multi-timeframe analysis
- Database persistence
