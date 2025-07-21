# Shagun Intelligence API Documentation

## Base URL
```
Production: https://api.shagunintelligence.com
Development: http://localhost:8000
```

## Authentication

Shagun Intelligence uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Obtain Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## API Endpoints

### Health Check

```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "redis": "connected",
    "kite": "connected"
  }
}
```

### Trading Operations

#### Start Trading System

```http
POST /api/v1/trading/start
Authorization: Bearer <token>
Content-Type: application/json

{
  "mode": "live",  // "live" or "paper"
  "symbols": ["RELIANCE", "TCS", "INFY"],
  "capital": 1000000,
  "risk_parameters": {
    "max_position_size": 100000,
    "max_daily_loss": 50000,
    "risk_per_trade": 2.0
  }
}
```

**Response:**
```json
{
  "status": "started",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "mode": "live",
  "active_symbols": ["RELIANCE", "TCS", "INFY"]
}
```

#### Stop Trading System

```http
POST /api/v1/trading/stop
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "stopped",
  "session_summary": {
    "total_trades": 15,
    "profitable_trades": 10,
    "total_pnl": 25000,
    "duration": "6h 30m"
  }
}
```

#### Place Order

```http
POST /api/v1/trading/order
Authorization: Bearer <token>
Content-Type: application/json

{
  "symbol": "RELIANCE",
  "exchange": "NSE",
  "order_type": "BUY",
  "quantity": 10,
  "product_type": "MIS",
  "price_type": "MARKET",
  "price": null,
  "trigger_price": null,
  "stop_loss": 2450.00,
  "take_profit": 2600.00,
  "use_ai_analysis": true
}
```

**Response:**
```json
{
  "order_id": "240115000012345",
  "status": "COMPLETE",
  "message": "Order placed successfully",
  "trade_id": 12345,
  "ai_confidence": 0.85,
  "execution_price": 2500.50,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Get Positions

```http
GET /api/v1/trading/positions
Authorization: Bearer <token>
```

**Response:**
```json
{
  "net_positions": [
    {
      "symbol": "RELIANCE",
      "quantity": 10,
      "average_price": 2500.50,
      "last_price": 2520.00,
      "pnl": 195.00,
      "pnl_percentage": 0.78,
      "value": 25200.00,
      "product": "MIS"
    }
  ],
  "day_positions": [],
  "total_pnl": 195.00,
  "total_value": 25200.00
}
```

#### Get Orders

```http
GET /api/v1/trading/orders?status=COMPLETE&symbol=RELIANCE
Authorization: Bearer <token>
```

**Query Parameters:**
- `status`: Filter by order status (COMPLETE, PENDING, CANCELLED, REJECTED)
- `symbol`: Filter by symbol
- `date`: Filter by date (YYYY-MM-DD)
- `limit`: Number of records (default: 100)

**Response:**
```json
{
  "orders": [
    {
      "order_id": "240115000012345",
      "symbol": "RELIANCE",
      "status": "COMPLETE",
      "order_type": "BUY",
      "quantity": 10,
      "price": 2500.50,
      "order_timestamp": "2024-01-15T10:30:00Z",
      "fill_timestamp": "2024-01-15T10:30:05Z"
    }
  ],
  "total_count": 25,
  "page": 1
}
```

### Market Data

#### Get Quote

```http
GET /api/v1/market/quote/{symbol}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "symbol": "RELIANCE",
  "last_price": 2520.50,
  "change": 20.50,
  "change_percent": 0.82,
  "volume": 1234567,
  "bid": 2520.25,
  "ask": 2520.75,
  "open": 2500.00,
  "high": 2530.00,
  "low": 2495.00,
  "close": 2500.00,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Get Multiple Quotes

```http
POST /api/v1/market/quotes
Authorization: Bearer <token>
Content-Type: application/json

{
  "symbols": ["RELIANCE", "TCS", "INFY", "HDFC"]
}
```

**Response:**
```json
{
  "quotes": {
    "RELIANCE": {
      "last_price": 2520.50,
      "change_percent": 0.82
    },
    "TCS": {
      "last_price": 3500.00,
      "change_percent": -0.50
    }
  }
}
```

#### Get Historical Data

```http
GET /api/v1/market/historical/{symbol}?interval=day&from=2024-01-01&to=2024-01-15
Authorization: Bearer <token>
```

**Query Parameters:**
- `interval`: minute, 5minute, 15minute, hour, day
- `from`: Start date (YYYY-MM-DD)
- `to`: End date (YYYY-MM-DD)

**Response:**
```json
{
  "symbol": "RELIANCE",
  "interval": "day",
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "open": 2480.00,
      "high": 2510.00,
      "low": 2475.00,
      "close": 2500.00,
      "volume": 1234567
    }
  ]
}
```

#### Get Market Depth

```http
GET /api/v1/market/depth/{symbol}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "symbol": "RELIANCE",
  "buy": [
    {"price": 2520.00, "quantity": 500, "orders": 5},
    {"price": 2519.75, "quantity": 1000, "orders": 8}
  ],
  "sell": [
    {"price": 2520.50, "quantity": 750, "orders": 6},
    {"price": 2520.75, "quantity": 1200, "orders": 10}
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Agent Operations

#### Get All Agents Status

```http
GET /api/v1/agents/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "agents": [
    {
      "name": "market_analyst",
      "status": "active",
      "last_execution": "2024-01-15T10:29:00Z",
      "execution_count": 1234,
      "success_rate": 0.95,
      "average_execution_time": 1.2
    },
    {
      "name": "risk_manager",
      "status": "active",
      "last_execution": "2024-01-15T10:29:30Z",
      "execution_count": 1000,
      "success_rate": 0.98,
      "average_execution_time": 0.8
    }
  ]
}
```

#### Get Agent Analysis

```http
GET /api/v1/agents/{agent_name}/analysis/{symbol}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "agent": "market_analyst",
  "symbol": "RELIANCE",
  "analysis": {
    "trend": "bullish",
    "confidence": 0.85,
    "support_levels": [2480, 2450, 2400],
    "resistance_levels": [2550, 2600, 2650],
    "indicators": {
      "rsi": 65,
      "macd": "bullish_crossover",
      "moving_averages": {
        "sma_20": 2490,
        "sma_50": 2470,
        "ema_20": 2495
      }
    },
    "recommendation": "buy",
    "reasoning": "Strong uptrend with RSI showing momentum..."
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Trigger Manual Analysis

```http
POST /api/v1/agents/analyze
Authorization: Bearer <token>
Content-Type: application/json

{
  "symbol": "RELIANCE",
  "agents": ["market_analyst", "technical_indicator", "sentiment_analyst"],
  "timeframe": "5minute",
  "deep_analysis": true
}
```

**Response:**
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "results": {
    "market_analyst": {
      "signal": "buy",
      "confidence": 0.85
    },
    "technical_indicator": {
      "signal": "buy",
      "confidence": 0.90
    },
    "sentiment_analyst": {
      "signal": "neutral",
      "confidence": 0.70
    }
  },
  "consensus": {
    "action": "buy",
    "confidence": 0.82,
    "entry_price": 2520.00,
    "stop_loss": 2480.00,
    "take_profit": 2600.00
  }
}
```

### Portfolio Management

#### Get Portfolio Summary

```http
GET /api/v1/portfolio/summary
Authorization: Bearer <token>
```

**Response:**
```json
{
  "total_value": 1025000,
  "available_cash": 775000,
  "invested_amount": 250000,
  "total_pnl": 25000,
  "total_pnl_percentage": 2.5,
  "today_pnl": 5000,
  "open_positions": 3,
  "realized_pnl": 15000,
  "unrealized_pnl": 10000
}
```

#### Get Performance Metrics

```http
GET /api/v1/portfolio/performance?period=month
Authorization: Bearer <token>
```

**Query Parameters:**
- `period`: day, week, month, year, all

**Response:**
```json
{
  "period": "month",
  "metrics": {
    "total_return": 5.2,
    "sharpe_ratio": 1.8,
    "max_drawdown": 3.5,
    "win_rate": 0.65,
    "profit_factor": 1.8,
    "average_win": 2500,
    "average_loss": 1200,
    "total_trades": 150,
    "winning_trades": 98,
    "losing_trades": 52
  },
  "daily_returns": [
    {"date": "2024-01-01", "return": 0.5},
    {"date": "2024-01-02", "return": -0.2}
  ]
}
```

#### Get Trade History

```http
GET /api/v1/portfolio/trades?from=2024-01-01&to=2024-01-15&symbol=RELIANCE
Authorization: Bearer <token>
```

**Response:**
```json
{
  "trades": [
    {
      "trade_id": 12345,
      "symbol": "RELIANCE",
      "entry_time": "2024-01-15T10:30:00Z",
      "exit_time": "2024-01-15T14:30:00Z",
      "entry_price": 2500.00,
      "exit_price": 2520.00,
      "quantity": 10,
      "pnl": 200.00,
      "pnl_percentage": 0.8,
      "holding_period": "4h",
      "trade_type": "long",
      "exit_reason": "take_profit"
    }
  ],
  "summary": {
    "total_trades": 25,
    "profitable_trades": 18,
    "total_pnl": 15000
  }
}
```

### System Configuration

#### Get Configuration

```http
GET /api/v1/system/config
Authorization: Bearer <token>
```

**Response:**
```json
{
  "trading": {
    "max_position_size": 100000,
    "max_daily_trades": 10,
    "risk_per_trade": 2.0,
    "stop_loss_percent": 2.0,
    "take_profit_percent": 4.0
  },
  "agents": {
    "market_analyst": {
      "enabled": true,
      "confidence_threshold": 0.7
    },
    "risk_manager": {
      "enabled": true,
      "max_portfolio_risk": 0.2
    }
  },
  "market_hours": {
    "start": "09:15",
    "end": "15:30",
    "timezone": "Asia/Kolkata"
  }
}
```

#### Update Configuration

```http
PUT /api/v1/system/config
Authorization: Bearer <token>
Content-Type: application/json

{
  "trading": {
    "max_position_size": 150000,
    "risk_per_trade": 1.5
  }
}
```

**Response:**
```json
{
  "status": "updated",
  "message": "Configuration updated successfully"
}
```

### WebSocket Endpoints

#### Market Data Stream

```javascript
const ws = new WebSocket('wss://api.shagunintelligence.com/ws/market');

ws.onopen = () => {
  // Subscribe to symbols
  ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['RELIANCE', 'TCS', 'INFY']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Tick:', data);
  // {
  //   "symbol": "RELIANCE",
  //   "ltp": 2520.50,
  //   "volume": 1234567,
  //   "bid": 2520.25,
  //   "ask": 2520.75,
  //   "timestamp": "2024-01-15T10:30:00.123Z"
  // }
};
```

#### Trading Signals Stream

```javascript
const ws = new WebSocket('wss://api.shagunintelligence.com/ws/signals');

ws.onmessage = (event) => {
  const signal = JSON.parse(event.data);
  console.log('Trading Signal:', signal);
  // {
  //   "type": "trade_signal",
  //   "symbol": "RELIANCE",
  //   "action": "buy",
  //   "confidence": 0.85,
  //   "entry_price": 2520.00,
  //   "stop_loss": 2480.00,
  //   "take_profit": 2600.00,
  //   "agents_consensus": {...},
  //   "timestamp": "2024-01-15T10:30:00Z"
  // }
};
```

## Error Responses

All error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Symbol INVALID not found",
    "details": {
      "field": "symbol",
      "value": "INVALID"
    }
  },
  "status_code": 400,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| UNAUTHORIZED | 401 | Invalid or missing authentication token |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| VALIDATION_ERROR | 400 | Invalid request parameters |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Internal server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

## Rate Limiting

API rate limits:
- **Authentication endpoints**: 5 requests per minute
- **Trading endpoints**: 100 requests per minute
- **Market data endpoints**: 300 requests per minute
- **WebSocket connections**: 10 concurrent connections

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705318800
```

## SDK Examples

### Python

```python
import requests

class Shagun IntelligenceClient:
    def __init__(self, api_key):
        self.base_url = "https://api.shagunintelligence.com"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def get_quote(self, symbol):
        response = requests.get(
            f"{self.base_url}/api/v1/market/quote/{symbol}",
            headers=self.headers
        )
        return response.json()
    
    def place_order(self, order_data):
        response = requests.post(
            f"{self.base_url}/api/v1/trading/order",
            json=order_data,
            headers=self.headers
        )
        return response.json()

# Usage
client = Shagun IntelligenceClient("your-api-key")
quote = client.get_quote("RELIANCE")
print(f"RELIANCE: {quote['last_price']}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

class Shagun IntelligenceClient {
  constructor(apiKey) {
    this.client = axios.create({
      baseURL: 'https://api.shagunintelligence.com',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });
  }
  
  async getQuote(symbol) {
    const response = await this.client.get(`/api/v1/market/quote/${symbol}`);
    return response.data;
  }
  
  async placeOrder(orderData) {
    const response = await this.client.post('/api/v1/trading/order', orderData);
    return response.data;
  }
}

// Usage
const client = new Shagun IntelligenceClient('your-api-key');
const quote = await client.getQuote('RELIANCE');
console.log(`RELIANCE: ${quote.last_price}`);
```

## Webhooks

Configure webhooks to receive real-time notifications:

```http
POST /api/v1/webhooks
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["order_complete", "position_closed", "signal_generated"],
  "secret": "your-webhook-secret"
}
```

Webhook payload example:
```json
{
  "event": "order_complete",
  "data": {
    "order_id": "240115000012345",
    "symbol": "RELIANCE",
    "status": "COMPLETE",
    "execution_price": 2520.50
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "signature": "sha256=..."
}
```

## Best Practices

1. **Authentication**
   - Store API keys securely
   - Rotate tokens regularly
   - Use environment variables

2. **Error Handling**
   - Implement exponential backoff for retries
   - Log all errors for debugging
   - Handle rate limits gracefully

3. **Performance**
   - Use WebSocket for real-time data
   - Batch API requests when possible
   - Cache frequently accessed data

4. **Security**
   - Always use HTTPS
   - Validate webhook signatures
   - Implement IP whitelisting for production