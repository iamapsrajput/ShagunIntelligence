# Shagun Intelligence API Documentation

## Base URL

```
Production: https://api.shagunintelligence.com
Development: http://localhost:8000
```

## Authentication

All API endpoints require authentication using JWT tokens.

### Get Access Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

### Use Token in Requests

```http
Authorization: Bearer <your_jwt_token>
```

## Core Endpoints

### Health Check

```http
GET /api/v1/health
```

Returns system health status.

### Trading Endpoints

#### Get Portfolio

```http
GET /api/v1/portfolio
```

#### Place Order

```http
POST /api/v1/trading/orders
Content-Type: application/json

{
  "symbol": "RELIANCE",
  "quantity": 10,
  "order_type": "BUY",
  "price": 2500.00
}
```

#### Get Market Data

```http
GET /api/v1/market/quote/{symbol}
```

### Automated Trading

#### Start Automated Trading

```http
POST /api/v1/automated-trading/start
```

#### Stop Automated Trading

```http
POST /api/v1/automated-trading/stop
```

#### Get Trading Status

```http
GET /api/v1/automated-trading/status
```

### AI Agents

#### Get Agent Analysis

```http
GET /api/v1/agents/analysis/{symbol}
```

#### Get Agent Status

```http
GET /api/v1/agents/status
```

### System Monitoring

#### Get System Health

```http
GET /api/v1/system/health/comprehensive
```

#### Get Circuit Breaker Status

```http
GET /api/v1/system/health/circuit-breakers
```

## WebSocket Endpoints

### Real-time Market Data

```
ws://localhost:8000/ws/market-data
```

### Trading Updates

```
ws://localhost:8000/ws/trading-updates
```

### System Status

```
ws://localhost:8000/ws/system-status
```

## Error Handling

All API endpoints return standardized error responses:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": "Additional error details"
  }
}
```

## Rate Limiting

- **General API**: 100 requests per minute
- **Market Data**: 500 requests per minute
- **Trading**: 50 requests per minute

## Response Formats

All responses are in JSON format with consistent structure:

```json
{
  "success": true,
  "data": {},
  "timestamp": "2025-08-07T12:00:00Z"
}
```

## Status Codes

- **200**: Success
- **201**: Created
- **400**: Bad Request
- **401**: Unauthorized
- **403**: Forbidden
- **404**: Not Found
- **429**: Rate Limited
- **500**: Internal Server Error

For complete API documentation with all endpoints, parameters, and examples, see the interactive API documentation at `/docs` when running the server.
