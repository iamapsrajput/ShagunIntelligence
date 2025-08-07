# 🎯 **Trading Modes Configuration Guide**

This guide explains the different trading modes available in the Shagun Intelligence Trading System and how to configure them safely.

## 📋 **Available Trading Modes**

### 1. 📝 **Paper Trading Mode**

- **Purpose**: Simulate trading with virtual money
- **Risk**: Zero financial risk
- **Use Case**: Strategy testing, system validation, learning

### 2. 🔴 **Live Trading Mode**

- **Purpose**: Execute real trades with real money
- **Risk**: Full financial risk
- **Use Case**: Production trading after thorough testing

### 3. 🔒 **Maintenance Mode**

- **Purpose**: Disable all trading while keeping system running
- **Risk**: Zero trading risk
- **Use Case**: System updates, troubleshooting

---

## ⚙️ **Configuration Settings**

### **Environment Variables (.env file)**

```bash
# =============================================================================
# TRADING MODE CONFIGURATION
# =============================================================================

# Primary trading mode (paper/live/maintenance)
TRADING_MODE=paper

# Individual mode toggles
PAPER_TRADING_ENABLED=true
LIVE_TRADING_ENABLED=false
MAINTENANCE_MODE=false

# =============================================================================
# PAPER TRADING SETTINGS
# =============================================================================

# Starting virtual capital for paper trading
PAPER_TRADING_CAPITAL=100000

# Enable realistic slippage simulation
PAPER_TRADING_SLIPPAGE=true
PAPER_TRADING_SLIPPAGE_PERCENT=0.1

# Enable realistic commission simulation
PAPER_TRADING_COMMISSION=true
PAPER_TRADING_COMMISSION_PERCENT=0.03

# Paper trading data source (live/historical)
PAPER_TRADING_DATA_SOURCE=live

# =============================================================================
# LIVE TRADING SETTINGS
# =============================================================================

# Maximum daily loss limit (percentage of capital)
MAX_DAILY_LOSS=0.05

# Maximum risk per trade (percentage of capital)
MAX_RISK_PER_TRADE=0.02

# Maximum number of concurrent positions
MAX_CONCURRENT_POSITIONS=5

# Enable pre-trade risk checks
PRE_TRADE_RISK_CHECK=true

# Enable post-trade monitoring
POST_TRADE_MONITORING=true

# =============================================================================
# SAFETY SETTINGS
# =============================================================================

# Require manual confirmation for large trades
LARGE_TRADE_CONFIRMATION=true
LARGE_TRADE_THRESHOLD=50000

# Enable circuit breaker for rapid losses
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_LOSS_PERCENT=0.03

# Trading hours enforcement
ENFORCE_TRADING_HOURS=true
TRADING_START_TIME=09:15
TRADING_END_TIME=15:30

# =============================================================================
# KITE CONNECT API SETTINGS (for live trading)
# =============================================================================

KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=your_access_token_here
KITE_REQUEST_TOKEN=your_request_token_here

# API rate limiting
KITE_RATE_LIMIT=180  # requests per minute
KITE_RETRY_ATTEMPTS=3
KITE_RETRY_DELAY=5   # seconds
```

---

## 🔄 **Mode Transition Guide**

### **From Paper to Live Trading**

**Step 1: Validate Paper Trading**

```bash
# Run comprehensive tests in paper mode
python scripts/test_system.py --mode paper --username your_user --password your_pass

# Verify paper trading results
curl -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/api/v1/portfolio/paper/performance
```

**Step 2: Configure Live Trading**

```bash
# Update .env file
TRADING_MODE=live
PAPER_TRADING_ENABLED=false
LIVE_TRADING_ENABLED=true

# Add real API credentials
KITE_API_KEY=your_real_api_key
KITE_API_SECRET=your_real_api_secret
# ... other credentials
```

**Step 3: Test Live Connection (No Trading)**

```bash
# Test API connection without trading
python scripts/test_system.py --mode live --confirm --username your_user --password your_pass

# Verify account access
curl -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/api/v1/portfolio/profile
```

**Step 4: Execute Small Test Trade**

```bash
# Place minimal quantity order
curl -X POST "http://127.0.0.1:8000/api/v1/trading/live/execute" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "action": "BUY",
    "quantity": 1,
    "order_type": "LIMIT",
    "price": 2500
  }'
```

### **Emergency Mode Switch**

**Immediate Stop All Trading:**

```bash
# Emergency stop (stops all trading immediately)
curl -X POST "http://127.0.0.1:8000/api/v1/system/emergency-stop" \
  -H "Authorization: Bearer $TOKEN"

# Or switch to maintenance mode
curl -X POST "http://127.0.0.1:8000/api/v1/system/maintenance-mode" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"enabled": true}'
```

---

## 🛡️ **Safety Features by Mode**

### **Paper Trading Safety Features**

- ✅ No real money at risk
- ✅ Realistic market simulation
- ✅ Commission and slippage simulation
- ✅ Full strategy testing capability
- ✅ Performance analytics
- ✅ Risk-free learning environment

### **Live Trading Safety Features**

- 🔒 Pre-trade risk validation
- 🔒 Position size limits
- 🔒 Daily loss limits
- 🔒 Circuit breaker protection
- 🔒 Trading hours enforcement
- 🔒 Large trade confirmations
- 🔒 Real-time monitoring
- 🔒 Emergency stop capability

### **Maintenance Mode Safety Features**

- 🚫 All trading disabled
- ✅ System monitoring continues
- ✅ Data collection continues
- ✅ Configuration changes allowed
- ✅ System updates possible

---

## 📊 **Mode Comparison Table**

| Feature | Paper Mode | Live Mode | Maintenance Mode |
|---------|------------|-----------|------------------|
| Financial Risk | None | Full | None |
| Market Data | Real/Simulated | Real | Real |
| Order Execution | Simulated | Real | Disabled |
| Strategy Testing | ✅ | ✅ | ❌ |
| Performance Analytics | ✅ | ✅ | ✅ |
| Risk Management | Simulated | Real | N/A |
| Learning Safe | ✅ | ❌ | ✅ |
| Production Ready | ❌ | ✅ | ❌ |

---

## 🎯 **Best Practices**

### **Development Workflow**

1. **Start with Paper Mode**: Always begin with paper trading
2. **Thorough Testing**: Run comprehensive tests before going live
3. **Small Live Tests**: Start with minimal quantities in live mode
4. **Gradual Scaling**: Slowly increase position sizes
5. **Continuous Monitoring**: Watch system closely during live trading

### **Risk Management**

1. **Set Conservative Limits**: Start with lower risk limits
2. **Enable All Safety Features**: Use circuit breakers and confirmations
3. **Regular Monitoring**: Check system health frequently
4. **Have Exit Strategy**: Know how to stop trading quickly
5. **Keep Records**: Maintain detailed logs of all activities

### **Configuration Management**

1. **Version Control**: Keep configuration files in version control
2. **Environment Separation**: Use different configs for different environments
3. **Secure Credentials**: Never commit API keys to version control
4. **Regular Backups**: Backup configuration and data regularly
5. **Documentation**: Document all configuration changes

---

## 🚨 **Emergency Procedures**

### **If Something Goes Wrong**

**Immediate Actions:**

```bash
# 1. Stop all trading immediately
curl -X POST "http://127.0.0.1:8000/api/v1/system/emergency-stop" \
  -H "Authorization: Bearer $TOKEN"

# 2. Check current positions
curl -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/api/v1/portfolio/positions

# 3. Review recent trades
curl -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/api/v1/trading/history?limit=50

# 4. Check system logs
curl -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/api/v1/system/logs/recent
```

**Manual Override:**

- Log into your Zerodha Kite account directly
- Manually close positions if needed
- Check for any pending orders
- Contact your broker if necessary

---

## 📞 **Support Checklist**

Before contacting support, gather this information:

- [ ] Current trading mode
- [ ] Recent error messages
- [ ] System health status
- [ ] Recent trading activity
- [ ] Configuration settings (without API keys)
- [ ] Log files from the last 24 hours

Remember: **Your safety and capital protection are the top priorities. When in doubt, stop trading and seek help.**
