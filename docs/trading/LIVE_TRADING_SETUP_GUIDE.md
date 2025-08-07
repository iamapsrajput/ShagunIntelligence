# 🔴 **Live Trading Setup Guide - ₹1000 Budget**

## ⚠️ **CRITICAL SAFETY WARNING**

This guide involves **REAL MONEY** trading. Maximum risk is limited to ₹100 daily loss and ₹50 per trade. Proceed with extreme caution.

---

## 📋 **Prerequisites Checklist**

Before starting, ensure you have:

- [ ] Active Zerodha trading account with KYC completed
- [ ] Kite Connect API subscription and credentials
- [ ] ₹1000+ available in your trading account
- [ ] System successfully running in paper mode
- [ ] All trading accuracy tests passing (9/9)
- [ ] Emergency stop procedures understood

---

## 🔧 **Step 1: Configure Live Trading Mode**

### Update Environment Configuration

Create or update `.env.live` file:

```env
# Live Trading Configuration
ENVIRONMENT=live
TRADING_MODE=live

# Budget and Risk Management
DEFAULT_POSITION_SIZE=1000  # ₹1000 total budget
MAX_POSITION_SIZE=300       # ₹300 max per trade
MAX_DAILY_LOSS=100         # ₹100 max daily loss
EMERGENCY_STOP_LOSS=80     # ₹80 emergency stop

# Trading Constraints
MAX_DAILY_TRADES=3         # Maximum 3 trades per day
MAX_CONCURRENT_POSITIONS=2 # Maximum 2 open positions
MIN_TRADE_AMOUNT=200       # Minimum ₹200 per trade

# Safety Features
ENFORCE_TRADING_HOURS=true
ENABLE_CIRCUIT_BREAKERS=true
REQUIRE_CONFIRMATION=false  # Set to true for manual confirmation
```

### Verify API Credentials

```bash
# Test live API connection
poetry run python scripts/validate_api_keys.py --live
```

Expected output:

```
✅ Zerodha Kite Connect (Live): ✅ Connected and authenticated
✅ Account Balance: ₹X,XXX available
✅ Trading permissions: Equity trading enabled
```

---

## 🚀 **Step 2: Initialize Live Trading System**

### Start the Trading Platform

```bash
# Load live environment and start
export ENV_FILE=.env.live
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Verify System Status

1. **Open Dashboard**: <http://localhost:3000>
2. **Check System Status**: All indicators should be green
3. **Verify Trading Mode**: Should show "LIVE TRADING" with red indicator
4. **Confirm Budget**: Should display ₹1000 total budget

---

## 🎯 **Step 3: Enable Automated Trading**

### Safety Checks Before Activation

Run comprehensive pre-trading validation:

```bash
poetry run python scripts/live_trading_1000_test.py
```

This script verifies:

- [ ] Account balance sufficient (₹1000+)
- [ ] API permissions correct
- [ ] Risk limits properly configured
- [ ] Emergency stops functional
- [ ] Market hours validation
- [ ] Circuit breakers active

### Activate Live Trading

1. **In Dashboard**: Click "Start Automated Trading"
2. **Confirm Safety Settings**: Review all risk parameters
3. **Acknowledge Risks**: Confirm understanding of real money trading
4. **Monitor Initial Trades**: Watch first few trades closely

---

## 📊 **Step 4: Monitoring and Safety**

### Real-Time Monitoring

**Dashboard Indicators to Watch:**

- 🟢 **System Status**: Should remain green
- 🔴 **Live Trading**: Active indicator
- 💰 **Available Balance**: Real-time balance
- 📈 **Daily P&L**: Current day profit/loss
- ⚠️ **Risk Alerts**: Any safety warnings

### Emergency Procedures

**Immediate Stop Trading:**

```bash
# Emergency stop all trading
curl -X POST http://localhost:8000/api/v1/automated-trading/emergency-stop
```

**Manual Position Exit:**

```bash
# Close all positions immediately
curl -X POST http://localhost:8000/api/v1/trading/close-all-positions
```

### Daily Monitoring Checklist

**Every Trading Day:**

- [ ] Check account balance before market open
- [ ] Verify system status (all green indicators)
- [ ] Monitor first trade execution
- [ ] Review daily P&L at market close
- [ ] Check for any error alerts

---

## 🛡️ **Safety Features Active**

### Automatic Risk Management

1. **Position Sizing**: Maximum ₹300 per trade
2. **Daily Loss Limit**: Stops at ₹100 daily loss
3. **Emergency Circuit Breaker**: Triggers at ₹80 total loss
4. **Trading Hours**: Only 9:15 AM - 3:30 PM IST
5. **Stock Filtering**: Only liquid, large-cap stocks
6. **Volume Validation**: Minimum 10,000 daily volume

### Manual Override Options

**Pause Trading:**

- Dashboard: "Pause Automated Trading" button
- API: `POST /api/v1/automated-trading/pause`

**Adjust Risk Limits:**

- Dashboard: Settings → Risk Management
- Requires system restart for changes

---

## 📈 **Expected Trading Behavior**

### Conservative Strategy

**Stock Selection:**

- Large-cap stocks (Nifty 50 primarily)
- High liquidity (>10,000 daily volume)
- Price range: ₹100-₹2000 per share

**Trade Execution:**

- 2-3 trades per day maximum
- ₹200-₹300 position sizes
- 1-2% profit targets
- 0.5-1% stop losses

**AI Decision Making:**

- Technical analysis using 5-minute charts
- Sentiment analysis from news/social media
- Risk assessment for each trade
- Conservative bias (prefers HOLD over risky trades)

---

## 🔍 **Monitoring Logs**

### Key Log Files

```bash
# Trading activity log
tail -f logs/trading_activity.log

# Error monitoring
tail -f logs/kite_trading_errors.log

# Agent decisions
tail -f logs/agent_decisions.log

# System performance
tail -f logs/performance.log
```

### Important Metrics to Track

**Daily Metrics:**

- Total trades executed
- Win/loss ratio
- Average profit per trade
- Maximum drawdown
- System uptime

**Weekly Review:**

- Overall P&L performance
- Strategy effectiveness
- Risk management performance
- System reliability

---

## 🚨 **Troubleshooting**

### Common Issues

**1. Trading Not Starting**

```bash
# Check system status
curl http://localhost:8000/api/v1/system/status

# Verify API connection
poetry run python scripts/validate_api_keys.py --live
```

**2. Unexpected Losses**

- Check if emergency stop triggered
- Review recent trade decisions in logs
- Verify risk limits are being enforced

**3. System Errors**

```bash
# Check error logs
tail -50 logs/kite_trading_errors.log

# Restart system if needed
docker-compose restart
```

### Emergency Contacts

**Immediate Issues:**

- Stop trading immediately using emergency stop
- Check Zerodha account directly
- Review all open positions

**System Issues:**

- Check logs for error patterns
- Verify network connectivity
- Restart services if necessary

---

## 📞 **Support and Resources**

### Documentation

- [API Key Setup Guide](../setup/API_KEY_SETUP_GUIDE.md)
- [Troubleshooting Guide](../support/TROUBLESHOOTING.md)
- [Security Best Practices](../security/SECURITY_BEST_PRACTICES.md)

### Monitoring Tools

- **Dashboard**: <http://localhost:3000>
- **API Health**: <http://localhost:8000/api/v1/health>
- **System Metrics**: <http://localhost:3000/monitoring>

---

## ✅ **Success Criteria**

After 1 week of live trading, you should see:

- [ ] System running without major errors
- [ ] Daily loss limits respected
- [ ] Profitable or break-even performance
- [ ] All safety features functioning
- [ ] Consistent trade execution

**Remember**: The goal is capital preservation and learning, not maximum profits. The ₹1000 budget is for gaining experience with live automated trading.

---

**⚠️ FINAL REMINDER**: This involves real money. Monitor closely, especially during the first week. The system is designed to be conservative, but market risks always exist.
