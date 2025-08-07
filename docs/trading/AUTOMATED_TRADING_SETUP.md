# 🤖 Automated Trading System Setup Guide

This guide will help you set up the fully automated trading system with live market data and AI-powered decision making.

## 📋 Prerequisites

### 1. Zerodha Kite Connect Account

- Active Zerodha trading account
- Kite Connect API subscription (₹2000/month)
- API credentials (API Key, API Secret, Access Token)

### 2. System Requirements

- Python 3.9+
- Minimum 4GB RAM
- Stable internet connection
- Linux/macOS/Windows

## 🚀 Quick Start (5 Minutes)

### Step 1: Get Your Kite API Credentials

1. **Login to Kite Connect Developer Console**

   ```
   https://developers.kite.trade/
   ```

2. **Create a new app** and note down:
   - API Key
   - API Secret

3. **Generate Access Token**
   - Use the login URL to get request token
   - Generate session to get access token

### Step 2: Configure the System

1. **Copy the configuration template**

   ```bash
   cp config/live_trading_test.env config/live_trading.env
   ```

2. **Edit the configuration file**

   ```bash
   nano config/live_trading.env
   ```

3. **Add your credentials**

   ```env
   KITE_API_KEY=your_actual_api_key
   KITE_API_SECRET=your_actual_api_secret
   KITE_ACCESS_TOKEN=your_actual_access_token
   ```

### Step 3: Test the Connection

```bash
# Load environment variables
export $(cat config/live_trading.env | xargs)

# Test the automated trading system
python scripts/test_automated_trading.py
```

### Step 4: Start the System

```bash
# Start the backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Start the dashboard (in another terminal)
cd dashboard
npm run dev
```

### Step 5: Access the Dashboard

Open your browser to: **<http://localhost:3000>**

## ⚙️ Configuration Options

### Budget & Risk Settings (₹1000 Budget Example)

```env
# Conservative settings for ₹1000 budget
TOTAL_BUDGET=1000
DEFAULT_POSITION_SIZE=200        # ₹200 per trade
MAX_POSITION_VALUE=300          # Maximum ₹300 per position
MAX_CONCURRENT_POSITIONS=3      # Maximum 3 positions
MAX_DAILY_LOSS=0.10            # 10% daily loss limit (₹100)
EMERGENCY_STOP_LOSS_AMOUNT=80   # Emergency stop at ₹80 loss
```

### Trading Hours

```env
ENFORCE_TRADING_HOURS=true
TRADING_START_TIME=09:15
TRADING_END_TIME=15:30
TRADING_TIMEZONE=Asia/Kolkata
```

### Automated Analysis

```env
MARKET_ANALYSIS_INTERVAL=300    # Analyze every 5 minutes
POSITION_MONITORING_INTERVAL=30 # Monitor positions every 30 seconds
RISK_CHECK_INTERVAL=60         # Risk checks every minute
```

## 🎯 How It Works

### 1. **Live Data Integration**

- Connects to Zerodha Kite API for real-time market data
- Streams live prices, volume, and market depth
- Validates data quality before analysis

### 2. **AI Agent Analysis**

- **Market Analyst**: Technical analysis with indicators
- **Sentiment Analyst**: News and social sentiment analysis
- **Risk Manager**: Position sizing and risk assessment
- **Trade Executor**: Order placement and management

### 3. **Automated Decision Making**

- Analyzes 10 liquid stocks every 5 minutes
- Makes BUY decisions based on:
  - Technical signals (RSI, MACD, Moving Averages)
  - Sentiment score > 0.6
  - Volume > 10,000 shares
  - Price range ₹50-₹5000
  - Available capital and position limits

### 4. **Risk Management**

- Automatic stop loss at 5%
- Automatic take profit at 10%
- Daily loss limits
- Emergency circuit breaker
- Position size limits

## 📊 Dashboard Features

### System Controls

- Start/Stop automated trading
- Emergency stop button
- Real-time system status
- Risk level monitoring

### Live Market Data

- Real-time price charts
- Technical indicators
- Volume analysis
- Market depth

### AI Agent Panel

- Agent decision logs
- Analysis results
- Confidence scores
- Trade recommendations

### Portfolio Monitoring

- Live P&L tracking
- Position status
- Trade history
- Performance metrics

## 🛡️ Safety Features

### Multiple Safety Layers

1. **Pre-trade validation**
   - API connection check
   - Market hours verification
   - Available funds check
   - Position limits check

2. **Real-time monitoring**
   - Continuous P&L tracking
   - Stop loss monitoring
   - Risk limit enforcement
   - Circuit breaker activation

3. **Emergency controls**
   - Manual emergency stop
   - Automatic loss limits
   - System health monitoring
   - Error handling and recovery

## 📈 Trading Strategy

### Stock Selection

- **Liquid stocks only**: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, etc.
- **Volume threshold**: Minimum 10,000 daily volume
- **Price range**: ₹50 - ₹5000 per share
- **Market cap**: Large and mid-cap stocks only

### Entry Criteria

- **Technical**: RSI < 70, MACD bullish crossover, price above 20-day MA
- **Sentiment**: Positive news sentiment score > 0.6
- **Risk**: Position size within limits, available capital
- **Market**: During trading hours, normal volatility

### Exit Strategy

- **Stop Loss**: 5% below entry price
- **Take Profit**: 10% above entry price
- **Time-based**: Close all positions before market close
- **Risk-based**: Emergency exit if daily loss limit reached

## 🔧 Troubleshooting

### Common Issues

1. **"API connection failed"**
   - Check your Kite API credentials
   - Ensure access token is valid (regenerate if needed)
   - Verify internet connection

2. **"Outside trading hours"**
   - System only trades 9:15 AM - 3:30 PM IST
   - Check system timezone settings
   - Verify market holidays

3. **"No trades executed"**
   - Check if market conditions meet entry criteria
   - Verify available capital
   - Review agent analysis logs

4. **"High risk detected"**
   - System automatically stops if risk limits exceeded
   - Review position sizes and loss limits
   - Check emergency stop settings

### Logs and Monitoring

```bash
# View trading logs
tail -f logs/live_trading.log

# Check system health
curl http://localhost:8000/api/v1/health

# Monitor agent decisions
curl http://localhost:8000/api/v1/agents/status
```

## 📞 Support

### Getting Help

- Check logs first: `logs/live_trading.log`
- Review configuration: `config/live_trading.env`
- Test API connection: `python scripts/test_automated_trading.py`
- Monitor dashboard: `http://localhost:3000`

### Important Notes

- **Start with paper trading** to test strategies
- **Use small amounts** initially (₹1000-5000)
- **Monitor closely** for the first few days
- **Keep emergency stop** easily accessible
- **Review trades daily** and adjust parameters

## 🎉 Success Checklist

- [ ] Kite API credentials configured
- [ ] System passes connection test
- [ ] Dashboard accessible and showing live data
- [ ] Automated trading starts without errors
- [ ] AI agents making analysis decisions
- [ ] Risk management working correctly
- [ ] Emergency stop accessible
- [ ] Logs showing trading activity

**You're now ready for fully automated trading! 🚀**
