# API Key Setup Guide

This comprehensive guide will help you set up all required API keys for the Shagun Intelligence Trading Platform.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Required API Keys](#required-api-keys)
3. [Zerodha Kite Connect Setup](#zerodha-kite-connect-setup)
4. [OpenAI API Setup](#openai-api-setup)
5. [Optional API Keys](#optional-api-keys)
6. [Environment Configuration](#environment-configuration)
7. [Security Best Practices](#security-best-practices)
8. [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Shagun Intelligence Trading Platform requires several API keys to function properly:

- **Zerodha Kite Connect** (Required): For live trading and market data
- **OpenAI API** (Required): For AI-powered market analysis
- **News APIs** (Optional): For sentiment analysis
- **Market Data APIs** (Optional): For additional data sources

## 🔑 Required API Keys

### Priority 1: Essential for Core Functionality

| API | Purpose | Cost | Required |
|-----|---------|------|----------|
| Zerodha Kite Connect | Trading & Market Data | ₹2000/month | ✅ Yes |
| OpenAI API | AI Analysis | $20-50/month | ✅ Yes |

### Priority 2: Enhanced Features

| API | Purpose | Cost | Required |
|-----|---------|------|----------|
| News API | News Sentiment | Free tier available | ❌ Optional |
| Alpha Vantage | Additional Market Data | Free tier available | ❌ Optional |

## 🏦 Zerodha Kite Connect Setup

### Step 1: Create Kite Connect App

1. **Login to Kite Connect Developer Console**
   - Visit: <https://developers.kite.trade/>
   - Login with your Zerodha credentials

2. **Create New App**
   - Click "Create new app"
   - Fill in app details:
     - **App name**: `Shagun Intelligence Trading`
     - **App type**: `Connect`
     - **Redirect URL**: `http://localhost:8000/api/v1/auth/kite/callback`
     - **Description**: `AI-powered algorithmic trading platform`

3. **Get API Credentials**
   - After approval, note down:
     - **API Key** (Consumer Key)
     - **API Secret** (Consumer Secret)

### Step 2: Configure Environment Variables

Add to your `.env` file:

```bash
# Zerodha Kite Connect API
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=  # Leave empty initially
KITE_REQUEST_TOKEN=  # Leave empty initially
```

### Step 3: Generate Access Token

Run the authentication helper:

```bash
python scripts/setup_kite_credentials.py
```

This will:

1. Open Kite login page in your browser
2. Guide you through the authentication process
3. Generate and save your access token
4. Validate the connection

### Step 4: Verify Setup

```bash
python scripts/validate_api_keys.py
```

Expected output:

```
✅ Kite API: Connected successfully
✅ Access Token: Valid
✅ Market Data: Available
✅ Trading: Enabled
```

## 🤖 OpenAI API Setup

### Step 1: Create OpenAI Account

1. Visit: <https://platform.openai.com/>
2. Sign up or login
3. Go to API Keys section
4. Create new API key

### Step 2: Configure Environment

Add to your `.env` file:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo for lower cost
```

### Step 3: Set Usage Limits

1. Go to OpenAI Usage & Billing
2. Set monthly spending limit (recommended: $50)
3. Enable usage notifications

## 📰 Optional API Keys

### News API (Free Tier Available)

1. Visit: <https://newsapi.org/>
2. Register for free account
3. Get API key (30 days free, then $449/month)

```bash
# News API
NEWS_API_KEY=your_news_api_key_here
NEWS_API_ENABLED=true
```

### Alpha Vantage (Free Tier Available)

1. Visit: <https://www.alphavantage.co/>
2. Get free API key (5 calls/minute, 500 calls/day)

```bash
# Alpha Vantage API
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
ALPHA_VANTAGE_ENABLED=true
```

### Finnhub (Free Tier Available)

1. Visit: <https://finnhub.io/>
2. Register for free account
3. Get API key (60 calls/minute free)

```bash
# Finnhub API
FINNHUB_API_KEY=your_finnhub_key_here
FINNHUB_ENABLED=true
```

## ⚙️ Environment Configuration

### Complete .env Template

Create `.env` file in project root:

```bash
# =============================================================================
# SHAGUN INTELLIGENCE TRADING PLATFORM - ENVIRONMENT CONFIGURATION
# =============================================================================

# =============================================================================
# BASIC CONFIGURATION
# =============================================================================
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-super-secret-key-change-in-production

# =============================================================================
# ZERODHA KITE CONNECT (REQUIRED)
# =============================================================================
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
KITE_ACCESS_TOKEN=your_kite_access_token
KITE_REQUEST_TOKEN=

# =============================================================================
# OPENAI API (REQUIRED)
# =============================================================================
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-4

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================
TRADING_MODE=paper  # paper or live
LIVE_TRADING_ENABLED=false
PAPER_TRADING_ENABLED=true
AUTOMATED_TRADING_ENABLED=false
MANUAL_APPROVAL_REQUIRED=true

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
DEFAULT_POSITION_SIZE=1000
MAX_POSITION_VALUE=300
MAX_DAILY_TRADES=3
MAX_DAILY_LOSS=100
EMERGENCY_STOP_LOSS_AMOUNT=80

# =============================================================================
# OPTIONAL APIS
# =============================================================================
NEWS_API_KEY=
NEWS_API_ENABLED=false

ALPHA_VANTAGE_API_KEY=
ALPHA_VANTAGE_ENABLED=false

FINNHUB_API_KEY=
FINNHUB_ENABLED=false

# =============================================================================
# DATABASE
# =============================================================================
DATABASE_URL=sqlite:///./trading.db

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL=INFO
```

### Environment-Specific Configurations

#### Development (.env.development)

```bash
ENVIRONMENT=development
DEBUG=true
TRADING_MODE=paper
LIVE_TRADING_ENABLED=false
LOG_LEVEL=DEBUG
```

#### Production (.env.production)

```bash
ENVIRONMENT=production
DEBUG=false
TRADING_MODE=live
LIVE_TRADING_ENABLED=true
LOG_LEVEL=INFO
```

## 🔒 Security Best Practices

### 1. API Key Security

- **Never commit API keys to version control**
- **Use environment variables only**
- **Rotate keys regularly (every 90 days)**
- **Use different keys for development and production**

### 2. File Permissions

```bash
# Secure your .env file
chmod 600 .env
chmod 600 .env.production

# Verify permissions
ls -la .env*
```

### 3. Access Control

- **Limit API key permissions to minimum required**
- **Use IP whitelisting where available**
- **Monitor API usage regularly**

### 4. Backup and Recovery

```bash
# Backup your configuration (without sensitive data)
cp .env.template .env.backup
# Store securely, separate from code
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Kite API Connection Failed

**Error**: `KiteAuthenticationError: Invalid API key`

**Solutions**:

- Verify API key and secret are correct
- Check if Kite Connect app is approved
- Ensure access token is not expired
- Run: `python scripts/setup_kite_credentials.py`

#### 2. OpenAI API Rate Limit

**Error**: `Rate limit exceeded`

**Solutions**:

- Check your OpenAI usage dashboard
- Increase rate limits or upgrade plan
- Implement request throttling

#### 3. Environment Variables Not Loading

**Error**: `Settings object has no attribute 'KITE_API_KEY'`

**Solutions**:

- Ensure `.env` file is in project root
- Check file permissions: `chmod 600 .env`
- Verify no typos in variable names
- Restart the application

### Validation Commands

```bash
# Test all API connections
python scripts/validate_api_keys.py

# Test specific API
python scripts/test_kite_connection.py
python scripts/test_openai_connection.py

# Check environment loading
python -c "from app.core.config import get_settings; print(get_settings().KITE_API_KEY[:10] + '...')"
```

### Getting Help

1. **Check logs**: `tail -f logs/trading.log`
2. **Run diagnostics**: `python scripts/system_diagnostics.py`
3. **Contact support**: Create issue on GitHub with logs (remove sensitive data)

## ✅ Final Verification

After setting up all API keys, run the complete validation:

```bash
# Run comprehensive validation
python scripts/validate_system.py

# Expected output:
# ✅ Environment: Loaded successfully
# ✅ Kite API: Connected and authenticated
# ✅ OpenAI API: Connected and functional
# ✅ Database: Connected and accessible
# ✅ All systems: Ready for trading
```

## 📞 Support

- **Documentation**: [docs/](../docs/)
- **GitHub Issues**: [Create Issue](https://github.com/iamapsrajput/shagunintelligence/issues)
- **Email**: <iamapsrajput@outlook.com>

---

**⚠️ Important**: Keep your API keys secure and never share them publicly. This platform handles real money - always test thoroughly in paper trading mode before going live.
