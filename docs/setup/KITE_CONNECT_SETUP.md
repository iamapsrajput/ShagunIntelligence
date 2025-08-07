# 🔐 Zerodha Kite Connect Authentication Setup Guide

This guide provides step-by-step instructions for setting up Zerodha Kite Connect API authentication, including obtaining the required access tokens.

## 📋 Prerequisites

1. **Active Zerodha Trading Account**: You must have an active Zerodha trading account
2. **Kite Connect Subscription**: Subscribe to Kite Connect API (₹2,000/month)
3. **API App Registration**: Register your application on Kite Connect platform

## 🚀 Step-by-Step Setup Process

### Step 1: Subscribe to Kite Connect API

1. Visit [Kite Connect](https://kite.trade/connect/)
2. Log in with your Zerodha credentials
3. Subscribe to Kite Connect API (₹2,000/month)
4. Wait for subscription activation (usually instant)

### Step 2: Create API App

1. Go to [Kite Connect Apps](https://developers.kite.trade/apps/)
2. Click "Create new app"
3. Fill in the application details:

   ```
   App Name: Shagun Intelligence Trading Platform
   App Type: Connect
   Redirect URL: http://localhost:8080/callback
   Description: AI-powered trading platform for automated market analysis
   ```

4. Submit the application
5. Note down your **API Key** and **API Secret**

### Step 3: Configure Environment Variables

Add your API credentials to `.env` file:

```env
# Zerodha Kite Connect API Configuration
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=your_access_token_here  # Will be obtained in Step 4
KITE_REQUEST_TOKEN=your_request_token_here  # Will be obtained in Step 4
```

### Step 4: Obtain Access Token (Authentication Flow)

The Kite Connect API uses OAuth 2.0 authentication. You need to complete the authentication flow to get access tokens.

#### Option A: Manual Authentication (Recommended for Development)

1. **Generate Login URL**:

   ```python
   from kiteconnect import KiteConnect

   kite = KiteConnect(api_key="your_api_key_here")
   login_url = kite.login_url()
   print(f"Visit this URL to authenticate: {login_url}")
   ```

2. **Visit the Login URL**:

   - Open the generated URL in your browser
   - Log in with your Zerodha credentials
   - Authorize the application
   - You'll be redirected to: `http://localhost:8080/callback?request_token=XXXXXX&action=login&status=success`

3. **Extract Request Token**:

   - Copy the `request_token` from the redirect URL
   - Add it to your `.env` file as `KITE_REQUEST_TOKEN`

4. **Generate Access Token**:

   ```python
   from kiteconnect import KiteConnect

   kite = KiteConnect(api_key="your_api_key_here")
   data = kite.generate_session(
       request_token="your_request_token_here",
       api_secret="your_api_secret_here"
   )

   access_token = data["access_token"]
   print(f"Access Token: {access_token}")
   ```

5. **Update .env File**:

   ```env
   KITE_ACCESS_TOKEN=your_generated_access_token_here
   ```

#### Option B: Automated Authentication Script

We provide an automated script for easier token generation:

```bash
# Run the authentication helper
poetry run python scripts/kite_auth_helper.py
```

This script will:

1. Generate the login URL
2. Open it in your browser
3. Wait for you to complete authentication
4. Automatically extract tokens
5. Update your `.env` file

### Step 5: Verify Authentication

Run the API validation script to verify your setup:

```bash
poetry run python scripts/validate_api_keys.py
```

You should see:

```
✅ Zerodha Kite Connect (Required): ✅ Connected and authenticated
```

## 🔄 Token Management

### Access Token Expiration

- **Access tokens expire daily** at 6:00 AM IST
- You need to regenerate them daily for continuous operation
- The platform includes automatic token refresh mechanisms

### Production Token Management

For production deployment, implement automated token refresh:

1. **Store refresh tokens securely**
2. **Set up daily token refresh job**
3. **Implement fallback mechanisms**
4. **Monitor token expiration**

## 🛠️ Troubleshooting

### Common Issues

1. **"Invalid API Key"**

   - Verify API key is correct
   - Ensure Kite Connect subscription is active
   - Check if app is approved

2. **"Invalid Request Token"**

   - Request tokens expire in 10 minutes
   - Generate a new request token
   - Complete authentication flow quickly

3. **"Access Token Expired"**

   - Access tokens expire daily
   - Generate new access token
   - Implement automated refresh

4. **"Redirect URL Mismatch"**
   - Ensure redirect URL matches exactly
   - Use `http://localhost:8080/callback` for development
   - Update app settings if needed

### Debug Commands

```bash
# Test API connection
poetry run python -c "
from kiteconnect import KiteConnect
kite = KiteConnect(api_key='your_api_key')
print('Profile:', kite.profile())
"

# Check token validity
poetry run python -c "
from app.core.config import get_settings
settings = get_settings()
print('API Key:', settings.KITE_API_KEY[:10] + '...')
print('Access Token:', settings.KITE_ACCESS_TOKEN[:10] + '...')
"
```

## 🔒 Security Best Practices

1. **Never commit tokens to version control**
2. **Use environment variables only**
3. **Rotate tokens regularly**
4. **Monitor API usage and costs**
5. **Implement rate limiting**
6. **Use HTTPS in production**

## 📞 Support

- **Zerodha Support**: [support@zerodha.com](mailto:support@zerodha.com)
- **Kite Connect Documentation**: [kite.trade/docs](https://kite.trade/docs)
- **Developer Forum**: [forum.zerodha.com](https://forum.zerodha.com)

## 🎯 Next Steps

After completing authentication:

1. Run the full API validation: `poetry run python scripts/validate_api_keys.py`
2. Test trading functionality: `poetry run python scripts/test_trading.py`
3. Start the platform: `poetry run uvicorn app.main:app --reload`

---

**Important**: Keep your API credentials secure and never share them publicly. The access token provides full access to your trading account.
