#!/usr/bin/env python3
"""
Kite API Credentials Setup Helper
Helps users set up their Zerodha Kite Connect API credentials
"""

import os
import sys
import webbrowser
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kiteconnect import KiteConnect


def print_banner():
    """Print setup banner"""
    print("🔑 Zerodha Kite Connect API Setup Helper")
    print("=" * 50)
    print("This script will help you set up your Kite API credentials")
    print("for automated trading with Shagun Intelligence.\n")


def get_api_credentials():
    """Get API key and secret from user"""
    print("📋 Step 1: Get your API credentials")
    print("1. Go to https://developers.kite.trade/")
    print("2. Login with your Zerodha credentials")
    print("3. Create a new app (if you haven't already)")
    print("4. Note down your API Key and API Secret\n")

    api_key = input("Enter your API Key: ").strip()
    api_secret = input("Enter your API Secret: ").strip()

    if not api_key or not api_secret:
        print("❌ API Key and Secret are required!")
        return None, None

    return api_key, api_secret


def generate_access_token(api_key, api_secret):
    """Generate access token using request token"""
    print("\n🔗 Step 2: Generate Access Token")

    try:
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()

        print("1. Opening login URL in your browser...")
        print(f"   URL: {login_url}")

        # Try to open in browser
        try:
            webbrowser.open(login_url)
            print("   ✅ Browser opened successfully")
        except:
            print("   ⚠️ Could not open browser automatically")
            print(f"   Please manually open: {login_url}")

        print("\n2. Complete the login process in your browser")
        print("3. After login, you'll be redirected to a URL like:")
        print(
            "   http://127.0.0.1:8080/?request_token=XXXXXX&action=login&status=success"
        )
        print("4. Copy the 'request_token' value from the URL\n")

        request_token = input("Enter the request token from the URL: ").strip()

        if not request_token:
            print("❌ Request token is required!")
            return None

        print("\n🔄 Generating access token...")

        # Generate session
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]

        print("✅ Access token generated successfully!")
        return access_token

    except Exception as e:
        print(f"❌ Failed to generate access token: {e}")
        return None


def test_credentials(api_key, api_secret, access_token):
    """Test the API credentials"""
    print("\n🧪 Step 3: Testing API credentials...")

    try:
        kite = KiteConnect(api_key=api_key, access_token=access_token)

        # Test by getting profile
        profile = kite.profile()

        print("✅ API credentials are working!")
        print(f"   User: {profile.get('user_name', 'N/A')}")
        print(f"   User ID: {profile.get('user_id', 'N/A')}")
        print(f"   Broker: {profile.get('broker', 'N/A')}")

        # Test getting a quote
        quote = kite.quote(["NSE:RELIANCE"])
        reliance_price = quote.get("NSE:RELIANCE", {}).get("last_price", "N/A")
        print(f"   Test Quote - RELIANCE: ₹{reliance_price}")

        return True

    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def save_credentials(api_key, api_secret, access_token):
    """Save credentials to configuration file"""
    print("\n💾 Step 4: Saving credentials...")

    config_dir = project_root / "config"
    config_file = config_dir / "live_trading.env"

    # Create config directory if it doesn't exist
    config_dir.mkdir(exist_ok=True)

    # Read template or create new config
    template_file = config_dir / "live_trading_test.env"

    if template_file.exists():
        with open(template_file) as f:
            config_content = f.read()
    else:
        config_content = f"""# Zerodha Kite Connect API Configuration
KITE_API_KEY={api_key}
KITE_API_SECRET={api_secret}
KITE_ACCESS_TOKEN={access_token}

# Trading Configuration
TRADING_MODE=live
LIVE_TRADING_ENABLED=true
AUTOMATED_TRADING_ENABLED=true
PAPER_TRADING_ENABLED=false
MANUAL_APPROVAL_REQUIRED=false

# Risk Management
DEFAULT_POSITION_SIZE=200
MAX_POSITION_VALUE=300
MAX_CONCURRENT_POSITIONS=3
MAX_DAILY_LOSS=0.10
EMERGENCY_STOP_LOSS_AMOUNT=80

# Position Management
AUTO_STOP_LOSS=true
AUTO_STOP_LOSS_PERCENT=0.05
AUTO_TAKE_PROFIT=true
AUTO_TAKE_PROFIT_PERCENT=0.10
"""

    # Update credentials in config
    config_content = config_content.replace(
        "KITE_API_KEY=your_api_key_here", f"KITE_API_KEY={api_key}"
    )
    config_content = config_content.replace(
        "KITE_API_SECRET=your_api_secret_here", f"KITE_API_SECRET={api_secret}"
    )
    config_content = config_content.replace(
        "KITE_ACCESS_TOKEN=your_access_token_here", f"KITE_ACCESS_TOKEN={access_token}"
    )

    # Save to file
    with open(config_file, "w") as f:
        f.write(config_content)

    print(f"✅ Credentials saved to: {config_file}")

    # Create environment export script
    export_script = config_dir / "export_env.sh"
    with open(export_script, "w") as f:
        f.write(
            f"""#!/bin/bash
# Export environment variables for automated trading
export $(cat {config_file} | grep -v '^#' | xargs)
echo "Environment variables loaded for automated trading"
"""
        )

    os.chmod(export_script, 0o755)
    print(f"✅ Environment script created: {export_script}")


def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup Complete!")
    print("=" * 30)
    print("Your Kite API credentials have been configured successfully.")
    print("\n📋 Next Steps:")
    print("1. Load environment variables:")
    print("   source config/export_env.sh")
    print("\n2. Test the automated trading system:")
    print("   python scripts/test_automated_trading.py")
    print("\n3. Start the trading system:")
    print("   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("\n4. Start the dashboard:")
    print("   cd dashboard && npm run dev")
    print("\n5. Access the dashboard:")
    print("   http://localhost:3000")
    print("\n⚠️ Important Notes:")
    print("- Start with small amounts for testing")
    print("- Monitor the system closely initially")
    print("- Keep the emergency stop button accessible")
    print("- Review all trades and adjust parameters as needed")


def main():
    """Main setup function"""
    print_banner()

    # Get API credentials
    api_key, api_secret = get_api_credentials()
    if not api_key or not api_secret:
        return

    # Generate access token
    access_token = generate_access_token(api_key, api_secret)
    if not access_token:
        return

    # Test credentials
    if not test_credentials(api_key, api_secret, access_token):
        return

    # Save credentials
    save_credentials(api_key, api_secret, access_token)

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please try again or check the documentation for manual setup.")
