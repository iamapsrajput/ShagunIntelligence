#!/usr/bin/env python3
"""
Test script for automated trading system
Tests the live data integration and automated trading functionality
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.services.automated_trading import AutomatedTradingService
from services.kite.client import KiteConnectService


async def test_kite_connection():
    """Test Kite Connect API connection"""
    print("🔗 Testing Kite Connect API connection...")

    try:
        kite_client = KiteConnectService()
        success = await kite_client.initialize()

        if success:
            print("✅ Kite Connect API connection successful")

            # Test getting a quote
            quote = await kite_client.get_quote("RELIANCE")
            print(f"📊 RELIANCE Quote: ₹{quote.get('last_price', 'N/A')}")

            return True
        else:
            print("❌ Kite Connect API connection failed")
            return False

    except Exception as e:
        print(f"❌ Kite Connect API error: {e}")
        return False


async def test_automated_trading_service():
    """Test the automated trading service"""
    print("\n🤖 Testing Automated Trading Service...")

    try:
        trading_service = AutomatedTradingService()

        # Test starting automated trading
        result = await trading_service.start_automated_trading()

        if result["status"] == "success":
            print("✅ Automated trading started successfully")
            print(f"   Config: {result.get('config', {})}")

            # Let it run for a short time
            print("⏳ Running automated trading for 30 seconds...")
            await asyncio.sleep(30)

            # Check status
            status = await trading_service.get_status()
            print(f"📊 Status: {status}")

            # Stop the service
            stop_result = await trading_service.stop_automated_trading()
            print(f"🛑 Stop result: {stop_result}")

            return True
        else:
            print(f"❌ Failed to start automated trading: {result['message']}")
            return False

    except Exception as e:
        print(f"❌ Automated trading service error: {e}")
        return False


async def test_live_data_flow():
    """Test live data flow to agents"""
    print("\n📡 Testing live data flow...")

    try:
        kite_client = KiteConnectService()
        await kite_client.initialize()

        # Test getting live quotes for trading symbols
        symbols = ["RELIANCE", "TCS", "INFY"]

        for symbol in symbols:
            quote = await kite_client.get_quote(symbol)
            print(
                f"📊 {symbol}: ₹{quote.get('last_price', 'N/A')} (Volume: {quote.get('volume', 'N/A')})"
            )

        return True

    except Exception as e:
        print(f"❌ Live data flow error: {e}")
        return False


def check_configuration():
    """Check if configuration is properly set"""
    print("⚙️ Checking configuration...")

    settings = get_settings()

    checks = {
        "KITE_API_KEY": bool(
            settings.KITE_API_KEY and settings.KITE_API_KEY != "your_api_key_here"
        ),
        "KITE_ACCESS_TOKEN": bool(
            settings.KITE_ACCESS_TOKEN
            and settings.KITE_ACCESS_TOKEN != "your_access_token_here"
        ),
        "LIVE_TRADING_ENABLED": settings.LIVE_TRADING_ENABLED,
        "AUTOMATED_TRADING_ENABLED": settings.AUTOMATED_TRADING_ENABLED,
        "TRADING_MODE": settings.TRADING_MODE == "live",
    }

    all_good = True
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check}: {status}")
        if not status:
            all_good = False

    if not all_good:
        print("\n⚠️ Configuration issues detected:")
        print(
            "   1. Make sure you've set your Kite API credentials in config/live_trading_test.env"
        )
        print("   2. Ensure TRADING_MODE=live and AUTOMATED_TRADING_ENABLED=true")
        print(
            "   3. Load the environment file: export $(cat config/live_trading_test.env | xargs)"
        )

    return all_good


async def main():
    """Main test function"""
    print("🚀 Shagun Intelligence - Automated Trading System Test")
    print("=" * 60)

    # Check configuration
    if not check_configuration():
        print("\n❌ Configuration check failed. Please fix the issues above.")
        return

    print("✅ Configuration check passed")

    # Test Kite connection
    if not await test_kite_connection():
        print(
            "\n❌ Kite connection test failed. Cannot proceed with automated trading test."
        )
        return

    # Test live data flow
    if not await test_live_data_flow():
        print("\n❌ Live data flow test failed.")
        return

    # Test automated trading service
    print("\n" + "=" * 60)
    print("🎯 All preliminary tests passed! Testing automated trading...")

    if await test_automated_trading_service():
        print("\n🎉 All tests passed! Automated trading system is working correctly.")
        print("\n📋 Next steps:")
        print("   1. Monitor the system logs for trading decisions")
        print("   2. Check your Zerodha account for any test trades")
        print("   3. Use the dashboard to monitor live trading activity")
        print("   4. Adjust risk parameters as needed")
    else:
        print("\n❌ Automated trading test failed.")


if __name__ == "__main__":
    # Load environment variables from test config
    env_file = project_root / "config" / "live_trading_test.env"
    if env_file.exists():
        print(f"📁 Loading configuration from {env_file}")
        # Note: In production, use python-dotenv or similar
        # For now, user should manually export the variables

    asyncio.run(main())
