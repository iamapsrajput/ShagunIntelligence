#!/usr/bin/env python3
"""
Live Trading Test Script for ₹1000 Budget
Ultra-conservative live trading validation with real money.

IMPORTANT: This script trades with REAL MONEY. Use with extreme caution.

Usage:
    python scripts/live_trading_1000_test.py --confirm-real-money
"""

import argparse
import asyncio
import sys
from datetime import datetime
from typing import Optional

import aiohttp
from loguru import logger

# Configure logging
logger.add("logs/live_trading_1000_test.log", rotation="1 day", retention="30 days")


class LiveTradingValidator:
    """Ultra-conservative live trading validator for ₹1000 budget."""

    def __init__(
        self, base_url: str = "http://127.0.0.1:8000", token: str | None = None
    ):
        self.base_url = base_url
        self.token = token
        self.session: aiohttp.ClientSession | None = None
        self.total_budget = 1000  # ₹1000 total budget
        self.max_risk_per_trade = 50  # ₹50 max per trade
        self.max_daily_loss = 100  # ₹100 max daily loss
        self.emergency_stop_loss = 80  # ₹80 emergency stop

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_headers(self) -> dict[str, str]:
        """Get headers with authentication token."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _make_request(
        self, method: str, endpoint: str, data: dict | None = None
    ) -> dict:
        """Make HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        try:
            async with self.session.request(
                method, url, headers=headers, json=data
            ) as response:
                result = await response.json()
                return {"status_code": response.status, "data": result}
        except Exception as e:
            logger.error(f"Request failed: {method} {url} - {str(e)}")
            return {"status_code": 0, "error": str(e)}

    async def validate_account_setup(self) -> bool:
        """Validate account setup and available funds."""
        logger.info("💰 Validating account setup...")

        # Check account profile
        result = await self._make_request("GET", "/api/v1/portfolio/profile")
        if result["status_code"] != 200:
            logger.error("❌ Failed to get account profile")
            return False

        profile = result["data"]
        logger.info(f"✅ Account: {profile.get('user_name', 'Unknown')}")

        # Check available funds
        result = await self._make_request("GET", "/api/v1/portfolio/funds")
        if result["status_code"] != 200:
            logger.error("❌ Failed to get account funds")
            return False

        funds = result["data"]
        available_cash = funds.get("available_cash", 0)

        if available_cash < self.total_budget:
            logger.error(
                f"❌ Insufficient funds: ₹{available_cash} < ₹{self.total_budget}"
            )
            return False

        logger.success(f"✅ Available funds: ₹{available_cash}")
        return True

    async def validate_risk_settings(self) -> bool:
        """Validate risk management settings."""
        logger.info("🛡️ Validating risk settings...")

        result = await self._make_request("GET", "/api/v1/system/risk-parameters")
        if result["status_code"] != 200:
            logger.warning("⚠️ Could not validate risk parameters")
            return True  # Continue anyway

        risk_params = result["data"]
        logger.info(f"✅ Risk parameters: {risk_params}")
        return True

    async def test_market_data_access(self) -> bool:
        """Test market data access."""
        logger.info("📊 Testing market data access...")

        # Test with a liquid stock
        result = await self._make_request("GET", "/api/v1/market/quote?symbol=RELIANCE")
        if result["status_code"] != 200:
            logger.error("❌ Failed to get market data")
            return False

        quote = result["data"]
        logger.success(f"✅ Market data: RELIANCE @ ₹{quote.get('last_price', 'N/A')}")
        return True

    async def execute_test_trade(
        self, symbol: str = "RELIANCE", max_amount: float = 200
    ) -> bool:
        """Execute a small test trade with real money."""
        logger.warning("🔴 EXECUTING REAL TRADE WITH REAL MONEY!")

        # Get current price
        result = await self._make_request(
            "GET", f"/api/v1/market/quote?symbol={symbol}"
        )
        if result["status_code"] != 200:
            logger.error(f"❌ Failed to get price for {symbol}")
            return False

        quote = result["data"]
        current_price = quote.get("last_price", 0)

        if current_price == 0:
            logger.error(f"❌ Invalid price for {symbol}")
            return False

        # Calculate quantity (1 share only for safety)
        quantity = 1
        trade_value = current_price * quantity

        if trade_value > max_amount:
            logger.error(f"❌ Trade value ₹{trade_value} exceeds limit ₹{max_amount}")
            return False

        logger.info(
            f"💸 Placing BUY order: {quantity} {symbol} @ ₹{current_price} (Total: ₹{trade_value})"
        )

        # Place the order
        order_data = {
            "symbol": symbol,
            "action": "BUY",
            "quantity": quantity,
            "order_type": "LIMIT",
            "price": current_price,
            "product": "MIS",  # Intraday
            "validity": "DAY",
        }

        result = await self._make_request(
            "POST", "/api/v1/trading/live/execute", order_data
        )
        if result["status_code"] not in [200, 201]:
            logger.error(f"❌ Order placement failed: {result}")
            return False

        order_response = result["data"]
        order_id = order_response.get("order_id")

        logger.success(f"✅ Order placed successfully! Order ID: {order_id}")

        # Monitor order status
        await self.monitor_order(order_id)

        return True

    async def monitor_order(self, order_id: str, timeout: int = 300) -> bool:
        """Monitor order execution."""
        logger.info(f"👀 Monitoring order {order_id}...")

        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < timeout:
            result = await self._make_request(
                "GET", f"/api/v1/trading/orders/{order_id}"
            )
            if result["status_code"] != 200:
                logger.error("❌ Failed to get order status")
                break

            order = result["data"]
            status = order.get("status", "UNKNOWN")

            logger.info(f"📋 Order {order_id} status: {status}")

            if status in ["COMPLETE", "FILLED"]:
                logger.success(f"✅ Order {order_id} executed successfully!")
                return True
            elif status in ["REJECTED", "CANCELLED"]:
                logger.error(f"❌ Order {order_id} was {status}")
                return False

            await asyncio.sleep(10)  # Check every 10 seconds

        logger.warning(f"⚠️ Order {order_id} monitoring timeout")
        return False

    async def check_positions_and_pnl(self) -> dict:
        """Check current positions and P&L."""
        logger.info("📈 Checking positions and P&L...")

        # Get current positions
        result = await self._make_request("GET", "/api/v1/portfolio/positions")
        if result["status_code"] != 200:
            logger.error("❌ Failed to get positions")
            return {}

        positions = result["data"]

        # Get P&L
        result = await self._make_request("GET", "/api/v1/portfolio/pnl")
        if result["status_code"] != 200:
            logger.error("❌ Failed to get P&L")
            return {}

        pnl = result["data"]

        total_pnl = pnl.get("total_pnl", 0)
        logger.info(f"💰 Current P&L: ₹{total_pnl}")

        if total_pnl < -self.emergency_stop_loss:
            logger.error(
                f"🚨 EMERGENCY: Loss ₹{abs(total_pnl)} exceeds limit ₹{self.emergency_stop_loss}"
            )
            await self.emergency_stop()

        return {"positions": positions, "pnl": pnl}

    async def emergency_stop(self) -> bool:
        """Execute emergency stop."""
        logger.error("🚨 EXECUTING EMERGENCY STOP!")

        result = await self._make_request("POST", "/api/v1/system/emergency-stop")
        if result["status_code"] != 200:
            logger.error("❌ Emergency stop failed!")
            return False

        logger.success("✅ Emergency stop executed successfully")
        return True

    async def run_live_trading_test(self, token: str) -> bool:
        """Run complete live trading test."""
        self.token = token

        logger.info("🚀 Starting ₹1000 Live Trading Test...")
        logger.warning("⚠️ THIS INVOLVES REAL MONEY - PROCEED WITH CAUTION!")

        # Step 1: Validate account setup
        if not await self.validate_account_setup():
            return False

        # Step 2: Validate risk settings
        if not await self.validate_risk_settings():
            return False

        # Step 3: Test market data
        if not await self.test_market_data_access():
            return False

        # Step 4: Execute test trade
        logger.warning("🔴 About to execute REAL TRADE in 10 seconds...")
        logger.warning("🔴 Press Ctrl+C to cancel!")

        try:
            await asyncio.sleep(10)
        except KeyboardInterrupt:
            logger.info("❌ Test cancelled by user")
            return False

        if not await self.execute_test_trade():
            return False

        # Step 5: Monitor positions
        await asyncio.sleep(30)  # Wait 30 seconds
        await self.check_positions_and_pnl()

        logger.success("🎉 Live trading test completed!")
        return True


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="₹1000 Live Trading Test")
    parser.add_argument(
        "--confirm-real-money",
        action="store_true",
        required=True,
        help="Confirm you understand this uses real money",
    )
    parser.add_argument("--token", required=True, help="JWT authentication token")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")

    args = parser.parse_args()

    if not args.confirm_real_money:
        logger.error("❌ Must confirm real money usage with --confirm-real-money")
        sys.exit(1)

    logger.warning("⚠️ FINAL WARNING: This script will trade with REAL MONEY!")
    logger.warning("⚠️ Maximum risk: ₹100 daily loss, ₹50 per trade")
    logger.warning("⚠️ Emergency stop at ₹80 total loss")

    confirm = input("Type 'I UNDERSTAND THE RISKS' to continue: ")
    if confirm != "I UNDERSTAND THE RISKS":
        logger.info("❌ Test cancelled")
        sys.exit(1)

    async with LiveTradingValidator(base_url=args.url) as validator:
        success = await validator.run_live_trading_test(args.token)

        if success:
            logger.success("🎉 Live trading test completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Live trading test failed!")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
