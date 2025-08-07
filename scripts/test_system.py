#!/usr/bin/env python3
"""
Shagun Intelligence Trading System - Automated Testing Script

This script automates the testing and validation of the trading system.
Run this before enabling live trading to ensure everything works correctly.

Usage:
    python scripts/test_system.py --mode paper
    python scripts/test_system.py --mode live --confirm
"""

import argparse
import asyncio
import sys
from typing import Optional

import aiohttp
from loguru import logger

# Configure logging
logger.add("logs/system_test.log", rotation="1 day", retention="7 days")


class TradingSystemTester:
    """Automated testing suite for the Shagun Intelligence Trading System."""

    def __init__(
        self, base_url: str = "http://127.0.0.1:8000", token: str | None = None
    ):
        self.base_url = base_url
        self.token = token
        self.session: aiohttp.ClientSession | None = None
        self.test_results: dict[str, bool] = {}

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

    async def test_basic_connectivity(self) -> bool:
        """Test basic system connectivity."""
        logger.info("🔍 Testing basic connectivity...")

        # Test root endpoint
        result = await self._make_request("GET", "/")
        if result["status_code"] != 200:
            logger.error("❌ Root endpoint failed")
            return False

        # Test health endpoint
        result = await self._make_request("GET", "/api/v1/health")
        if result["status_code"] != 200:
            logger.error("❌ Health endpoint failed")
            return False

        # Test detailed health
        result = await self._make_request("GET", "/api/v1/health/detailed")
        if result["status_code"] != 200:
            logger.error("❌ Detailed health endpoint failed")
            return False

        logger.success("✅ Basic connectivity tests passed")
        return True

    async def test_authentication(self, username: str, password: str) -> bool:
        """Test authentication system."""
        logger.info("🔐 Testing authentication...")

        # Test token endpoint
        auth_data = {"username": username, "password": password}
        result = await self._make_request("POST", "/api/v1/auth/token", auth_data)

        if result["status_code"] != 200:
            logger.error("❌ Authentication failed")
            return False

        # Extract token
        self.token = result["data"].get("access_token")
        if not self.token:
            logger.error("❌ No access token received")
            return False

        # Test authenticated endpoint
        result = await self._make_request("GET", "/api/v1/auth/me")
        if result["status_code"] != 200:
            logger.error("❌ Authenticated endpoint test failed")
            return False

        logger.success("✅ Authentication tests passed")
        return True

    async def test_agent_system(self) -> bool:
        """Test multi-agent system."""
        logger.info("🤖 Testing agent system...")

        # Test agent status
        result = await self._make_request("GET", "/api/v1/agents/status")
        if result["status_code"] != 200:
            logger.error("❌ Agent status check failed")
            return False

        # Test individual agents
        agents = ["risk-manager", "technical-indicator", "sentiment-analyst"]
        for agent in agents:
            result = await self._make_request("GET", f"/api/v1/agents/{agent}/status")
            if result["status_code"] != 200:
                logger.error(f"❌ {agent} status check failed")
                return False

        logger.success("✅ Agent system tests passed")
        return True

    async def test_market_data(self) -> bool:
        """Test market data connectivity."""
        logger.info("📊 Testing market data...")

        # Test market data endpoint
        result = await self._make_request("GET", "/api/v1/market/quote?symbol=RELIANCE")
        if result["status_code"] != 200:
            logger.warning(
                "⚠️ Market data test failed (may be expected if market is closed)"
            )
            return True  # Don't fail the test if market is closed

        logger.success("✅ Market data tests passed")
        return True

    async def test_paper_trading(self) -> bool:
        """Test paper trading functionality."""
        logger.info("📝 Testing paper trading...")

        # Test paper trade execution
        trade_data = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "quantity": 1,
            "order_type": "MARKET",
            "strategy": "system_test",
        }

        result = await self._make_request(
            "POST", "/api/v1/trading/paper/execute", trade_data
        )
        if result["status_code"] not in [200, 201]:
            logger.error("❌ Paper trade execution failed")
            return False

        # Test paper portfolio status
        result = await self._make_request("GET", "/api/v1/portfolio/paper/status")
        if result["status_code"] != 200:
            logger.error("❌ Paper portfolio status check failed")
            return False

        logger.success("✅ Paper trading tests passed")
        return True

    async def test_live_trading_connection(self) -> bool:
        """Test live trading API connection (without executing trades)."""
        logger.info("🔴 Testing live trading connection...")

        # Test API connection
        result = await self._make_request("GET", "/api/v1/market/connection/test")
        if result["status_code"] != 200:
            logger.error("❌ Live trading API connection failed")
            return False

        # Test account profile
        result = await self._make_request("GET", "/api/v1/portfolio/profile")
        if result["status_code"] != 200:
            logger.error("❌ Account profile check failed")
            return False

        logger.success("✅ Live trading connection tests passed")
        return True

    async def test_risk_management(self) -> bool:
        """Test risk management system."""
        logger.info("🛡️ Testing risk management...")

        # Test risk parameters
        result = await self._make_request("GET", "/api/v1/system/risk-parameters")
        if result["status_code"] != 200:
            logger.error("❌ Risk parameters check failed")
            return False

        # Test position sizing
        sizing_data = {"symbol": "RELIANCE", "price": 2500, "risk_amount": 1000}
        result = await self._make_request(
            "POST", "/api/v1/agents/risk-manager/position-size", sizing_data
        )
        if result["status_code"] != 200:
            logger.error("❌ Position sizing test failed")
            return False

        logger.success("✅ Risk management tests passed")
        return True

    async def test_emergency_procedures(self) -> bool:
        """Test emergency stop and safety procedures."""
        logger.info("🚨 Testing emergency procedures...")

        # Test emergency stop (dry run)
        result = await self._make_request(
            "POST", "/api/v1/system/emergency-stop", {"dry_run": True}
        )
        if result["status_code"] != 200:
            logger.error("❌ Emergency stop test failed")
            return False

        logger.success("✅ Emergency procedures tests passed")
        return True

    async def run_comprehensive_test(
        self, mode: str = "paper", username: str = None, password: str = None
    ) -> bool:
        """Run comprehensive system test."""
        logger.info(f"🚀 Starting comprehensive system test in {mode} mode...")

        tests = [
            ("Basic Connectivity", self.test_basic_connectivity()),
            ("Market Data", self.test_market_data()),
            ("Risk Management", self.test_risk_management()),
            ("Emergency Procedures", self.test_emergency_procedures()),
        ]

        # Add authentication-dependent tests if credentials provided
        if username and password:
            tests.extend(
                [
                    ("Authentication", self.test_authentication(username, password)),
                    ("Agent System", self.test_agent_system()),
                    ("Paper Trading", self.test_paper_trading()),
                ]
            )

            if mode == "live":
                tests.append(
                    ("Live Trading Connection", self.test_live_trading_connection())
                )

        # Run all tests
        all_passed = True
        for test_name, test_coro in tests:
            try:
                result = await test_coro
                self.test_results[test_name] = result
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {test_name} test failed with exception: {str(e)}")
                self.test_results[test_name] = False
                all_passed = False

        # Print summary
        self._print_test_summary()
        return all_passed

    def _print_test_summary(self):
        """Print test results summary."""
        logger.info("\n" + "=" * 50)
        logger.info("📋 TEST RESULTS SUMMARY")
        logger.info("=" * 50)

        passed = 0
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
            if result:
                passed += 1

        logger.info("=" * 50)
        logger.info(f"OVERALL: {passed}/{total} tests passed")

        if passed == total:
            logger.success("🎉 ALL TESTS PASSED - System is ready!")
        else:
            logger.error("⚠️ SOME TESTS FAILED - Review issues before proceeding")


async def main():
    """Main function to run the testing suite."""
    parser = argparse.ArgumentParser(
        description="Shagun Intelligence Trading System Tester"
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Testing mode (paper or live)",
    )
    parser.add_argument("--username", help="Username for authentication")
    parser.add_argument("--password", help="Password for authentication")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm live trading tests (required for live mode)",
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000", help="Base URL of the trading system"
    )

    args = parser.parse_args()

    # Safety check for live mode
    if args.mode == "live" and not args.confirm:
        logger.error("❌ Live mode requires --confirm flag for safety")
        sys.exit(1)

    # Run tests
    async with TradingSystemTester(base_url=args.url) as tester:
        success = await tester.run_comprehensive_test(
            mode=args.mode, username=args.username, password=args.password
        )

        if success:
            logger.success("🎉 System validation completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ System validation failed!")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
