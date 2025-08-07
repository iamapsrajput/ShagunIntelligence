#!/usr/bin/env python3
"""
API Key Validation Script for Shagun Intelligence Trading Platform

This script validates all configured API keys and tests connectivity
to external services to ensure the platform is properly configured.
"""

import asyncio
import os
import sys

import httpx

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_settings


class APIKeyValidator:
    """Validates API keys and service connectivity."""

    def __init__(self):
        self.settings = get_settings()
        self.results: dict[str, dict[str, any]] = {}

    async def validate_openai(self) -> tuple[bool, str]:
        """Validate OpenAI API key."""
        if not self.settings.OPENAI_API_KEY:
            return False, "API key not configured"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.settings.OPENAI_API_KEY}"},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    models = response.json()
                    gpt4_available = any(
                        "gpt-4" in model["id"] for model in models["data"]
                    )
                    return True, f"✅ Connected. GPT-4 available: {gpt4_available}"
                else:
                    return False, f"❌ HTTP {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"❌ Connection failed: {str(e)}"

    async def validate_alpha_vantage(self) -> tuple[bool, str]:
        """Validate Alpha Vantage API key."""
        if not self.settings.ALPHA_VANTAGE_API_KEY:
            return False, "API key not configured (optional)"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={self.settings.ALPHA_VANTAGE_API_KEY}",
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    if "Global Quote" in data:
                        return True, "✅ Connected and working"
                    elif "Note" in data:
                        return False, "❌ Rate limit exceeded"
                    else:
                        return False, f"❌ Unexpected response: {data}"
                else:
                    return False, f"❌ HTTP {response.status_code}"
        except Exception as e:
            return False, f"❌ Connection failed: {str(e)}"

    async def validate_finnhub(self) -> tuple[bool, str]:
        """Validate Finnhub API key."""
        finnhub_key = self.settings.FINNHUB_API_KEY
        if not finnhub_key:
            return False, "API key not configured (optional)"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={finnhub_key}",
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    if "c" in data:  # Current price
                        return True, "✅ Connected and working"
                    else:
                        return False, f"❌ Invalid response: {data}"
                else:
                    return False, f"❌ HTTP {response.status_code}"
        except Exception as e:
            return False, f"❌ Connection failed: {str(e)}"

    async def validate_newsapi(self) -> tuple[bool, str]:
        """Validate NewsAPI key."""
        newsapi_key = self.settings.NEWS_API_KEY
        if not newsapi_key:
            return False, "API key not configured (optional)"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey={newsapi_key}",
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok":
                        return (
                            True,
                            f"✅ Connected. Articles: {len(data.get('articles', []))}",
                        )
                    else:
                        return (
                            False,
                            f"❌ API error: {data.get('message', 'Unknown error')}",
                        )
                else:
                    return False, f"❌ HTTP {response.status_code}"
        except Exception as e:
            return False, f"❌ Connection failed: {str(e)}"

    async def validate_kite_connect(self) -> tuple[bool, str]:
        """Validate Zerodha Kite Connect credentials."""
        if not all([self.settings.KITE_API_KEY, self.settings.KITE_API_SECRET]):
            return False, "API credentials not configured"

        # Note: Full validation requires access token which expires daily
        # This just checks if credentials are provided
        if self.settings.KITE_ACCESS_TOKEN:
            return True, "✅ Credentials configured (access token provided)"
        else:
            return (
                False,
                "⚠️ Credentials configured but access token missing (expires daily)",
            )

    async def validate_database(self) -> tuple[bool, str]:
        """Validate database connectivity."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(self.settings.DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True, "✅ Database connection successful"
        except Exception as e:
            return False, f"❌ Database connection failed: {str(e)}"

    async def validate_redis(self) -> tuple[bool, str]:
        """Validate Redis connectivity."""
        try:
            import redis

            r = redis.from_url(self.settings.REDIS_URL)
            r.ping()
            return True, "✅ Redis connection successful"
        except Exception as e:
            return False, f"❌ Redis connection failed: {str(e)}"

    async def run_all_validations(self) -> dict[str, dict[str, any]]:
        """Run all API validations."""
        validations = {
            "OpenAI (Required)": self.validate_openai(),
            "Zerodha Kite Connect (Required)": self.validate_kite_connect(),
            "Alpha Vantage (Optional)": self.validate_alpha_vantage(),
            "Finnhub (Optional)": self.validate_finnhub(),
            "NewsAPI (Optional)": self.validate_newsapi(),
            "Database": self.validate_database(),
            "Redis": self.validate_redis(),
        }

        results = {}
        for name, validation_coro in validations.items():
            try:
                success, message = await validation_coro
                results[name] = {"success": success, "message": message}
                status = "✅" if success else "❌"
                print(f"{status} {name}: {message}")
            except Exception as e:
                results[name] = {
                    "success": False,
                    "message": f"Validation error: {str(e)}",
                }
                print(f"❌ {name}: Validation error: {str(e)}")

        return results

    def print_summary(self, results: dict[str, dict[str, any]]):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("🔍 API KEY VALIDATION SUMMARY")
        print("=" * 60)

        required_services = [
            "OpenAI (Required)",
            "Zerodha Kite Connect (Required)",
            "Database",
        ]
        optional_services = [k for k in results.keys() if k not in required_services]

        # Check required services
        required_ok = all(
            results[service]["success"]
            for service in required_services
            if service in results
        )

        print(
            f"\n📋 Required Services: {'✅ ALL OK' if required_ok else '❌ ISSUES FOUND'}"
        )
        for service in required_services:
            if service in results:
                status = "✅" if results[service]["success"] else "❌"
                print(f"  {status} {service}")

        print("\n🔧 Optional Services:")
        for service in optional_services:
            status = "✅" if results[service]["success"] else "⚠️"
            print(f"  {status} {service}")

        if required_ok:
            print("\n🚀 Platform is ready for trading!")
        else:
            print("\n⚠️  Please configure required services before trading.")

        print("\n📖 For setup instructions, see: docs/API_KEYS_GUIDE.md")


async def main():
    """Main validation function."""
    print("🔑 Shagun Intelligence API Key Validator")
    print("=" * 50)

    validator = APIKeyValidator()
    results = await validator.run_all_validations()
    validator.print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
