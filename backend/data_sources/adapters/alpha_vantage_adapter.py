"""
Alpha Vantage adapter for the multi-source data manager.
Example implementation for backup market data source.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any

import aiohttp

from ..base import (
    DataSourceConfig,
    DataSourceHealth,
    DataSourceStatus,
    MarketDataSource,
)


class AlphaVantageMarketDataSource(MarketDataSource):
    """Alpha Vantage implementation of MarketDataSource."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://www.alphavantage.co/query"

    async def connect(self) -> bool:
        """Connect to Alpha Vantage API."""
        try:
            self.update_health_status(DataSourceStatus.CONNECTING)

            # Create aiohttp session
            self.session = aiohttp.ClientSession()

            # Validate API key with a simple query
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "RELIANCE.BSE",
                "apikey": self.config.api_key,
            }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "Error Message" in data or "Note" in data:
                        raise Exception(data.get("Error Message", data.get("Note")))

                    self._is_connected = True
                    self.update_health_status(DataSourceStatus.HEALTHY)
                    self.logger.info("Connected to Alpha Vantage")
                    return True
                else:
                    raise Exception(f"API returned status {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to connect to Alpha Vantage: {e}")
            self.update_health_status(DataSourceStatus.DISCONNECTED, str(e))
            if self.session:
                await self.session.close()
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpha Vantage."""
        try:
            self._is_connected = False

            if self.session:
                await self.session.close()
                self.session = None

            self.update_health_status(DataSourceStatus.DISCONNECTED)
            self.logger.info("Disconnected from Alpha Vantage")

        except Exception as e:
            self.logger.error(f"Error disconnecting from Alpha Vantage: {e}")

    async def health_check(self) -> DataSourceHealth:
        """Check Alpha Vantage API health."""
        try:
            if not self.session:
                self.update_health_status(DataSourceStatus.DISCONNECTED)
                return self.health

            # Simple health check with lightweight endpoint
            start_time = asyncio.get_event_loop().time()

            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "RELIANCE.BSE",
                "apikey": self.config.api_key,
            }

            async with self.session.get(
                self.base_url, params=params, timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                latency = (asyncio.get_event_loop().time() - start_time) * 1000

                if response.status == 200:
                    data = await response.json()
                    if "Global Quote" in data:
                        self.record_request_metrics(True, latency)
                        self.update_health_status(DataSourceStatus.HEALTHY)
                    else:
                        # Rate limited or error
                        self.update_health_status(DataSourceStatus.DEGRADED)
                else:
                    self.record_request_metrics(False, latency)
                    self.update_health_status(DataSourceStatus.UNHEALTHY)

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))

        return self.health

    async def validate_credentials(self) -> bool:
        """Validate Alpha Vantage API credentials."""
        try:
            if not self.session or not self.config.api_key:
                return False

            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "IBM",
                "apikey": self.config.api_key,
            }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return "Global Quote" in data

            return False

        except Exception as e:
            self.logger.error(f"Credential validation failed: {e}")
            return False

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get current quote for a symbol."""
        return await self.execute_with_retry(self._get_quote, symbol)

    async def _get_quote(self, symbol: str) -> dict[str, Any]:
        """Internal method to get quote."""
        if not self.session:
            raise Exception("Not connected to Alpha Vantage")

        # Convert symbol to Alpha Vantage format
        av_symbol = self._convert_symbol(symbol)

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": av_symbol,
            "apikey": self.config.api_key,
        }

        async with self.session.get(self.base_url, params=params) as response:
            data = await response.json()

            if "Global Quote" not in data:
                raise Exception(f"Invalid response: {data}")

            quote_data = data["Global Quote"]
            return self._normalize_quote(symbol, quote_data)

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get quotes for multiple symbols."""
        # Alpha Vantage doesn't support batch quotes, so we need to fetch individually
        # with rate limiting
        results = {}

        for symbol in symbols:
            try:
                quote = await self.get_quote(symbol)
                results[symbol] = quote

                # Rate limiting - Alpha Vantage free tier is 5 calls/minute
                if len(symbols) > 1:
                    await asyncio.sleep(
                        12
                    )  # 5 calls per minute = 12 seconds between calls

            except Exception as e:
                self.logger.error(f"Failed to get quote for {symbol}: {e}")

        return results

    async def get_historical_data(
        self, symbol: str, interval: str, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """Get historical data for a symbol."""
        return await self.execute_with_retry(
            self._get_historical_data, symbol, interval, from_date, to_date
        )

    async def _get_historical_data(
        self, symbol: str, interval: str, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """Internal method to get historical data."""
        if not self.session:
            raise Exception("Not connected to Alpha Vantage")

        av_symbol = self._convert_symbol(symbol)
        function, av_interval = self._get_function_and_interval(interval)

        params = {
            "function": function,
            "symbol": av_symbol,
            "apikey": self.config.api_key,
            "outputsize": "full",
        }

        if av_interval:
            params["interval"] = av_interval

        async with self.session.get(self.base_url, params=params) as response:
            data = await response.json()

            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break

            if not time_series_key:
                raise Exception(f"No time series data in response: {data}")

            time_series = data[time_series_key]

            # Convert to standard format and filter by date range
            candles = []
            for timestamp_str, candle_data in time_series.items():
                timestamp = datetime.fromisoformat(timestamp_str)

                if from_date <= timestamp <= to_date:
                    candles.append(self._normalize_candle(timestamp, candle_data))

            # Sort by timestamp
            candles.sort(key=lambda x: x["timestamp"])
            return candles

    async def get_market_depth(self, symbol: str) -> dict[str, Any]:
        """Get market depth - not available in Alpha Vantage."""
        # Alpha Vantage doesn't provide order book data
        return {
            "bids": [],
            "asks": [],
            "error": "Market depth not available from Alpha Vantage",
        }

    async def subscribe_live_data(
        self, symbols: list[str], callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Alpha Vantage doesn't support WebSocket streaming."""
        raise NotImplementedError("Alpha Vantage does not support live streaming data")

    async def unsubscribe_live_data(self, symbols: list[str]) -> None:
        """Alpha Vantage doesn't support WebSocket streaming."""
        pass

    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol to Alpha Vantage format."""
        # For Indian stocks, append .BSE or .NSE
        if ":" in symbol:
            exchange, sym = symbol.split(":")
            if exchange in ["NSE", "BSE"]:
                return f"{sym}.{exchange}"
            return sym

        # Default to BSE for Indian stocks
        return f"{symbol}.BSE"

    def _get_function_and_interval(self, interval: str) -> tuple:
        """Get Alpha Vantage function and interval."""
        if interval in ["1min", "5min", "15min", "30min", "60min"]:
            return "TIME_SERIES_INTRADAY", interval
        elif interval == "day":
            return "TIME_SERIES_DAILY", None
        elif interval == "week":
            return "TIME_SERIES_WEEKLY", None
        elif interval == "month":
            return "TIME_SERIES_MONTHLY", None
        else:
            return "TIME_SERIES_DAILY", None

    def _normalize_quote(self, symbol: str, quote_data: dict) -> dict[str, Any]:
        """Normalize Alpha Vantage quote to standard format."""
        return {
            "symbol": symbol,
            "last_price": float(quote_data.get("05. price", 0)),
            "open": float(quote_data.get("02. open", 0)),
            "high": float(quote_data.get("03. high", 0)),
            "low": float(quote_data.get("04. low", 0)),
            "close": float(quote_data.get("05. price", 0)),
            "volume": int(quote_data.get("06. volume", 0)),
            "change": float(quote_data.get("09. change", 0)),
            "change_percent": quote_data.get("10. change percent", "0%").rstrip("%"),
            "timestamp": datetime.fromisoformat(
                quote_data.get("07. latest trading day", "")
            ),
        }

    def _normalize_candle(
        self, timestamp: datetime, candle_data: dict
    ) -> dict[str, Any]:
        """Normalize Alpha Vantage candle to standard format."""
        return {
            "timestamp": timestamp,
            "open": float(candle_data.get("1. open", 0)),
            "high": float(candle_data.get("2. high", 0)),
            "low": float(candle_data.get("3. low", 0)),
            "close": float(candle_data.get("4. close", 0)),
            "volume": int(candle_data.get("5. volume", 0)),
        }
