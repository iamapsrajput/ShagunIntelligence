"""
Alpha Vantage real-time streaming handler.
"""

import asyncio
from datetime import datetime
from typing import Any

import aiohttp
from loguru import logger

from ..realtime_pipeline import (
    DataStreamHandler,
    StreamConfig,
    StreamMessage,
    StreamStatus,
)


class AlphaVantageStreamHandler(DataStreamHandler):
    """Handler for Alpha Vantage real-time data streaming."""

    def __init__(self, config: StreamConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.session = None
        self.polling_tasks = {}  # symbol -> task mapping
        self.last_prices = {}  # symbol -> last price for change detection

        # Alpha Vantage specific settings
        self.base_url = "https://www.alphavantage.co/query"
        self.polling_interval = 5  # seconds (respecting rate limits)

    async def connect(self) -> bool:
        """Initialize HTTP session for Alpha Vantage."""
        try:
            self.status = StreamStatus.CONNECTING

            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Test connection with a simple quote
            test_response = await self._fetch_quote("AAPL")

            if test_response:
                self.status = StreamStatus.CONNECTED
                self.metrics.last_message_time = datetime.now()
                logger.info("Connected to Alpha Vantage API")
                return True
            else:
                self.status = StreamStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Alpha Vantage: {e}")
            self.status = StreamStatus.ERROR
            self.metrics.errors_count += 1
            return False

    async def disconnect(self):
        """Close the HTTP session."""
        try:
            # Cancel all polling tasks
            for task in self.polling_tasks.values():
                task.cancel()
            self.polling_tasks.clear()

            # Close session
            if self.session:
                await self.session.close()

            self.status = StreamStatus.DISCONNECTED
            logger.info("Disconnected from Alpha Vantage")

        except Exception as e:
            logger.error(f"Error disconnecting from Alpha Vantage: {e}")

    async def subscribe(self, symbols: list[str]):
        """Start polling for the given symbols."""
        try:
            for symbol in symbols:
                if symbol not in self.polling_tasks:
                    # Start polling task for this symbol
                    task = asyncio.create_task(self._poll_symbol(symbol))
                    self.polling_tasks[symbol] = task
                    logger.info(f"Started polling for {symbol}")

        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
            self.metrics.errors_count += 1

    async def process_message(self, message: Any) -> StreamMessage | None:
        """Process is handled by polling tasks."""
        # This method is not used for Alpha Vantage
        # Data is pushed directly from polling tasks
        return None

    async def send_heartbeat(self):
        """Check if polling tasks are healthy."""
        self.last_heartbeat = datetime.now()

        # Check and restart dead tasks
        for symbol, task in list(self.polling_tasks.items()):
            if task.done():
                logger.warning(f"Restarting polling task for {symbol}")
                new_task = asyncio.create_task(self._poll_symbol(symbol))
                self.polling_tasks[symbol] = new_task

    async def _poll_symbol(self, symbol: str):
        """Poll for real-time data for a specific symbol."""
        consecutive_errors = 0

        while self.status == StreamStatus.CONNECTED:
            try:
                start_time = datetime.now()

                # Fetch quote data
                quote_data = await self._fetch_quote(symbol)

                if quote_data:
                    # Calculate latency
                    latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                    # Check for price changes
                    current_price = quote_data.get("price", 0)
                    has_changed = True

                    if symbol in self.last_prices:
                        has_changed = current_price != self.last_prices[symbol]

                    self.last_prices[symbol] = current_price

                    # Only create message if data has changed
                    if has_changed:
                        # Update metrics
                        self.metrics.messages_received += 1
                        self.metrics.messages_processed += 1
                        self.metrics.last_message_time = datetime.now()
                        self.metrics.average_latency_ms = (
                            self.metrics.average_latency_ms
                            * (self.metrics.messages_processed - 1)
                            + latency_ms
                        ) / self.metrics.messages_processed

                        # Calculate quality
                        quality_score = self.calculate_quality_score(latency_ms)

                        # Create stream message
                        StreamMessage(
                            stream_name=self.config.name,
                            symbol=symbol,
                            data=quote_data,
                            timestamp=datetime.now(),
                            latency_ms=latency_ms,
                            quality_score=quality_score,
                        )

                        # This would normally be sent to the pipeline
                        # In practice, you'd use a callback or queue
                        logger.debug(
                            f"Alpha Vantage data for {symbol}: ${current_price}"
                        )

                    consecutive_errors = 0

                else:
                    consecutive_errors += 1
                    self.metrics.errors_count += 1

                    if consecutive_errors > 5:
                        logger.error(f"Too many errors polling {symbol}, stopping")
                        break

                # Wait before next poll
                await asyncio.sleep(self.polling_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error polling {symbol}: {e}")
                self.metrics.errors_count += 1
                consecutive_errors += 1

                if consecutive_errors > 5:
                    break

                await asyncio.sleep(self.polling_interval * 2)  # Back off on error

    async def _fetch_quote(self, symbol: str) -> dict[str, Any] | None:
        """Fetch quote data from Alpha Vantage."""
        if not self.session:
            return None

        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key,
            }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Check for rate limit error
                    if "Note" in data or "Error Message" in data:
                        logger.warning(f"Alpha Vantage API limit or error: {data}")
                        return None

                    # Extract quote data
                    quote = data.get("Global Quote", {})

                    if quote:
                        return {
                            "symbol": quote.get("01. symbol", symbol),
                            "price": float(quote.get("05. price", 0)),
                            "open": float(quote.get("02. open", 0)),
                            "high": float(quote.get("03. high", 0)),
                            "low": float(quote.get("04. low", 0)),
                            "volume": int(quote.get("06. volume", 0)),
                            "latest_trading_day": quote.get("07. latest trading day"),
                            "previous_close": float(quote.get("08. previous close", 0)),
                            "change": float(quote.get("09. change", 0)),
                            "change_percent": quote.get("10. change percent", "0%"),
                        }

                    return None

                else:
                    logger.error(f"Alpha Vantage API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage quote: {e}")
            return None
