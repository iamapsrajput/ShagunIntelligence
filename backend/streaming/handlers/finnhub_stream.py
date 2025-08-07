"""
Finnhub WebSocket streaming handler for real-time market data and news.
"""

import asyncio
import json
from datetime import datetime
from typing import Any

import websockets
from loguru import logger

from ..realtime_pipeline import (
    DataStreamHandler,
    StreamConfig,
    StreamMessage,
    StreamStatus,
)


class FinnhubStreamHandler(DataStreamHandler):
    """Handler for Finnhub WebSocket streaming."""

    def __init__(self, config: StreamConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.ws_url = f"wss://ws.finnhub.io?token={api_key}"
        self.subscribed_symbols = set()
        self.message_buffer = asyncio.Queue(maxsize=1000)
        self.processing_task = None

    async def connect(self) -> bool:
        """Connect to Finnhub WebSocket."""
        try:
            self.status = StreamStatus.CONNECTING

            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.ws_url, ping_interval=20, ping_timeout=10
            )

            # Start message processing
            self.processing_task = asyncio.create_task(self._process_messages())

            self.status = StreamStatus.CONNECTED
            self.metrics.last_message_time = datetime.now()
            logger.info("Connected to Finnhub WebSocket")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Finnhub: {e}")
            self.status = StreamStatus.ERROR
            self.metrics.errors_count += 1
            return False

    async def disconnect(self):
        """Disconnect from Finnhub WebSocket."""
        try:
            if self.processing_task:
                self.processing_task.cancel()

            if self.websocket:
                await self.websocket.close()

            self.status = StreamStatus.DISCONNECTED
            logger.info("Disconnected from Finnhub WebSocket")

        except Exception as e:
            logger.error(f"Error disconnecting from Finnhub: {e}")

    async def subscribe(self, symbols: list[str]):
        """Subscribe to symbols on Finnhub."""
        try:
            for symbol in symbols:
                if symbol not in self.subscribed_symbols:
                    # Subscribe to trades
                    subscribe_msg = {"type": "subscribe", "symbol": symbol}
                    await self.websocket.send(json.dumps(subscribe_msg))

                    # Subscribe to news
                    news_msg = {"type": "subscribe", "symbol": f"news:{symbol}"}
                    await self.websocket.send(json.dumps(news_msg))

                    self.subscribed_symbols.add(symbol)
                    logger.info(f"Subscribed to {symbol} on Finnhub")

        except Exception as e:
            logger.error(f"Error subscribing to Finnhub symbols: {e}")
            self.metrics.errors_count += 1

    async def process_message(self, message: Any) -> StreamMessage | None:
        """Process a message from Finnhub."""
        try:
            # Get message from buffer
            if not self.message_buffer.empty():
                return await self.message_buffer.get()
            return None

        except Exception as e:
            logger.error(f"Error processing Finnhub message: {e}")
            self.metrics.errors_count += 1
            return None

    async def send_heartbeat(self):
        """Send ping to keep connection alive."""
        try:
            if self.websocket:
                await self.websocket.ping()
                self.last_heartbeat = datetime.now()
        except Exception as e:
            logger.error(f"Error sending Finnhub heartbeat: {e}")
            self.status = StreamStatus.ERROR

    async def _process_messages(self):
        """Process incoming WebSocket messages."""
        while self.status == StreamStatus.CONNECTED:
            try:
                # Receive message
                message = await self.websocket.recv()
                data = json.loads(message)

                # Process based on message type
                if data.get("type") == "trade":
                    await self._process_trade(data)
                elif data.get("type") == "news":
                    await self._process_news(data)
                elif data.get("type") == "ping":
                    # Respond to ping
                    await self.websocket.send(json.dumps({"type": "pong"}))

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Finnhub WebSocket connection closed")
                self.status = StreamStatus.DISCONNECTED
                break
            except Exception as e:
                logger.error(f"Error in Finnhub message processing: {e}")
                self.metrics.errors_count += 1

    async def _process_trade(self, data: dict[str, Any]):
        """Process trade data from Finnhub."""
        try:
            receive_time = datetime.now()

            # Extract trade data
            trades = data.get("data", [])

            for trade in trades:
                symbol = trade.get("s")
                if not symbol:
                    continue

                # Calculate latency
                trade_time = trade.get("t", 0) / 1000  # Convert ms to seconds
                if trade_time > 0:
                    trade_datetime = datetime.fromtimestamp(trade_time)
                    latency_ms = (receive_time - trade_datetime).total_seconds() * 1000
                else:
                    latency_ms = 0

                # Update metrics
                self.metrics.messages_received += 1
                self.metrics.messages_processed += 1
                self.metrics.last_message_time = receive_time

                # Update average latency
                if latency_ms > 0:
                    self.metrics.average_latency_ms = (
                        self.metrics.average_latency_ms
                        * (self.metrics.messages_processed - 1)
                        + latency_ms
                    ) / self.metrics.messages_processed

                # Calculate quality score
                quality_score = self.calculate_quality_score(latency_ms)

                # Create stream message
                message = StreamMessage(
                    stream_name=self.config.name,
                    symbol=symbol,
                    data={
                        "price": trade.get("p"),
                        "volume": trade.get("v"),
                        "timestamp": trade_time,
                        "conditions": trade.get("c", []),
                        "type": "trade",
                    },
                    timestamp=receive_time,
                    latency_ms=latency_ms,
                    quality_score=quality_score,
                )

                # Add to buffer
                if not self.message_buffer.full():
                    await self.message_buffer.put(message)

        except Exception as e:
            logger.error(f"Error processing Finnhub trade: {e}")

    async def _process_news(self, data: dict[str, Any]):
        """Process news data from Finnhub."""
        try:
            receive_time = datetime.now()

            # Extract news data
            news_items = data.get("data", [])

            for news in news_items:
                # Extract relevant symbols
                related_symbols = news.get("related", "").split(",")

                for symbol in related_symbols:
                    if symbol and symbol in self.subscribed_symbols:
                        # Create news message
                        message = StreamMessage(
                            stream_name=self.config.name,
                            symbol=symbol,
                            data={
                                "type": "news",
                                "headline": news.get("headline"),
                                "summary": news.get("summary"),
                                "source": news.get("source"),
                                "url": news.get("url"),
                                "datetime": news.get("datetime"),
                                "category": news.get("category"),
                                "id": news.get("id"),
                            },
                            timestamp=receive_time,
                            latency_ms=0,  # News doesn't have real-time latency
                            quality_score=1.0,  # News quality is based on source reliability
                        )

                        # Add to buffer
                        if not self.message_buffer.full():
                            await self.message_buffer.put(message)

        except Exception as e:
            logger.error(f"Error processing Finnhub news: {e}")
