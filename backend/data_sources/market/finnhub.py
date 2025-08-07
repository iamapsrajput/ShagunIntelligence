import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from loguru import logger

from backend.data_sources.base import (
    DataSourceConfig,
    DataSourceStatus,
    MarketDataSource,
)

from .models import (
    DataCostTier,
    DataSourceCost,
    HistoricalBar,
    MarketData,
    MarketDataQuality,
)


class FinnhubSource(MarketDataSource):
    """Finnhub - Real-time market data with built-in sentiment analysis"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.api_key = config.credentials.get("api_key")
        self.base_url = "https://finnhub.io/api/v1"
        self.ws_url = "wss://ws.finnhub.io"

        self.session: aiohttp.ClientSession | None = None
        self.ws_session: aiohttp.ClientWebSocketResponse | None = None

        # Cost information
        self.cost_info = DataSourceCost(
            tier=DataCostTier.FREE,  # Has free tier
            monthly_cost=0.0,  # Free tier
            per_request_cost=0.0,
            free_requests=60,  # 60 requests/min free
            requests_per_minute=60,
            includes_realtime=True,
            includes_historical=True,
            includes_forex=True,
            includes_crypto=True,
        )

        # Premium tiers
        self.premium_tiers = {
            "starter": DataSourceCost(
                tier=DataCostTier.LOW,
                monthly_cost=49.0,
                per_request_cost=0.0,
                free_requests=0,
                requests_per_minute=300,
                includes_realtime=True,
            ),
            "growth": DataSourceCost(
                tier=DataCostTier.MEDIUM,
                monthly_cost=249.0,
                per_request_cost=0.0,
                free_requests=0,
                requests_per_minute=600,
                includes_realtime=True,
            ),
            "enterprise": DataSourceCost(
                tier=DataCostTier.HIGH,
                monthly_cost=999.0,
                per_request_cost=0.0,
                free_requests=0,
                requests_per_minute=None,  # Unlimited
                includes_realtime=True,
            ),
        }

        # Rate limiting
        self.request_times = []

        # Sentiment cache
        self.sentiment_cache = {}
        self.sentiment_cache_ttl = 300  # 5 minutes

        logger.info("Initialized FinnhubSource")

    async def connect(self) -> bool:
        """Connect to Finnhub API"""
        try:
            if not self.api_key:
                raise ValueError("Finnhub API key not provided")

            self.session = aiohttp.ClientSession()

            # Test connection
            url = f"{self.base_url}/quote"
            params = {"symbol": "AAPL", "token": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "error" not in data:
                        self.update_health_status(DataSourceStatus.HEALTHY)
                        logger.info("Connected to Finnhub")
                        return True
                    else:
                        raise Exception(f"API error: {data['error']}")
                elif response.status == 429:
                    raise Exception("Rate limit exceeded")
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Failed to connect to Finnhub: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from Finnhub"""
        if self.ws_session:
            await self.ws_session.close()
        if self.session:
            await self.session.close()
        self.update_health_status(DataSourceStatus.DISCONNECTED)
        logger.info("Disconnected from Finnhub")

    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        # Remove old request times
        self.request_times = [t for t in self.request_times if t > minute_ago]

        # Check limit
        if len(self.request_times) >= self.cost_info.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Record request
        self.request_times.append(now)

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            await self._check_rate_limit()

            url = f"{self.base_url}/quote"
            params = {"symbol": symbol, "token": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "error" in data:
                        raise Exception(f"API error: {data['error']}")

                    # Convert to standardized format
                    market_data = MarketData(
                        symbol=symbol,
                        exchange=self._detect_exchange(symbol),
                        timestamp=datetime.fromtimestamp(
                            data.get("t", datetime.utcnow().timestamp())
                        ),
                        source="finnhub",
                        current_price=float(data.get("c", 0)),
                        open=float(data.get("o", 0)),
                        high=float(data.get("h", 0)),
                        low=float(data.get("l", 0)),
                        close=float(data.get("c", 0)),
                        previous_close=float(data.get("pc", 0)),
                        change=float(data.get("d", 0)),
                        change_percent=float(data.get("dp", 0)),
                        data_quality=MarketDataQuality.PROFESSIONAL,
                        latency_ms=100,  # Good latency
                        is_delayed=False,
                    )

                    # Get sentiment if available
                    sentiment = await self.get_sentiment(symbol)
                    if sentiment:
                        market_data.sentiment_score = sentiment.get("score")
                        market_data.sentiment_buzz = sentiment.get("buzz")

                    return market_data.to_dict()
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching quote from Finnhub: {e}")
            raise

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get quotes for multiple symbols"""
        quotes = {}

        # Process in parallel with rate limiting
        batch_size = min(
            10, self.cost_info.requests_per_minute // 6
        )  # Leave room for other requests

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            tasks = [self.get_quote(symbol) for symbol in batch]

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for symbol, result in zip(batch, results, strict=False):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching {symbol}: {result}")
                        quotes[symbol] = None
                    else:
                        quotes[symbol] = result

                # Small delay between batches
                if i + batch_size < len(symbols):
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in batch quote fetch: {e}")

        return quotes

    async def get_sentiment(self, symbol: str) -> dict[str, Any] | None:
        """Get sentiment analysis for a symbol - Finnhub's unique feature"""
        try:
            # Check cache
            cache_key = f"{symbol}_sentiment"
            if cache_key in self.sentiment_cache:
                cached_time, cached_data = self.sentiment_cache[cache_key]
                if (datetime.utcnow() - cached_time).seconds < self.sentiment_cache_ttl:
                    return cached_data

            await self._check_rate_limit()

            url = f"{self.base_url}/news-sentiment"
            params = {"symbol": symbol, "token": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "error" not in data:
                        sentiment_data = {
                            "symbol": symbol,
                            "score": data.get("sentiment", {}).get("score", 0),
                            "buzz": data.get("buzz", {}).get("buzz", 0),
                            "articles_in_last_week": data.get("buzz", {}).get(
                                "articlesInLastWeek", 0
                            ),
                            "weekly_average": data.get("buzz", {}).get(
                                "weeklyAverage", 0
                            ),
                            "sector_average_sentiment": data.get(
                                "sectorAverageSentiment", 0
                            ),
                            "company_news_score": data.get("companyNewsScore", 0),
                            "sector_news_score": data.get("sectorNewsScore", 0),
                            "timestamp": datetime.utcnow().isoformat(),
                        }

                        # Cache result
                        self.sentiment_cache[cache_key] = (
                            datetime.utcnow(),
                            sentiment_data,
                        )

                        return sentiment_data

        except Exception as e:
            logger.error(f"Error fetching sentiment from Finnhub: {e}")
            return None

    async def get_company_news(
        self, symbol: str, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """Get company news with sentiment"""
        try:
            await self._check_rate_limit()

            url = f"{self.base_url}/company-news"
            params = {
                "symbol": symbol,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "token": self.api_key,
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    news_items = await response.json()

                    return [
                        {
                            "id": str(item.get("id")),
                            "symbol": symbol,
                            "headline": item.get("headline"),
                            "summary": item.get("summary"),
                            "url": item.get("url"),
                            "source": item.get("source"),
                            "datetime": datetime.fromtimestamp(
                                item.get("datetime", 0)
                            ).isoformat(),
                            "category": item.get("category"),
                            "related": item.get("related", symbol),
                            "image": item.get("image"),
                        }
                        for item in news_items[:50]  # Limit to 50 items
                    ]
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching company news: {e}")
            return []

    async def get_market_depth(self, symbol: str) -> dict[str, Any]:
        """Get market depth - limited in Finnhub"""
        # Finnhub doesn't provide full order book
        # Return basic bid/ask from quote
        try:
            quote = await self.get_quote(symbol)

            if quote:
                # Estimate depth based on typical spreads
                current_price = quote.get("current_price", 0)
                spread_percent = 0.0005  # 0.05% typical spread

                return {
                    "symbol": symbol,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "finnhub",
                    "bids": [
                        {"price": current_price * (1 - spread_percent), "size": 1000},
                        {
                            "price": current_price * (1 - spread_percent * 2),
                            "size": 2000,
                        },
                        {
                            "price": current_price * (1 - spread_percent * 3),
                            "size": 3000,
                        },
                    ],
                    "asks": [
                        {"price": current_price * (1 + spread_percent), "size": 1000},
                        {
                            "price": current_price * (1 + spread_percent * 2),
                            "size": 2000,
                        },
                        {
                            "price": current_price * (1 + spread_percent * 3),
                            "size": 3000,
                        },
                    ],
                    "note": "Estimated depth - full order book not available",
                }
            else:
                return {"bids": [], "asks": []}

        except Exception as e:
            logger.error(f"Error getting market depth: {e}")
            return {"bids": [], "asks": []}

    async def get_historical_data(
        self, symbol: str, interval: str, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """Get historical data for a symbol"""
        try:
            await self._check_rate_limit()

            # Map interval to Finnhub resolution
            resolution_map = {
                "minute": "1",
                "1min": "1",
                "5min": "5",
                "15min": "15",
                "30min": "30",
                "60min": "60",
                "hour": "60",
                "day": "D",
                "week": "W",
                "month": "M",
            }

            resolution = resolution_map.get(interval, "D")

            url = f"{self.base_url}/stock/candle"
            params = {
                "symbol": symbol,
                "resolution": resolution,
                "from": int(from_date.timestamp()),
                "to": int(to_date.timestamp()),
                "token": self.api_key,
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("s") == "ok":
                        bars = []

                        timestamps = data.get("t", [])
                        opens = data.get("o", [])
                        highs = data.get("h", [])
                        lows = data.get("l", [])
                        closes = data.get("c", [])
                        volumes = data.get("v", [])

                        for i in range(len(timestamps)):
                            bar = HistoricalBar(
                                timestamp=datetime.fromtimestamp(timestamps[i]),
                                open=opens[i],
                                high=highs[i],
                                low=lows[i],
                                close=closes[i],
                                volume=int(volumes[i]) if i < len(volumes) else 0,
                            )
                            bars.append(bar.to_dict())

                        return bars
                    else:
                        logger.warning(f"No data available for {symbol}")
                        return []
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching historical data from Finnhub: {e}")
            return []

    async def stream_quotes(self, symbols: list[str], callback: Any) -> None:
        """Stream real-time quotes via WebSocket"""
        try:
            # Connect to WebSocket
            ws_url = f"{self.ws_url}?token={self.api_key}"
            self.ws_session = await self.session.ws_connect(ws_url)

            # Subscribe to symbols
            for symbol in symbols:
                subscribe_msg = {"type": "subscribe", "symbol": symbol}
                await self.ws_session.send_json(subscribe_msg)

            # Listen for updates
            async for msg in self.ws_session:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if data.get("type") == "trade":
                        trades = data.get("data", [])

                        for trade in trades:
                            symbol = trade.get("s")

                            # Convert to market data
                            market_data = MarketData(
                                symbol=symbol,
                                exchange=self._detect_exchange(symbol),
                                timestamp=datetime.fromtimestamp(
                                    trade.get("t", 0) / 1000
                                ),
                                source="finnhub",
                                current_price=float(trade.get("p", 0)),
                                volume=int(trade.get("v", 0)),
                                data_quality=MarketDataQuality.INSTITUTIONAL,
                                latency_ms=50,  # Low latency WebSocket
                                is_delayed=False,
                            )

                            # Get latest sentiment
                            sentiment = await self.get_sentiment(symbol)
                            if sentiment:
                                market_data.sentiment_score = sentiment.get("score")

                            await callback(symbol, market_data.to_dict())

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break

        except Exception as e:
            logger.error(f"Error in quote streaming: {e}")
            raise

    async def get_economic_calendar(self) -> list[dict[str, Any]]:
        """Get economic calendar events"""
        try:
            await self._check_rate_limit()

            url = f"{self.base_url}/calendar/economic"
            params = {"token": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "economicCalendar" in data:
                        events = data["economicCalendar"]

                        return [
                            {
                                "event": event.get("event"),
                                "country": event.get("country"),
                                "currency": event.get("currency"),
                                "impact": event.get("impact"),
                                "forecast": event.get("forecast"),
                                "previous": event.get("previous"),
                                "actual": event.get("actual"),
                                "time": event.get("time"),
                                "unit": event.get("unit"),
                            }
                            for event in events[:100]  # Limit results
                        ]
                    else:
                        return []
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return []

    async def get_forex_rate(
        self, from_currency: str, to_currency: str
    ) -> dict[str, Any]:
        """Get forex exchange rate"""
        try:
            symbol = f"{from_currency}{to_currency}"
            quote = await self.get_quote(symbol)

            if quote:
                return {
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "exchange_rate": quote.get("current_price", 0),
                    "bid": quote.get("bid", quote.get("current_price", 0)),
                    "ask": quote.get("ask", quote.get("current_price", 0)),
                    "timestamp": quote.get("timestamp"),
                    "source": "finnhub",
                }
            else:
                raise Exception("Unable to fetch forex rate")

        except Exception as e:
            logger.error(f"Error fetching forex rate: {e}")
            raise

    async def get_crypto_quote(
        self, symbol: str, exchange: str = "BINANCE"
    ) -> dict[str, Any]:
        """Get cryptocurrency quote"""
        try:
            # Format symbol for crypto
            crypto_symbol = f"{exchange}:{symbol}"
            quote = await self.get_quote(crypto_symbol)

            if quote:
                # Add crypto-specific fields
                quote["exchange"] = exchange
                quote["is_crypto"] = True

                return quote
            else:
                raise Exception(f"Unable to fetch crypto quote for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching crypto quote: {e}")
            raise

    def _detect_exchange(self, symbol: str) -> str:
        """Detect exchange from symbol format"""
        if ":" in symbol:
            return symbol.split(":")[0]
        elif "." in symbol:
            return symbol.split(".")[-1]
        else:
            return "US"  # Default to US markets

    def get_cost_info(self) -> DataSourceCost:
        """Get cost information for this data source"""
        return self.cost_info

    def upgrade_tier(self, tier: str) -> DataSourceCost:
        """Get upgraded tier information"""
        return self.premium_tiers.get(tier, self.cost_info)

    def supports_market(self, market: str) -> bool:
        """Check if this source supports a market"""
        supported = ["US", "NYSE", "NASDAQ", "FOREX", "CRYPTO", "EU", "UK", "JP"]
        return market.upper() in supported

    def supports_sentiment(self) -> bool:
        """Check if this source supports sentiment analysis"""
        return True  # Finnhub has built-in sentiment
