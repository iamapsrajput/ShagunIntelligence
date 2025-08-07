import asyncio
import json
from datetime import date, datetime, timedelta
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
    MarketDepth,
)


class PolygonSource(MarketDataSource):
    """Polygon.io - Institutional-grade market data with comprehensive coverage"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.api_key = config.credentials.get("api_key")
        self.base_url = "https://api.polygon.io"
        self.ws_url = "wss://socket.polygon.io"

        self.session: aiohttp.ClientSession | None = None
        self.ws_sessions: dict[str, aiohttp.ClientWebSocketResponse] = {}

        # Cost information - Polygon has multiple tiers
        self.cost_info = DataSourceCost(
            tier=DataCostTier.FREE,  # Basic tier
            monthly_cost=0.0,
            per_request_cost=0.0,
            free_requests=None,  # No hard limit, but rate limited
            requests_per_minute=5,  # Free tier limit
            includes_realtime=False,  # Delayed data only
            includes_historical=True,
            includes_options=False,
            includes_forex=True,
            includes_crypto=True,
        )

        # Premium tiers with institutional features
        self.premium_tiers = {
            "stocks_starter": DataSourceCost(
                tier=DataCostTier.MEDIUM,
                monthly_cost=199.0,
                per_request_cost=0.0,
                free_requests=None,
                requests_per_minute=100,
                includes_realtime=True,
                includes_historical=True,
                includes_options=False,
            ),
            "stocks_developer": DataSourceCost(
                tier=DataCostTier.HIGH,
                monthly_cost=799.0,
                per_request_cost=0.0,
                free_requests=None,
                requests_per_minute=1000,
                includes_realtime=True,
                includes_historical=True,
                includes_options=True,
            ),
            "stocks_advanced": DataSourceCost(
                tier=DataCostTier.PREMIUM,
                monthly_cost=1999.0,
                per_request_cost=0.0,
                free_requests=None,
                requests_per_minute=10000,
                includes_realtime=True,
                includes_historical=True,
                includes_options=True,
            ),
        }

        # Rate limiting
        self.request_times = []

        # WebSocket subscription management
        self.subscriptions = set()

        logger.info("Initialized PolygonSource")

    async def connect(self) -> bool:
        """Connect to Polygon API"""
        try:
            if not self.api_key:
                raise ValueError("Polygon API key not provided")

            self.session = aiohttp.ClientSession()

            # Test connection with account details
            url = f"{self.base_url}/v1/marketstatus/now"
            params = {"apiKey": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "status" in data:
                        self.update_health_status(DataSourceStatus.HEALTHY)
                        logger.info(
                            f"Connected to Polygon. Market status: {data.get('market')}"
                        )
                        return True
                    else:
                        raise Exception("Invalid response format")
                elif response.status == 401:
                    raise Exception("Invalid API key")
                elif response.status == 429:
                    raise Exception("Rate limit exceeded")
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Failed to connect to Polygon: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from Polygon"""
        # Close all WebSocket connections
        for ws in self.ws_sessions.values():
            await ws.close()
        self.ws_sessions.clear()

        if self.session:
            await self.session.close()

        self.update_health_status(DataSourceStatus.DISCONNECTED)
        logger.info("Disconnected from Polygon")

    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        # Remove old request times
        self.request_times = [t for t in self.request_times if t > minute_ago]

        # Check limit
        if (
            self.cost_info.requests_per_minute
            and len(self.request_times) >= self.cost_info.requests_per_minute
        ):
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

            # Get last trade
            url = f"{self.base_url}/v2/last/trade/{symbol}"
            params = {"apiKey": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "OK":
                        trade = data.get("results", {})

                        # Get aggregates for OHLC
                        agg_data = await self._get_daily_aggregate(symbol)

                        # Convert to standardized format
                        market_data = MarketData(
                            symbol=symbol,
                            exchange=trade.get("exchange", "US"),
                            timestamp=datetime.fromtimestamp(
                                trade.get("t", 0) / 1000000000
                            ),  # Nanoseconds
                            source="polygon",
                            current_price=float(trade.get("p", 0)),
                            volume=int(trade.get("s", 0)),
                            data_quality=MarketDataQuality.INSTITUTIONAL,
                            latency_ms=10,  # Ultra-low latency
                            is_delayed=self.cost_info.tier
                            == DataCostTier.FREE,  # Free tier is delayed
                        )

                        # Add OHLC from aggregates if available
                        if agg_data:
                            market_data.open = agg_data.get("o")
                            market_data.high = agg_data.get("h")
                            market_data.low = agg_data.get("l")
                            market_data.close = agg_data.get("c")
                            market_data.volume = agg_data.get("v")
                            market_data.vwap = agg_data.get("vw")

                        # Get quote data for bid/ask
                        quote_data = await self._get_last_quote(symbol)
                        if quote_data:
                            market_data.bid = quote_data.get("bid_price")
                            market_data.ask = quote_data.get("ask_price")
                            market_data.bid_size = quote_data.get("bid_size")
                            market_data.ask_size = quote_data.get("ask_size")

                        return market_data.to_dict()
                    else:
                        raise Exception(
                            f"API error: {data.get('message', 'Unknown error')}"
                        )
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching quote from Polygon: {e}")
            raise

    async def _get_last_quote(self, symbol: str) -> dict[str, Any] | None:
        """Get last quote (bid/ask) data"""
        try:
            url = f"{self.base_url}/v2/last/nbbo/{symbol}"
            params = {"apiKey": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "OK":
                        quote = data.get("results", {})
                        return {
                            "bid_price": float(quote.get("P", 0)),
                            "ask_price": float(quote.get("P", 0)),
                            "bid_size": int(quote.get("S", 0)),
                            "ask_size": int(quote.get("S", 0)),
                        }
            return None
        except:
            return None

    async def _get_daily_aggregate(self, symbol: str) -> dict[str, Any] | None:
        """Get daily aggregate data"""
        try:
            today = date.today()
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{today}/{today}"
            params = {"apiKey": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "OK" and data.get("results"):
                        return data["results"][0]
            return None
        except:
            return None

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get quotes for multiple symbols"""
        quotes = {}

        # Polygon supports grouped daily bars for efficiency
        if len(symbols) > 10:
            # Use grouped endpoint for large batches
            quotes = await self._get_grouped_quotes(symbols)
        else:
            # Get individual quotes for small batches
            tasks = [self.get_quote(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for symbol, result in zip(symbols, results, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching {symbol}: {result}")
                    quotes[symbol] = None
                else:
                    quotes[symbol] = result

        return quotes

    async def _get_grouped_quotes(
        self, symbols: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Get grouped daily bars for multiple symbols"""
        try:
            await self._check_rate_limit()

            today = date.today()
            url = f"{self.base_url}/v2/aggs/grouped/locale/us/market/stocks/{today}"
            params = {"apiKey": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "OK":
                        results = data.get("results", [])
                        quotes = {}

                        # Create a map for quick lookup
                        result_map = {r["T"]: r for r in results}

                        for symbol in symbols:
                            if symbol in result_map:
                                bar = result_map[symbol]

                                market_data = MarketData(
                                    symbol=symbol,
                                    exchange="US",
                                    timestamp=datetime.fromtimestamp(
                                        bar.get("t", 0) / 1000
                                    ),
                                    source="polygon",
                                    current_price=float(bar.get("c", 0)),
                                    open=float(bar.get("o", 0)),
                                    high=float(bar.get("h", 0)),
                                    low=float(bar.get("l", 0)),
                                    close=float(bar.get("c", 0)),
                                    volume=int(bar.get("v", 0)),
                                    vwap=float(bar.get("vw", 0)),
                                    data_quality=MarketDataQuality.INSTITUTIONAL,
                                    latency_ms=100,
                                    is_delayed=False,
                                )

                                quotes[symbol] = market_data.to_dict()
                            else:
                                quotes[symbol] = None

                        return quotes

        except Exception as e:
            logger.error(f"Error fetching grouped quotes: {e}")
            return {symbol: None for symbol in symbols}

    async def get_market_depth(self, symbol: str) -> dict[str, Any]:
        """Get market depth (order book) data"""
        try:
            await self._check_rate_limit()

            # Get snapshot including full book
            url = (
                f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            )
            params = {"apiKey": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "OK":
                        ticker = data.get("ticker", {})

                        # Polygon provides level 2 data in premium tiers
                        # For now, return basic bid/ask
                        last_quote = ticker.get("lastQuote", {})

                        market_depth = MarketDepth(
                            symbol=symbol,
                            timestamp=datetime.utcnow(),
                            source="polygon",
                            bids=[
                                {
                                    "price": float(last_quote.get("P", 0)),
                                    "size": int(last_quote.get("S", 0)),
                                }
                            ],
                            asks=[
                                {
                                    "price": float(last_quote.get("P", 0)),
                                    "size": int(last_quote.get("S", 0)),
                                }
                            ],
                        )

                        return market_depth.to_dict()

        except Exception as e:
            logger.error(f"Error fetching market depth: {e}")
            return {"bids": [], "asks": []}

    async def get_historical_data(
        self, symbol: str, interval: str, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """Get historical data for a symbol"""
        try:
            await self._check_rate_limit()

            # Map interval to Polygon timespan/multiplier
            interval_map = {
                "minute": (1, "minute"),
                "1min": (1, "minute"),
                "5min": (5, "minute"),
                "15min": (15, "minute"),
                "30min": (30, "minute"),
                "60min": (1, "hour"),
                "hour": (1, "hour"),
                "day": (1, "day"),
                "week": (1, "week"),
                "month": (1, "month"),
            }

            multiplier, timespan = interval_map.get(interval, (1, "day"))

            # Format dates
            from_str = from_date.strftime("%Y-%m-%d")
            to_str = to_date.strftime("%Y-%m-%d")

            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
            params = {
                "apiKey": self.api_key,
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,  # Max limit
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "OK":
                        results = data.get("results", [])

                        bars = []
                        for bar in results:
                            historical_bar = HistoricalBar(
                                timestamp=datetime.fromtimestamp(bar["t"] / 1000),
                                open=float(bar["o"]),
                                high=float(bar["h"]),
                                low=float(bar["l"]),
                                close=float(bar["c"]),
                                volume=int(bar["v"]),
                                vwap=float(bar.get("vw", 0)),
                                trades=int(bar.get("n", 0)),
                            )
                            bars.append(historical_bar.to_dict())

                        return bars
                    else:
                        logger.warning(f"No data for {symbol}: {data.get('message')}")
                        return []
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching historical data from Polygon: {e}")
            return []

    async def stream_quotes(self, symbols: list[str], callback: Any) -> None:
        """Stream real-time quotes via WebSocket"""
        try:
            # Polygon uses different WebSocket endpoints for different asset classes
            ws_url = f"{self.ws_url}/stocks"

            # Create authenticated connection
            headers = {"Authorization": f"Bearer {self.api_key}"}
            ws = await self.session.ws_connect(ws_url, headers=headers)
            self.ws_sessions["stocks"] = ws

            # Send authentication
            auth_msg = {"action": "auth", "params": self.api_key}
            await ws.send_json(auth_msg)

            # Wait for auth confirmation
            auth_response = await ws.receive_json()
            if auth_response[0].get("status") != "auth_success":
                raise Exception("WebSocket authentication failed")

            # Subscribe to symbols
            # Polygon uses different channels: T.* for trades, Q.* for quotes, A.* for aggregates
            subscribe_msg = {
                "action": "subscribe",
                "params": ",".join([f"T.{symbol},Q.{symbol}" for symbol in symbols]),
            }
            await ws.send_json(subscribe_msg)

            # Store subscriptions
            self.subscriptions.update(symbols)

            # Listen for updates
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    for item in data:
                        ev_type = item.get("ev")

                        if ev_type == "T":  # Trade
                            symbol = item.get("sym")

                            market_data = MarketData(
                                symbol=symbol,
                                exchange=item.get("x", "US"),
                                timestamp=datetime.fromtimestamp(
                                    item.get("t", 0) / 1000000000
                                ),
                                source="polygon",
                                current_price=float(item.get("p", 0)),
                                volume=int(item.get("s", 0)),
                                data_quality=MarketDataQuality.INSTITUTIONAL,
                                latency_ms=5,  # Ultra-low latency
                                is_delayed=False,
                            )

                            await callback(symbol, market_data.to_dict())

                        elif ev_type == "Q":  # Quote
                            symbol = item.get("sym")

                            # Update bid/ask data
                            quote_update = {
                                "symbol": symbol,
                                "bid": float(item.get("bp", 0)),
                                "ask": float(item.get("ap", 0)),
                                "bid_size": int(item.get("bs", 0)),
                                "ask_size": int(item.get("as", 0)),
                                "timestamp": datetime.fromtimestamp(
                                    item.get("t", 0) / 1000000000
                                ).isoformat(),
                                "type": "quote_update",
                            }

                            await callback(symbol, quote_update)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break

        except Exception as e:
            logger.error(f"Error in quote streaming: {e}")
            raise

    async def get_options_chain(
        self, underlying: str, expiration: str | None = None
    ) -> dict[str, Any]:
        """Get options chain for a symbol"""
        try:
            await self._check_rate_limit()

            # Get contract list
            url = f"{self.base_url}/v3/reference/options/contracts"
            params = {
                "apiKey": self.api_key,
                "underlying_ticker": underlying,
                "limit": 1000,
            }

            if expiration:
                params["expiration_date"] = expiration

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "OK":
                        contracts = data.get("results", [])

                        # Group by expiration and strike
                        chain = {}
                        for contract in contracts:
                            exp = contract.get("expiration_date")
                            strike = contract.get("strike_price")
                            contract_type = contract.get("contract_type")

                            if exp not in chain:
                                chain[exp] = {"calls": {}, "puts": {}}

                            option_data = {
                                "symbol": contract.get("ticker"),
                                "strike": strike,
                                "type": contract_type,
                                "expiration": exp,
                            }

                            if contract_type == "call":
                                chain[exp]["calls"][strike] = option_data
                            else:
                                chain[exp]["puts"][strike] = option_data

                        return {
                            "underlying": underlying,
                            "chain": chain,
                            "source": "polygon",
                        }

        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return {"chain": {}}

    async def get_forex_rate(
        self, from_currency: str, to_currency: str
    ) -> dict[str, Any]:
        """Get forex exchange rate"""
        try:
            # Get real-time forex quote
            url = f"{self.base_url}/v1/last/forex/{from_currency}/{to_currency}"
            params = {"apiKey": self.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "success":
                        last = data.get("last", {})

                        return {
                            "from_currency": from_currency,
                            "to_currency": to_currency,
                            "exchange_rate": float(last.get("exchange", 0)),
                            "bid": float(last.get("bid", 0)),
                            "ask": float(last.get("ask", 0)),
                            "timestamp": datetime.utcnow().isoformat(),
                            "source": "polygon",
                        }

        except Exception as e:
            logger.error(f"Error fetching forex rate: {e}")
            raise

    async def get_crypto_quote(
        self, symbol: str, exchange: str = "X:BTCUSD"
    ) -> dict[str, Any]:
        """Get cryptocurrency quote"""
        try:
            # Format for Polygon crypto tickers
            crypto_symbol = f"X:{symbol}" if not symbol.startswith("X:") else symbol

            quote = await self.get_quote(crypto_symbol)

            if quote:
                quote["is_crypto"] = True
                quote["exchange"] = "CRYPTO"

            return quote

        except Exception as e:
            logger.error(f"Error fetching crypto quote: {e}")
            raise

    async def get_market_snapshot(self) -> dict[str, Any]:
        """Get market-wide snapshot - institutional feature"""
        try:
            await self._check_rate_limit()

            url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
            params = {
                "apiKey": self.api_key,
                "tickers.any_of": "SPY,QQQ,IWM,DIA",  # Major indices
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "OK":
                        tickers = data.get("tickers", [])

                        snapshot = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "source": "polygon",
                            "indices": {},
                        }

                        for ticker in tickers:
                            symbol = ticker.get("ticker")
                            day = ticker.get("day", {})

                            snapshot["indices"][symbol] = {
                                "price": float(day.get("c", 0)),
                                "change": float(day.get("c", 0))
                                - float(day.get("o", 0)),
                                "change_percent": (
                                    (float(day.get("c", 0)) - float(day.get("o", 0)))
                                    / float(day.get("o", 1))
                                )
                                * 100,
                                "volume": int(day.get("v", 0)),
                                "high": float(day.get("h", 0)),
                                "low": float(day.get("l", 0)),
                            }

                        return snapshot

        except Exception as e:
            logger.error(f"Error fetching market snapshot: {e}")
            return {"indices": {}}

    def get_cost_info(self) -> DataSourceCost:
        """Get cost information for this data source"""
        return self.cost_info

    def upgrade_tier(self, tier: str) -> DataSourceCost:
        """Get upgraded tier information"""
        return self.premium_tiers.get(tier, self.cost_info)

    def supports_market(self, market: str) -> bool:
        """Check if this source supports a market"""
        supported = ["US", "NYSE", "NASDAQ", "FOREX", "CRYPTO", "OPTIONS"]
        return market.upper() in supported

    def supports_institutional_features(self) -> bool:
        """Check if this source supports institutional features"""
        return True  # Polygon has institutional-grade features
