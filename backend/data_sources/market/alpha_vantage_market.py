import asyncio
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


class AlphaVantageMarketSource(MarketDataSource):
    """Alpha Vantage - Global market data with technical indicators"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.api_key = config.credentials.get("api_key")
        self.base_url = "https://www.alphavantage.co/query"
        self.session: aiohttp.ClientSession | None = None

        # Cost information
        self.cost_info = DataSourceCost(
            tier=DataCostTier.FREE,  # Has free tier
            monthly_cost=0.0,  # Free tier
            per_request_cost=0.0,
            free_requests=500,  # 5 requests/min, 500/day
            requests_per_minute=5,
            requests_per_day=500,
            includes_realtime=True,
            includes_historical=True,
            includes_forex=True,
            includes_crypto=True,
        )

        # Premium tiers available
        self.premium_tiers = {
            "premium": DataSourceCost(
                tier=DataCostTier.LOW,
                monthly_cost=49.99,
                per_request_cost=0.0,
                free_requests=0,
                requests_per_minute=75,
                requests_per_day=None,  # Unlimited
            ),
            "enterprise": DataSourceCost(
                tier=DataCostTier.MEDIUM,
                monthly_cost=249.99,
                per_request_cost=0.0,
                free_requests=0,
                requests_per_minute=1200,
            ),
        }

        # Request tracking for rate limiting
        self.request_times = []
        self.daily_requests = 0
        self.last_request_date = datetime.utcnow().date()

        logger.info("Initialized AlphaVantageMarketSource")

    async def connect(self) -> bool:
        """Connect to Alpha Vantage API"""
        try:
            if not self.api_key:
                raise ValueError("Alpha Vantage API key not provided")

            self.session = aiohttp.ClientSession()

            # Test connection with a simple request
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "MSFT",
                "apikey": self.api_key,
            }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "Error Message" in data:
                        raise Exception(f"API error: {data['Error Message']}")
                    elif "Note" in data:
                        logger.warning(f"API note: {data['Note']}")

                    self.update_health_status(DataSourceStatus.HEALTHY)
                    logger.info("Connected to Alpha Vantage")
                    return True
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Failed to connect to Alpha Vantage: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpha Vantage"""
        if self.session:
            await self.session.close()
        self.update_health_status(DataSourceStatus.DISCONNECTED)
        logger.info("Disconnected from Alpha Vantage")

    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        now = datetime.utcnow()

        # Reset daily counter if new day
        if now.date() != self.last_request_date:
            self.daily_requests = 0
            self.last_request_date = now.date()

        # Check daily limit (free tier)
        if self.cost_info.tier == DataCostTier.FREE:
            if self.daily_requests >= self.cost_info.free_requests:
                raise Exception("Daily request limit reached")

        # Check per-minute limit
        minute_ago = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > minute_ago]

        if len(self.request_times) >= self.cost_info.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Record request
        self.request_times.append(now)
        self.daily_requests += 1

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            await self._check_rate_limit()

            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key,
            }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "Global Quote" not in data:
                        raise Exception(f"Invalid response: {data}")

                    quote = data["Global Quote"]

                    # Convert to standardized format
                    market_data = MarketData(
                        symbol=symbol,
                        exchange=self._detect_exchange(symbol),
                        timestamp=datetime.utcnow(),
                        source="alpha_vantage",
                        current_price=float(quote.get("05. price", 0)),
                        open=float(quote.get("02. open", 0)),
                        high=float(quote.get("03. high", 0)),
                        low=float(quote.get("04. low", 0)),
                        close=float(quote.get("05. price", 0)),  # Latest price as close
                        previous_close=float(quote.get("08. previous close", 0)),
                        volume=int(quote.get("06. volume", 0)),
                        change=float(quote.get("09. change", 0)),
                        change_percent=self._parse_percent(
                            quote.get("10. change percent", "0%")
                        ),
                        data_quality=MarketDataQuality.STANDARD,
                        latency_ms=1000,  # API has some latency
                        is_delayed=False,  # Real-time for US stocks
                    )

                    # Calculate bid/ask spread estimate
                    spread_percent = 0.001  # 0.1% typical spread
                    market_data.bid = market_data.current_price * (1 - spread_percent)
                    market_data.ask = market_data.current_price * (1 + spread_percent)

                    return market_data.to_dict()
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching quote from Alpha Vantage: {e}")
            raise

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get quotes for multiple symbols"""
        quotes = {}

        # Alpha Vantage doesn't support batch quotes in free tier
        # Process sequentially with rate limiting
        for symbol in symbols:
            try:
                quote = await self.get_quote(symbol)
                quotes[symbol] = quote

                # Small delay to respect rate limits
                if len(symbols) > 1:
                    await asyncio.sleep(12.1)  # 5 requests per minute

            except Exception as e:
                logger.error(f"Error fetching quote for {symbol}: {e}")
                quotes[symbol] = None

        return quotes

    async def get_market_depth(self, symbol: str) -> dict[str, Any]:
        """Get market depth - not available in Alpha Vantage"""
        # Alpha Vantage doesn't provide order book data
        # Return empty depth
        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "alpha_vantage",
            "bids": [],
            "asks": [],
            "total_bid_volume": 0,
            "total_ask_volume": 0,
            "note": "Market depth not available from Alpha Vantage",
        }

    async def get_historical_data(
        self, symbol: str, interval: str, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """Get historical data for a symbol"""
        try:
            await self._check_rate_limit()

            # Map interval to Alpha Vantage function
            if interval in ["minute", "1min", "5min", "15min", "30min", "60min"]:
                function = "TIME_SERIES_INTRADAY"
                av_interval = (
                    interval
                    if interval in ["1min", "5min", "15min", "30min", "60min"]
                    else "1min"
                )
                time_key = f"Time Series ({av_interval})"
            else:
                function = "TIME_SERIES_DAILY"
                av_interval = None
                time_key = "Time Series (Daily)"

            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full",  # Get more data
            }

            if av_interval:
                params["interval"] = av_interval

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if time_key not in data:
                        raise Exception(f"Invalid response: {data}")

                    time_series = data[time_key]
                    bars = []

                    for timestamp_str, candle in time_series.items():
                        # Parse timestamp
                        if function == "TIME_SERIES_INTRADAY":
                            timestamp = datetime.strptime(
                                timestamp_str, "%Y-%m-%d %H:%M:%S"
                            )
                        else:
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")

                        # Filter by date range
                        if from_date <= timestamp <= to_date:
                            bar = HistoricalBar(
                                timestamp=timestamp,
                                open=float(candle["1. open"]),
                                high=float(candle["2. high"]),
                                low=float(candle["3. low"]),
                                close=float(candle["4. close"]),
                                volume=int(candle["5. volume"]),
                            )
                            bars.append(bar.to_dict())

                    # Sort by timestamp
                    bars.sort(key=lambda x: x["timestamp"])

                    return bars
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching historical data from Alpha Vantage: {e}")
            return []

    async def get_technical_indicators(
        self, symbol: str, indicator: str, interval: str = "daily", **kwargs
    ) -> dict[str, Any]:
        """Get technical indicators - unique Alpha Vantage feature"""
        try:
            await self._check_rate_limit()

            # Map indicator names
            indicator_map = {
                "sma": "SMA",
                "ema": "EMA",
                "rsi": "RSI",
                "macd": "MACD",
                "bbands": "BBANDS",
                "stoch": "STOCH",
                "adx": "ADX",
            }

            function = indicator_map.get(indicator.lower(), indicator.upper())

            params = {
                "function": function,
                "symbol": symbol,
                "interval": interval,
                "apikey": self.api_key,
                "series_type": kwargs.get("series_type", "close"),
            }

            # Add indicator-specific parameters
            if "time_period" in kwargs:
                params["time_period"] = kwargs["time_period"]

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract indicator data
                    tech_key = f"Technical Analysis: {function}"
                    if tech_key in data:
                        return {
                            "symbol": symbol,
                            "indicator": indicator,
                            "interval": interval,
                            "data": data[tech_key],
                            "metadata": data.get("Meta Data", {}),
                        }
                    else:
                        raise Exception(f"Invalid response: {data}")
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching technical indicators: {e}")
            raise

    async def get_forex_rate(
        self, from_currency: str, to_currency: str
    ) -> dict[str, Any]:
        """Get forex exchange rate"""
        try:
            await self._check_rate_limit()

            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_currency,
                "to_currency": to_currency,
                "apikey": self.api_key,
            }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "Realtime Currency Exchange Rate" in data:
                        rate_data = data["Realtime Currency Exchange Rate"]

                        return {
                            "from_currency": from_currency,
                            "to_currency": to_currency,
                            "exchange_rate": float(rate_data["5. Exchange Rate"]),
                            "last_refreshed": rate_data["6. Last Refreshed"],
                            "bid": float(
                                rate_data.get(
                                    "8. Bid Price", rate_data["5. Exchange Rate"]
                                )
                            ),
                            "ask": float(
                                rate_data.get(
                                    "9. Ask Price", rate_data["5. Exchange Rate"]
                                )
                            ),
                            "source": "alpha_vantage",
                        }
                    else:
                        raise Exception(f"Invalid response: {data}")
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching forex rate: {e}")
            raise

    async def get_crypto_quote(
        self, symbol: str, market: str = "USD"
    ) -> dict[str, Any]:
        """Get cryptocurrency quote"""
        try:
            await self._check_rate_limit()

            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": symbol,
                "to_currency": market,
                "apikey": self.api_key,
            }

            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "Realtime Currency Exchange Rate" in data:
                        rate_data = data["Realtime Currency Exchange Rate"]

                        # Convert to standard market data format
                        market_data = MarketData(
                            symbol=f"{symbol}/{market}",
                            exchange="CRYPTO",
                            timestamp=datetime.utcnow(),
                            source="alpha_vantage",
                            current_price=float(rate_data["5. Exchange Rate"]),
                            bid=float(
                                rate_data.get(
                                    "8. Bid Price", rate_data["5. Exchange Rate"]
                                )
                            ),
                            ask=float(
                                rate_data.get(
                                    "9. Ask Price", rate_data["5. Exchange Rate"]
                                )
                            ),
                            data_quality=MarketDataQuality.STANDARD,
                            latency_ms=1000,
                            is_delayed=False,
                        )

                        return market_data.to_dict()
                    else:
                        raise Exception(f"Invalid response: {data}")
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching crypto quote: {e}")
            raise

    def _detect_exchange(self, symbol: str) -> str:
        """Detect exchange from symbol format"""
        # Simple detection based on common patterns
        if "." in symbol:
            return symbol.split(".")[-1]
        elif symbol.endswith("X"):  # TSX symbols
            return "TSX"
        elif symbol.endswith("L"):  # LSE symbols
            return "LSE"
        else:
            return "US"  # Default to US markets

    def _parse_percent(self, percent_str: str) -> float:
        """Parse percentage string to float"""
        try:
            return float(percent_str.strip("%"))
        except:
            return 0.0

    def get_cost_info(self) -> DataSourceCost:
        """Get cost information for this data source"""
        return self.cost_info

    def upgrade_tier(self, tier: str) -> DataSourceCost:
        """Get upgraded tier information"""
        return self.premium_tiers.get(tier, self.cost_info)

    def supports_market(self, market: str) -> bool:
        """Check if this source supports a market"""
        supported = ["US", "NYSE", "NASDAQ", "FOREX", "CRYPTO", "TSX", "LSE"]
        return market.upper() in supported
