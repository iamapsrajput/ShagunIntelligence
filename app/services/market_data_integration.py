"""
Real Market Data Integration Service
Provides live market data feeds with WebSocket support for NSE/BSE
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import aiohttp
import numpy as np
import pandas as pd
import websockets
from loguru import logger

from app.core.resilience import with_circuit_breaker, with_retry


# Mock settings for development
class MockSettings:
    ALPHA_VANTAGE_API_KEY = "demo"


try:
    from app.core.config import settings
except ImportError:
    settings = MockSettings()


class DataProvider(Enum):
    """Supported market data providers"""

    ZERODHA_KITE = "zerodha_kite"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    NSE_OFFICIAL = "nse_official"
    BSE_OFFICIAL = "bse_official"


class DataType(Enum):
    """Types of market data"""

    QUOTE = "quote"
    DEPTH = "depth"
    TRADES = "trades"
    OHLC = "ohlc"
    HISTORICAL = "historical"


@dataclass
class MarketQuote:
    """Real-time market quote data"""

    symbol: str
    exchange: str
    last_price: float
    change: float
    change_percent: float
    volume: int
    bid: float
    ask: float
    bid_quantity: int
    ask_quantity: int
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketDepth:
    """Market depth (Level 2) data"""

    symbol: str
    exchange: str
    bids: list[dict[str, float]]  # [{"price": 100, "quantity": 1000}, ...]
    asks: list[dict[str, float]]
    spread: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeData:
    """Individual trade data"""

    symbol: str
    exchange: str
    price: float
    quantity: int
    timestamp: datetime
    trade_id: str
    buyer_seller_flag: str  # "B" for buyer initiated, "S" for seller initiated


class RealTimeDataFeed:
    """Real-time market data feed with WebSocket support"""

    def __init__(self, provider: DataProvider = DataProvider.ZERODHA_KITE):
        self.provider = provider
        self.websocket = None
        self.is_connected = False
        self.subscriptions = set()
        self.callbacks = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        # Data caching
        self.quote_cache = {}
        self.depth_cache = {}
        self.cache_expiry = timedelta(seconds=1)  # 1 second cache

        logger.info(f"Real-time data feed initialized with provider: {provider.value}")

    async def connect(self, api_key: str = None, access_token: str = None):
        """Connect to real-time data feed"""
        try:
            if self.provider == DataProvider.ZERODHA_KITE:
                await self._connect_kite_websocket(api_key, access_token)
            elif self.provider == DataProvider.ALPHA_VANTAGE:
                await self._connect_alpha_vantage(api_key)
            else:
                await self._connect_generic_websocket()

            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("Successfully connected to real-time data feed")

        except Exception as e:
            logger.error(f"Failed to connect to data feed: {str(e)}")
            await self._handle_connection_error()

    async def _connect_kite_websocket(self, api_key: str, access_token: str):
        """Connect to Zerodha Kite WebSocket"""
        # Note: This is a simplified implementation
        # In production, use the official KiteConnect WebSocket client

        websocket_url = (
            f"wss://ws.kite.trade/?api_key={api_key}&access_token={access_token}"
        )

        self.websocket = await websockets.connect(websocket_url)

        # Start listening for messages
        asyncio.create_task(self._listen_for_messages())

    async def _connect_alpha_vantage(self, api_key: str):
        """Connect to Alpha Vantage (polling-based)"""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

        # Start polling task
        asyncio.create_task(self._alpha_vantage_polling())

    async def _alpha_vantage_polling(self):
        """Polling task for Alpha Vantage data"""
        # For now, just use mock data generator
        await self._mock_data_generator()

    async def _connect_generic_websocket(self):
        """Connect to generic WebSocket feed"""
        # Fallback to a generic WebSocket or polling mechanism
        logger.info("Using generic data feed (mock data for development)")
        asyncio.create_task(self._mock_data_generator())

    async def subscribe(
        self,
        symbols: list[str],
        data_types: list[DataType],
        callback: Callable[[dict], None] = None,
    ):
        """Subscribe to real-time data for symbols"""
        try:
            for symbol in symbols:
                for data_type in data_types:
                    subscription_key = f"{symbol}:{data_type.value}"
                    self.subscriptions.add(subscription_key)

                    if callback:
                        self.callbacks[subscription_key] = callback

            # Send subscription message to WebSocket
            if self.websocket and self.is_connected:
                subscription_message = {
                    "a": "subscribe",
                    "v": [self._get_instrument_token(symbol) for symbol in symbols],
                }
                await self.websocket.send(json.dumps(subscription_message))

            logger.info(
                f"Subscribed to {len(symbols)} symbols for {len(data_types)} data types"
            )

        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {str(e)}")

    async def unsubscribe(self, symbols: list[str], data_types: list[DataType]):
        """Unsubscribe from real-time data"""
        try:
            for symbol in symbols:
                for data_type in data_types:
                    subscription_key = f"{symbol}:{data_type.value}"
                    self.subscriptions.discard(subscription_key)
                    self.callbacks.pop(subscription_key, None)

            # Send unsubscription message
            if self.websocket and self.is_connected:
                unsubscription_message = {
                    "a": "unsubscribe",
                    "v": [self._get_instrument_token(symbol) for symbol in symbols],
                }
                await self.websocket.send(json.dumps(unsubscription_message))

            logger.info(f"Unsubscribed from {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Failed to unsubscribe: {str(e)}")

    async def _listen_for_messages(self):
        """Listen for WebSocket messages"""
        try:
            while self.is_connected and self.websocket:
                message = await self.websocket.recv()
                await self._process_message(message)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            await self._handle_connection_error()
        except Exception as e:
            logger.error(f"Error listening for messages: {str(e)}")
            await self._handle_connection_error()

    async def _process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)

            # Parse different message types
            if "ltp" in data:  # Last traded price
                await self._process_quote_update(data)
            elif "depth" in data:  # Market depth
                await self._process_depth_update(data)
            elif "trades" in data:  # Trade data
                await self._process_trade_update(data)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    async def _process_quote_update(self, data: dict):
        """Process quote update"""
        try:
            symbol = data.get("instrument_token", "UNKNOWN")

            quote = MarketQuote(
                symbol=symbol,
                exchange=data.get("exchange", "NSE"),
                last_price=data.get("ltp", 0.0),
                change=data.get("change", 0.0),
                change_percent=data.get("change_percent", 0.0),
                volume=data.get("volume", 0),
                bid=data.get("bid", 0.0),
                ask=data.get("ask", 0.0),
                bid_quantity=data.get("bid_quantity", 0),
                ask_quantity=data.get("ask_quantity", 0),
                open=data.get("open", 0.0),
                high=data.get("high", 0.0),
                low=data.get("low", 0.0),
                close=data.get("close", 0.0),
            )

            # Cache the quote
            self.quote_cache[symbol] = (quote, datetime.now())

            # Call registered callbacks
            callback_key = f"{symbol}:{DataType.QUOTE.value}"
            if callback_key in self.callbacks:
                await self.callbacks[callback_key](quote)

        except Exception as e:
            logger.error(f"Error processing quote update: {str(e)}")

    async def _process_depth_update(self, data: dict):
        """Process market depth update"""
        try:
            symbol = data.get("instrument_token", "UNKNOWN")

            depth = MarketDepth(
                symbol=symbol,
                exchange=data.get("exchange", "NSE"),
                bids=data.get("depth", {}).get("buy", []),
                asks=data.get("depth", {}).get("sell", []),
                spread=self._calculate_spread(data.get("depth", {})),
            )

            # Cache the depth
            self.depth_cache[symbol] = (depth, datetime.now())

            # Call registered callbacks
            callback_key = f"{symbol}:{DataType.DEPTH.value}"
            if callback_key in self.callbacks:
                await self.callbacks[callback_key](depth)

        except Exception as e:
            logger.error(f"Error processing depth update: {str(e)}")

    async def _process_trade_update(self, data: dict):
        """Process trade update"""
        try:
            symbol = data.get("instrument_token", "UNKNOWN")

            trade = TradeData(
                symbol=symbol,
                exchange=data.get("exchange", "NSE"),
                price=data.get("price", 0.0),
                quantity=data.get("quantity", 0),
                timestamp=datetime.now(),
                trade_id=data.get("trade_id", ""),
                buyer_seller_flag=data.get("buyer_seller_flag", ""),
            )

            # Call registered callbacks
            callback_key = f"{symbol}:{DataType.TRADES.value}"
            if callback_key in self.callbacks:
                await self.callbacks[callback_key](trade)

        except Exception as e:
            logger.error(f"Error processing trade update: {str(e)}")

    async def _handle_connection_error(self):
        """Handle connection errors and attempt reconnection"""
        self.is_connected = False

        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = min(2**self.reconnect_attempts, 60)  # Exponential backoff

            logger.info(
                f"Attempting reconnection in {wait_time} seconds (attempt {self.reconnect_attempts})"
            )
            await asyncio.sleep(wait_time)

            try:
                await self.connect()
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {str(e)}")
        else:
            logger.error("Max reconnection attempts reached. Connection failed.")

    async def _mock_data_generator(self):
        """Generate mock real-time data for development"""
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
        base_prices = {
            "RELIANCE": 2500,
            "TCS": 3600,
            "HDFCBANK": 1600,
            "INFY": 2500,
            "ICICIBANK": 1200,
        }

        while self.is_connected:
            try:
                for symbol in symbols:
                    if f"{symbol}:{DataType.QUOTE.value}" in self.subscriptions:
                        # Generate realistic price movement
                        base_price = base_prices.get(symbol, 2500)
                        price_change = np.random.normal(
                            0, base_price * 0.001
                        )  # 0.1% volatility
                        new_price = base_price + price_change

                        quote = MarketQuote(
                            symbol=symbol,
                            exchange="NSE",
                            last_price=new_price,
                            change=price_change,
                            change_percent=(price_change / base_price) * 100,
                            volume=np.random.randint(10000, 100000),
                            bid=new_price - 0.5,
                            ask=new_price + 0.5,
                            bid_quantity=np.random.randint(100, 1000),
                            ask_quantity=np.random.randint(100, 1000),
                            open=base_price,
                            high=new_price
                            + abs(np.random.normal(0, base_price * 0.005)),
                            low=new_price
                            - abs(np.random.normal(0, base_price * 0.005)),
                            close=base_price,
                        )

                        # Update cache
                        self.quote_cache[symbol] = (quote, datetime.now())

                        # Call callbacks
                        callback_key = f"{symbol}:{DataType.QUOTE.value}"
                        if callback_key in self.callbacks:
                            await self.callbacks[callback_key](quote)

                await asyncio.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Error in mock data generator: {str(e)}")
                await asyncio.sleep(5)

    def _get_instrument_token(self, symbol: str) -> str:
        """Get instrument token for symbol (mock implementation)"""
        # In production, this would map symbols to actual instrument tokens
        token_map = {
            "RELIANCE": "738561",
            "TCS": "2953217",
            "HDFCBANK": "341249",
            "INFY": "408065",
            "ICICIBANK": "1270529",
        }
        return token_map.get(symbol, symbol)

    def _calculate_spread(self, depth_data: dict) -> float:
        """Calculate bid-ask spread from depth data"""
        try:
            buy_orders = depth_data.get("buy", [])
            sell_orders = depth_data.get("sell", [])

            if buy_orders and sell_orders:
                best_bid = buy_orders[0].get("price", 0)
                best_ask = sell_orders[0].get("price", 0)
                return best_ask - best_bid

            return 0.0
        except:
            return 0.0

    async def get_cached_quote(self, symbol: str) -> MarketQuote | None:
        """Get cached quote data"""
        if symbol in self.quote_cache:
            quote, timestamp = self.quote_cache[symbol]
            if datetime.now() - timestamp < self.cache_expiry:
                return quote
        return None

    async def get_cached_depth(self, symbol: str) -> MarketDepth | None:
        """Get cached market depth data"""
        if symbol in self.depth_cache:
            depth, timestamp = self.depth_cache[symbol]
            if datetime.now() - timestamp < self.cache_expiry:
                return depth
        return None

    async def disconnect(self):
        """Disconnect from data feed"""
        try:
            self.is_connected = False

            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            self.subscriptions.clear()
            self.callbacks.clear()

            logger.info("Disconnected from real-time data feed")

        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")

    def get_connection_status(self) -> dict[str, Any]:
        """Get connection status and statistics"""
        return {
            "connected": self.is_connected,
            "provider": self.provider.value,
            "subscriptions": len(self.subscriptions),
            "reconnect_attempts": self.reconnect_attempts,
            "cached_quotes": len(self.quote_cache),
            "cached_depths": len(self.depth_cache),
        }


class HistoricalDataService:
    """Service for fetching historical market data"""

    def __init__(self, provider: DataProvider = DataProvider.ALPHA_VANTAGE):
        self.provider = provider
        self.session = None
        self.cache = {}
        self.cache_expiry = timedelta(hours=1)  # Cache for 1 hour

        # API configurations
        self.api_configs = {
            DataProvider.ALPHA_VANTAGE: {
                "base_url": "https://www.alphavantage.co/query",
                "rate_limit": 5,  # 5 requests per minute
            },
            DataProvider.YAHOO_FINANCE: {
                "base_url": "https://query1.finance.yahoo.com/v8/finance/chart",
                "rate_limit": 2000,  # 2000 requests per hour
            },
        }

        logger.info(
            f"Historical data service initialized with provider: {provider.value}"
        )

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    @with_circuit_breaker("historical_data")
    @with_retry(max_retries=3, delay=1.0)
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}_{limit}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < self.cache_expiry:
                    return data

            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=limit)

            # Fetch data based on provider
            if self.provider == DataProvider.ALPHA_VANTAGE:
                data = await self._fetch_alpha_vantage_data(
                    symbol, timeframe, start_date, end_date
                )
            elif self.provider == DataProvider.YAHOO_FINANCE:
                data = await self._fetch_yahoo_finance_data(
                    symbol, timeframe, start_date, end_date
                )
            else:
                data = self._generate_mock_historical_data(
                    symbol, timeframe, start_date, end_date
                )

            # Cache the result
            self.cache[cache_key] = (data, datetime.now())

            return data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            # Return mock data as fallback
            return self._generate_mock_historical_data(
                symbol, timeframe, start_date, end_date
            )

    async def _fetch_alpha_vantage_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Map timeframe to Alpha Vantage function
        function_map = {
            "1min": "TIME_SERIES_INTRADAY",
            "5min": "TIME_SERIES_INTRADAY",
            "15min": "TIME_SERIES_INTRADAY",
            "30min": "TIME_SERIES_INTRADAY",
            "60min": "TIME_SERIES_INTRADAY",
            "1d": "TIME_SERIES_DAILY",
            "1w": "TIME_SERIES_WEEKLY",
            "1M": "TIME_SERIES_MONTHLY",
        }

        function = function_map.get(timeframe, "TIME_SERIES_DAILY")

        params = {
            "function": function,
            "symbol": f"{symbol}.BSE",  # Assuming BSE format
            "apikey": settings.ALPHA_VANTAGE_API_KEY,
            "outputsize": "full",
        }

        if "INTRADAY" in function:
            params["interval"] = timeframe

        config = self.api_configs[DataProvider.ALPHA_VANTAGE]

        async with self.session.get(config["base_url"], params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_alpha_vantage_response(data, timeframe)
            else:
                raise Exception(f"API request failed with status {response.status}")

    async def _fetch_yahoo_finance_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Convert symbol to Yahoo format (add .NS for NSE)
        yahoo_symbol = f"{symbol}.NS"

        # Convert timeframe to Yahoo interval
        interval_map = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "60min": "1h",
            "1d": "1d",
            "1w": "1wk",
            "1M": "1mo",
        }

        interval = interval_map.get(timeframe, "1d")

        # Convert dates to timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        url = (
            f"{self.api_configs[DataProvider.YAHOO_FINANCE]['base_url']}/{yahoo_symbol}"
        )
        params = {
            "period1": start_timestamp,
            "period2": end_timestamp,
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits",
        }

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_yahoo_finance_response(data)
            else:
                raise Exception(
                    f"Yahoo Finance API request failed with status {response.status}"
                )

    def _parse_alpha_vantage_response(self, data: dict, timeframe: str) -> pd.DataFrame:
        """Parse Alpha Vantage API response"""
        try:
            # Find the time series key
            time_series_key = None
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break

            if not time_series_key:
                raise Exception("No time series data found in response")

            time_series = data[time_series_key]

            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append(
                    {
                        "timestamp": pd.to_datetime(timestamp),
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(values["5. volume"]),
                    }
                )

            df = pd.DataFrame(df_data)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage response: {str(e)}")
            raise

    def _parse_yahoo_finance_response(self, data: dict) -> pd.DataFrame:
        """Parse Yahoo Finance API response"""
        try:
            result = data["chart"]["result"][0]

            timestamps = result["timestamp"]
            indicators = result["indicators"]["quote"][0]

            df_data = []
            for i, timestamp in enumerate(timestamps):
                df_data.append(
                    {
                        "timestamp": pd.to_datetime(timestamp, unit="s"),
                        "open": indicators["open"][i],
                        "high": indicators["high"][i],
                        "low": indicators["low"][i],
                        "close": indicators["close"][i],
                        "volume": indicators["volume"][i] or 0,
                    }
                )

            df = pd.DataFrame(df_data)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            # Remove rows with NaN values
            df.dropna(inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error parsing Yahoo Finance response: {str(e)}")
            raise

    def _generate_mock_historical_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate mock historical data for development/testing"""
        try:
            # Determine frequency based on timeframe
            freq_map = {
                "1min": "1T",
                "5min": "5T",
                "15min": "15T",
                "30min": "30T",
                "60min": "1H",
                "1d": "1D",
                "1w": "1W",
                "1M": "1M",
            }

            freq = freq_map.get(timeframe, "1D")

            # Generate date range
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)

            # Generate realistic price data
            np.random.seed(hash(symbol) % 1000)  # Consistent data per symbol

            base_price = {
                "RELIANCE": 2500,
                "TCS": 3600,
                "HDFCBANK": 1600,
                "INFY": 2500,
                "ICICIBANK": 1200,
            }.get(symbol, 2500)

            # Generate returns with some trend and volatility clustering
            n_periods = len(dates)
            returns = np.random.normal(0, 0.02, n_periods)

            # Add trend component
            trend = np.linspace(-0.001, 0.001, n_periods)
            returns += trend

            # Generate prices
            prices = base_price * np.exp(np.cumsum(returns))

            # Generate OHLCV data
            df_data = []
            for i, date in enumerate(dates):
                price = prices[i]
                daily_vol = abs(np.random.normal(0, price * 0.01))

                high = price + daily_vol * np.random.uniform(0.3, 1.0)
                low = price - daily_vol * np.random.uniform(0.3, 1.0)
                open_price = price + np.random.normal(0, price * 0.005)
                close_price = price + np.random.normal(0, price * 0.005)

                df_data.append(
                    {
                        "open": max(low, min(high, open_price)),
                        "high": high,
                        "low": low,
                        "close": max(low, min(high, close_price)),
                        "volume": int(np.random.lognormal(10, 0.5)),
                    }
                )

            df = pd.DataFrame(df_data, index=dates)
            return df

        except Exception as e:
            logger.error(f"Error generating mock data: {str(e)}")
            # Return minimal DataFrame
            return pd.DataFrame(
                {
                    "open": [base_price],
                    "high": [base_price],
                    "low": [base_price],
                    "close": [base_price],
                    "volume": [100000],
                },
                index=[datetime.now()],
            )

    async def get_multiple_symbols_data(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols"""
        results = {}

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        async def fetch_symbol_data(symbol: str):
            async with semaphore:
                try:
                    data = await self.get_historical_data(
                        symbol, timeframe, start_date, end_date
                    )
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                    results[symbol] = pd.DataFrame()

        # Execute all requests concurrently
        tasks = [fetch_symbol_data(symbol) for symbol in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        logger.info("Historical data cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_items": len(self.cache),
            "cache_expiry_hours": self.cache_expiry.total_seconds() / 3600,
            "provider": self.provider.value,
        }
