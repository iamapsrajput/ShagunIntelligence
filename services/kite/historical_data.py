"""Historical data service for OHLC and candlestick data"""

import asyncio
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

from .auth import KiteAuthManager
from .exceptions import KiteDataError
from .rate_limiter import RateLimiter


class Interval(Enum):
    """Supported time intervals for historical data"""
    MINUTE = "minute"
    MINUTE_3 = "3minute"
    MINUTE_5 = "5minute"
    MINUTE_10 = "10minute"
    MINUTE_15 = "15minute"
    MINUTE_30 = "30minute"
    HOUR = "60minute"
    DAY = "day"


@dataclass
class OHLC:
    """OHLC data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: Optional[int] = None  # Open Interest for derivatives
    

@dataclass
class HistoricalDataRequest:
    """Request parameters for historical data"""
    instrument_token: int
    from_date: Union[date, datetime]
    to_date: Union[date, datetime]
    interval: Interval
    continuous: bool = False
    oi: bool = False


class HistoricalDataService:
    """Service for fetching and processing historical market data"""
    
    def __init__(self, auth_manager: KiteAuthManager):
        self.auth_manager = auth_manager
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60)  # 10 requests per minute
        self._instruments_cache: Dict[str, Dict] = {}
        self._cache_expiry: Optional[datetime] = None
        
    async def get_ohlc_data(
        self, 
        symbol: str, 
        from_date: Union[date, datetime], 
        to_date: Union[date, datetime],
        interval: Interval = Interval.DAY,
        exchange: str = "NSE"
    ) -> List[OHLC]:
        """Get OHLC data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "NIFTY50")
            from_date: Start date
            to_date: End date
            interval: Time interval
            exchange: Exchange name
            
        Returns:
            List of OHLC data points
        """
        try:
            # Get instrument token
            instrument_token = await self._get_instrument_token(symbol, exchange)
            
            # Create request
            request = HistoricalDataRequest(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            # Fetch data
            raw_data = await self._fetch_historical_data(request)
            
            # Convert to OHLC objects
            ohlc_data = [
                OHLC(
                    timestamp=record["date"],
                    open=float(record["open"]),
                    high=float(record["high"]),
                    low=float(record["low"]),
                    close=float(record["close"]),
                    volume=int(record["volume"]),
                    oi=record.get("oi")
                )
                for record in raw_data
            ]
            
            logger.info(f"Retrieved {len(ohlc_data)} OHLC records for {symbol}")
            return ohlc_data
            
        except Exception as e:
            logger.error(f"Failed to get OHLC data for {symbol}: {str(e)}")
            raise KiteDataError(f"OHLC data fetch failed: {str(e)}")
    
    async def get_candlestick_data(
        self,
        symbol: str,
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        interval: Interval = Interval.DAY,
        exchange: str = "NSE"
    ) -> pd.DataFrame:
        """Get candlestick data as pandas DataFrame
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlc_data = await self.get_ohlc_data(symbol, from_date, to_date, interval, exchange)
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': ohlc.timestamp,
                    'open': ohlc.open,
                    'high': ohlc.high,
                    'low': ohlc.low,
                    'close': ohlc.close,
                    'volume': ohlc.volume,
                    'oi': ohlc.oi
                }
                for ohlc in ohlc_data
            ])
            
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            logger.info(f"Created DataFrame with {len(df)} candlestick records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get candlestick data for {symbol}: {str(e)}")
            raise KiteDataError(f"Candlestick data fetch failed: {str(e)}")
    
    async def get_intraday_data(
        self,
        symbol: str,
        interval: Interval = Interval.MINUTE_5,
        exchange: str = "NSE",
        days_back: int = 1
    ) -> pd.DataFrame:
        """Get intraday data for recent days
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            exchange: Exchange name
            days_back: Number of days to fetch (max 60 for intraday)
            
        Returns:
            DataFrame with intraday data
        """
        try:
            to_date = datetime.now().date()
            from_date = to_date - timedelta(days=days_back)
            
            return await self.get_candlestick_data(
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                exchange=exchange
            )
            
        except Exception as e:
            logger.error(f"Failed to get intraday data for {symbol}: {str(e)}")
            raise KiteDataError(f"Intraday data fetch failed: {str(e)}")
    
    async def get_historical_data_bulk(
        self,
        symbols: List[str],
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        interval: Interval = Interval.DAY,
        exchange: str = "NSE"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        try:
            results = {}
            
            # Process symbols in batches to respect rate limits
            batch_size = 5
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                # Create tasks for parallel processing
                tasks = [
                    self.get_candlestick_data(symbol, from_date, to_date, interval, exchange)
                    for symbol in batch
                ]
                
                # Execute batch with rate limiting
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for symbol, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to fetch data for {symbol}: {str(result)}")
                        results[symbol] = pd.DataFrame()
                    else:
                        results[symbol] = result
                
                # Rate limiting delay between batches
                if i + batch_size < len(symbols):
                    await asyncio.sleep(1)
            
            logger.info(f"Fetched historical data for {len(results)} symbols")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get bulk historical data: {str(e)}")
            raise KiteDataError(f"Bulk data fetch failed: {str(e)}")
    
    async def get_continuous_data(
        self,
        symbol: str,
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        interval: Interval = Interval.DAY,
        exchange: str = "NSE"
    ) -> pd.DataFrame:
        """Get continuous data (adjusted for splits/bonuses)"""
        try:
            instrument_token = await self._get_instrument_token(symbol, exchange)
            
            request = HistoricalDataRequest(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=True
            )
            
            raw_data = await self._fetch_historical_data(request)
            
            df = pd.DataFrame(raw_data)
            if not df.empty:
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
            
            logger.info(f"Retrieved continuous data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get continuous data for {symbol}: {str(e)}")
            raise KiteDataError(f"Continuous data fetch failed: {str(e)}")
    
    async def _fetch_historical_data(self, request: HistoricalDataRequest) -> List[Dict[str, Any]]:
        """Fetch raw historical data from Kite API"""
        try:
            # Wait for rate limiter
            await self.rate_limiter.acquire()
            
            # Map interval enum to Kite constants
            interval_map = {
                Interval.MINUTE: self.auth_manager.kite.INTERVAL_MINUTE,
                Interval.MINUTE_3: self.auth_manager.kite.INTERVAL_3MINUTE,
                Interval.MINUTE_5: self.auth_manager.kite.INTERVAL_5MINUTE,
                Interval.MINUTE_10: self.auth_manager.kite.INTERVAL_10MINUTE,
                Interval.MINUTE_15: self.auth_manager.kite.INTERVAL_15MINUTE,
                Interval.MINUTE_30: self.auth_manager.kite.INTERVAL_30MINUTE,
                Interval.HOUR: self.auth_manager.kite.INTERVAL_HOUR,
                Interval.DAY: self.auth_manager.kite.INTERVAL_DAY
            }
            
            kite_interval = interval_map[request.interval]
            
            # Fetch data
            data = await asyncio.to_thread(
                self.auth_manager.kite.historical_data,
                request.instrument_token,
                request.from_date,
                request.to_date,
                kite_interval,
                continuous=request.continuous,
                oi=request.oi
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {str(e)}")
            raise KiteDataError(f"Historical data fetch failed: {str(e)}")
    
    async def _get_instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        """Get instrument token for a symbol"""
        try:
            # Check cache first
            if await self._is_cache_valid():
                cache_key = f"{exchange}:{symbol}"
                if cache_key in self._instruments_cache:
                    return self._instruments_cache[cache_key]["instrument_token"]
            
            # Refresh instruments cache
            await self._refresh_instruments_cache(exchange)
            
            # Look up symbol
            cache_key = f"{exchange}:{symbol}"
            if cache_key in self._instruments_cache:
                return self._instruments_cache[cache_key]["instrument_token"]
            else:
                raise ValueError(f"Symbol {symbol} not found in {exchange}")
                
        except Exception as e:
            logger.error(f"Failed to get instrument token for {symbol}: {str(e)}")
            raise KiteDataError(f"Instrument token lookup failed: {str(e)}")
    
    async def _refresh_instruments_cache(self, exchange: str = None):
        """Refresh instruments cache"""
        try:
            logger.info(f"Refreshing instruments cache for {exchange or 'all exchanges'}")
            
            if exchange:
                instruments = await asyncio.to_thread(
                    self.auth_manager.kite.instruments, exchange
                )
            else:
                instruments = await asyncio.to_thread(
                    self.auth_manager.kite.instruments
                )
            
            # Build cache
            self._instruments_cache.clear()
            for instrument in instruments:
                cache_key = f"{instrument['exchange']}:{instrument['tradingsymbol']}"
                self._instruments_cache[cache_key] = instrument
            
            # Set cache expiry (instruments data is updated daily)
            self._cache_expiry = datetime.now() + timedelta(hours=6)
            
            logger.info(f"Cached {len(self._instruments_cache)} instruments")
            
        except Exception as e:
            logger.error(f"Failed to refresh instruments cache: {str(e)}")
            raise KiteDataError(f"Instruments cache refresh failed: {str(e)}")
    
    async def _is_cache_valid(self) -> bool:
        """Check if instruments cache is valid"""
        if not self._instruments_cache or not self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry
    
    async def get_available_instruments(self, exchange: str = None) -> List[Dict[str, Any]]:
        """Get list of available instruments"""
        try:
            if not await self._is_cache_valid():
                await self._refresh_instruments_cache(exchange)
            
            if exchange:
                return [
                    instrument for key, instrument in self._instruments_cache.items()
                    if key.startswith(f"{exchange}:")
                ]
            else:
                return list(self._instruments_cache.values())
                
        except Exception as e:
            logger.error(f"Failed to get available instruments: {str(e)}")
            raise KiteDataError(f"Instruments fetch failed: {str(e)}")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators for OHLC data"""
        try:
            if df.empty:
                return df
            
            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            logger.debug("Calculated technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {str(e)}")
            raise KiteDataError(f"Technical indicators calculation failed: {str(e)}")