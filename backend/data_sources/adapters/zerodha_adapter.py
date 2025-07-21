"""
Zerodha Kite Connect adapter for the multi-source data manager.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging

from kiteconnect import KiteConnect, KiteTicker

from ..base import MarketDataSource, DataSourceStatus, DataSourceConfig


class ZerodhaMarketDataSource(MarketDataSource):
    """Zerodha Kite Connect implementation of MarketDataSource."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.kite: Optional[KiteConnect] = None
        self.ticker: Optional[KiteTicker] = None
        self._ticker_callbacks = {}
        self._ticker_task = None
        
    async def connect(self) -> bool:
        """Connect to Zerodha Kite."""
        try:
            self.update_health_status(DataSourceStatus.CONNECTING)
            
            # Initialize KiteConnect
            self.kite = KiteConnect(
                api_key=self.config.api_key,
                access_token=self.config.extra_config.get('access_token')
            )
            
            # Validate connection
            profile = await asyncio.to_thread(self.kite.profile)
            self.logger.info(f"Connected to Zerodha as {profile.get('user_name')}")
            
            # Initialize ticker for live data
            if self.config.extra_config.get('enable_websocket', True):
                self.ticker = KiteTicker(
                    api_key=self.config.api_key,
                    access_token=self.config.extra_config.get('access_token')
                )
                self._setup_ticker_callbacks()
            
            self._is_connected = True
            self.update_health_status(DataSourceStatus.HEALTHY)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Zerodha: {e}")
            self.update_health_status(DataSourceStatus.DISCONNECTED, str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Zerodha."""
        try:
            self._is_connected = False
            
            if self.ticker and self.ticker.is_connected():
                await asyncio.to_thread(self.ticker.stop)
            
            if self._ticker_task:
                self._ticker_task.cancel()
                try:
                    await self._ticker_task
                except asyncio.CancelledError:
                    pass
            
            self.update_health_status(DataSourceStatus.DISCONNECTED)
            self.logger.info("Disconnected from Zerodha")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from Zerodha: {e}")
    
    async def health_check(self) -> DataSourceHealth:
        """Check Zerodha connection health."""
        try:
            if not self.kite:
                self.update_health_status(DataSourceStatus.DISCONNECTED)
                return self.health
            
            # Try to get profile as health check
            start_time = asyncio.get_event_loop().time()
            profile = await asyncio.to_thread(self.kite.profile)
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            
            self.record_request_metrics(True, latency)
            self.update_health_status(DataSourceStatus.HEALTHY)
            
            # Add metadata
            self.health.metadata = {
                'user_id': profile.get('user_id'),
                'user_name': profile.get('user_name'),
                'broker': profile.get('broker'),
                'websocket_connected': self.ticker.is_connected() if self.ticker else False
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
        
        return self.health
    
    async def validate_credentials(self) -> bool:
        """Validate Zerodha API credentials."""
        try:
            if not self.kite:
                return False
            
            profile = await asyncio.to_thread(self.kite.profile)
            return bool(profile)
            
        except Exception as e:
            self.logger.error(f"Credential validation failed: {e}")
            return False
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol."""
        return await self.execute_with_retry(self._get_quote, symbol)
    
    async def _get_quote(self, symbol: str) -> Dict[str, Any]:
        """Internal method to get quote."""
        if not self.kite:
            raise Exception("Not connected to Zerodha")
        
        # Convert symbol to Zerodha format if needed
        zerodha_symbol = self._convert_symbol(symbol)
        
        quotes = await asyncio.to_thread(self.kite.quote, [zerodha_symbol])
        quote_data = quotes.get(zerodha_symbol, {})
        
        # Normalize to standard format
        return self._normalize_quote(zerodha_symbol, quote_data)
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        return await self.execute_with_retry(self._get_quotes, symbols)
    
    async def _get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Internal method to get multiple quotes."""
        if not self.kite:
            raise Exception("Not connected to Zerodha")
        
        # Convert symbols
        zerodha_symbols = [self._convert_symbol(s) for s in symbols]
        
        quotes = await asyncio.to_thread(self.kite.quote, zerodha_symbols)
        
        # Normalize results
        result = {}
        for i, symbol in enumerate(symbols):
            zerodha_symbol = zerodha_symbols[i]
            if zerodha_symbol in quotes:
                result[symbol] = self._normalize_quote(zerodha_symbol, quotes[zerodha_symbol])
        
        return result
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: datetime,
        to_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get historical data for a symbol."""
        return await self.execute_with_retry(
            self._get_historical_data,
            symbol,
            interval,
            from_date,
            to_date
        )
    
    async def _get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: datetime,
        to_date: datetime
    ) -> List[Dict[str, Any]]:
        """Internal method to get historical data."""
        if not self.kite:
            raise Exception("Not connected to Zerodha")
        
        zerodha_symbol = self._convert_symbol(symbol)
        instrument_token = await self._get_instrument_token(zerodha_symbol)
        
        # Convert interval to Zerodha format
        zerodha_interval = self._convert_interval(interval)
        
        data = await asyncio.to_thread(
            self.kite.historical_data,
            instrument_token,
            from_date,
            to_date,
            zerodha_interval
        )
        
        # Normalize to standard format
        return [self._normalize_candle(candle) for candle in data]
    
    async def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """Get market depth for a symbol."""
        return await self.execute_with_retry(self._get_market_depth, symbol)
    
    async def _get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """Internal method to get market depth."""
        if not self.kite:
            raise Exception("Not connected to Zerodha")
        
        zerodha_symbol = self._convert_symbol(symbol)
        
        quote = await asyncio.to_thread(self.kite.quote, [zerodha_symbol])
        depth_data = quote.get(zerodha_symbol, {}).get('depth', {})
        
        # Normalize to standard format
        return self._normalize_depth(depth_data)
    
    async def subscribe_live_data(
        self,
        symbols: List[str],
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Subscribe to live market data."""
        if not self.ticker:
            raise Exception("WebSocket not enabled for live data")
        
        # Convert symbols and get tokens
        tokens = []
        for symbol in symbols:
            zerodha_symbol = self._convert_symbol(symbol)
            token = await self._get_instrument_token(zerodha_symbol)
            tokens.append(token)
            self._ticker_callbacks[token] = callback
        
        # Subscribe with ticker
        await asyncio.to_thread(self.ticker.subscribe, tokens)
        await asyncio.to_thread(self.ticker.set_mode, self.ticker.MODE_FULL, tokens)
        
        # Start ticker if not running
        if not self.ticker.is_connected():
            self._ticker_task = asyncio.create_task(self._run_ticker())
    
    async def unsubscribe_live_data(self, symbols: List[str]) -> None:
        """Unsubscribe from live market data."""
        if not self.ticker:
            return
        
        # Convert symbols and get tokens
        tokens = []
        for symbol in symbols:
            zerodha_symbol = self._convert_symbol(symbol)
            token = await self._get_instrument_token(zerodha_symbol)
            tokens.append(token)
            self._ticker_callbacks.pop(token, None)
        
        # Unsubscribe
        if tokens:
            await asyncio.to_thread(self.ticker.unsubscribe, tokens)
    
    def _setup_ticker_callbacks(self) -> None:
        """Setup WebSocket ticker callbacks."""
        def on_ticks(ws, ticks):
            # Process ticks in async context
            for tick in ticks:
                token = tick.get('instrument_token')
                if token in self._ticker_callbacks:
                    normalized_tick = self._normalize_tick(tick)
                    callback = self._ticker_callbacks[token]
                    
                    # Schedule callback in event loop
                    asyncio.create_task(
                        asyncio.to_thread(callback, normalized_tick)
                    )
        
        def on_connect(ws, response):
            self.logger.info("WebSocket connected")
        
        def on_close(ws, code, reason):
            self.logger.warning(f"WebSocket closed: {code} - {reason}")
        
        def on_error(ws, code, reason):
            self.logger.error(f"WebSocket error: {code} - {reason}")
        
        self.ticker.on_ticks = on_ticks
        self.ticker.on_connect = on_connect
        self.ticker.on_close = on_close
        self.ticker.on_error = on_error
    
    async def _run_ticker(self) -> None:
        """Run ticker in background."""
        try:
            await asyncio.to_thread(self.ticker.connect)
        except Exception as e:
            self.logger.error(f"Ticker error: {e}")
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol to Zerodha format."""
        # Example: RELIANCE -> NSE:RELIANCE
        if ':' in symbol:
            return symbol
        
        # Default to NSE
        return f"NSE:{symbol}"
    
    async def _get_instrument_token(self, symbol: str) -> int:
        """Get instrument token for a symbol."""
        # This would normally query Zerodha's instrument list
        # For now, return a placeholder
        # In production, maintain a cache of symbol -> token mapping
        return hash(symbol) % 1000000
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval to Zerodha format."""
        interval_map = {
            '1min': 'minute',
            '5min': '5minute',
            '15min': '15minute',
            '30min': '30minute',
            '60min': '60minute',
            '1hour': '60minute',
            'day': 'day',
            'week': 'week',
            'month': 'month'
        }
        return interval_map.get(interval, interval)
    
    def _normalize_quote(self, symbol: str, quote_data: Dict) -> Dict[str, Any]:
        """Normalize Zerodha quote to standard format."""
        return {
            'symbol': symbol,
            'last_price': quote_data.get('last_price'),
            'open': quote_data.get('ohlc', {}).get('open'),
            'high': quote_data.get('ohlc', {}).get('high'),
            'low': quote_data.get('ohlc', {}).get('low'),
            'close': quote_data.get('ohlc', {}).get('close'),
            'volume': quote_data.get('volume'),
            'bid': quote_data.get('depth', {}).get('buy', [{}])[0].get('price'),
            'ask': quote_data.get('depth', {}).get('sell', [{}])[0].get('price'),
            'bid_size': quote_data.get('depth', {}).get('buy', [{}])[0].get('quantity'),
            'ask_size': quote_data.get('depth', {}).get('sell', [{}])[0].get('quantity'),
            'change': quote_data.get('change'),
            'change_percent': quote_data.get('change_percent'),
            'timestamp': quote_data.get('last_trade_time', datetime.now())
        }
    
    def _normalize_candle(self, candle: Dict) -> Dict[str, Any]:
        """Normalize Zerodha candle to standard format."""
        return {
            'timestamp': candle.get('date'),
            'open': candle.get('open'),
            'high': candle.get('high'),
            'low': candle.get('low'),
            'close': candle.get('close'),
            'volume': candle.get('volume')
        }
    
    def _normalize_depth(self, depth_data: Dict) -> Dict[str, Any]:
        """Normalize Zerodha depth to standard format."""
        return {
            'bids': [
                {'price': level.get('price'), 'quantity': level.get('quantity')}
                for level in depth_data.get('buy', [])
            ],
            'asks': [
                {'price': level.get('price'), 'quantity': level.get('quantity')}
                for level in depth_data.get('sell', [])
            ]
        }
    
    def _normalize_tick(self, tick: Dict) -> Dict[str, Any]:
        """Normalize Zerodha tick to standard format."""
        return {
            'symbol': tick.get('tradingsymbol'),
            'last_price': tick.get('last_price'),
            'volume': tick.get('volume'),
            'bid': tick.get('depth', {}).get('buy', [{}])[0].get('price'),
            'ask': tick.get('depth', {}).get('sell', [{}])[0].get('price'),
            'timestamp': tick.get('timestamp', datetime.now())
        }