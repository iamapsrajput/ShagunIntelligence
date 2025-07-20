"""Real-time market data processor for Market Analyst Agent"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
from loguru import logger

from services.kite.client import KiteConnectService
from services.kite.websocket_client import KiteWebSocketClient


@dataclass
class TickData:
    """Real-time tick data structure"""
    instrument_token: int
    symbol: str
    last_price: float
    volume: int
    last_trade_time: datetime
    change: float
    change_percent: float
    ohlc: Dict[str, float]
    depth: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'instrument_token': self.instrument_token,
            'symbol': self.symbol,
            'last_price': self.last_price,
            'volume': self.volume,
            'last_trade_time': self.last_trade_time,
            'change': self.change,
            'change_percent': self.change_percent,
            'open': self.ohlc.get('open', 0),
            'high': self.ohlc.get('high', 0),
            'low': self.ohlc.get('low', 0),
            'close': self.ohlc.get('close', 0)
        }


@dataclass
class CandleData:
    """Candlestick data for analysis"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol
        }


class RealTimeDataProcessor:
    """Processes real-time market data from WebSocket feeds"""
    
    def __init__(self, kite_service: KiteConnectService, max_history: int = 1000):
        self.kite_service = kite_service
        self.max_history = max_history
        
        # Data storage
        self.tick_data: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.current_ticks: Dict[int, TickData] = {}
        self.candle_data: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=500)))
        
        # Symbol mapping
        self.token_to_symbol: Dict[int, str] = {}
        self.symbol_to_token: Dict[str, int] = {}
        
        # Data processing locks
        self._data_lock = threading.RLock()
        
        # Callbacks for data updates
        self.tick_callbacks: List[Callable] = []
        self.candle_callbacks: List[Callable] = []
        
        # Candle building
        self.candle_builders: Dict[str, Dict[int, 'CandleBuilder']] = defaultdict(dict)
        
        # Processing statistics
        self.stats = {
            'ticks_processed': 0,
            'candles_built': 0,
            'last_update': None,
            'symbols_count': 0
        }
        
    async def initialize(self, symbols: List[str]) -> bool:
        """Initialize data processor with symbols"""
        try:
            logger.info(f"Initializing real-time data processor for {len(symbols)} symbols")
            
            # Get instrument tokens for symbols
            instruments = await self.kite_service.get_instruments("NSE")
            
            for symbol in symbols:
                for instrument in instruments:
                    if instrument['tradingsymbol'] == symbol and instrument['exchange'] == 'NSE':
                        token = instrument['instrument_token']
                        self.token_to_symbol[token] = symbol
                        self.symbol_to_token[symbol] = token
                        
                        # Initialize candle builders for different timeframes
                        self.candle_builders['1min'][token] = CandleBuilder(symbol, token, 60)  # 1 minute
                        self.candle_builders['5min'][token] = CandleBuilder(symbol, token, 300)  # 5 minutes
                        self.candle_builders['15min'][token] = CandleBuilder(symbol, token, 900)  # 15 minutes
                        break
            
            self.stats['symbols_count'] = len(self.symbol_to_token)
            
            # Set up WebSocket callback
            if hasattr(self.kite_service, 'websocket_client') and self.kite_service.websocket_client:
                self.kite_service.websocket_client.set_on_tick_callback(self._on_tick_received)
            
            logger.info(f"Data processor initialized for {self.stats['symbols_count']} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize data processor: {str(e)}")
            return False
    
    async def start_monitoring(self, symbols: List[str]) -> bool:
        """Start monitoring symbols"""
        try:
            # Connect WebSocket if not connected
            if not self.kite_service.websocket_client.is_connected:
                await self.kite_service.connect_websocket()
            
            # Subscribe to symbols
            success = await self.kite_service.subscribe_symbols(symbols, mode="full")
            
            if success:
                logger.info(f"Started monitoring {len(symbols)} symbols")
                return True
            else:
                logger.error("Failed to subscribe to symbols")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
            return False
    
    def _on_tick_received(self, ticks: List[Dict[str, Any]]):
        """Handle incoming tick data"""
        try:
            with self._data_lock:
                for tick in ticks:
                    self._process_tick(tick)
                
                self.stats['last_update'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error processing ticks: {str(e)}")
    
    def _process_tick(self, tick: Dict[str, Any]):
        """Process individual tick data"""
        try:
            instrument_token = tick.get('instrument_token')
            if not instrument_token or instrument_token not in self.token_to_symbol:
                return
            
            symbol = self.token_to_symbol[instrument_token]
            
            # Create TickData object
            tick_data = TickData(
                instrument_token=instrument_token,
                symbol=symbol,
                last_price=tick.get('last_price', 0),
                volume=tick.get('volume_traded', 0),
                last_trade_time=tick.get('last_trade_time', datetime.now()),
                change=tick.get('change', 0),
                change_percent=tick.get('change', 0) / tick.get('last_price', 1) * 100 if tick.get('last_price') else 0,
                ohlc={
                    'open': tick.get('ohlc', {}).get('open', 0),
                    'high': tick.get('ohlc', {}).get('high', 0),
                    'low': tick.get('ohlc', {}).get('low', 0),
                    'close': tick.get('ohlc', {}).get('close', 0)
                },
                depth=tick.get('depth', {})
            )
            
            # Store tick data
            self.tick_data[instrument_token].append(tick_data)
            self.current_ticks[instrument_token] = tick_data
            
            # Update candle builders
            for timeframe, builders in self.candle_builders.items():
                if instrument_token in builders:
                    candle = builders[instrument_token].add_tick(tick_data)
                    if candle:
                        self.candle_data[timeframe][symbol].append(candle)
                        self.stats['candles_built'] += 1
                        self._notify_candle_callbacks(timeframe, candle)
            
            self.stats['ticks_processed'] += 1
            
            # Notify tick callbacks
            self._notify_tick_callbacks(tick_data)
            
        except Exception as e:
            logger.error(f"Error processing tick for {tick.get('instrument_token')}: {str(e)}")
    
    def _notify_tick_callbacks(self, tick_data: TickData):
        """Notify registered tick callbacks"""
        for callback in self.tick_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(tick_data))
                else:
                    callback(tick_data)
            except Exception as e:
                logger.error(f"Error in tick callback: {str(e)}")
    
    def _notify_candle_callbacks(self, timeframe: str, candle: CandleData):
        """Notify registered candle callbacks"""
        for callback in self.candle_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(timeframe, candle))
                else:
                    callback(timeframe, candle)
            except Exception as e:
                logger.error(f"Error in candle callback: {str(e)}")
    
    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest tick for a symbol"""
        token = self.symbol_to_token.get(symbol)
        if token and token in self.current_ticks:
            return self.current_ticks[token]
        return None
    
    def get_tick_history(self, symbol: str, count: int = 100) -> List[TickData]:
        """Get tick history for a symbol"""
        token = self.symbol_to_token.get(symbol)
        if token and token in self.tick_data:
            with self._data_lock:
                return list(self.tick_data[token])[-count:]
        return []
    
    def get_candle_data(self, symbol: str, timeframe: str = "1min", count: int = 100) -> pd.DataFrame:
        """Get candle data as DataFrame"""
        try:
            with self._data_lock:
                if timeframe in self.candle_data and symbol in self.candle_data[timeframe]:
                    candles = list(self.candle_data[timeframe][symbol])[-count:]
                    
                    if candles:
                        df = pd.DataFrame([candle.to_dict() for candle in candles])
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                        return df
                
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting candle data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all monitored symbols"""
        prices = {}
        with self._data_lock:
            for token, tick_data in self.current_ticks.items():
                symbol = self.token_to_symbol.get(token)
                if symbol:
                    prices[symbol] = tick_data.last_price
        return prices
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            'active_symbols': list(self.symbol_to_token.keys()),
            'memory_usage': {
                'tick_data_size': sum(len(deque_data) for deque_data in self.tick_data.values()),
                'candle_data_size': sum(
                    sum(len(symbol_data) for symbol_data in timeframe_data.values())
                    for timeframe_data in self.candle_data.values()
                )
            }
        }
    
    def add_tick_callback(self, callback: Callable):
        """Add callback for tick updates"""
        self.tick_callbacks.append(callback)
    
    def add_candle_callback(self, callback: Callable):
        """Add callback for candle updates"""
        self.candle_callbacks.append(callback)
    
    def remove_tick_callback(self, callback: Callable):
        """Remove tick callback"""
        if callback in self.tick_callbacks:
            self.tick_callbacks.remove(callback)
    
    def remove_candle_callback(self, callback: Callable):
        """Remove candle callback"""
        if callback in self.candle_callbacks:
            self.candle_callbacks.remove(callback)


class CandleBuilder:
    """Builds candlestick data from tick data"""
    
    def __init__(self, symbol: str, instrument_token: int, interval_seconds: int):
        self.symbol = symbol
        self.instrument_token = instrument_token
        self.interval_seconds = interval_seconds
        
        self.current_candle: Optional[CandleData] = None
        self.current_interval_start: Optional[datetime] = None
        
    def add_tick(self, tick: TickData) -> Optional[CandleData]:
        """Add tick data and return completed candle if any"""
        try:
            tick_time = tick.last_trade_time
            interval_start = self._get_interval_start(tick_time)
            
            # Check if we need to start a new candle
            if self.current_interval_start != interval_start:
                completed_candle = self.current_candle
                
                # Start new candle
                self.current_candle = CandleData(
                    timestamp=interval_start,
                    open=tick.last_price,
                    high=tick.last_price,
                    low=tick.last_price,
                    close=tick.last_price,
                    volume=tick.volume,
                    symbol=self.symbol
                )
                self.current_interval_start = interval_start
                
                return completed_candle
            
            # Update current candle
            if self.current_candle:
                self.current_candle.high = max(self.current_candle.high, tick.last_price)
                self.current_candle.low = min(self.current_candle.low, tick.last_price)
                self.current_candle.close = tick.last_price
                # Note: Volume from tick is cumulative for the day, not for the interval
                # In real implementation, you'd need to calculate interval volume differently
                
            return None
            
        except Exception as e:
            logger.error(f"Error in candle builder for {self.symbol}: {str(e)}")
            return None
    
    def _get_interval_start(self, timestamp: datetime) -> datetime:
        """Get the start time of the interval for the given timestamp"""
        # Round down to the nearest interval
        seconds_since_midnight = (timestamp.hour * 3600 + 
                                timestamp.minute * 60 + 
                                timestamp.second)
        
        interval_number = seconds_since_midnight // self.interval_seconds
        interval_start_seconds = interval_number * self.interval_seconds
        
        return datetime.combine(
            timestamp.date(),
            datetime.min.time()
        ) + timedelta(seconds=interval_start_seconds)
    
    def get_current_candle(self) -> Optional[CandleData]:
        """Get current incomplete candle"""
        return self.current_candle


class MarketDataAggregator:
    """Aggregates and provides market data across multiple timeframes"""
    
    def __init__(self, data_processor: RealTimeDataProcessor):
        self.data_processor = data_processor
        
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        timeframes = ['1min', '5min', '15min']
        data = {}
        
        for timeframe in timeframes:
            df = self.data_processor.get_candle_data(symbol, timeframe)
            if not df.empty:
                data[timeframe] = df
        
        return data
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overview of all monitored symbols"""
        overview = {
            'symbols': [],
            'total_volume': 0,
            'advancing': 0,
            'declining': 0,
            'unchanged': 0,
            'timestamp': datetime.now()
        }
        
        current_prices = self.data_processor.get_current_prices()
        
        for symbol in current_prices.keys():
            latest_tick = self.data_processor.get_latest_tick(symbol)
            if latest_tick:
                symbol_data = {
                    'symbol': symbol,
                    'price': latest_tick.last_price,
                    'change': latest_tick.change,
                    'change_percent': latest_tick.change_percent,
                    'volume': latest_tick.volume
                }
                overview['symbols'].append(symbol_data)
                overview['total_volume'] += latest_tick.volume
                
                if latest_tick.change > 0:
                    overview['advancing'] += 1
                elif latest_tick.change < 0:
                    overview['declining'] += 1
                else:
                    overview['unchanged'] += 1
        
        return overview
    
    def get_symbol_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive snapshot of a symbol"""
        latest_tick = self.data_processor.get_latest_tick(symbol)
        if not latest_tick:
            return {}
        
        # Get recent tick history for micro analysis
        recent_ticks = self.data_processor.get_tick_history(symbol, 20)
        
        # Get candle data
        candle_1min = self.data_processor.get_candle_data(symbol, "1min", 50)
        candle_5min = self.data_processor.get_candle_data(symbol, "5min", 20)
        
        snapshot = {
            'symbol': symbol,
            'current_price': latest_tick.last_price,
            'change': latest_tick.change,
            'change_percent': latest_tick.change_percent,
            'volume': latest_tick.volume,
            'ohlc': latest_tick.ohlc,
            'last_update': latest_tick.last_trade_time,
            'tick_count': len(recent_ticks),
            'candle_data': {
                '1min_count': len(candle_1min),
                '5min_count': len(candle_5min)
            }
        }
        
        # Add basic statistics if we have enough data
        if len(recent_ticks) > 5:
            prices = [tick.last_price for tick in recent_ticks]
            snapshot['recent_stats'] = {
                'min_price': min(prices),
                'max_price': max(prices),
                'avg_price': np.mean(prices),
                'price_volatility': np.std(prices)
            }
        
        return snapshot