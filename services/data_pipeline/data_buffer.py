import logging
from typing import Dict, Any, List, Optional, Deque
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import threading
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Structured tick data"""
    instrument_token: int
    timestamp: datetime
    last_price: float
    volume: int = 0
    last_quantity: int = 0
    average_price: float = 0.0
    buy_quantity: int = 0
    sell_quantity: int = 0
    ohlc: Dict[str, float] = field(default_factory=dict)
    bid_depth: List[Dict[str, Any]] = field(default_factory=list)
    ask_depth: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "full"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "instrument_token": self.instrument_token,
            "timestamp": self.timestamp.isoformat(),
            "last_price": self.last_price,
            "volume": self.volume,
            "last_quantity": self.last_quantity,
            "average_price": self.average_price,
            "buy_quantity": self.buy_quantity,
            "sell_quantity": self.sell_quantity,
            "ohlc": self.ohlc,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "mode": self.mode
        }


@dataclass
class OHLCData:
    """OHLC candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    
    def update(self, price: float, volume: int = 0) -> None:
        """Update OHLC with new price"""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }


class DataBuffer:
    """High-performance in-memory data buffer with historical storage"""
    
    def __init__(self, max_size: int = 10000, history_minutes: int = 60):
        self.max_size = max_size
        self.history_minutes = history_minutes
        
        # Thread-safe data structures
        self.lock = threading.RLock()
        
        # Tick data storage (per symbol)
        self.tick_buffers: Dict[str, Deque[TickData]] = defaultdict(
            lambda: deque(maxlen=max_size)
        )
        
        # OHLC data storage (per symbol, per timeframe)
        self.ohlc_buffers: Dict[str, Dict[str, Deque[OHLCData]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1440))  # 24 hours of 1-min candles
        )
        
        # Latest tick cache for fast access
        self.latest_ticks: Dict[str, TickData] = {}
        
        # Statistics
        self.stats = {
            "total_ticks": 0,
            "buffer_overflows": 0,
            "symbols_tracked": 0
        }
        
        # Supported OHLC intervals
        self.ohlc_intervals = {
            "1min": timedelta(minutes=1),
            "5min": timedelta(minutes=5),
            "15min": timedelta(minutes=15),
            "30min": timedelta(minutes=30),
            "1hour": timedelta(hours=1)
        }
        
        logger.info(f"DataBuffer initialized with max_size={max_size}, "
                   f"history_minutes={history_minutes}")
    
    def add_tick(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        """Add new tick data to buffer"""
        with self.lock:
            try:
                # Convert to TickData
                tick = self._create_tick_data(tick_data)
                
                # Add to buffer
                self.tick_buffers[symbol].append(tick)
                
                # Update latest tick
                self.latest_ticks[symbol] = tick
                
                # Update OHLC candles
                self._update_ohlc(symbol, tick)
                
                # Update statistics
                self.stats["total_ticks"] += 1
                self.stats["symbols_tracked"] = len(self.tick_buffers)
                
                # Check for buffer overflow
                if len(self.tick_buffers[symbol]) == self.max_size:
                    self.stats["buffer_overflows"] += 1
                    
            except Exception as e:
                logger.error(f"Error adding tick for {symbol}: {str(e)}")
    
    def get_latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest tick for a symbol"""
        with self.lock:
            tick = self.latest_ticks.get(symbol)
            return tick.to_dict() if tick else None
    
    def get_history(self, symbol: str, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get historical ticks for a symbol"""
        with self.lock:
            if symbol not in self.tick_buffers:
                return []
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # Filter ticks by time
            history = []
            for tick in self.tick_buffers[symbol]:
                if tick.timestamp >= cutoff_time:
                    history.append(tick.to_dict())
            
            return history
    
    def get_ohlc(self, symbol: str, interval: str = "1min",
                 count: int = 100) -> List[Dict[str, Any]]:
        """Get OHLC data for a symbol"""
        with self.lock:
            if symbol not in self.ohlc_buffers:
                return []
            
            if interval not in self.ohlc_intervals:
                logger.warning(f"Invalid interval: {interval}")
                return []
            
            ohlc_data = self.ohlc_buffers[symbol].get(interval, deque())
            
            # Return last 'count' candles
            result = []
            for candle in list(ohlc_data)[-count:]:
                result.append(candle.to_dict())
            
            return result
    
    def get_spread(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current bid-ask spread"""
        with self.lock:
            tick = self.latest_ticks.get(symbol)
            if not tick or not tick.bid_depth or not tick.ask_depth:
                return None
            
            best_bid = tick.bid_depth[0]["price"] if tick.bid_depth else 0
            best_ask = tick.ask_depth[0]["price"] if tick.ask_depth else 0
            
            if best_bid and best_ask:
                spread = best_ask - best_bid
                spread_percent = (spread / best_ask) * 100
                
                return {
                    "bid": best_bid,
                    "ask": best_ask,
                    "spread": spread,
                    "spread_percent": spread_percent
                }
            
            return None
    
    def get_volume_profile(self, symbol: str, minutes: int = 30) -> Dict[str, Any]:
        """Get volume profile for a symbol"""
        with self.lock:
            if symbol not in self.tick_buffers:
                return {}
            
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # Aggregate volume by price levels
            price_volumes = defaultdict(int)
            total_volume = 0
            
            for tick in self.tick_buffers[symbol]:
                if tick.timestamp >= cutoff_time:
                    # Round price to nearest 0.05
                    price_level = round(tick.last_price * 20) / 20
                    price_volumes[price_level] += tick.last_quantity
                    total_volume += tick.last_quantity
            
            if not price_volumes:
                return {}
            
            # Find point of control (POC)
            poc_price = max(price_volumes, key=price_volumes.get)
            poc_volume = price_volumes[poc_price]
            
            return {
                "price_levels": dict(sorted(price_volumes.items())),
                "total_volume": total_volume,
                "poc_price": poc_price,
                "poc_volume": poc_volume,
                "poc_percent": (poc_volume / total_volume * 100) if total_volume > 0 else 0
            }
    
    def _create_tick_data(self, data: Dict[str, Any]) -> TickData:
        """Create TickData from raw tick dictionary"""
        # Parse timestamp
        timestamp_str = data.get("timestamp", datetime.now().isoformat())
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str
        
        return TickData(
            instrument_token=data.get("instrument_token", 0),
            timestamp=timestamp,
            last_price=float(data.get("last_price", 0)),
            volume=int(data.get("volume", 0)),
            last_quantity=int(data.get("last_quantity", 0)),
            average_price=float(data.get("average_price", 0)),
            buy_quantity=int(data.get("buy_quantity", 0)),
            sell_quantity=int(data.get("sell_quantity", 0)),
            ohlc=data.get("ohlc", {}),
            bid_depth=data.get("bid_depth", []),
            ask_depth=data.get("ask_depth", []),
            mode=data.get("mode", "full")
        )
    
    def _update_ohlc(self, symbol: str, tick: TickData) -> None:
        """Update OHLC candles with new tick"""
        for interval_name, interval_delta in self.ohlc_intervals.items():
            # Get candle timestamp (floor to interval)
            candle_time = self._floor_timestamp(tick.timestamp, interval_delta)
            
            # Get or create candle
            ohlc_buffer = self.ohlc_buffers[symbol][interval_name]
            
            # Find existing candle or create new one
            if ohlc_buffer and ohlc_buffer[-1].timestamp == candle_time:
                # Update existing candle
                ohlc_buffer[-1].update(tick.last_price, tick.last_quantity)
            else:
                # Create new candle
                new_candle = OHLCData(
                    timestamp=candle_time,
                    open=tick.last_price,
                    high=tick.last_price,
                    low=tick.last_price,
                    close=tick.last_price,
                    volume=tick.last_quantity
                )
                ohlc_buffer.append(new_candle)
    
    def _floor_timestamp(self, timestamp: datetime, interval: timedelta) -> datetime:
        """Floor timestamp to interval boundary"""
        if interval == timedelta(minutes=1):
            return timestamp.replace(second=0, microsecond=0)
        elif interval == timedelta(minutes=5):
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif interval == timedelta(minutes=15):
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif interval == timedelta(minutes=30):
            minute = (timestamp.minute // 30) * 30
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif interval == timedelta(hours=1):
            return timestamp.replace(minute=0, second=0, microsecond=0)
        
        return timestamp
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove all data for a symbol"""
        with self.lock:
            self.tick_buffers.pop(symbol, None)
            self.ohlc_buffers.pop(symbol, None)
            self.latest_ticks.pop(symbol, None)
            
            self.stats["symbols_tracked"] = len(self.tick_buffers)
            
            logger.info(f"Removed data for symbol: {symbol}")
    
    def clear_old_data(self) -> int:
        """Clear data older than history_minutes"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=self.history_minutes)
            removed_count = 0
            
            for symbol in list(self.tick_buffers.keys()):
                # Remove old ticks
                original_size = len(self.tick_buffers[symbol])
                
                # Filter ticks
                self.tick_buffers[symbol] = deque(
                    (tick for tick in self.tick_buffers[symbol] 
                     if tick.timestamp >= cutoff_time),
                    maxlen=self.max_size
                )
                
                removed_count += original_size - len(self.tick_buffers[symbol])
                
                # Remove symbol if no data left
                if not self.tick_buffers[symbol]:
                    self.remove_symbol(symbol)
            
            logger.info(f"Cleared {removed_count} old ticks")
            return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            symbol_stats = {}
            
            for symbol, buffer in self.tick_buffers.items():
                if buffer:
                    symbol_stats[symbol] = {
                        "tick_count": len(buffer),
                        "oldest_tick": buffer[0].timestamp.isoformat(),
                        "latest_tick": buffer[-1].timestamp.isoformat(),
                        "buffer_usage": len(buffer) / self.max_size * 100
                    }
            
            return {
                **self.stats,
                "symbol_stats": symbol_stats,
                "total_memory_usage": self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> str:
        """Estimate memory usage of buffers"""
        # Rough estimation
        tick_size = 500  # bytes per tick
        ohlc_size = 100  # bytes per candle
        
        total_ticks = sum(len(buffer) for buffer in self.tick_buffers.values())
        total_ohlc = sum(
            sum(len(candles) for candles in symbol_ohlc.values())
            for symbol_ohlc in self.ohlc_buffers.values()
        )
        
        total_bytes = (total_ticks * tick_size) + (total_ohlc * ohlc_size)
        
        # Convert to human-readable format
        if total_bytes < 1024:
            return f"{total_bytes} B"
        elif total_bytes < 1024 * 1024:
            return f"{total_bytes / 1024:.2f} KB"
        else:
            return f"{total_bytes / (1024 * 1024):.2f} MB"
    
    def get_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all buffered data (for export/backup)"""
        with self.lock:
            all_data = {}
            
            for symbol, buffer in self.tick_buffers.items():
                all_data[symbol] = [tick.to_dict() for tick in buffer]
            
            return all_data