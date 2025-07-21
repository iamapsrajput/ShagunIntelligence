import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass
import json

from .websocket_client import ZerodhaWebSocketClient
from .data_buffer import DataBuffer
from .data_synchronizer import DataSynchronizer
from .data_validator import DataValidator
from .data_distributor import DataDistributor
from .performance_tracker import PerformanceTracker
from .redis_cache import RedisCacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline"""
    websocket_url: str = "wss://ws.kite.trade"
    buffer_size: int = 10000  # Per symbol
    history_minutes: int = 60  # Keep 1 hour of tick data
    max_symbols: int = 100
    enable_redis: bool = False
    redis_url: str = "redis://localhost:6379"
    performance_interval: int = 60  # Report performance every minute
    validation_enabled: bool = True
    reconnect_attempts: int = 5
    reconnect_delay: int = 5  # seconds


class DataPipeline:
    """Main orchestrator for real-time market data processing"""
    
    def __init__(self, kite_client, config: Optional[PipelineConfig] = None):
        self.kite_client = kite_client
        self.config = config or PipelineConfig()
        
        # Core components
        self.websocket_client = None
        self.data_buffer = DataBuffer(
            max_size=self.config.buffer_size,
            history_minutes=self.config.history_minutes
        )
        self.data_synchronizer = DataSynchronizer()
        self.data_validator = DataValidator()
        self.data_distributor = DataDistributor()
        self.performance_tracker = PerformanceTracker()
        
        # Optional Redis cache
        self.redis_cache = None
        if self.config.enable_redis:
            self.redis_cache = RedisCacheManager(self.config.redis_url)
        
        # State management
        self.is_running = False
        self.subscribed_symbols: Set[str] = set()
        self.subscribers: Dict[str, List[Callable]] = {}
        
        # Performance monitoring
        self.performance_task = None
        
        # Error tracking
        self.error_count = 0
        self.last_error_time = None
        
        logger.info("Data pipeline initialized")
    
    async def start(self, symbols: List[str]) -> None:
        """Start the data pipeline with given symbols"""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        try:
            logger.info(f"Starting data pipeline for {len(symbols)} symbols")
            
            # Initialize WebSocket client
            self.websocket_client = ZerodhaWebSocketClient(
                self.kite_client,
                on_tick=self._handle_tick,
                on_connect=self._on_connect,
                on_disconnect=self._on_disconnect,
                on_error=self._on_error
            )
            
            # Subscribe to symbols
            self.subscribed_symbols = set(symbols[:self.config.max_symbols])
            
            # Start components
            await self._start_components()
            
            # Connect WebSocket
            await self.websocket_client.connect()
            
            self.is_running = True
            logger.info("Data pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {str(e)}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the data pipeline gracefully"""
        logger.info("Stopping data pipeline")
        
        self.is_running = False
        
        # Stop WebSocket
        if self.websocket_client:
            await self.websocket_client.disconnect()
        
        # Stop performance monitoring
        if self.performance_task:
            self.performance_task.cancel()
        
        # Flush buffers
        await self._flush_buffers()
        
        # Close Redis connection
        if self.redis_cache:
            await self.redis_cache.close()
        
        logger.info("Data pipeline stopped")
    
    async def _start_components(self) -> None:
        """Start all pipeline components"""
        # Start performance monitoring
        self.performance_task = asyncio.create_task(
            self._monitor_performance()
        )
        
        # Initialize Redis if enabled
        if self.redis_cache:
            await self.redis_cache.connect()
        
        # Start data distributor
        await self.data_distributor.start()
    
    async def _handle_tick(self, tick_data: Dict[str, Any]) -> None:
        """Handle incoming tick data"""
        try:
            # Track performance
            start_time = asyncio.get_event_loop().time()
            
            # Validate data
            if self.config.validation_enabled:
                is_valid, cleaned_data = self.data_validator.validate_tick(tick_data)
                if not is_valid:
                    self.performance_tracker.record_invalid_tick()
                    return
                tick_data = cleaned_data
            
            # Extract symbol
            symbol = tick_data.get("instrument_token")
            if not symbol:
                return
            
            # Store in buffer
            self.data_buffer.add_tick(symbol, tick_data)
            
            # Synchronize with agents
            await self.data_synchronizer.sync_tick(symbol, tick_data)
            
            # Distribute to subscribers
            await self.data_distributor.distribute_tick(symbol, tick_data)
            
            # Cache in Redis if enabled
            if self.redis_cache:
                await self.redis_cache.cache_tick(symbol, tick_data)
            
            # Track performance
            processing_time = asyncio.get_event_loop().time() - start_time
            self.performance_tracker.record_tick_processed(processing_time)
            
        except Exception as e:
            logger.error(f"Error handling tick: {str(e)}")
            self.performance_tracker.record_error()
    
    async def _on_connect(self) -> None:
        """Handle WebSocket connection"""
        logger.info("WebSocket connected, subscribing to symbols")
        
        # Subscribe to all symbols
        if self.subscribed_symbols:
            instrument_tokens = await self._get_instrument_tokens(
                list(self.subscribed_symbols)
            )
            await self.websocket_client.subscribe(instrument_tokens)
            
            logger.info(f"Subscribed to {len(instrument_tokens)} instruments")
    
    async def _on_disconnect(self, code: int, reason: str) -> None:
        """Handle WebSocket disconnection"""
        logger.warning(f"WebSocket disconnected: {code} - {reason}")
        
        if self.is_running:
            # Attempt reconnection
            await self._handle_reconnection()
    
    async def _on_error(self, error: Exception) -> None:
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {str(error)}")
        
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        # Track error
        self.performance_tracker.record_error()
    
    async def _handle_reconnection(self) -> None:
        """Handle reconnection logic"""
        for attempt in range(self.config.reconnect_attempts):
            if not self.is_running:
                break
            
            logger.info(f"Reconnection attempt {attempt + 1}/{self.config.reconnect_attempts}")
            
            try:
                await asyncio.sleep(self.config.reconnect_delay * (attempt + 1))
                await self.websocket_client.connect()
                
                logger.info("Reconnection successful")
                return
                
            except Exception as e:
                logger.error(f"Reconnection failed: {str(e)}")
        
        logger.error("Max reconnection attempts reached")
        await self.stop()
    
    async def _get_instrument_tokens(self, symbols: List[str]) -> List[int]:
        """Convert symbols to instrument tokens"""
        try:
            # This would use Kite API to get instrument tokens
            # For now, returning mock tokens
            return list(range(len(symbols)))
        except Exception as e:
            logger.error(f"Error getting instrument tokens: {str(e)}")
            return []
    
    def subscribe_to_symbol(self, symbol: str, callback: Callable) -> None:
        """Subscribe to updates for a specific symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        
        self.subscribers[symbol].append(callback)
        self.data_distributor.add_subscriber(symbol, callback)
        
        logger.info(f"Added subscriber for {symbol}")
    
    def unsubscribe_from_symbol(self, symbol: str, callback: Callable) -> None:
        """Unsubscribe from symbol updates"""
        if symbol in self.subscribers:
            self.subscribers[symbol].remove(callback)
            self.data_distributor.remove_subscriber(symbol, callback)
    
    def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest tick for a symbol"""
        return self.data_buffer.get_latest(symbol)
    
    def get_historical_ticks(self, symbol: str, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get historical ticks for a symbol"""
        return self.data_buffer.get_history(symbol, minutes)
    
    def get_ohlc(self, symbol: str, interval: str = "1min") -> Optional[Dict[str, Any]]:
        """Get OHLC data for a symbol"""
        return self.data_buffer.get_ohlc(symbol, interval)
    
    async def add_symbols(self, symbols: List[str]) -> None:
        """Add new symbols to the pipeline"""
        new_symbols = set(symbols) - self.subscribed_symbols
        
        if not new_symbols:
            return
        
        # Check capacity
        total_symbols = len(self.subscribed_symbols) + len(new_symbols)
        if total_symbols > self.config.max_symbols:
            logger.warning(f"Cannot add {len(new_symbols)} symbols. "
                         f"Max capacity: {self.config.max_symbols}")
            return
        
        # Add symbols
        self.subscribed_symbols.update(new_symbols)
        
        # Subscribe if connected
        if self.websocket_client and self.websocket_client.is_connected():
            instrument_tokens = await self._get_instrument_tokens(list(new_symbols))
            await self.websocket_client.subscribe(instrument_tokens)
            
            logger.info(f"Added {len(new_symbols)} symbols to pipeline")
    
    async def remove_symbols(self, symbols: List[str]) -> None:
        """Remove symbols from the pipeline"""
        symbols_to_remove = set(symbols) & self.subscribed_symbols
        
        if not symbols_to_remove:
            return
        
        # Remove symbols
        self.subscribed_symbols -= symbols_to_remove
        
        # Unsubscribe if connected
        if self.websocket_client and self.websocket_client.is_connected():
            instrument_tokens = await self._get_instrument_tokens(
                list(symbols_to_remove)
            )
            await self.websocket_client.unsubscribe(instrument_tokens)
        
        # Clean up buffers
        for symbol in symbols_to_remove:
            self.data_buffer.remove_symbol(symbol)
            if symbol in self.subscribers:
                del self.subscribers[symbol]
        
        logger.info(f"Removed {len(symbols_to_remove)} symbols from pipeline")
    
    async def _monitor_performance(self) -> None:
        """Monitor and report pipeline performance"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.performance_interval)
                
                # Get performance metrics
                metrics = self.performance_tracker.get_metrics()
                
                # Log performance
                logger.info(f"Pipeline performance: {json.dumps(metrics, indent=2)}")
                
                # Check for performance issues
                if metrics.get("avg_processing_time", 0) > 0.001:  # 1ms threshold
                    logger.warning("High processing latency detected")
                
                if metrics.get("error_rate", 0) > 0.01:  # 1% error threshold
                    logger.warning("High error rate detected")
                
                # Reset metrics
                self.performance_tracker.reset_interval_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
    
    async def _flush_buffers(self) -> None:
        """Flush all data buffers"""
        try:
            # Save buffer data if needed
            buffer_data = self.data_buffer.get_all_data()
            
            if self.redis_cache:
                for symbol, data in buffer_data.items():
                    await self.redis_cache.save_buffer(symbol, data)
            
            logger.info("Buffers flushed successfully")
            
        except Exception as e:
            logger.error(f"Error flushing buffers: {str(e)}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "is_running": self.is_running,
            "subscribed_symbols": len(self.subscribed_symbols),
            "total_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "buffer_stats": self.data_buffer.get_stats(),
            "performance_metrics": self.performance_tracker.get_metrics(),
            "error_count": self.error_count,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "websocket_connected": (
                self.websocket_client.is_connected() 
                if self.websocket_client else False
            ),
            "redis_enabled": self.config.enable_redis
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the pipeline"""
        health = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check WebSocket
        health["checks"]["websocket"] = (
            "connected" if self.websocket_client and self.websocket_client.is_connected()
            else "disconnected"
        )
        
        # Check data flow
        metrics = self.performance_tracker.get_metrics()
        ticks_per_second = metrics.get("ticks_per_second", 0)
        health["checks"]["data_flow"] = "active" if ticks_per_second > 0 else "inactive"
        
        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        health["checks"]["error_rate"] = "normal" if error_rate < 0.01 else "high"
        
        # Check Redis if enabled
        if self.redis_cache:
            redis_connected = await self.redis_cache.ping()
            health["checks"]["redis"] = "connected" if redis_connected else "disconnected"
        
        # Overall status
        if (health["checks"]["websocket"] == "disconnected" or
            health["checks"]["data_flow"] == "inactive" or
            health["checks"]["error_rate"] == "high"):
            health["status"] = "unhealthy"
        
        return health