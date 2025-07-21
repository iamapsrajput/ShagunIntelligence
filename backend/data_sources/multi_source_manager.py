"""
Multi-source data manager with intelligent failover and orchestration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Type, Union, Tuple
from collections import defaultdict
import time
from functools import wraps

from .base import (
    BaseDataSource,
    MarketDataSource,
    SentimentDataSource,
    DataSourceStatus,
    DataSourceType,
    DataSourceConfig,
    DataSourceHealth
)
from .data_quality_validator import (
    DataQualityValidator,
    QualityMetrics,
    QualityGrade
)
from .models import MarketData


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window  # seconds
        self.calls = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            now = time.time()
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                # Need to wait
                oldest_call = self.calls[0]
                wait_time = self.time_window - (now - oldest_call) + 0.1
                await asyncio.sleep(wait_time)
                # Recursive call to recheck
                await self.acquire()
            else:
                self.calls.append(now)


class ConnectionPool:
    """Connection pool for data sources."""
    
    def __init__(self, create_func: Callable, size: int = 10):
        self.create_func = create_func
        self.size = size
        self.pool = asyncio.Queue(maxsize=size)
        self.created = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        # Try to get from pool
        try:
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            pass
        
        # Create new if under limit
        async with self._lock:
            if self.created < self.size:
                conn = await self.create_func()
                self.created += 1
                return conn
        
        # Wait for available connection
        return await self.pool.get()
    
    async def release(self, conn: Any) -> None:
        """Release connection back to pool."""
        try:
            self.pool.put_nowait(conn)
        except asyncio.QueueFull:
            # Pool is full, close the connection
            if hasattr(conn, 'close'):
                await conn.close()


class MultiSourceDataManager:
    """
    Central manager for coordinating multiple data sources with intelligent failover.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._sources: Dict[DataSourceType, List[BaseDataSource]] = defaultdict(list)
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._connection_pools: Dict[str, ConnectionPool] = {}
        self._failover_callbacks: List[Callable] = []
        self._is_running = False
        self._monitor_task = None
        self._source_locks: Dict[str, asyncio.Lock] = {}
        
        # Initialize data quality validator
        self._quality_validator = DataQualityValidator()
        self._quality_metrics: Dict[str, QualityMetrics] = {}
        self._quality_callbacks: List[Callable] = []
        
        # Register quality alert callback
        self._quality_validator.register_alert_callback(self._handle_quality_alert)
        
    def add_source(self, source: BaseDataSource) -> None:
        """Add a data source to the manager."""
        source_type = source.source_type
        self._sources[source_type].append(source)
        
        # Sort by priority (lower number = higher priority)
        self._sources[source_type].sort(key=lambda s: s.config.priority)
        
        # Create rate limiter if configured
        if source.config.rate_limit:
            self._rate_limiters[source.config.name] = RateLimiter(
                max_calls=source.config.rate_limit,
                time_window=60
            )
        
        # Create lock for thread-safe operations
        self._source_locks[source.config.name] = asyncio.Lock()
        
        self.logger.info(f"Added data source: {source.config.name} ({source_type.value})")
    
    def remove_source(self, source_name: str) -> None:
        """Remove a data source."""
        for source_type, sources in self._sources.items():
            self._sources[source_type] = [
                s for s in sources if s.config.name != source_name
            ]
        
        # Clean up associated resources
        self._rate_limiters.pop(source_name, None)
        self._connection_pools.pop(source_name, None)
        self._source_locks.pop(source_name, None)
        
        self.logger.info(f"Removed data source: {source_name}")
    
    async def start(self) -> None:
        """Start the data manager and connect all sources."""
        self.logger.info("Starting MultiSourceDataManager")
        self._is_running = True
        
        # Connect all sources
        connect_tasks = []
        for sources in self._sources.values():
            for source in sources:
                if source.config.enabled:
                    connect_tasks.append(self._connect_source(source))
        
        results = await asyncio.gather(*connect_tasks, return_exceptions=True)
        
        # Log connection results
        for source, result in zip(
            [s for sources in self._sources.values() for s in sources if s.config.enabled],
            results
        ):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to connect {source.config.name}: {result}")
            else:
                self.logger.info(f"Connected to {source.config.name}")
        
        # Start monitoring
        self._monitor_task = asyncio.create_task(self._monitor_sources())
        
        self.logger.info("MultiSourceDataManager started")
    
    async def stop(self) -> None:
        """Stop the data manager and disconnect all sources."""
        self.logger.info("Stopping MultiSourceDataManager")
        self._is_running = False
        
        # Stop monitoring
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all sources
        disconnect_tasks = []
        for sources in self._sources.values():
            for source in sources:
                disconnect_tasks.append(self._disconnect_source(source))
        
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        self.logger.info("MultiSourceDataManager stopped")
    
    async def _connect_source(self, source: BaseDataSource) -> bool:
        """Connect to a single source."""
        try:
            async with self._source_locks[source.config.name]:
                success = await source.connect()
                if success:
                    await source.start_health_monitoring()
                return success
        except Exception as e:
            self.logger.error(f"Error connecting to {source.config.name}: {e}")
            return False
    
    async def _disconnect_source(self, source: BaseDataSource) -> None:
        """Disconnect from a single source."""
        try:
            async with self._source_locks[source.config.name]:
                await source.stop_health_monitoring()
                await source.disconnect()
        except Exception as e:
            self.logger.error(f"Error disconnecting from {source.config.name}: {e}")
    
    async def _monitor_sources(self) -> None:
        """Monitor all sources and handle failovers."""
        while self._is_running:
            try:
                for source_type, sources in self._sources.items():
                    # Check if primary source is healthy
                    if sources and sources[0].health.status != DataSourceStatus.HEALTHY:
                        # Find next healthy source
                        for i, source in enumerate(sources[1:], 1):
                            if source.health.status == DataSourceStatus.HEALTHY:
                                # Perform failover
                                await self._perform_failover(source_type, 0, i)
                                break
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in source monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _perform_failover(
        self,
        source_type: DataSourceType,
        from_index: int,
        to_index: int
    ) -> None:
        """Perform failover from one source to another."""
        sources = self._sources[source_type]
        from_source = sources[from_index]
        to_source = sources[to_index]
        
        self.logger.warning(
            f"Performing failover for {source_type.value}: "
            f"{from_source.config.name} -> {to_source.config.name}"
        )
        
        # Swap priorities temporarily
        sources[from_index], sources[to_index] = sources[to_index], sources[from_index]
        
        # Notify callbacks
        for callback in self._failover_callbacks:
            try:
                await callback(source_type, from_source, to_source)
            except Exception as e:
                self.logger.error(f"Error in failover callback: {e}")
    
    def add_failover_callback(self, callback: Callable) -> None:
        """Add a callback to be notified on failovers."""
        self._failover_callbacks.append(callback)
    
    async def get_healthy_source(
        self,
        source_type: DataSourceType,
        required_status: DataSourceStatus = DataSourceStatus.HEALTHY
    ) -> Optional[BaseDataSource]:
        """Get a healthy source of the specified type."""
        sources = self._sources.get(source_type, [])
        
        for source in sources:
            if source.config.enabled and source.health.status == required_status:
                return source
        
        # If no healthy source, try degraded
        if required_status == DataSourceStatus.HEALTHY:
            for source in sources:
                if source.config.enabled and source.health.status == DataSourceStatus.DEGRADED:
                    return source
        
        return None
    
    async def execute_with_failover(
        self,
        source_type: DataSourceType,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with automatic failover."""
        sources = self._sources.get(source_type, [])
        last_error = None
        
        for source in sources:
            if not source.config.enabled:
                continue
                
            try:
                # Apply rate limiting if configured
                if source.config.name in self._rate_limiters:
                    await self._rate_limiters[source.config.name].acquire()
                
                # Get the method
                method = getattr(source, operation, None)
                if not method:
                    self.logger.error(f"Method {operation} not found on {source.config.name}")
                    continue
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    method(*args, **kwargs),
                    timeout=source.config.timeout
                )
                
                # Success - update health if needed
                if source.health.status != DataSourceStatus.HEALTHY:
                    source.update_health_status(DataSourceStatus.HEALTHY)
                
                return result
                
            except asyncio.TimeoutError:
                last_error = f"Timeout on {source.config.name}"
                source.update_health_status(DataSourceStatus.DEGRADED, last_error)
                self.logger.warning(f"{last_error} for operation {operation}")
                
            except Exception as e:
                last_error = str(e)
                source.update_health_status(DataSourceStatus.UNHEALTHY, last_error)
                self.logger.error(
                    f"Error on {source.config.name} for operation {operation}: {e}"
                )
        
        # All sources failed
        raise Exception(
            f"All sources failed for {source_type.value}.{operation}: {last_error}"
        )
    
    # Convenience methods for market data operations
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote with automatic failover."""
        return await self.execute_with_failover(
            DataSourceType.MARKET_DATA,
            'get_quote',
            symbol
        )
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get multiple quotes with automatic failover."""
        return await self.execute_with_failover(
            DataSourceType.MARKET_DATA,
            'get_quotes',
            symbols
        )
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: datetime,
        to_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get historical data with automatic failover."""
        return await self.execute_with_failover(
            DataSourceType.MARKET_DATA,
            'get_historical_data',
            symbol,
            interval,
            from_date,
            to_date
        )
    
    async def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """Get market depth with automatic failover."""
        return await self.execute_with_failover(
            DataSourceType.MARKET_DATA,
            'get_market_depth',
            symbol
        )
    
    # Convenience methods for sentiment operations
    
    async def get_sentiment_score(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment score with automatic failover."""
        return await self.execute_with_failover(
            DataSourceType.SENTIMENT,
            'get_sentiment_score',
            symbol
        )
    
    async def get_news_sentiment(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get news sentiment with automatic failover."""
        return await self.execute_with_failover(
            DataSourceType.SENTIMENT,
            'get_news_sentiment',
            symbol,
            from_date,
            to_date
        )
    
    # Health and monitoring methods
    
    def get_all_sources_health(self) -> Dict[str, DataSourceHealth]:
        """Get health status of all sources."""
        health_status = {}
        
        for sources in self._sources.values():
            for source in sources:
                health_status[source.config.name] = source.health
        
        return health_status
    
    def get_source_metrics(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific source."""
        for sources in self._sources.values():
            for source in sources:
                if source.config.name == source_name:
                    return {
                        'health': source.health,
                        'config': source.config,
                        'metrics': source._performance_metrics
                    }
        return None
    
    async def force_health_check(self, source_name: Optional[str] = None) -> Dict[str, DataSourceHealth]:
        """Force health check on sources."""
        health_results = {}
        
        for sources in self._sources.values():
            for source in sources:
                if source_name and source.config.name != source_name:
                    continue
                    
                try:
                    health = await source.health_check()
                    health_results[source.config.name] = health
                except Exception as e:
                    self.logger.error(f"Health check failed for {source.config.name}: {e}")
                    source.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
                    health_results[source.config.name] = source.health
        
        return health_results
    
    # Data quality methods
    
    async def _handle_quality_alert(self, source: DataSourceType, metrics: QualityMetrics):
        """Handle quality alerts from the validator."""
        self.logger.warning(
            f"Quality alert for {source.value}: {metrics.grade.value} "
            f"(score: {metrics.overall_score:.2f})"
        )
        
        # Update source health if quality is poor
        if metrics.grade == QualityGrade.FAILED:
            for s in self._sources.get(source, []):
                if s.health.status == DataSourceStatus.HEALTHY:
                    s.update_health_status(
                        DataSourceStatus.DEGRADED,
                        f"Data quality: {metrics.grade.value}"
                    )
        
        # Notify callbacks
        for callback in self._quality_callbacks:
            try:
                await callback(source, metrics)
            except Exception as e:
                self.logger.error(f"Error in quality callback: {e}")
    
    def add_quality_callback(self, callback: Callable) -> None:
        """Add a callback for quality monitoring."""
        self._quality_callbacks.append(callback)
    
    async def validate_and_get_quote(self, symbol: str) -> Tuple[Dict[str, Any], QualityMetrics]:
        """Get quote with quality validation."""
        source = await self.get_healthy_source(DataSourceType.MARKET_DATA)
        if not source:
            raise Exception("No healthy market data source available")
        
        # Get the quote
        quote_data = await self.execute_with_failover(
            DataSourceType.MARKET_DATA,
            'get_quote',
            symbol
        )
        
        # Convert to MarketData for validation
        market_data = MarketData.from_dict(quote_data)
        
        # Get reference data from other sources for cross-validation
        reference_data = {}
        for other_source in self._sources[DataSourceType.MARKET_DATA]:
            if other_source != source and other_source.health.status == DataSourceStatus.HEALTHY:
                try:
                    ref_quote = await other_source.get_quote(symbol)
                    reference_data[other_source.config.name] = MarketData.from_dict(ref_quote)
                except Exception:
                    pass
        
        # Validate the data
        quality_metrics = self._quality_validator.validate_data(
            market_data,
            source.source_type,
            reference_data
        )
        
        # Store metrics
        metric_key = f"{source.config.name}:{symbol}"
        self._quality_metrics[metric_key] = quality_metrics
        
        return quote_data, quality_metrics
    
    async def get_quote_with_best_quality(self, symbol: str) -> Dict[str, Any]:
        """Get quote from the source with best quality score."""
        best_quote = None
        best_metrics = None
        best_score = 0.0
        
        # Try all healthy sources
        for source in self._sources[DataSourceType.MARKET_DATA]:
            if source.health.status != DataSourceStatus.HEALTHY:
                continue
            
            try:
                # Apply rate limiting
                if source.config.name in self._rate_limiters:
                    await self._rate_limiters[source.config.name].acquire()
                
                # Get quote
                quote_data = await asyncio.wait_for(
                    source.get_quote(symbol),
                    timeout=source.config.timeout
                )
                
                # Validate
                market_data = MarketData.from_dict(quote_data)
                metrics = self._quality_validator.validate_data(
                    market_data,
                    source.source_type
                )
                
                # Check if this is the best so far
                if metrics.overall_score > best_score:
                    best_quote = quote_data
                    best_metrics = metrics
                    best_score = metrics.overall_score
                
                # Early exit if we found excellent quality
                if metrics.grade == QualityGrade.EXCELLENT:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error getting quote from {source.config.name}: {e}")
        
        if best_quote is None:
            raise Exception(f"No valid quote data available for {symbol}")
        
        return best_quote
    
    def get_quality_metrics(self, source_name: Optional[str] = None) -> Dict[str, QualityMetrics]:
        """Get quality metrics for sources."""
        if source_name:
            return {
                k: v for k, v in self._quality_metrics.items()
                if k.startswith(f"{source_name}:")
            }
        return self._quality_metrics.copy()
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report."""
        report = {
            "source_reliability": self._quality_validator.get_source_reliability_report(),
            "recent_metrics": {},
            "alerts": []
        }
        
        # Get recent metrics grouped by source
        for key, metrics in self._quality_metrics.items():
            source_name = key.split(":")[0]
            if source_name not in report["recent_metrics"]:
                report["recent_metrics"][source_name] = []
            
            report["recent_metrics"][source_name].append({
                "symbol": key.split(":")[1] if ":" in key else "unknown",
                "metrics": metrics.to_dict()
            })
        
        # Include recent alerts
        for key, metrics in self._quality_metrics.items():
            if metrics.anomalies:
                report["alerts"].append({
                    "source": key.split(":")[0],
                    "symbol": key.split(":")[1] if ":" in key else "unknown",
                    "anomalies": metrics.anomalies,
                    "timestamp": metrics.timestamp.isoformat()
                })
        
        return report
    
    def get_symbol_quality_trend(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get quality trend for a specific symbol."""
        return self._quality_validator.get_symbol_quality_trend(symbol, hours)