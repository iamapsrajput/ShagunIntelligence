"""
Real-time data streaming pipeline for aggregating multiple data sources.

This module provides a high-performance streaming system that:
- Manages multiple concurrent WebSocket connections
- Monitors data quality in real-time
- Provides automatic failover and reconnection
- Broadcasts updates to subscribed agents
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import time
from abc import ABC, abstractmethod

from loguru import logger
import websockets
from websockets.exceptions import WebSocketException
import aioredis
import numpy as np


class StreamStatus(Enum):
    """Status of a data stream."""
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    DEGRADED = "degraded"


class DataQualityStatus(Enum):
    """Real-time data quality status."""
    EXCELLENT = "excellent"  # <50ms latency, no gaps
    GOOD = "good"           # <100ms latency, minimal gaps
    DEGRADED = "degraded"   # <500ms latency, some gaps
    POOR = "poor"           # >500ms latency, significant gaps
    CRITICAL = "critical"   # Stream unusable


@dataclass
class StreamMetrics:
    """Metrics for monitoring stream health."""
    messages_received: int = 0
    messages_processed: int = 0
    errors_count: int = 0
    reconnection_count: int = 0
    average_latency_ms: float = 0.0
    last_message_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    data_gaps: int = 0
    quality_score: float = 1.0


@dataclass
class StreamConfig:
    """Configuration for a data stream."""
    name: str
    url: str
    priority: int = 5  # 1-10, higher is more important
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10
    heartbeat_interval: float = 30.0
    buffer_size: int = 1000
    quality_threshold: float = 0.8
    latency_threshold_ms: float = 100.0


@dataclass
class StreamMessage:
    """A message from a data stream."""
    stream_name: str
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    latency_ms: float
    quality_score: float
    sequence_number: Optional[int] = None


class DataStreamHandler(ABC):
    """Abstract base class for handling specific data streams."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.status = StreamStatus.DISCONNECTED
        self.metrics = StreamMetrics()
        self.websocket = None
        self.last_heartbeat = datetime.now()
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data stream."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close the connection."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        """Subscribe to specific symbols."""
        pass
    
    @abstractmethod
    async def process_message(self, message: Any) -> Optional[StreamMessage]:
        """Process a raw message from the stream."""
        pass
    
    @abstractmethod
    async def send_heartbeat(self):
        """Send heartbeat to keep connection alive."""
        pass
    
    def calculate_quality_score(self, latency_ms: float, has_gaps: bool = False) -> float:
        """Calculate quality score based on latency and data completeness."""
        # Latency component (0-0.7)
        if latency_ms < 50:
            latency_score = 0.7
        elif latency_ms < 100:
            latency_score = 0.6
        elif latency_ms < 200:
            latency_score = 0.4
        elif latency_ms < 500:
            latency_score = 0.2
        else:
            latency_score = 0.0
        
        # Reliability component (0-0.3)
        uptime_ratio = min(self.metrics.uptime_seconds / 3600, 1.0)  # Cap at 1 hour
        error_penalty = min(self.metrics.errors_count * 0.05, 0.2)
        reliability_score = (uptime_ratio * 0.3) - error_penalty
        
        # Gap penalty
        gap_penalty = 0.1 if has_gaps else 0.0
        
        total_score = max(0.0, latency_score + reliability_score - gap_penalty)
        return min(1.0, total_score)


class RealTimeDataPipeline:
    """
    High-performance real-time data aggregation pipeline.
    
    Features:
    - Multiple concurrent WebSocket streams
    - Automatic quality monitoring and alerting
    - Smart stream switching based on quality
    - Memory-efficient circular buffers
    - Agent notification system
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.streams: Dict[str, DataStreamHandler] = {}
        self.redis_url = redis_url
        self.redis_client = None
        
        # Data buffers (symbol -> deque of messages)
        self.data_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Stream quality tracking
        self.stream_quality: Dict[str, DataQualityStatus] = {}
        self.quality_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Subscribers for real-time updates
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        
        # Performance metrics
        self.pipeline_metrics = {
            'total_messages': 0,
            'messages_per_second': 0.0,
            'active_streams': 0,
            'total_subscribers': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_propagation_ms': 0.0
        }
        
        # Control flags
        self.running = False
        self.monitoring_task = None
        self.broadcasting_task = None
        
        # Quality thresholds for alerting
        self.quality_thresholds = {
            'critical_quality': 0.3,
            'poor_quality': 0.5,
            'degraded_quality': 0.7,
            'good_quality': 0.85
        }
        
        # Stream prioritization
        self.stream_priorities: Dict[str, int] = {}
        self.primary_streams: Dict[str, str] = {}  # symbol -> primary stream
        
        logger.info("RealTimeDataPipeline initialized")
    
    async def initialize(self):
        """Initialize the pipeline and connections."""
        try:
            # Connect to Redis for caching
            self.redis_client = await aioredis.create_redis_pool(self.redis_url)
            logger.info("Connected to Redis for caching")
            
            # Start monitoring tasks
            self.monitoring_task = asyncio.create_task(self._monitor_streams())
            self.broadcasting_task = asyncio.create_task(self._broadcast_updates())
            
            self.running = True
            logger.info("Pipeline initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def add_stream(self, handler: DataStreamHandler):
        """Add a new data stream handler."""
        self.streams[handler.config.name] = handler
        self.stream_priorities[handler.config.name] = handler.config.priority
        logger.info(f"Added stream: {handler.config.name} (priority: {handler.config.priority})")
    
    async def start_stream(self, stream_name: str, symbols: List[str]):
        """Start a specific data stream."""
        if stream_name not in self.streams:
            logger.error(f"Stream {stream_name} not found")
            return
        
        handler = self.streams[stream_name]
        
        try:
            # Connect to stream
            connected = await handler.connect()
            if not connected:
                raise Exception("Failed to connect")
            
            # Subscribe to symbols
            await handler.subscribe(symbols)
            
            # Start processing messages
            asyncio.create_task(self._process_stream(stream_name))
            
            logger.info(f"Started stream: {stream_name} for symbols: {symbols}")
            
        except Exception as e:
            logger.error(f"Failed to start stream {stream_name}: {e}")
            handler.status = StreamStatus.ERROR
    
    async def _process_stream(self, stream_name: str):
        """Process messages from a specific stream."""
        handler = self.streams[stream_name]
        reconnect_attempts = 0
        
        while self.running:
            try:
                if handler.status != StreamStatus.CONNECTED:
                    # Attempt reconnection
                    if reconnect_attempts >= handler.config.max_reconnect_attempts:
                        logger.error(f"Max reconnection attempts reached for {stream_name}")
                        handler.status = StreamStatus.ERROR
                        break
                    
                    await asyncio.sleep(handler.config.reconnect_interval)
                    connected = await handler.connect()
                    
                    if connected:
                        handler.metrics.reconnection_count += 1
                        reconnect_attempts = 0
                        logger.info(f"Reconnected to {stream_name}")
                    else:
                        reconnect_attempts += 1
                        continue
                
                # Process messages
                if handler.websocket:
                    try:
                        # Set timeout for receiving messages
                        message = await asyncio.wait_for(
                            handler.websocket.recv(),
                            timeout=handler.config.heartbeat_interval
                        )
                        
                        # Process the message
                        processed = await handler.process_message(message)
                        
                        if processed:
                            # Update metrics
                            handler.metrics.messages_received += 1
                            handler.metrics.last_message_time = datetime.now()
                            
                            # Add to buffer
                            self.data_buffers[processed.symbol].append(processed)
                            
                            # Update quality score
                            quality = self._calculate_stream_quality(handler)
                            self.stream_quality[stream_name] = quality
                            
                            # Cache in Redis
                            await self._cache_message(processed)
                            
                            # Notify subscribers
                            await self._notify_subscribers(processed)
                            
                            # Update pipeline metrics
                            self.pipeline_metrics['total_messages'] += 1
                            
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await handler.send_heartbeat()
                        
            except WebSocketException as e:
                logger.warning(f"WebSocket error in {stream_name}: {e}")
                handler.status = StreamStatus.DISCONNECTED
                
            except Exception as e:
                logger.error(f"Error processing stream {stream_name}: {e}")
                handler.metrics.errors_count += 1
                
                if handler.metrics.errors_count > 10:
                    handler.status = StreamStatus.ERROR
    
    def _calculate_stream_quality(self, handler: DataStreamHandler) -> DataQualityStatus:
        """Calculate real-time quality status for a stream."""
        metrics = handler.metrics
        
        # Check if stream is alive
        if handler.status != StreamStatus.CONNECTED:
            return DataQualityStatus.CRITICAL
        
        # Check message recency
        if metrics.last_message_time:
            time_since_last = (datetime.now() - metrics.last_message_time).total_seconds()
            if time_since_last > 60:  # No message for 1 minute
                return DataQualityStatus.POOR
        
        # Check latency
        if metrics.average_latency_ms < 50:
            latency_status = DataQualityStatus.EXCELLENT
        elif metrics.average_latency_ms < 100:
            latency_status = DataQualityStatus.GOOD
        elif metrics.average_latency_ms < 500:
            latency_status = DataQualityStatus.DEGRADED
        else:
            latency_status = DataQualityStatus.POOR
        
        # Check error rate
        if metrics.messages_received > 0:
            error_rate = metrics.errors_count / metrics.messages_received
            if error_rate > 0.1:  # >10% errors
                return DataQualityStatus.POOR
        
        # Check for data gaps
        if metrics.data_gaps > 5:
            return DataQualityStatus.DEGRADED
        
        return latency_status
    
    async def _monitor_streams(self):
        """Monitor stream health and quality."""
        while self.running:
            try:
                for stream_name, handler in self.streams.items():
                    # Update uptime
                    if handler.status == StreamStatus.CONNECTED:
                        handler.metrics.uptime_seconds += 5
                    
                    # Check stream health
                    quality = self._calculate_stream_quality(handler)
                    old_quality = self.stream_quality.get(stream_name)
                    
                    if old_quality != quality:
                        logger.info(f"Stream {stream_name} quality changed: {old_quality} -> {quality}")
                        
                        # Trigger quality alerts
                        if quality in [DataQualityStatus.POOR, DataQualityStatus.CRITICAL]:
                            await self._handle_quality_degradation(stream_name, quality)
                    
                    # Update quality history
                    self.quality_history[stream_name].append({
                        'timestamp': datetime.now(),
                        'quality': quality,
                        'metrics': handler.metrics.__dict__.copy()
                    })
                
                # Update pipeline metrics
                self._update_pipeline_metrics()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in stream monitoring: {e}")
    
    async def _handle_quality_degradation(self, stream_name: str, quality: DataQualityStatus):
        """Handle stream quality degradation."""
        logger.warning(f"Quality degradation detected for {stream_name}: {quality}")
        
        # Find symbols using this stream as primary
        affected_symbols = [
            symbol for symbol, primary in self.primary_streams.items()
            if primary == stream_name
        ]
        
        if not affected_symbols:
            return
        
        # Switch to backup stream for affected symbols
        for symbol in affected_symbols:
            backup_stream = await self._find_best_stream_for_symbol(symbol, exclude=[stream_name])
            
            if backup_stream:
                self.primary_streams[symbol] = backup_stream
                logger.info(f"Switched {symbol} from {stream_name} to {backup_stream}")
                
                # Notify subscribers of stream switch
                await self._notify_stream_switch(symbol, stream_name, backup_stream)
            else:
                logger.error(f"No backup stream available for {symbol}")
    
    async def _find_best_stream_for_symbol(
        self, 
        symbol: str, 
        exclude: List[str] = None
    ) -> Optional[str]:
        """Find the best quality stream for a symbol."""
        exclude = exclude or []
        best_stream = None
        best_score = 0.0
        
        for stream_name, handler in self.streams.items():
            if stream_name in exclude:
                continue
            
            if handler.status != StreamStatus.CONNECTED:
                continue
            
            # Calculate combined score (quality + priority)
            quality = self.stream_quality.get(stream_name, DataQualityStatus.POOR)
            quality_score = {
                DataQualityStatus.EXCELLENT: 1.0,
                DataQualityStatus.GOOD: 0.8,
                DataQualityStatus.DEGRADED: 0.5,
                DataQualityStatus.POOR: 0.2,
                DataQualityStatus.CRITICAL: 0.0
            }.get(quality, 0.0)
            
            priority_score = handler.config.priority / 10.0
            combined_score = (quality_score * 0.7) + (priority_score * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_stream = stream_name
        
        return best_stream
    
    async def _cache_message(self, message: StreamMessage):
        """Cache message in Redis for fast access."""
        if not self.redis_client:
            return
        
        try:
            # Create cache key
            cache_key = f"realtime:{message.symbol}:{message.stream_name}"
            
            # Serialize message
            message_data = {
                'symbol': message.symbol,
                'data': message.data,
                'timestamp': message.timestamp.isoformat(),
                'quality_score': message.quality_score,
                'latency_ms': message.latency_ms
            }
            
            # Store with expiration
            await self.redis_client.setex(
                cache_key,
                60,  # 1 minute expiration
                json.dumps(message_data)
            )
            
            # Also store in sorted set for time-series queries
            timestamp_score = message.timestamp.timestamp()
            series_key = f"timeseries:{message.symbol}:{message.stream_name}"
            
            await self.redis_client.zadd(
                series_key,
                timestamp_score,
                json.dumps(message_data)
            )
            
            # Trim old data (keep last 1000 points)
            await self.redis_client.zremrangebyrank(series_key, 0, -1001)
            
        except Exception as e:
            logger.error(f"Failed to cache message: {e}")
    
    async def _notify_subscribers(self, message: StreamMessage):
        """Notify subscribers of new data."""
        symbol_subscribers = self.subscribers.get(message.symbol, set())
        all_subscribers = self.subscribers.get('*', set())  # Wildcard subscribers
        
        all_callbacks = symbol_subscribers | all_subscribers
        
        if all_callbacks:
            # Create notification tasks
            notification_tasks = []
            for callback in all_callbacks:
                task = asyncio.create_task(self._safe_callback(callback, message))
                notification_tasks.append(task)
            
            # Wait for all notifications (with timeout)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*notification_tasks, return_exceptions=True),
                    timeout=0.1  # 100ms timeout for notifications
                )
            except asyncio.TimeoutError:
                logger.warning(f"Some notifications timed out for {message.symbol}")
    
    async def _safe_callback(self, callback: Callable, message: StreamMessage):
        """Safely execute a callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            logger.error(f"Error in subscriber callback: {e}")
    
    async def _notify_stream_switch(self, symbol: str, old_stream: str, new_stream: str):
        """Notify subscribers of stream switch."""
        notification = {
            'type': 'stream_switch',
            'symbol': symbol,
            'old_stream': old_stream,
            'new_stream': new_stream,
            'timestamp': datetime.now().isoformat(),
            'reason': 'quality_degradation'
        }
        
        # Notify quality monitoring subscribers
        quality_subscribers = self.subscribers.get('_quality_', set())
        for callback in quality_subscribers:
            await self._safe_callback(callback, notification)
    
    async def _broadcast_updates(self):
        """Broadcast pipeline status updates."""
        while self.running:
            try:
                # Prepare status update
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_metrics': self.pipeline_metrics,
                    'stream_status': {
                        name: {
                            'status': handler.status.value,
                            'quality': self.stream_quality.get(name, DataQualityStatus.CRITICAL).value,
                            'metrics': {
                                'messages': handler.metrics.messages_received,
                                'errors': handler.metrics.errors_count,
                                'latency_ms': handler.metrics.average_latency_ms,
                                'uptime_seconds': handler.metrics.uptime_seconds
                            }
                        }
                        for name, handler in self.streams.items()
                    }
                }
                
                # Notify status subscribers
                status_subscribers = self.subscribers.get('_status_', set())
                for callback in status_subscribers:
                    await self._safe_callback(callback, status)
                
                await asyncio.sleep(10)  # Broadcast every 10 seconds
                
            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")
    
    def _update_pipeline_metrics(self):
        """Update pipeline-level metrics."""
        # Calculate messages per second
        active_streams = sum(1 for h in self.streams.values() if h.status == StreamStatus.CONNECTED)
        total_subscribers = sum(len(subs) for subs in self.subscribers.values())
        
        self.pipeline_metrics.update({
            'active_streams': active_streams,
            'total_subscribers': total_subscribers,
            'timestamp': datetime.now().isoformat()
        })
        
        # Calculate average propagation time
        latencies = []
        for handler in self.streams.values():
            if handler.metrics.average_latency_ms > 0:
                latencies.append(handler.metrics.average_latency_ms)
        
        if latencies:
            self.pipeline_metrics['average_propagation_ms'] = np.mean(latencies)
    
    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to real-time updates for a symbol."""
        self.subscribers[symbol].add(callback)
        logger.info(f"Added subscriber for {symbol}")
    
    def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from updates."""
        if symbol in self.subscribers:
            self.subscribers[symbol].discard(callback)
    
    async def get_latest_data(self, symbol: str) -> Optional[StreamMessage]:
        """Get the latest data for a symbol."""
        # Check buffer first
        if symbol in self.data_buffers and self.data_buffers[symbol]:
            self.pipeline_metrics['cache_hits'] += 1
            return self.data_buffers[symbol][-1]
        
        # Check Redis cache
        if self.redis_client:
            try:
                primary_stream = self.primary_streams.get(symbol)
                if primary_stream:
                    cache_key = f"realtime:{symbol}:{primary_stream}"
                    cached = await self.redis_client.get(cache_key)
                    
                    if cached:
                        self.pipeline_metrics['cache_hits'] += 1
                        data = json.loads(cached)
                        return StreamMessage(
                            stream_name=primary_stream,
                            symbol=symbol,
                            data=data['data'],
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            latency_ms=data['latency_ms'],
                            quality_score=data['quality_score']
                        )
            except Exception as e:
                logger.error(f"Error retrieving from cache: {e}")
        
        self.pipeline_metrics['cache_misses'] += 1
        return None
    
    async def get_historical_buffer(
        self, 
        symbol: str, 
        seconds: int = 60
    ) -> List[StreamMessage]:
        """Get historical data from buffer."""
        if symbol not in self.data_buffers:
            return []
        
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        return [
            msg for msg in self.data_buffers[symbol]
            if msg.timestamp >= cutoff_time
        ]
    
    def get_stream_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report for all streams."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': self._calculate_overall_health(),
            'streams': {}
        }
        
        for stream_name, handler in self.streams.items():
            quality = self.stream_quality.get(stream_name, DataQualityStatus.CRITICAL)
            
            report['streams'][stream_name] = {
                'status': handler.status.value,
                'quality': quality.value,
                'metrics': {
                    'uptime_percentage': (handler.metrics.uptime_seconds / 3600) * 100,
                    'messages_per_minute': handler.metrics.messages_received / max(handler.metrics.uptime_seconds / 60, 1),
                    'error_rate': handler.metrics.errors_count / max(handler.metrics.messages_received, 1),
                    'average_latency_ms': handler.metrics.average_latency_ms,
                    'reconnections': handler.metrics.reconnection_count,
                    'last_message': handler.metrics.last_message_time.isoformat() if handler.metrics.last_message_time else None
                },
                'quality_history': [
                    {
                        'timestamp': h['timestamp'].isoformat(),
                        'quality': h['quality'].value
                    }
                    for h in list(self.quality_history[stream_name])[-10:]  # Last 10 entries
                ]
            }
        
        return report
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall pipeline health."""
        if not self.streams:
            return "no_streams"
        
        # Count streams by quality
        quality_counts = defaultdict(int)
        for quality in self.stream_quality.values():
            quality_counts[quality] += 1
        
        total_streams = len(self.streams)
        
        # Determine overall health
        if quality_counts[DataQualityStatus.CRITICAL] > 0:
            return "critical"
        elif quality_counts[DataQualityStatus.POOR] / total_streams > 0.3:
            return "poor"
        elif quality_counts[DataQualityStatus.DEGRADED] / total_streams > 0.5:
            return "degraded"
        elif quality_counts[DataQualityStatus.GOOD] / total_streams > 0.5:
            return "good"
        elif quality_counts[DataQualityStatus.EXCELLENT] / total_streams > 0.5:
            return "excellent"
        else:
            return "fair"
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline."""
        logger.info("Shutting down RealTimeDataPipeline")
        self.running = False
        
        # Cancel monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.broadcasting_task:
            self.broadcasting_task.cancel()
        
        # Disconnect all streams
        for stream_name, handler in self.streams.items():
            try:
                await handler.disconnect()
                logger.info(f"Disconnected stream: {stream_name}")
            except Exception as e:
                logger.error(f"Error disconnecting {stream_name}: {e}")
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        logger.info("Pipeline shutdown complete")