"""
Real-time Data Processing Pipeline for Shagun Intelligence

This module provides a high-performance data pipeline for processing live market data
from Zerodha WebSocket, including:
- Asyncio-based WebSocket client
- In-memory data buffering with historical storage
- Shared memory for inter-agent communication
- Low-latency data distribution
- Data validation and error recovery
- Performance monitoring and optimization
- Optional Redis caching
"""

from .pipeline import DataPipeline, PipelineConfig
from .websocket_client import ZerodhaWebSocketClient, TickMode, MessageType
from .data_buffer import DataBuffer, TickData, OHLCData
from .data_synchronizer import DataSynchronizer, SharedDataSegment
from .data_validator import DataValidator, ValidationRule
from .data_distributor import DataDistributor, DistributionMode, Subscriber
from .performance_tracker import PerformanceTracker, PerformanceMetric
from .redis_cache import RedisCacheManager

__all__ = [
    # Main pipeline
    "DataPipeline",
    "PipelineConfig",
    
    # WebSocket client
    "ZerodhaWebSocketClient",
    "TickMode",
    "MessageType",
    
    # Data buffer
    "DataBuffer",
    "TickData",
    "OHLCData",
    
    # Synchronizer
    "DataSynchronizer",
    "SharedDataSegment",
    
    # Validator
    "DataValidator",
    "ValidationRule",
    
    # Distributor
    "DataDistributor",
    "DistributionMode",
    "Subscriber",
    
    # Performance
    "PerformanceTracker",
    "PerformanceMetric",
    
    # Redis cache
    "RedisCacheManager"
]