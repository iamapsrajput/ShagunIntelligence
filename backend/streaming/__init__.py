"""
Real-time streaming system for Shagun Intelligence.

This module provides high-performance data streaming from multiple sources
with quality monitoring, automatic failover, and agent integration.
"""

from .stream_manager import StreamManager, StreamManagerConfig, get_stream_manager
from .agent_integration import StreamingAgentBridge, AgentNotification
from .performance_monitor import StreamingPerformanceMonitor, PerformanceAlert
from .realtime_pipeline import (
    RealTimeDataPipeline,
    StreamConfig,
    StreamMessage,
    StreamStatus,
    DataQualityStatus,
    StreamMetrics,
    DataStreamHandler
)

__all__ = [
    # Main components
    'StreamManager',
    'StreamManagerConfig',
    'get_stream_manager',
    'StreamingAgentBridge',
    'AgentNotification',
    'StreamingPerformanceMonitor',
    'PerformanceAlert',
    
    # Pipeline components
    'RealTimeDataPipeline',
    'StreamConfig',
    'StreamMessage',
    'StreamStatus',
    'DataQualityStatus',
    'StreamMetrics',
    'DataStreamHandler'
]

# Version
__version__ = '1.0.0'