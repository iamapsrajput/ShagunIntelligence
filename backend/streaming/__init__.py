"""
Real-time streaming system for Shagun Intelligence.

This module provides high-performance data streaming from multiple sources
with quality monitoring, automatic failover, and agent integration.
"""

from .agent_integration import AgentNotification, StreamingAgentBridge
from .performance_monitor import PerformanceAlert, StreamingPerformanceMonitor
from .realtime_pipeline import (
    DataQualityStatus,
    DataStreamHandler,
    RealTimeDataPipeline,
    StreamConfig,
    StreamMessage,
    StreamMetrics,
    StreamStatus,
)
from .stream_manager import StreamManager, StreamManagerConfig, get_stream_manager

__all__ = [
    # Main components
    "StreamManager",
    "StreamManagerConfig",
    "get_stream_manager",
    "StreamingAgentBridge",
    "AgentNotification",
    "StreamingPerformanceMonitor",
    "PerformanceAlert",
    # Pipeline components
    "RealTimeDataPipeline",
    "StreamConfig",
    "StreamMessage",
    "StreamStatus",
    "DataQualityStatus",
    "StreamMetrics",
    "DataStreamHandler",
]

# Version
__version__ = "1.0.0"
