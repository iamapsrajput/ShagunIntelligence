"""Comprehensive logging and monitoring for Kite Connect service"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import psutil
from loguru import logger

from .exceptions import KiteException


class MetricType(Enum):
    """Types of metrics to track"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class LogLevel(Enum):
    """Log levels for operations"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class MetricPoint:
    """Single metric data point"""

    name: str
    value: float
    timestamp: float
    tags: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class PerformanceMetrics:
    """Performance metrics for API operations"""

    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_type: str | None = None
    error_message: str | None = None
    request_size: int | None = None
    response_size: int | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class APIStats:
    """API usage statistics"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float("inf")
    requests_per_minute: float = 0.0
    error_rate: float = 0.0


class KiteLogger:
    """Enhanced logging system for Kite Connect operations"""

    def __init__(self, log_file: str = "logs/kite_trading.log"):
        self.log_file = log_file
        self.setup_logger()

    def setup_logger(self):
        """Configure loguru logger"""
        # Remove default handler
        logger.remove()

        # Add console handler with colors
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG",
            colorize=True,
        )

        # Add file handler
        logger.add(
            sink=self.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )

        # Add error file handler
        logger.add(
            sink=self.log_file.replace(".log", "_errors.log"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="50 MB",
            retention="30 days",
        )

    def log_api_call(
        self, operation: str, params: dict[str, Any], level: LogLevel = LogLevel.INFO
    ):
        """Log API call with parameters"""
        sanitized_params = self._sanitize_params(params)
        message = f"API Call: {operation} | Params: {sanitized_params}"

        if level == LogLevel.DEBUG:
            logger.debug(message)
        elif level == LogLevel.INFO:
            logger.info(message)
        elif level == LogLevel.WARNING:
            logger.warning(message)
        elif level == LogLevel.ERROR:
            logger.error(message)
        elif level == LogLevel.CRITICAL:
            logger.critical(message)

    def log_api_response(
        self, operation: str, response_size: int, duration: float, success: bool
    ):
        """Log API response details"""
        status = "SUCCESS" if success else "FAILED"
        message = f"API Response: {operation} | Status: {status} | Duration: {duration:.3f}s | Size: {response_size} bytes"

        if success:
            logger.info(message)
        else:
            logger.error(message)

    def log_order_event(self, event_type: str, order_data: dict[str, Any]):
        """Log order-related events"""
        sanitized_data = self._sanitize_order_data(order_data)
        message = f"Order Event: {event_type} | Data: {sanitized_data}"
        logger.info(message)

    def log_websocket_event(self, event_type: str, details: dict[str, Any]):
        """Log WebSocket events"""
        message = f"WebSocket Event: {event_type} | Details: {details}"
        logger.info(message)

    def log_error(
        self, operation: str, error: Exception, context: dict[str, Any] = None
    ):
        """Log errors with context"""
        context = context or {}
        error_type = type(error).__name__
        message = f"Error in {operation}: {error_type} - {str(error)}"

        if context:
            sanitized_context = self._sanitize_params(context)
            message += f" | Context: {sanitized_context}"

        if isinstance(error, KiteException):
            logger.error(message)
        else:
            logger.exception(message)

    def _sanitize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive data from parameters"""
        sensitive_keys = {"access_token", "api_secret", "password", "api_key"}
        sanitized = {}

        for key, value in params.items():
            if key.lower() in sensitive_keys:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_params(value)
            else:
                sanitized[key] = value

        return sanitized

    def _sanitize_order_data(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize order data for logging"""
        # Include important fields but exclude sensitive ones
        important_fields = [
            "order_id",
            "symbol",
            "transaction_type",
            "quantity",
            "price",
            "order_type",
            "status",
            "average_price",
        ]

        sanitized = {}
        for field in important_fields:
            if field in order_data:
                sanitized[field] = order_data[field]

        return sanitized


class MetricsCollector:
    """Metrics collection and aggregation system"""

    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.api_stats: dict[str, APIStats] = defaultdict(APIStats)
        self.performance_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

        # Start background aggregation
        self._start_aggregation_task()

    def record_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] = None,
        metric_type: MetricType = MetricType.GAUGE,
    ):
        """Record a metric point"""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metric_type=metric_type,
        )

        with self._lock:
            self.metrics.append(metric)

    def record_api_performance(self, metrics: PerformanceMetrics):
        """Record API performance metrics"""
        with self._lock:
            self.performance_history.append(metrics)

            # Update API stats
            stats = self.api_stats[metrics.operation]
            stats.total_requests += 1

            if metrics.success:
                stats.successful_requests += 1
            else:
                stats.failed_requests += 1

            # Update response time stats
            if metrics.duration < stats.min_response_time:
                stats.min_response_time = metrics.duration
            if metrics.duration > stats.max_response_time:
                stats.max_response_time = metrics.duration

            # Calculate average response time
            total_duration = (
                stats.avg_response_time * (stats.total_requests - 1) + metrics.duration
            )
            stats.avg_response_time = total_duration / stats.total_requests

            # Calculate error rate
            stats.error_rate = (stats.failed_requests / stats.total_requests) * 100

    def get_metrics_summary(self, time_window: int = 3600) -> dict[str, Any]:
        """Get metrics summary for the specified time window (seconds)"""
        cutoff_time = time.time() - time_window

        with self._lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            recent_performance = [
                p for p in self.performance_history if p.start_time >= cutoff_time
            ]

        summary = {
            "total_metrics": len(recent_metrics),
            "total_api_calls": len(recent_performance),
            "successful_calls": sum(1 for p in recent_performance if p.success),
            "failed_calls": sum(1 for p in recent_performance if not p.success),
            "avg_response_time": 0.0,
            "operations": {},
            "system_metrics": self._get_system_metrics(),
        }

        if recent_performance:
            summary["avg_response_time"] = sum(
                p.duration for p in recent_performance
            ) / len(recent_performance)

            # Group by operation
            operations = defaultdict(list)
            for perf in recent_performance:
                operations[perf.operation].append(perf)

            for operation, perfs in operations.items():
                summary["operations"][operation] = {
                    "total_calls": len(perfs),
                    "successful_calls": sum(1 for p in perfs if p.success),
                    "avg_duration": sum(p.duration for p in perfs) / len(perfs),
                    "error_rate": (sum(1 for p in perfs if not p.success) / len(perfs))
                    * 100,
                }

        return summary

    def get_api_stats(self) -> dict[str, APIStats]:
        """Get current API statistics"""
        with self._lock:
            return {op: asdict(stats) for op, stats in self.api_stats.items()}

    def _get_system_metrics(self) -> dict[str, float]:
        """Get system resource metrics"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "network_sent": psutil.net_io_counters().bytes_sent,
                "network_recv": psutil.net_io_counters().bytes_recv,
            }
        except Exception:
            return {}

    def _start_aggregation_task(self):
        """Start background task for metrics aggregation"""

        def aggregate_metrics():
            while True:
                try:
                    # Calculate requests per minute for each operation
                    cutoff_time = time.time() - 60  # Last minute

                    with self._lock:
                        recent_calls = [
                            p
                            for p in self.performance_history
                            if p.start_time >= cutoff_time
                        ]

                    operations = defaultdict(int)
                    for call in recent_calls:
                        operations[call.operation] += 1

                    for operation, count in operations.items():
                        self.api_stats[operation].requests_per_minute = count

                    time.sleep(60)  # Run every minute

                except Exception as e:
                    logger.error(f"Error in metrics aggregation: {str(e)}")
                    time.sleep(60)

        thread = threading.Thread(target=aggregate_metrics, daemon=True)
        thread.start()


class PerformanceMonitor:
    """Performance monitoring and alerting"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_thresholds = {
            "response_time": 5.0,  # seconds
            "error_rate": 10.0,  # percentage
            "memory_usage": 80.0,  # percentage
            "cpu_usage": 80.0,  # percentage
        }
        self.alert_callbacks: list[Callable] = []

    def monitor_operation(self, operation_name: str):
        """Decorator to monitor operation performance"""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                success = False
                error_type = None
                error_message = None

                try:
                    result = await func(*args, **kwargs)
                    success = True
                    return result

                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    raise

                finally:
                    end_time = time.time()
                    duration = end_time - start_time

                    # Record performance metrics
                    perf_metrics = PerformanceMetrics(
                        operation=operation_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        success=success,
                        error_type=error_type,
                        error_message=error_message,
                    )

                    self.metrics_collector.record_api_performance(perf_metrics)

                    # Check for alerts
                    await self._check_alerts(operation_name, duration, success)

            return wrapper

        return decorator

    async def _check_alerts(self, operation: str, duration: float, success: bool):
        """Check if any alert thresholds are breached"""
        alerts = []

        # Response time alert
        if duration > self.alert_thresholds["response_time"]:
            alerts.append(
                {
                    "type": "response_time",
                    "operation": operation,
                    "value": duration,
                    "threshold": self.alert_thresholds["response_time"],
                    "message": f"Slow response time for {operation}: {duration:.2f}s",
                }
            )

        # Error rate alert
        stats = self.metrics_collector.get_api_stats().get(operation)
        if stats and stats["error_rate"] > self.alert_thresholds["error_rate"]:
            alerts.append(
                {
                    "type": "error_rate",
                    "operation": operation,
                    "value": stats["error_rate"],
                    "threshold": self.alert_thresholds["error_rate"],
                    "message": f"High error rate for {operation}: {stats['error_rate']:.1f}%",
                }
            )

        # System resource alerts
        system_metrics = self.metrics_collector._get_system_metrics()

        if (
            system_metrics.get("memory_percent", 0)
            > self.alert_thresholds["memory_usage"]
        ):
            alerts.append(
                {
                    "type": "memory_usage",
                    "value": system_metrics["memory_percent"],
                    "threshold": self.alert_thresholds["memory_usage"],
                    "message": f"High memory usage: {system_metrics['memory_percent']:.1f}%",
                }
            )

        if system_metrics.get("cpu_percent", 0) > self.alert_thresholds["cpu_usage"]:
            alerts.append(
                {
                    "type": "cpu_usage",
                    "value": system_metrics["cpu_percent"],
                    "threshold": self.alert_thresholds["cpu_usage"],
                    "message": f"High CPU usage: {system_metrics['cpu_percent']:.1f}%",
                }
            )

        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)

    async def _send_alert(self, alert: dict[str, Any]):
        """Send alert to registered callbacks"""
        logger.warning(f"ALERT: {alert['message']}")

        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")

    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)

    def set_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold"""
        self.alert_thresholds[metric_name] = threshold


class KiteMonitoringService:
    """Complete monitoring service for Kite Connect operations"""

    def __init__(self):
        self.logger = KiteLogger()
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        self.is_monitoring = False

    def start_monitoring(self):
        """Start the monitoring service"""
        self.is_monitoring = True
        logger.info("Kite monitoring service started")

    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.is_monitoring = False
        logger.info("Kite monitoring service stopped")

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status"""
        try:
            summary = self.metrics_collector.get_metrics_summary()
            api_stats = self.metrics_collector.get_api_stats()

            # Determine overall health
            health_score = 100.0
            issues = []

            # Check error rates
            for operation, stats in api_stats.items():
                if stats["error_rate"] > 10:
                    health_score -= 10
                    issues.append(
                        f"High error rate for {operation}: {stats['error_rate']:.1f}%"
                    )

            # Check response times
            if summary["avg_response_time"] > 3.0:
                health_score -= 20
                issues.append(
                    f"Slow average response time: {summary['avg_response_time']:.2f}s"
                )

            # Check system resources
            system_metrics = summary.get("system_metrics", {})
            if system_metrics.get("memory_percent", 0) > 80:
                health_score -= 15
                issues.append(
                    f"High memory usage: {system_metrics['memory_percent']:.1f}%"
                )

            if system_metrics.get("cpu_percent", 0) > 80:
                health_score -= 15
                issues.append(f"High CPU usage: {system_metrics['cpu_percent']:.1f}%")

            # Determine status
            if health_score >= 90:
                status = "healthy"
            elif health_score >= 70:
                status = "warning"
            else:
                status = "critical"

            return {
                "status": status,
                "health_score": max(0, health_score),
                "issues": issues,
                "metrics_summary": summary,
                "api_statistics": api_stats,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                "status": "error",
                "health_score": 0,
                "issues": [f"Monitoring error: {str(e)}"],
                "timestamp": datetime.now().isoformat(),
            }


# Global monitoring service instance
monitoring_service = KiteMonitoringService()
