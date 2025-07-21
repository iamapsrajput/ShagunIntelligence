import logging
from typing import Dict, Any, Deque, Optional
from datetime import datetime, timedelta
from collections import deque, defaultdict
import time
import asyncio
from dataclasses import dataclass, field
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric with time window"""
    name: str
    window_size: int  # seconds
    values: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=10000))
    
    def add_value(self, value: float, timestamp: Optional[float] = None):
        """Add a value with timestamp"""
        if timestamp is None:
            timestamp = time.time()
        self.values.append((timestamp, value))
    
    def get_current_value(self, window_seconds: Optional[int] = None) -> Dict[str, float]:
        """Get current metric values within time window"""
        if not self.values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        
        current_time = time.time()
        window = window_seconds or self.window_size
        cutoff_time = current_time - window
        
        # Filter values within window
        recent_values = [v for t, v in self.values if t >= cutoff_time]
        
        if not recent_values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        
        values_array = np.array(recent_values)
        
        return {
            "count": len(recent_values),
            "mean": np.mean(values_array),
            "min": np.min(values_array),
            "max": np.max(values_array),
            "p50": np.percentile(values_array, 50),
            "p95": np.percentile(values_array, 95),
            "p99": np.percentile(values_array, 99),
            "std": np.std(values_array)
        }


class PerformanceTracker:
    """Tracks and monitors data pipeline performance metrics"""
    
    def __init__(self):
        # Performance metrics
        self.metrics = {
            "tick_processing_time": PerformanceMetric("tick_processing_time", 60),
            "websocket_latency": PerformanceMetric("websocket_latency", 60),
            "distribution_latency": PerformanceMetric("distribution_latency", 60),
            "validation_time": PerformanceMetric("validation_time", 60),
            "buffer_write_time": PerformanceMetric("buffer_write_time", 60),
            "sync_time": PerformanceMetric("sync_time", 60)
        }
        
        # Counters
        self.counters = defaultdict(int)
        self.counter_start_time = time.time()
        
        # Rate metrics
        self.rate_metrics = {
            "ticks_per_second": deque(maxlen=300),  # 5 minutes
            "errors_per_minute": deque(maxlen=60),  # 1 hour
            "bytes_per_second": deque(maxlen=300)
        }
        
        # System metrics
        self.system_metrics = {
            "memory_usage_mb": deque(maxlen=300),
            "cpu_percent": deque(maxlen=300),
            "active_connections": deque(maxlen=300)
        }
        
        # Performance thresholds
        self.thresholds = {
            "tick_processing_time_ms": 5,  # 5ms warning threshold
            "websocket_latency_ms": 10,    # 10ms warning threshold
            "error_rate_percent": 1,        # 1% error rate warning
            "memory_usage_mb": 500          # 500MB warning threshold
        }
        
        # Alert tracking
        self.alerts = deque(maxlen=100)
        self.last_alert_time = {}
        self.alert_cooldown = 60  # seconds
        
        # Monitoring task
        self.monitoring_task = None
        self.is_monitoring = False
        
        logger.info("PerformanceTracker initialized")
    
    def record_tick_processed(self, processing_time_seconds: float) -> None:
        """Record tick processing time"""
        processing_time_ms = processing_time_seconds * 1000
        self.metrics["tick_processing_time"].add_value(processing_time_ms)
        self.counters["ticks_processed"] += 1
        
        # Check threshold
        if processing_time_ms > self.thresholds["tick_processing_time_ms"]:
            self._raise_alert(
                "high_processing_time",
                f"Tick processing time {processing_time_ms:.2f}ms exceeds threshold"
            )
    
    def record_websocket_latency(self, latency_ms: float) -> None:
        """Record WebSocket latency"""
        self.metrics["websocket_latency"].add_value(latency_ms)
        
        if latency_ms > self.thresholds["websocket_latency_ms"]:
            self._raise_alert(
                "high_websocket_latency",
                f"WebSocket latency {latency_ms:.2f}ms exceeds threshold"
            )
    
    def record_distribution_latency(self, latency_ms: float) -> None:
        """Record distribution latency"""
        self.metrics["distribution_latency"].add_value(latency_ms)
    
    def record_validation_time(self, time_ms: float) -> None:
        """Record data validation time"""
        self.metrics["validation_time"].add_value(time_ms)
    
    def record_buffer_write(self, time_ms: float) -> None:
        """Record buffer write time"""
        self.metrics["buffer_write_time"].add_value(time_ms)
    
    def record_sync_time(self, time_ms: float) -> None:
        """Record synchronization time"""
        self.metrics["sync_time"].add_value(time_ms)
    
    def record_error(self, error_type: str = "general") -> None:
        """Record an error occurrence"""
        self.counters["errors_total"] += 1
        self.counters[f"errors_{error_type}"] += 1
    
    def record_invalid_tick(self) -> None:
        """Record invalid tick data"""
        self.counters["invalid_ticks"] += 1
    
    def record_bytes_processed(self, bytes_count: int) -> None:
        """Record bytes processed"""
        self.counters["bytes_processed"] += bytes_count
    
    def record_connection_count(self, count: int) -> None:
        """Record active connection count"""
        self.system_metrics["active_connections"].append((time.time(), count))
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitor_loop())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        last_tick_count = 0
        last_byte_count = 0
        last_error_count = 0
        
        while self.is_monitoring:
            try:
                await asyncio.sleep(1)  # Update every second
                
                current_time = time.time()
                
                # Calculate rates
                tick_count = self.counters["ticks_processed"]
                ticks_per_second = tick_count - last_tick_count
                self.rate_metrics["ticks_per_second"].append((current_time, ticks_per_second))
                last_tick_count = tick_count
                
                byte_count = self.counters["bytes_processed"]
                bytes_per_second = byte_count - last_byte_count
                self.rate_metrics["bytes_per_second"].append((current_time, bytes_per_second))
                last_byte_count = byte_count
                
                # Error rate (per minute)
                if int(current_time) % 60 == 0:  # Every minute
                    error_count = self.counters["errors_total"]
                    errors_per_minute = error_count - last_error_count
                    self.rate_metrics["errors_per_minute"].append((current_time, errors_per_minute))
                    last_error_count = error_count
                    
                    # Check error rate threshold
                    total_operations = self.counters["ticks_processed"] + self.counters["errors_total"]
                    if total_operations > 0:
                        error_rate = (self.counters["errors_total"] / total_operations) * 100
                        if error_rate > self.thresholds["error_rate_percent"]:
                            self._raise_alert(
                                "high_error_rate",
                                f"Error rate {error_rate:.2f}% exceeds threshold"
                            )
                
                # Collect system metrics
                await self._collect_system_metrics()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        try:
            import psutil
            import os
            
            # Memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.system_metrics["memory_usage_mb"].append((time.time(), memory_mb))
            
            if memory_mb > self.thresholds["memory_usage_mb"]:
                self._raise_alert(
                    "high_memory_usage",
                    f"Memory usage {memory_mb:.2f}MB exceeds threshold"
                )
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            self.system_metrics["cpu_percent"].append((time.time(), cpu_percent))
            
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.debug(f"Error collecting system metrics: {str(e)}")
    
    def _raise_alert(self, alert_type: str, message: str) -> None:
        """Raise a performance alert"""
        current_time = time.time()
        
        # Check cooldown
        if alert_type in self.last_alert_time:
            if current_time - self.last_alert_time[alert_type] < self.alert_cooldown:
                return
        
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message
        }
        
        self.alerts.append(alert)
        self.last_alert_time[alert_type] = current_time
        
        logger.warning(f"Performance alert: {alert_type} - {message}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics_summary = {}
        
        # Get metric summaries
        for name, metric in self.metrics.items():
            metrics_summary[name] = metric.get_current_value()
        
        # Calculate rates
        current_time = time.time()
        runtime_seconds = current_time - self.counter_start_time
        
        # Get recent rates
        ticks_per_second = 0
        if self.rate_metrics["ticks_per_second"]:
            recent_tps = [v for t, v in self.rate_metrics["ticks_per_second"] 
                         if t > current_time - 10]  # Last 10 seconds
            if recent_tps:
                ticks_per_second = np.mean(recent_tps)
        
        errors_per_minute = 0
        if self.rate_metrics["errors_per_minute"]:
            recent_epm = [v for t, v in self.rate_metrics["errors_per_minute"] 
                         if t > current_time - 300]  # Last 5 minutes
            if recent_epm:
                errors_per_minute = np.mean(recent_epm)
        
        # Calculate error rate
        total_operations = self.counters["ticks_processed"] + self.counters["errors_total"]
        error_rate = (self.counters["errors_total"] / total_operations * 100) if total_operations > 0 else 0
        
        return {
            "metrics": metrics_summary,
            "counters": dict(self.counters),
            "rates": {
                "ticks_per_second": ticks_per_second,
                "errors_per_minute": errors_per_minute,
                "error_rate": error_rate,
                "bytes_per_second": self._get_recent_rate("bytes_per_second", 10)
            },
            "system": {
                "memory_usage_mb": self._get_recent_value("memory_usage_mb"),
                "cpu_percent": self._get_recent_value("cpu_percent"),
                "active_connections": self._get_recent_value("active_connections")
            },
            "runtime_seconds": runtime_seconds,
            "recent_alerts": list(self.alerts)[-10:]  # Last 10 alerts
        }
    
    def _get_recent_rate(self, metric_name: str, window_seconds: int) -> float:
        """Get recent rate for a metric"""
        if metric_name not in self.rate_metrics:
            return 0
        
        current_time = time.time()
        recent_values = [v for t, v in self.rate_metrics[metric_name] 
                        if t > current_time - window_seconds]
        
        return np.mean(recent_values) if recent_values else 0
    
    def _get_recent_value(self, metric_name: str) -> float:
        """Get most recent value for a system metric"""
        if metric_name not in self.system_metrics or not self.system_metrics[metric_name]:
            return 0
        
        return self.system_metrics[metric_name][-1][1]
    
    def reset_interval_metrics(self) -> None:
        """Reset metrics for a new interval"""
        # Keep counters but reset rate calculations
        self.rate_metrics["ticks_per_second"].clear()
        self.rate_metrics["errors_per_minute"].clear()
        self.rate_metrics["bytes_per_second"].clear()
    
    def get_performance_report(self) -> str:
        """Generate a performance report"""
        metrics = self.get_metrics()
        
        report = f"""
=== Data Pipeline Performance Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Runtime: {metrics['runtime_seconds']:.0f} seconds

THROUGHPUT
----------
Ticks/Second: {metrics['rates']['ticks_per_second']:.2f}
Bytes/Second: {metrics['rates']['bytes_per_second']:.2f}
Total Ticks: {metrics['counters']['ticks_processed']:,}

LATENCY (ms)
-----------
Tick Processing: {metrics['metrics']['tick_processing_time']['mean']:.2f} (p95: {metrics['metrics']['tick_processing_time']['p95']:.2f})
WebSocket: {metrics['metrics']['websocket_latency']['mean']:.2f} (p95: {metrics['metrics']['websocket_latency']['p95']:.2f})
Distribution: {metrics['metrics']['distribution_latency']['mean']:.2f} (p95: {metrics['metrics']['distribution_latency']['p95']:.2f})

ERROR METRICS
------------
Error Rate: {metrics['rates']['error_rate']:.2f}%
Total Errors: {metrics['counters']['errors_total']}
Invalid Ticks: {metrics['counters']['invalid_ticks']}

SYSTEM METRICS
-------------
Memory Usage: {metrics['system']['memory_usage_mb']:.2f} MB
CPU Usage: {metrics['system']['cpu_percent']:.1f}%
Active Connections: {metrics['system']['active_connections']:.0f}
"""
        
        # Add recent alerts if any
        if metrics["recent_alerts"]:
            report += "\nRECENT ALERTS\n-------------\n"
            for alert in metrics["recent_alerts"][-5:]:
                report += f"{alert['timestamp']}: {alert['type']} - {alert['message']}\n"
        
        return report