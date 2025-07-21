"""
Performance monitoring for the real-time streaming system.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import numpy as np
from loguru import logger
from dataclasses import dataclass, field
import psutil
import time


@dataclass
class PerformanceMetric:
    """A single performance measurement."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert when thresholds are breached."""
    timestamp: datetime
    severity: str  # 'warning', 'critical'
    metric_name: str
    current_value: float
    threshold: float
    message: str


class StreamingPerformanceMonitor:
    """
    Monitors performance of the real-time streaming system.
    
    Tracks:
    - Message throughput
    - Latency distribution
    - Memory usage
    - CPU usage
    - Stream health metrics
    - Data quality trends
    """
    
    def __init__(self, alert_callback: Optional[callable] = None):
        self.alert_callback = alert_callback
        
        # Metric storage (circular buffers)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance thresholds
        self.thresholds = {
            'latency_p95_ms': 100,  # 95th percentile latency
            'latency_p99_ms': 200,  # 99th percentile latency
            'messages_per_second': 10,  # Minimum throughput
            'error_rate': 0.05,  # 5% error rate
            'memory_usage_mb': 1000,  # 1GB memory
            'cpu_percent': 80,  # 80% CPU
            'stream_quality_score': 0.7,  # Minimum quality
            'data_gaps_per_minute': 5  # Maximum gaps
        }
        
        # Tracking variables
        self.start_time = datetime.now()
        self.total_messages = 0
        self.total_errors = 0
        self.message_timestamps = deque(maxlen=1000)
        self.latency_samples = deque(maxlen=1000)
        
        # Alert history
        self.alerts: List[PerformanceAlert] = []
        self.alert_cooldown: Dict[str, datetime] = {}
        
        # Monitoring task
        self.monitoring_task = None
        self.running = False
        
        logger.info("StreamingPerformanceMonitor initialized")
    
    def start_monitoring(self):
        """Start the performance monitoring loop."""
        if not self.running:
            self.running = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Calculate derived metrics
                await self._calculate_performance_metrics()
                
                # Check thresholds and raise alerts
                await self._check_thresholds()
                
                # Wait before next collection
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        timestamp = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self._record_metric('cpu_percent', cpu_percent, '%', timestamp)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        self._record_metric('memory_usage_mb', memory_mb, 'MB', timestamp)
        self._record_metric('memory_percent', memory.percent, '%', timestamp)
        
        # Process-specific metrics
        try:
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024
            process_cpu = process.cpu_percent()
            
            self._record_metric('process_memory_mb', process_memory, 'MB', timestamp)
            self._record_metric('process_cpu_percent', process_cpu, '%', timestamp)
        except:
            pass
    
    async def _calculate_performance_metrics(self):
        """Calculate derived performance metrics."""
        timestamp = datetime.now()
        
        # Message throughput
        if self.message_timestamps:
            # Calculate messages per second over last 60 seconds
            cutoff = timestamp - timedelta(seconds=60)
            recent_messages = [t for t in self.message_timestamps if t > cutoff]
            
            if len(recent_messages) > 1:
                duration = (recent_messages[-1] - recent_messages[0]).total_seconds()
                if duration > 0:
                    mps = len(recent_messages) / duration
                    self._record_metric('messages_per_second', mps, 'msg/s', timestamp)
        
        # Latency percentiles
        if self.latency_samples:
            latencies = list(self.latency_samples)
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            
            self._record_metric('latency_p50_ms', p50, 'ms', timestamp)
            self._record_metric('latency_p95_ms', p95, 'ms', timestamp)
            self._record_metric('latency_p99_ms', p99, 'ms', timestamp)
        
        # Error rate
        if self.total_messages > 0:
            error_rate = self.total_errors / self.total_messages
            self._record_metric('error_rate', error_rate, 'ratio', timestamp)
    
    def record_message(self, latency_ms: float, quality_score: float, stream_name: str):
        """Record a processed message."""
        timestamp = datetime.now()
        
        self.total_messages += 1
        self.message_timestamps.append(timestamp)
        self.latency_samples.append(latency_ms)
        
        # Record per-stream metrics
        self._record_metric(
            f'stream_{stream_name}_latency_ms', 
            latency_ms, 
            'ms', 
            timestamp
        )
        self._record_metric(
            f'stream_{stream_name}_quality', 
            quality_score, 
            'score', 
            timestamp
        )
    
    def record_error(self, error_type: str, stream_name: str):
        """Record an error occurrence."""
        self.total_errors += 1
        timestamp = datetime.now()
        
        self._record_metric(
            f'errors_{error_type}', 
            1, 
            'count', 
            timestamp,
            {'stream': stream_name}
        )
    
    def record_stream_metrics(self, stream_name: str, metrics: Dict[str, Any]):
        """Record stream-specific metrics."""
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self._record_metric(
                    f'stream_{stream_name}_{metric_name}',
                    value,
                    'auto',
                    timestamp
                )
    
    def _record_metric(
        self, 
        name: str, 
        value: float, 
        unit: str, 
        timestamp: datetime,
        metadata: Dict[str, Any] = None
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        self.metrics[name].append(metric)
    
    async def _check_thresholds(self):
        """Check metrics against thresholds and raise alerts."""
        timestamp = datetime.now()
        
        for metric_name, threshold in self.thresholds.items():
            # Get latest metric value
            if metric_name in self.metrics and self.metrics[metric_name]:
                latest = self.metrics[metric_name][-1]
                
                # Check if threshold is breached
                if self._is_threshold_breached(metric_name, latest.value, threshold):
                    # Check cooldown
                    if metric_name in self.alert_cooldown:
                        if timestamp - self.alert_cooldown[metric_name] < timedelta(minutes=5):
                            continue
                    
                    # Create alert
                    severity = 'critical' if self._is_critical(metric_name, latest.value) else 'warning'
                    
                    alert = PerformanceAlert(
                        timestamp=timestamp,
                        severity=severity,
                        metric_name=metric_name,
                        current_value=latest.value,
                        threshold=threshold,
                        message=self._get_alert_message(metric_name, latest.value, threshold)
                    )
                    
                    await self._raise_alert(alert)
                    self.alert_cooldown[metric_name] = timestamp
    
    def _is_threshold_breached(self, metric_name: str, value: float, threshold: float) -> bool:
        """Check if a threshold is breached."""
        # For some metrics, lower is worse
        if metric_name in ['messages_per_second', 'stream_quality_score']:
            return value < threshold
        else:
            return value > threshold
    
    def _is_critical(self, metric_name: str, value: float) -> bool:
        """Determine if a breach is critical."""
        threshold = self.thresholds[metric_name]
        
        # Define critical multipliers
        critical_multipliers = {
            'latency_p95_ms': 2.0,  # 2x threshold
            'latency_p99_ms': 2.0,
            'error_rate': 2.0,
            'memory_usage_mb': 1.5,
            'cpu_percent': 1.1,  # 110%
        }
        
        multiplier = critical_multipliers.get(metric_name, 1.5)
        
        if metric_name in ['messages_per_second', 'stream_quality_score']:
            return value < threshold / multiplier
        else:
            return value > threshold * multiplier
    
    def _get_alert_message(self, metric_name: str, value: float, threshold: float) -> str:
        """Generate alert message."""
        messages = {
            'latency_p95_ms': f"95th percentile latency is {value:.1f}ms (threshold: {threshold}ms)",
            'latency_p99_ms': f"99th percentile latency is {value:.1f}ms (threshold: {threshold}ms)",
            'messages_per_second': f"Message throughput dropped to {value:.1f} msg/s (minimum: {threshold})",
            'error_rate': f"Error rate increased to {value:.1%} (threshold: {threshold:.1%})",
            'memory_usage_mb': f"Memory usage is {value:.0f}MB (limit: {threshold}MB)",
            'cpu_percent': f"CPU usage is {value:.0f}% (limit: {threshold}%)",
            'stream_quality_score': f"Stream quality dropped to {value:.2f} (minimum: {threshold})",
        }
        
        return messages.get(metric_name, f"{metric_name} is {value} (threshold: {threshold})")
    
    async def _raise_alert(self, alert: PerformanceAlert):
        """Raise a performance alert."""
        self.alerts.append(alert)
        
        logger.warning(f"Performance Alert ({alert.severity}): {alert.message}")
        
        # Call alert callback if registered
        if self.alert_callback:
            try:
                if asyncio.iscoroutinefunction(self.alert_callback):
                    await self.alert_callback(alert)
                else:
                    self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {}
        
        # Get latest value for each metric
        for metric_name, values in self.metrics.items():
            if values:
                latest = values[-1]
                metrics[metric_name] = {
                    'value': latest.value,
                    'unit': latest.unit,
                    'timestamp': latest.timestamp.isoformat()
                }
        
        # Add summary statistics
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        metrics['summary'] = {
            'uptime_seconds': uptime,
            'total_messages': self.total_messages,
            'total_errors': self.total_errors,
            'average_throughput': self.total_messages / uptime if uptime > 0 else 0,
            'overall_error_rate': self.total_errors / self.total_messages if self.total_messages > 0 else 0
        }
        
        return metrics
    
    def get_metric_history(
        self, 
        metric_name: str, 
        duration_seconds: int = 300
    ) -> List[Dict[str, Any]]:
        """Get history of a specific metric."""
        if metric_name not in self.metrics:
            return []
        
        cutoff = datetime.now() - timedelta(seconds=duration_seconds)
        
        history = []
        for metric in self.metrics[metric_name]:
            if metric.timestamp > cutoff:
                history.append({
                    'timestamp': metric.timestamp.isoformat(),
                    'value': metric.value,
                    'unit': metric.unit,
                    'metadata': metric.metadata
                })
        
        return history
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_metrics = self.get_current_metrics()
        
        # Calculate statistics for key metrics
        stats = {}
        
        for metric_name in ['latency_p95_ms', 'messages_per_second', 'error_rate']:
            if metric_name in self.metrics and self.metrics[metric_name]:
                values = [m.value for m in self.metrics[metric_name]]
                stats[metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std_dev': np.std(values)
                }
        
        # Get recent alerts
        recent_alerts = []
        cutoff = datetime.now() - timedelta(minutes=30)
        
        for alert in self.alerts:
            if alert.timestamp > cutoff:
                recent_alerts.append({
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity,
                    'metric': alert.metric_name,
                    'message': alert.message
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'statistics': stats,
            'recent_alerts': recent_alerts,
            'health_score': self._calculate_health_score(current_metrics),
            'recommendations': self._generate_recommendations(current_metrics, stats)
        }
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # Deduct points for threshold breaches
        deductions = {
            'latency_p95_ms': 20,
            'error_rate': 30,
            'messages_per_second': 25,
            'cpu_percent': 15,
            'memory_usage_mb': 10
        }
        
        for metric_name, deduction in deductions.items():
            if metric_name in metrics and metric_name in self.thresholds:
                value = metrics[metric_name].get('value', 0)
                threshold = self.thresholds[metric_name]
                
                if self._is_threshold_breached(metric_name, value, threshold):
                    # Calculate severity of breach
                    if metric_name in ['messages_per_second', 'stream_quality_score']:
                        severity = max(0, 1 - (value / threshold))
                    else:
                        severity = max(0, (value - threshold) / threshold)
                    
                    score -= deduction * min(1, severity)
        
        return max(0, score)
    
    def _generate_recommendations(
        self, 
        current_metrics: Dict[str, Any], 
        stats: Dict[str, Any]
    ) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Check latency
        if 'latency_p95_ms' in stats:
            if stats['latency_p95_ms']['current'] > self.thresholds['latency_p95_ms']:
                recommendations.append(
                    "High latency detected. Consider: "
                    "1) Switching to lower latency data sources, "
                    "2) Optimizing data processing pipeline, "
                    "3) Upgrading network connection"
                )
        
        # Check throughput
        if 'messages_per_second' in current_metrics:
            mps = current_metrics['messages_per_second'].get('value', 0)
            if mps < self.thresholds['messages_per_second']:
                recommendations.append(
                    "Low message throughput. Consider: "
                    "1) Checking data source connections, "
                    "2) Increasing stream buffer sizes, "
                    "3) Scaling up processing capacity"
                )
        
        # Check errors
        if 'error_rate' in current_metrics:
            error_rate = current_metrics['error_rate'].get('value', 0)
            if error_rate > self.thresholds['error_rate']:
                recommendations.append(
                    "High error rate detected. Consider: "
                    "1) Reviewing error logs for patterns, "
                    "2) Implementing retry mechanisms, "
                    "3) Checking API rate limits"
                )
        
        # Check resources
        if 'cpu_percent' in current_metrics:
            cpu = current_metrics['cpu_percent'].get('value', 0)
            if cpu > self.thresholds['cpu_percent']:
                recommendations.append(
                    "High CPU usage. Consider: "
                    "1) Optimizing data processing algorithms, "
                    "2) Implementing caching, "
                    "3) Scaling horizontally"
                )
        
        return recommendations
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export performance metrics to file."""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'performance_report': self.get_performance_report(),
                'metric_history': {}
            }
            
            # Export last hour of data for key metrics
            for metric_name in ['latency_p95_ms', 'messages_per_second', 'error_rate']:
                data['metric_history'][metric_name] = self.get_metric_history(metric_name, 3600)
            
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            logger.info(f"Exported performance metrics to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")