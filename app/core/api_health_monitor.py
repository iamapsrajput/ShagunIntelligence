"""
API health monitoring and alerting system.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import statistics
from loguru import logger
import json

from .api_config import APIProvider, APIConfig, get_api_config
from .api_rate_limiter import get_rate_limiter


class HealthStatus(str, Enum):
    """API health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    provider: str
    status: HealthStatus
    response_time_ms: float
    status_code: Optional[int]
    error: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthAlert:
    """Health alert for an API."""
    provider: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    health_status: HealthStatus
    details: Dict[str, Any] = field(default_factory=dict)


class APIHealthMonitor:
    """
    Monitors API health and availability.
    
    Features:
    - Periodic health checks
    - Response time monitoring
    - Availability tracking
    - Automatic alerting
    - Historical health data
    """
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.alert_callback = alert_callback
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Health check results
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        self.max_history_size = 1000
        
        # Current status
        self.current_status: Dict[str, HealthStatus] = {}
        self.last_check: Dict[str, datetime] = {}
        
        # Monitoring settings
        self.check_interval = 300  # 5 minutes
        self.monitoring_task = None
        self.running = False
        
        # Alert management
        self.alerts: List[HealthAlert] = []
        self.alert_cooldown: Dict[str, datetime] = {}
        self.alert_cooldown_minutes = 30
        
        # Health thresholds
        self.thresholds = {
            'response_time_ms': 2000,  # 2 seconds
            'error_rate': 0.1,  # 10%
            'availability': 0.95,  # 95%
        }
        
        logger.info("APIHealthMonitor initialized")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self.running:
            return
        
        # Initialize session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started API health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.session:
            await self.session.close()
        
        logger.info("Stopped API health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check all enabled APIs
                tasks = []
                for provider in APIProvider:
                    config = get_api_config().get_api_config(provider)
                    if config and config.enabled:
                        task = self._check_api_health(provider, config)
                        tasks.append(task)
                
                # Run health checks concurrently
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Analyze results and generate alerts
                await self._analyze_health_status()
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait on error
    
    async def _check_api_health(self, provider: APIProvider, config: APIConfig):
        """Perform health check for an API."""
        provider_str = provider.value
        
        # Check rate limits first
        rate_limiter = await get_rate_limiter()
        allowed, limit_info = await rate_limiter.check_rate_limit(provider, "health_check")
        
        if not allowed:
            # Can't check due to rate limits
            result = HealthCheckResult(
                provider=provider_str,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0,
                status_code=None,
                error="Rate limited",
                timestamp=datetime.now(),
                metadata={'reason': 'rate_limit'}
            )
        else:
            # Perform actual health check
            result = await self._perform_health_check(provider, config)
        
        # Store result
        if provider_str not in self.health_history:
            self.health_history[provider_str] = []
        
        self.health_history[provider_str].append(result)
        
        # Trim history
        if len(self.health_history[provider_str]) > self.max_history_size:
            self.health_history[provider_str].pop(0)
        
        # Update current status
        self.current_status[provider_str] = result.status
        self.last_check[provider_str] = result.timestamp
    
    async def _perform_health_check(
        self,
        provider: APIProvider,
        config: APIConfig
    ) -> HealthCheckResult:
        """Perform actual health check request."""
        provider_str = provider.value
        start_time = datetime.now()
        
        try:
            # Build health check URL
            if config.health_check_endpoint:
                url = f"{config.base_url}{config.health_check_endpoint}"
            else:
                # Use default endpoints based on provider
                url = self._get_default_health_endpoint(provider, config)
            
            if not url:
                return HealthCheckResult(
                    provider=provider_str,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0,
                    status_code=None,
                    error="No health endpoint",
                    timestamp=start_time
                )
            
            # Make request
            headers = await self._get_auth_headers(provider)
            
            async with self.session.get(url, headers=headers) as response:
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Determine health status
                if response.status == 200:
                    if response_time < self.thresholds['response_time_ms']:
                        status = HealthStatus.HEALTHY
                    else:
                        status = HealthStatus.DEGRADED
                elif response.status < 500:
                    status = HealthStatus.HEALTHY  # Client errors don't indicate API health
                else:
                    status = HealthStatus.UNHEALTHY
                
                return HealthCheckResult(
                    provider=provider_str,
                    status=status,
                    response_time_ms=response_time,
                    status_code=response.status,
                    error=None,
                    timestamp=start_time
                )
                
        except asyncio.TimeoutError:
            return HealthCheckResult(
                provider=provider_str,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=config.timeout_seconds * 1000,
                status_code=None,
                error="Timeout",
                timestamp=start_time
            )
        except Exception as e:
            return HealthCheckResult(
                provider=provider_str,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                status_code=None,
                error=str(e),
                timestamp=start_time
            )
    
    def _get_default_health_endpoint(
        self,
        provider: APIProvider,
        config: APIConfig
    ) -> Optional[str]:
        """Get default health check endpoint for provider."""
        endpoints = {
            APIProvider.KITE_CONNECT: "/session/token",
            APIProvider.ALPHA_VANTAGE: "?function=GLOBAL_QUOTE&symbol=IBM",
            APIProvider.FINNHUB: "/stock/profile2?symbol=AAPL",
            APIProvider.POLYGON: "/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09",
            APIProvider.TWITTER: "/users/by/username/Twitter",
            APIProvider.NEWSAPI: "/everything?q=test&pageSize=1",
        }
        
        endpoint = endpoints.get(provider)
        if endpoint:
            return f"{config.base_url}{endpoint}"
        
        return None
    
    async def _get_auth_headers(self, provider: APIProvider) -> Dict[str, str]:
        """Get authentication headers for provider."""
        from .api_key_manager import get_api_key_manager
        
        key_manager = get_api_key_manager()
        headers = {}
        
        # Get API key
        api_key = key_manager.get_key(provider, 'api_key')
        
        if provider == APIProvider.ALPHA_VANTAGE and api_key:
            # Alpha Vantage uses query parameter, but we'll add for completeness
            headers['X-API-Key'] = api_key
        elif provider == APIProvider.FINNHUB and api_key:
            headers['X-Finnhub-Token'] = api_key
        elif provider == APIProvider.POLYGON and api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        elif provider == APIProvider.TWITTER:
            bearer_token = key_manager.get_key(provider, 'bearer_token')
            if bearer_token:
                headers['Authorization'] = f'Bearer {bearer_token}'
        elif api_key:
            # Generic API key header
            headers['X-API-Key'] = api_key
        
        return headers
    
    async def _analyze_health_status(self):
        """Analyze health status and generate alerts."""
        for provider_str, history in self.health_history.items():
            if not history:
                continue
            
            # Get recent results (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            recent_results = [r for r in history if r.timestamp > cutoff]
            
            if not recent_results:
                continue
            
            # Calculate metrics
            total_checks = len(recent_results)
            healthy_checks = sum(1 for r in recent_results if r.status == HealthStatus.HEALTHY)
            error_checks = sum(1 for r in recent_results if r.error is not None)
            
            availability = healthy_checks / total_checks if total_checks > 0 else 0
            error_rate = error_checks / total_checks if total_checks > 0 else 0
            
            # Calculate average response time
            response_times = [r.response_time_ms for r in recent_results if r.response_time_ms > 0]
            avg_response_time = statistics.mean(response_times) if response_times else 0
            
            # Determine overall status
            if availability < self.thresholds['availability']:
                overall_status = HealthStatus.UNHEALTHY
            elif error_rate > self.thresholds['error_rate']:
                overall_status = HealthStatus.DEGRADED
            elif avg_response_time > self.thresholds['response_time_ms']:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
            
            # Generate alerts if status changed
            previous_status = self.current_status.get(provider_str)
            if previous_status != overall_status:
                await self._generate_health_alert(
                    provider_str,
                    overall_status,
                    previous_status,
                    {
                        'availability': availability,
                        'error_rate': error_rate,
                        'avg_response_time_ms': avg_response_time,
                        'total_checks': total_checks
                    }
                )
    
    async def _generate_health_alert(
        self,
        provider: str,
        new_status: HealthStatus,
        previous_status: Optional[HealthStatus],
        metrics: Dict[str, Any]
    ):
        """Generate health alert."""
        # Check cooldown
        if provider in self.alert_cooldown:
            if datetime.now() < self.alert_cooldown[provider]:
                return
        
        # Determine severity
        if new_status == HealthStatus.UNHEALTHY:
            severity = AlertSeverity.ERROR
        elif new_status == HealthStatus.DEGRADED:
            severity = AlertSeverity.WARNING
        elif previous_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
            severity = AlertSeverity.INFO  # Recovery
        else:
            return  # No alert for healthy -> healthy
        
        # Create message
        if new_status == HealthStatus.UNHEALTHY:
            message = f"{provider} API is unhealthy - Availability: {metrics['availability']:.1%}"
        elif new_status == HealthStatus.DEGRADED:
            message = f"{provider} API is degraded - Response time: {metrics['avg_response_time_ms']:.0f}ms"
        else:
            message = f"{provider} API has recovered"
        
        # Create alert
        alert = HealthAlert(
            provider=provider,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            health_status=new_status,
            details=metrics
        )
        
        # Store alert
        self.alerts.append(alert)
        
        # Set cooldown
        self.alert_cooldown[provider] = datetime.now() + timedelta(minutes=self.alert_cooldown_minutes)
        
        # Send alert
        await self._send_alert(alert)
    
    async def _send_alert(self, alert: HealthAlert):
        """Send health alert."""
        logger.warning(
            f"Health Alert [{alert.severity.upper()}] - {alert.provider}: {alert.message}"
        )
        
        if self.alert_callback:
            try:
                if asyncio.iscoroutinefunction(self.alert_callback):
                    await self.alert_callback(alert)
                else:
                    self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_health_status(self, provider: Optional[APIProvider] = None) -> Dict[str, Any]:
        """Get current health status."""
        if provider:
            provider_str = provider.value
            return self._get_provider_health_status(provider_str)
        
        # Get all providers
        status = {}
        for provider in APIProvider:
            provider_str = provider.value
            if provider_str in self.current_status:
                status[provider_str] = self._get_provider_health_status(provider_str)
        
        return status
    
    def _get_provider_health_status(self, provider: str) -> Dict[str, Any]:
        """Get health status for a single provider."""
        # Get recent history
        history = self.health_history.get(provider, [])
        recent = history[-10:] if history else []  # Last 10 checks
        
        if not recent:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'last_check': None,
                'metrics': {}
            }
        
        # Calculate metrics
        response_times = [r.response_time_ms for r in recent if r.response_time_ms > 0]
        error_count = sum(1 for r in recent if r.error is not None)
        
        metrics = {
            'avg_response_time_ms': statistics.mean(response_times) if response_times else 0,
            'min_response_time_ms': min(response_times) if response_times else 0,
            'max_response_time_ms': max(response_times) if response_times else 0,
            'error_rate': error_count / len(recent) if recent else 0,
            'last_error': next((r.error for r in reversed(recent) if r.error), None),
            'checks_performed': len(recent)
        }
        
        return {
            'status': self.current_status.get(provider, HealthStatus.UNKNOWN).value,
            'last_check': self.last_check.get(provider).isoformat() if provider in self.last_check else None,
            'metrics': metrics
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'providers': self.get_health_status(),
            'summary': {
                'total_providers': len(self.current_status),
                'healthy': sum(1 for s in self.current_status.values() if s == HealthStatus.HEALTHY),
                'degraded': sum(1 for s in self.current_status.values() if s == HealthStatus.DEGRADED),
                'unhealthy': sum(1 for s in self.current_status.values() if s == HealthStatus.UNHEALTHY),
                'unknown': sum(1 for s in self.current_status.values() if s == HealthStatus.UNKNOWN)
            },
            'recent_alerts': [
                {
                    'provider': a.provider,
                    'severity': a.severity.value,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat(),
                    'status': a.health_status.value
                }
                for a in self.alerts[-10:]  # Last 10 alerts
            ]
        }
        
        # Add overall health score
        total = report['summary']['total_providers']
        if total > 0:
            healthy_score = (report['summary']['healthy'] / total) * 100
            degraded_penalty = (report['summary']['degraded'] / total) * 25
            unhealthy_penalty = (report['summary']['unhealthy'] / total) * 50
            
            report['summary']['health_score'] = max(0, healthy_score - degraded_penalty - unhealthy_penalty)
        else:
            report['summary']['health_score'] = 0
        
        return report
    
    async def force_health_check(self, provider: APIProvider) -> HealthCheckResult:
        """Force an immediate health check for a provider."""
        config = get_api_config().get_api_config(provider)
        if not config:
            raise ValueError(f"Provider {provider.value} not configured")
        
        result = await self._perform_health_check(provider, config)
        
        # Store result
        provider_str = provider.value
        if provider_str not in self.health_history:
            self.health_history[provider_str] = []
        
        self.health_history[provider_str].append(result)
        self.current_status[provider_str] = result.status
        self.last_check[provider_str] = result.timestamp
        
        return result
    
    def export_health_data(self, filepath: str = "api_health.json"):
        """Export health data for analysis."""
        data = {
            'export_time': datetime.now().isoformat(),
            'health_report': self.get_health_report(),
            'detailed_history': {}
        }
        
        # Add last 100 results for each provider
        for provider, history in self.health_history.items():
            data['detailed_history'][provider] = [
                {
                    'status': r.status.value,
                    'response_time_ms': r.response_time_ms,
                    'status_code': r.status_code,
                    'error': r.error,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in history[-100:]
            ]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported health data to {filepath}")


# Singleton instance
_health_monitor_instance: Optional[APIHealthMonitor] = None

def get_health_monitor(alert_callback: Optional[Callable] = None) -> APIHealthMonitor:
    """Get the health monitor instance."""
    global _health_monitor_instance
    
    if _health_monitor_instance is None:
        _health_monitor_instance = APIHealthMonitor(alert_callback)
    
    return _health_monitor_instance