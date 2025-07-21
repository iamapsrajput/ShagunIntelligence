"""
API rate limiting and cost management for Shagun Intelligence.
"""

import asyncio
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time
from dataclasses import dataclass, field
from enum import Enum
import json
import aioredis
from loguru import logger

from .api_config import APIProvider, APIConfig, get_api_config


class RateLimitStatus(str, Enum):
    """Rate limit status."""
    OK = "ok"
    THROTTLED = "throttled"
    QUOTA_EXCEEDED = "quota_exceeded"
    BLOCKED = "blocked"


@dataclass
class RateLimitInfo:
    """Rate limit information for an API."""
    provider: str
    requests_made: int
    requests_limit: int
    window_start: datetime
    window_type: str  # 'minute', 'hour', 'day'
    status: RateLimitStatus
    reset_time: datetime
    cost_incurred: float = 0.0


@dataclass
class APIUsageRecord:
    """Record of API usage."""
    provider: str
    endpoint: str
    timestamp: datetime
    response_time_ms: float
    status_code: int
    cost: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class APIRateLimiter:
    """
    Rate limiter for API calls with cost tracking.
    
    Features:
    - Token bucket algorithm for rate limiting
    - Multi-window rate limiting (per minute/hour/day)
    - Cost tracking and budget enforcement
    - Redis-backed for distributed systems
    - Automatic throttling and backoff
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or "redis://localhost:6379"
        self.redis_client = None
        
        # Local rate limit tracking (fallback if Redis unavailable)
        self.local_buckets: Dict[str, Dict[str, deque]] = defaultdict(lambda: {
            'minute': deque(),
            'hour': deque(),
            'day': deque()
        })
        
        # Cost tracking
        self.cost_tracker: Dict[str, float] = defaultdict(float)
        self.monthly_costs: Dict[str, float] = defaultdict(float)
        
        # Usage history
        self.usage_history: List[APIUsageRecord] = []
        self.max_history_size = 10000
        
        # Throttle states
        self.throttle_until: Dict[str, datetime] = {}
        self.backoff_multipliers: Dict[str, float] = defaultdict(lambda: 1.0)
        
        logger.info("APIRateLimiter initialized")
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = await aioredis.create_redis_pool(self.redis_url)
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using local rate limiting.")
    
    async def check_rate_limit(
        self,
        provider: APIProvider,
        endpoint: str = "default"
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check if API call is allowed under rate limits.
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        provider_str = provider.value
        
        # Get API configuration
        config = get_api_config().get_api_config(provider)
        if not config:
            return False, self._create_limit_info(provider_str, RateLimitStatus.BLOCKED)
        
        # Check if throttled
        if provider_str in self.throttle_until:
            if datetime.now() < self.throttle_until[provider_str]:
                remaining = (self.throttle_until[provider_str] - datetime.now()).seconds
                info = self._create_limit_info(
                    provider_str,
                    RateLimitStatus.THROTTLED,
                    reset_seconds=remaining
                )
                return False, info
            else:
                del self.throttle_until[provider_str]
        
        # Check multiple rate limit windows
        now = datetime.now()
        allowed = True
        limiting_window = None
        
        # Check per-minute limit
        if config.rate_limit_per_minute:
            minute_count = await self._get_request_count(provider_str, 'minute', 60)
            if minute_count >= config.rate_limit_per_minute:
                allowed = False
                limiting_window = ('minute', config.rate_limit_per_minute, 60)
        
        # Check per-hour limit
        if allowed and config.rate_limit_per_hour:
            hour_count = await self._get_request_count(provider_str, 'hour', 3600)
            if hour_count >= config.rate_limit_per_hour:
                allowed = False
                limiting_window = ('hour', config.rate_limit_per_hour, 3600)
        
        # Check per-day limit
        if allowed and config.rate_limit_per_day:
            day_count = await self._get_request_count(provider_str, 'day', 86400)
            if day_count >= config.rate_limit_per_day:
                allowed = False
                limiting_window = ('day', config.rate_limit_per_day, 86400)
        
        # Check budget limit
        if allowed and config.monthly_budget:
            if self.monthly_costs[provider_str] >= config.monthly_budget:
                allowed = False
                info = self._create_limit_info(
                    provider_str,
                    RateLimitStatus.QUOTA_EXCEEDED,
                    message="Monthly budget exceeded"
                )
                return False, info
        
        # Create rate limit info
        if allowed:
            info = self._create_limit_info(provider_str, RateLimitStatus.OK)
        else:
            window_type, limit, window_seconds = limiting_window
            info = self._create_limit_info(
                provider_str,
                RateLimitStatus.THROTTLED,
                window_type=window_type,
                limit=limit,
                reset_seconds=window_seconds
            )
        
        return allowed, info
    
    async def record_request(
        self,
        provider: APIProvider,
        endpoint: str,
        response_time_ms: float,
        status_code: int,
        error: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record an API request."""
        provider_str = provider.value
        config = get_api_config().get_api_config(provider)
        
        if not config:
            return
        
        # Calculate cost
        cost = 0.0
        if config.cost_per_request > 0:
            # Check if within free tier
            month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0)
            month_requests = await self._get_request_count(
                provider_str,
                'month',
                int((datetime.now() - month_start).total_seconds())
            )
            
            if month_requests > config.free_requests_per_month:
                cost = config.cost_per_request
        
        # Create usage record
        record = APIUsageRecord(
            provider=provider_str,
            endpoint=endpoint,
            timestamp=datetime.now(),
            response_time_ms=response_time_ms,
            status_code=status_code,
            cost=cost,
            error=error,
            metadata=metadata or {}
        )
        
        # Store in history
        self.usage_history.append(record)
        if len(self.usage_history) > self.max_history_size:
            self.usage_history.pop(0)
        
        # Update costs
        self.cost_tracker[provider_str] += cost
        self.monthly_costs[provider_str] += cost
        
        # Update rate limit buckets
        await self._add_request(provider_str)
        
        # Handle rate limit errors
        if status_code == 429:  # Too Many Requests
            await self._handle_rate_limit_error(provider_str)
        elif status_code >= 500:  # Server errors
            await self._handle_server_error(provider_str)
        elif error:
            await self._handle_api_error(provider_str, error)
    
    async def _get_request_count(
        self,
        provider: str,
        window: str,
        window_seconds: int
    ) -> int:
        """Get request count for a time window."""
        if self.redis_client:
            try:
                key = f"rate_limit:{provider}:{window}"
                count = await self.redis_client.zcount(
                    key,
                    min=time.time() - window_seconds,
                    max=time.time()
                )
                return count
            except:
                pass
        
        # Fallback to local tracking
        bucket = self.local_buckets[provider][window]
        cutoff = time.time() - window_seconds
        
        # Remove old entries
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        
        return len(bucket)
    
    async def _add_request(self, provider: str):
        """Add a request to rate limit tracking."""
        timestamp = time.time()
        
        if self.redis_client:
            try:
                # Add to Redis sorted sets
                for window in ['minute', 'hour', 'day']:
                    key = f"rate_limit:{provider}:{window}"
                    await self.redis_client.zadd(key, timestamp, timestamp)
                    
                    # Set expiration
                    if window == 'minute':
                        await self.redis_client.expire(key, 120)
                    elif window == 'hour':
                        await self.redis_client.expire(key, 7200)
                    elif window == 'day':
                        await self.redis_client.expire(key, 172800)
                
                # Clean old entries
                for window, seconds in [('minute', 60), ('hour', 3600), ('day', 86400)]:
                    key = f"rate_limit:{provider}:{window}"
                    await self.redis_client.zremrangebyscore(
                        key, 0, timestamp - seconds
                    )
                return
            except:
                pass
        
        # Fallback to local tracking
        for window in ['minute', 'hour', 'day']:
            self.local_buckets[provider][window].append(timestamp)
    
    async def _handle_rate_limit_error(self, provider: str):
        """Handle rate limit error from API."""
        logger.warning(f"Rate limit hit for {provider}")
        
        # Exponential backoff
        current_backoff = self.backoff_multipliers[provider]
        backoff_seconds = int(60 * current_backoff)  # Start with 1 minute
        
        self.throttle_until[provider] = datetime.now() + timedelta(seconds=backoff_seconds)
        self.backoff_multipliers[provider] = min(current_backoff * 2, 32)  # Max 32 minutes
        
        logger.info(f"Throttling {provider} for {backoff_seconds} seconds")
    
    async def _handle_server_error(self, provider: str):
        """Handle server error from API."""
        # Light throttle for server errors
        self.throttle_until[provider] = datetime.now() + timedelta(seconds=30)
        logger.warning(f"Server error for {provider}, throttling for 30 seconds")
    
    async def _handle_api_error(self, provider: str, error: str):
        """Handle general API error."""
        # Reset backoff on successful requests
        if provider in self.backoff_multipliers:
            self.backoff_multipliers[provider] = 1.0
    
    def _create_limit_info(
        self,
        provider: str,
        status: RateLimitStatus,
        window_type: str = 'minute',
        limit: int = 0,
        reset_seconds: int = 60,
        message: str = None
    ) -> RateLimitInfo:
        """Create rate limit info object."""
        now = datetime.now()
        
        return RateLimitInfo(
            provider=provider,
            requests_made=0,  # Will be filled by caller if needed
            requests_limit=limit,
            window_start=now,
            window_type=window_type,
            status=status,
            reset_time=now + timedelta(seconds=reset_seconds),
            cost_incurred=self.cost_tracker.get(provider, 0.0)
        )
    
    def get_usage_summary(self, provider: Optional[APIProvider] = None) -> Dict[str, Any]:
        """Get usage summary for provider(s)."""
        if provider:
            providers = [provider.value]
        else:
            providers = [p.value for p in APIProvider]
        
        summary = {}
        
        for provider_str in providers:
            # Get recent usage
            recent_usage = [
                r for r in self.usage_history
                if r.provider == provider_str and
                r.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            if recent_usage:
                # Calculate statistics
                response_times = [r.response_time_ms for r in recent_usage]
                error_count = sum(1 for r in recent_usage if r.error)
                
                summary[provider_str] = {
                    'requests_24h': len(recent_usage),
                    'errors_24h': error_count,
                    'error_rate': error_count / len(recent_usage) if recent_usage else 0,
                    'avg_response_time_ms': sum(response_times) / len(response_times),
                    'min_response_time_ms': min(response_times),
                    'max_response_time_ms': max(response_times),
                    'cost_24h': sum(r.cost for r in recent_usage),
                    'cost_month': self.monthly_costs.get(provider_str, 0.0),
                    'is_throttled': provider_str in self.throttle_until
                }
            else:
                summary[provider_str] = {
                    'requests_24h': 0,
                    'errors_24h': 0,
                    'error_rate': 0,
                    'avg_response_time_ms': 0,
                    'cost_24h': 0,
                    'cost_month': self.monthly_costs.get(provider_str, 0.0),
                    'is_throttled': provider_str in self.throttle_until
                }
        
        return summary
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for all APIs."""
        status = {}
        
        for provider in APIProvider:
            provider_str = provider.value
            config = get_api_config().get_api_config(provider)
            
            if not config or not config.enabled:
                continue
            
            # Get current counts (synchronous version for display)
            minute_count = len([
                t for t in self.local_buckets[provider_str]['minute']
                if t > time.time() - 60
            ])
            
            hour_count = len([
                t for t in self.local_buckets[provider_str]['hour']
                if t > time.time() - 3600
            ])
            
            day_count = len([
                t for t in self.local_buckets[provider_str]['day']
                if t > time.time() - 86400
            ])
            
            status[provider_str] = {
                'limits': {
                    'per_minute': config.rate_limit_per_minute,
                    'per_hour': config.rate_limit_per_hour,
                    'per_day': config.rate_limit_per_day
                },
                'current': {
                    'minute': minute_count,
                    'hour': hour_count,
                    'day': day_count
                },
                'remaining': {
                    'minute': max(0, (config.rate_limit_per_minute or float('inf')) - minute_count),
                    'hour': max(0, (config.rate_limit_per_hour or float('inf')) - hour_count),
                    'day': max(0, (config.rate_limit_per_day or float('inf')) - day_count)
                },
                'throttled': provider_str in self.throttle_until,
                'throttle_until': self.throttle_until.get(provider_str).isoformat() if provider_str in self.throttle_until else None
            }
        
        return status
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost report for all APIs."""
        report = {
            'current_month': datetime.now().strftime('%Y-%m'),
            'providers': {},
            'total_cost': 0.0,
            'total_budget': 0.0,
            'budget_utilization': 0.0
        }
        
        for provider in APIProvider:
            provider_str = provider.value
            config = get_api_config().get_api_config(provider)
            
            if not config or not config.enabled:
                continue
            
            cost = self.monthly_costs.get(provider_str, 0.0)
            budget = config.monthly_budget or 0.0
            
            report['providers'][provider_str] = {
                'cost': cost,
                'budget': budget,
                'utilization': (cost / budget * 100) if budget > 0 else 0,
                'cost_per_request': config.cost_per_request,
                'free_requests': config.free_requests_per_month,
                'status': 'over_budget' if budget > 0 and cost > budget else 'ok'
            }
            
            report['total_cost'] += cost
            report['total_budget'] += budget
        
        if report['total_budget'] > 0:
            report['budget_utilization'] = (report['total_cost'] / report['total_budget']) * 100
        
        return report
    
    async def reset_monthly_costs(self):
        """Reset monthly cost tracking (call at month start)."""
        logger.info("Resetting monthly API costs")
        self.monthly_costs.clear()
        
        # Store cost history if needed
        if self.cost_tracker:
            month = datetime.now().strftime('%Y-%m')
            history_file = f"api_costs_{month}.json"
            
            with open(history_file, 'w') as f:
                json.dump({
                    'month': month,
                    'costs': dict(self.cost_tracker),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Saved cost history to {history_file}")
            self.cost_tracker.clear()
    
    def export_usage_data(self, filepath: str = "api_usage.json"):
        """Export usage data for analysis."""
        data = {
            'export_time': datetime.now().isoformat(),
            'usage_summary': self.get_usage_summary(),
            'rate_limit_status': self.get_rate_limit_status(),
            'cost_report': self.get_cost_report(),
            'recent_requests': [
                {
                    'provider': r.provider,
                    'endpoint': r.endpoint,
                    'timestamp': r.timestamp.isoformat(),
                    'response_time_ms': r.response_time_ms,
                    'status_code': r.status_code,
                    'cost': r.cost,
                    'error': r.error
                }
                for r in self.usage_history[-100:]  # Last 100 requests
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported usage data to {filepath}")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()


# Singleton instance
_rate_limiter_instance: Optional[APIRateLimiter] = None

async def get_rate_limiter() -> APIRateLimiter:
    """Get the rate limiter instance."""
    global _rate_limiter_instance
    
    if _rate_limiter_instance is None:
        _rate_limiter_instance = APIRateLimiter()
        await _rate_limiter_instance.initialize()
    
    return _rate_limiter_instance