import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for AI API calls with token tracking"""

    def __init__(
        self, max_requests_per_minute: int = 60, max_tokens_per_minute: int = 150000
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute

        # Request tracking per provider
        self.request_history: dict[str, deque] = defaultdict(lambda: deque())
        self.token_history: dict[str, deque] = defaultdict(lambda: deque())

        # Lock for thread safety
        self.lock = asyncio.Lock()

        # Statistics
        self.stats = defaultdict(
            lambda: {
                "total_requests": 0,
                "blocked_requests": 0,
                "total_tokens": 0,
                "rate_limit_hits": 0,
            }
        )

        # Provider-specific limits (can be customized)
        self.provider_limits = {
            "openai": {
                "requests_per_minute": 60,
                "tokens_per_minute": 150000,
                "requests_per_day": 10000,
                "tokens_per_day": 2000000,
            },
            "anthropic": {
                "requests_per_minute": 50,
                "tokens_per_minute": 100000,
                "requests_per_day": 5000,
                "tokens_per_day": 1000000,
            },
        }

        # Daily usage tracking
        self.daily_usage = defaultdict(
            lambda: {"requests": 0, "tokens": 0, "reset_time": self._next_day_reset()}
        )

        logger.info(
            f"RateLimiter initialized - Requests: {max_requests_per_minute}/min, "
            f"Tokens: {max_tokens_per_minute}/min"
        )

    async def check_request(self, provider: str, estimated_tokens: int) -> bool:
        """Check if request can proceed within rate limits"""
        async with self.lock:
            provider_name = (
                provider.value if hasattr(provider, "value") else str(provider)
            )
            current_time = time.time()

            # Get provider-specific limits
            limits = self.provider_limits.get(
                provider_name.lower(),
                {
                    "requests_per_minute": self.max_requests_per_minute,
                    "tokens_per_minute": self.max_tokens_per_minute,
                    "requests_per_day": 10000,
                    "tokens_per_day": 2000000,
                },
            )

            # Clean old entries (older than 1 minute)
            self._clean_old_entries(provider_name, current_time - 60)

            # Check daily limits
            if not self._check_daily_limits(provider_name, estimated_tokens, limits):
                self.stats[provider_name]["rate_limit_hits"] += 1
                logger.warning(f"Daily limit exceeded for {provider_name}")
                return False

            # Count requests in the last minute
            recent_requests = len(self.request_history[provider_name])

            # Count tokens in the last minute
            recent_tokens = sum(self.token_history[provider_name]) + estimated_tokens

            # Check limits
            if recent_requests >= limits["requests_per_minute"]:
                self.stats[provider_name]["blocked_requests"] += 1
                self.stats[provider_name]["rate_limit_hits"] += 1
                logger.warning(
                    f"Request rate limit exceeded for {provider_name}: "
                    f"{recent_requests}/{limits['requests_per_minute']}"
                )
                return False

            if recent_tokens > limits["tokens_per_minute"]:
                self.stats[provider_name]["blocked_requests"] += 1
                self.stats[provider_name]["rate_limit_hits"] += 1
                logger.warning(
                    f"Token rate limit exceeded for {provider_name}: "
                    f"{recent_tokens}/{limits['tokens_per_minute']}"
                )
                return False

            return True

    async def record_request(self, provider: str, actual_tokens: int) -> None:
        """Record a completed request"""
        async with self.lock:
            provider_name = (
                provider.value if hasattr(provider, "value") else str(provider)
            )
            current_time = time.time()

            # Add to history
            self.request_history[provider_name].append(current_time)
            self.token_history[provider_name].append(actual_tokens)

            # Update statistics
            self.stats[provider_name]["total_requests"] += 1
            self.stats[provider_name]["total_tokens"] += actual_tokens

            # Update daily usage
            self.daily_usage[provider_name]["requests"] += 1
            self.daily_usage[provider_name]["tokens"] += actual_tokens

            logger.debug(
                f"Recorded request for {provider_name}: {actual_tokens} tokens"
            )

    def _clean_old_entries(self, provider: str, cutoff_time: float) -> None:
        """Remove entries older than cutoff time"""
        # Clean request history
        while (
            self.request_history[provider]
            and self.request_history[provider][0] < cutoff_time
        ):
            self.request_history[provider].popleft()

        # Clean token history (paired with request history)
        while len(self.token_history[provider]) > len(self.request_history[provider]):
            self.token_history[provider].popleft()

    def _check_daily_limits(
        self, provider: str, estimated_tokens: int, limits: dict[str, int]
    ) -> bool:
        """Check daily usage limits"""
        # Reset daily usage if needed
        if datetime.now() >= self.daily_usage[provider]["reset_time"]:
            self.daily_usage[provider] = {
                "requests": 0,
                "tokens": 0,
                "reset_time": self._next_day_reset(),
            }

        # Check limits
        daily_requests = self.daily_usage[provider]["requests"]
        daily_tokens = self.daily_usage[provider]["tokens"] + estimated_tokens

        if daily_requests >= limits["requests_per_day"]:
            return False

        if daily_tokens > limits["tokens_per_day"]:
            return False

        return True

    def _next_day_reset(self) -> datetime:
        """Get next daily reset time (midnight UTC)"""
        tomorrow = datetime.utcnow().date() + timedelta(days=1)
        return datetime.combine(tomorrow, datetime.min.time())

    async def wait_if_needed(self, provider: str, estimated_tokens: int) -> float:
        """Wait if rate limited and return wait time"""
        if await self.check_request(provider, estimated_tokens):
            return 0.0

        provider_name = provider.value if hasattr(provider, "value") else str(provider)

        # Calculate wait time
        async with self.lock:
            if not self.request_history[provider_name]:
                return 0.0

            # Find the oldest request time
            oldest_request = self.request_history[provider_name][0]
            current_time = time.time()

            # Wait until the oldest request is older than 1 minute
            wait_time = max(0, 60 - (current_time - oldest_request))

            if wait_time > 0:
                logger.info(
                    f"Rate limited for {provider_name}, waiting {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)

            return wait_time

    def get_current_usage(self, provider: str | None = None) -> dict[str, Any]:
        """Get current usage statistics"""
        if provider:
            provider_name = (
                provider.value if hasattr(provider, "value") else str(provider)
            )

            # Count recent usage
            current_time = time.time()
            cutoff_time = current_time - 60

            recent_requests = sum(
                1 for t in self.request_history[provider_name] if t >= cutoff_time
            )
            recent_tokens = sum(
                tokens
                for t, tokens in zip(
                    self.request_history[provider_name],
                    self.token_history[provider_name],
                    strict=False,
                )
                if t >= cutoff_time
            )

            limits = self.provider_limits.get(
                provider_name.lower(),
                {
                    "requests_per_minute": self.max_requests_per_minute,
                    "tokens_per_minute": self.max_tokens_per_minute,
                },
            )

            return {
                "provider": provider_name,
                "requests": {
                    "current": recent_requests,
                    "limit": limits["requests_per_minute"],
                    "percentage": (
                        recent_requests / limits["requests_per_minute"] * 100
                    ),
                },
                "tokens": {
                    "current": recent_tokens,
                    "limit": limits["tokens_per_minute"],
                    "percentage": (recent_tokens / limits["tokens_per_minute"] * 100),
                },
                "daily": self.daily_usage[provider_name],
            }
        else:
            # Return all providers
            usage = {}
            for provider_name in self.stats.keys():
                usage[provider_name] = self.get_current_usage(provider_name)
            return usage

    def get_status(self) -> dict[str, Any]:
        """Get rate limiter status"""
        status = {
            "providers": {},
            "total_stats": {
                "total_requests": 0,
                "blocked_requests": 0,
                "total_tokens": 0,
                "rate_limit_hits": 0,
            },
        }

        for provider, stats in self.stats.items():
            current_usage = self.get_current_usage(provider)

            status["providers"][provider] = {
                "stats": stats,
                "current_usage": current_usage,
                "health": self._calculate_health(current_usage),
            }

            # Aggregate totals
            for key in [
                "total_requests",
                "blocked_requests",
                "total_tokens",
                "rate_limit_hits",
            ]:
                status["total_stats"][key] += stats[key]

        return status

    def _calculate_health(self, usage: dict[str, Any]) -> str:
        """Calculate health status based on usage"""
        request_pct = usage["requests"]["percentage"]
        token_pct = usage["tokens"]["percentage"]

        max_usage = max(request_pct, token_pct)

        if max_usage < 50:
            return "healthy"
        elif max_usage < 80:
            return "moderate"
        elif max_usage < 95:
            return "high"
        else:
            return "critical"

    def reset_provider(self, provider: str) -> None:
        """Reset rate limit tracking for a provider"""
        provider_name = provider.value if hasattr(provider, "value") else str(provider)

        self.request_history[provider_name].clear()
        self.token_history[provider_name].clear()
        self.daily_usage[provider_name] = {
            "requests": 0,
            "tokens": 0,
            "reset_time": self._next_day_reset(),
        }

        logger.info(f"Reset rate limits for {provider_name}")

    def update_limits(self, provider: str, limits: dict[str, int]) -> None:
        """Update provider-specific limits"""
        provider_name = provider.value if hasattr(provider, "value") else str(provider)

        self.provider_limits[provider_name.lower()].update(limits)

        logger.info(f"Updated limits for {provider_name}: {limits}")
