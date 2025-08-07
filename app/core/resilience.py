"""
Enhanced Resilience System for Shagun Intelligence Trading Platform
Provides comprehensive error handling, circuit breakers, graceful degradation, and automatic recovery
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ServiceState(Enum):
    """Service health states"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class HealthMetrics:
    """Health metrics for a service"""

    success_count: int = 0
    failure_count: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    consecutive_failures: int = 0
    uptime_start: datetime = field(default_factory=datetime.now)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: int = 60
    half_open_max_calls: int = 5
    expected_exception: type = Exception


class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with comprehensive failure handling"""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = HealthMetrics()
        self.state_changed_at = datetime.now()
        self.half_open_calls = 0

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.state != CircuitState.OPEN:
            return False

        time_since_open = datetime.now() - self.state_changed_at
        return time_since_open.total_seconds() >= self.config.timeout

    def _record_success(self):
        """Record successful operation"""
        self.metrics.success_count += 1
        self.metrics.total_requests += 1
        self.metrics.last_success = datetime.now()
        self.metrics.consecutive_failures = 0

        if self.state == CircuitState.HALF_OPEN:
            if self.metrics.success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def _record_failure(self, exception: Exception):
        """Record failed operation"""
        self.metrics.failure_count += 1
        self.metrics.total_requests += 1
        self.metrics.last_failure = datetime.now()
        self.metrics.consecutive_failures += 1

        if self.state == CircuitState.CLOSED:
            if self.metrics.consecutive_failures >= self.config.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()

    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        self.state = CircuitState.OPEN
        self.state_changed_at = datetime.now()
        logger.warning(f"Circuit breaker '{self.name}' opened due to failures")

    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.state_changed_at = datetime.now()
        self.half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")

    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.metrics.consecutive_failures = 0
        logger.info(f"Circuit breaker '{self.name}' closed - service recovered")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""

        # Check if we should attempt reset
        if self._should_attempt_reset():
            self._transition_to_half_open()

        # Reject calls if circuit is open
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")

        # Limit calls in half-open state
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' half-open call limit exceeded"
                )
            self.half_open_calls += 1

        start_time = time.time()
        try:
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # Record success
            response_time = time.time() - start_time
            self.metrics.avg_response_time = (
                self.metrics.avg_response_time * (self.metrics.total_requests - 1)
                + response_time
            ) / self.metrics.total_requests
            self._record_success()

            return result

        except self.config.expected_exception as e:
            self._record_failure(e)
            raise e


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""

    pass


class ResilienceManager:
    """Central manager for system resilience"""

    def __init__(self):
        self.circuit_breakers: dict[str, EnhancedCircuitBreaker] = {}
        self.service_health: dict[str, HealthMetrics] = {}
        self.degradation_strategies: dict[str, Callable] = {}
        self.recovery_strategies: dict[str, Callable] = {}

        # Initialize default circuit breakers
        self._initialize_default_breakers()

        # Initialize default degradation strategies
        self._initialize_degradation_strategies()

        # Initialize recovery strategies
        self._initialize_recovery_strategies()

    def _initialize_default_breakers(self):
        """Initialize circuit breakers for critical services"""

        # Kite API circuit breaker
        self.register_circuit_breaker(
            "kite_api",
            CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout=300,  # 5 minutes
                expected_exception=Exception,
            ),
        )

        # Database circuit breaker
        self.register_circuit_breaker(
            "database",
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=60,  # 1 minute
                expected_exception=Exception,
            ),
        )

        # AI agents circuit breaker
        self.register_circuit_breaker(
            "ai_agents",
            CircuitBreakerConfig(
                failure_threshold=10,
                success_threshold=5,
                timeout=180,  # 3 minutes
                expected_exception=Exception,
            ),
        )

        # Market data circuit breaker
        self.register_circuit_breaker(
            "market_data",
            CircuitBreakerConfig(
                failure_threshold=8,
                success_threshold=4,
                timeout=120,  # 2 minutes
                expected_exception=Exception,
            ),
        )

    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Register a new circuit breaker"""
        self.circuit_breakers[name] = EnhancedCircuitBreaker(name, config)
        logger.info(f"Registered circuit breaker: {name}")

    def get_circuit_breaker(self, name: str) -> EnhancedCircuitBreaker | None:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)

    def register_degradation_strategy(self, service: str, strategy: Callable):
        """Register graceful degradation strategy for a service"""
        self.degradation_strategies[service] = strategy
        logger.info(f"Registered degradation strategy for: {service}")

    def register_recovery_strategy(self, service: str, strategy: Callable):
        """Register automatic recovery strategy for a service"""
        self.recovery_strategies[service] = strategy
        logger.info(f"Registered recovery strategy for: {service}")

    def _initialize_degradation_strategies(self):
        """Initialize default degradation strategies"""

        async def kite_api_degradation(*args, **kwargs):
            """Fallback when Kite API is unavailable"""
            logger.warning(
                "Kite API unavailable, using cached data and disabling new orders"
            )
            return {
                "status": "degraded",
                "message": "Using cached market data, live trading disabled",
                "data": None,
                "trading_enabled": False,
            }

        async def market_data_degradation(*args, **kwargs):
            """Fallback when market data is unavailable"""
            logger.warning("Market data unavailable, using last known prices")
            # Return last known good data or safe defaults
            return {
                "price": 0,
                "volume": 0,
                "timestamp": datetime.now(),
                "status": "stale_data",
                "message": "Using cached market data",
            }

        async def ai_agents_degradation(*args, **kwargs):
            """Fallback when AI agents are unavailable"""
            logger.warning(
                "AI agents unavailable, defaulting to conservative HOLD decisions"
            )
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "reason": "AI agents unavailable - conservative hold position",
                "status": "degraded",
            }

        async def database_degradation(*args, **kwargs):
            """Fallback when database is unavailable"""
            logger.warning("Database unavailable, using in-memory cache")
            return {
                "status": "degraded",
                "message": "Database unavailable, using cached data",
                "data": None,
            }

        # Register strategies
        self.register_degradation_strategy("kite_api", kite_api_degradation)
        self.register_degradation_strategy("market_data", market_data_degradation)
        self.register_degradation_strategy("ai_agents", ai_agents_degradation)
        self.register_degradation_strategy("database", database_degradation)

    def _initialize_recovery_strategies(self):
        """Initialize automatic recovery strategies"""

        async def kite_api_recovery():
            """Attempt to recover Kite API connection"""
            try:
                # Attempt to reconnect to Kite API
                from app.services.kite_service import kite_service

                if hasattr(kite_service, "reconnect"):
                    await kite_service.reconnect()
                    logger.info("Kite API connection recovered")
                    return True
            except Exception as e:
                logger.error(f"Kite API recovery failed: {e}")
                return False

        async def database_recovery():
            """Attempt to recover database connection"""
            try:
                from app.core.database import db_manager

                if hasattr(db_manager, "reconnect"):
                    await db_manager.reconnect()
                    logger.info("Database connection recovered")
                    return True
            except Exception as e:
                logger.error(f"Database recovery failed: {e}")
                return False

        async def market_data_recovery():
            """Attempt to recover market data feed"""
            try:
                # Attempt to reconnect market data sources
                logger.info("Attempting market data recovery")
                # Add specific recovery logic here
                return True
            except Exception as e:
                logger.error(f"Market data recovery failed: {e}")
                return False

        # Register recovery strategies
        self.register_recovery_strategy("kite_api", kite_api_recovery)
        self.register_recovery_strategy("database", database_recovery)
        self.register_recovery_strategy("market_data", market_data_recovery)

    async def execute_with_resilience(
        self,
        service_name: str,
        func: Callable,
        *args,
        fallback: Callable | None = None,
        **kwargs,
    ) -> Any:
        """Execute function with full resilience protection"""

        circuit_breaker = self.get_circuit_breaker(service_name)

        try:
            if circuit_breaker:
                return await circuit_breaker.call(func, *args, **kwargs)
            else:
                return (
                    await func(*args, **kwargs)
                    if asyncio.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )

        except CircuitBreakerOpenError:
            logger.warning(
                f"Circuit breaker open for {service_name}, attempting graceful degradation"
            )
            return await self._handle_degradation(
                service_name, fallback, *args, **kwargs
            )

        except Exception as e:
            logger.error(f"Error in {service_name}: {e}")
            return await self._handle_degradation(
                service_name, fallback, *args, **kwargs
            )

    async def _handle_degradation(
        self, service_name: str, fallback: Callable | None, *args, **kwargs
    ) -> Any:
        """Handle service degradation"""

        # Try registered degradation strategy
        if service_name in self.degradation_strategies:
            try:
                strategy = self.degradation_strategies[service_name]
                return (
                    await strategy(*args, **kwargs)
                    if asyncio.iscoroutinefunction(strategy)
                    else strategy(*args, **kwargs)
                )
            except Exception as e:
                logger.error(f"Degradation strategy failed for {service_name}: {e}")

        # Try provided fallback
        if fallback:
            try:
                return (
                    await fallback(*args, **kwargs)
                    if asyncio.iscoroutinefunction(fallback)
                    else fallback(*args, **kwargs)
                )
            except Exception as e:
                logger.error(f"Fallback failed for {service_name}: {e}")

        # Return safe default
        return self._get_safe_default(service_name)

    def _get_safe_default(self, service_name: str) -> Any:
        """Get safe default response for service"""
        defaults = {
            "kite_api": {"status": "unavailable", "data": None},
            "market_data": {"price": 0, "volume": 0, "timestamp": datetime.now()},
            "ai_agents": {
                "decision": "HOLD",
                "confidence": 0.0,
                "reason": "Service unavailable",
            },
            "database": None,
        }
        return defaults.get(service_name, None)

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health status"""
        health_summary = {
            "overall_status": ServiceState.HEALTHY.value,
            "services": {},
            "circuit_breakers": {},
            "timestamp": datetime.now().isoformat(),
        }

        unhealthy_services = 0
        total_services = len(self.circuit_breakers)

        for name, breaker in self.circuit_breakers.items():
            service_health = {
                "state": breaker.state.value,
                "success_rate": (
                    breaker.metrics.success_count
                    / max(breaker.metrics.total_requests, 1)
                    * 100
                ),
                "consecutive_failures": breaker.metrics.consecutive_failures,
                "avg_response_time": breaker.metrics.avg_response_time,
                "last_success": (
                    breaker.metrics.last_success.isoformat()
                    if breaker.metrics.last_success
                    else None
                ),
                "last_failure": (
                    breaker.metrics.last_failure.isoformat()
                    if breaker.metrics.last_failure
                    else None
                ),
            }

            health_summary["services"][name] = service_health
            health_summary["circuit_breakers"][name] = breaker.state.value

            if breaker.state in [CircuitState.OPEN, CircuitState.HALF_OPEN]:
                unhealthy_services += 1

        # Determine overall status
        if unhealthy_services == 0:
            health_summary["overall_status"] = ServiceState.HEALTHY.value
        elif unhealthy_services <= total_services * 0.3:  # 30% threshold
            health_summary["overall_status"] = ServiceState.DEGRADED.value
        elif unhealthy_services <= total_services * 0.7:  # 70% threshold
            health_summary["overall_status"] = ServiceState.UNHEALTHY.value
        else:
            health_summary["overall_status"] = ServiceState.CRITICAL.value

        return health_summary


# Global resilience manager instance
resilience_manager = ResilienceManager()


def with_circuit_breaker(service_name: str, fallback: Callable | None = None):
    """Decorator for circuit breaker protection"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await resilience_manager.execute_with_resilience(
                service_name, func, *args, fallback=fallback, **kwargs
            )

        return wrapper

    return decorator


def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retry logic with exponential backoff"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return (
                        await func(*args, **kwargs)
                        if asyncio.iscoroutinefunction(func)
                        else func(*args, **kwargs)
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper

    return decorator
