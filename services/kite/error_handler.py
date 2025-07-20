"""Comprehensive error handling and recovery system"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import traceback
from loguru import logger

from .exceptions import (
    KiteException, 
    KiteAuthenticationError, 
    KiteTokenExpiredError,
    KiteRateLimitError, 
    KiteNetworkError,
    KiteOrderError,
    KiteDataError,
    KiteWebSocketError
)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"


@dataclass
class ErrorInfo:
    """Error information structure"""
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: float
    retry_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None


@dataclass
class RetryConfig:
    """Retry configuration"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


class CircuitBreaker:
    """Circuit breaker for API calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func):
        """Decorator for circuit breaker functionality"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise KiteException("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.reset()
                return result
            except self.expected_exception as e:
                self.record_failure()
                raise e
        
        return wrapper
    
    def record_failure(self):
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def reset(self):
        """Reset circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        logger.info("Circuit breaker reset to CLOSED state")


class KiteErrorHandler:
    """Comprehensive error handling system for Kite Connect"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_callbacks: List[Callable] = []
        self.default_retry_config = RetryConfig()
        
        # Initialize circuit breakers for different operations
        self.circuit_breakers['api'] = CircuitBreaker(failure_threshold=10, recovery_timeout=300)
        self.circuit_breakers['websocket'] = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.circuit_breakers['orders'] = CircuitBreaker(failure_threshold=3, recovery_timeout=120)
        
    def with_error_handling(self, 
                          operation_type: str = "api",
                          retry_config: Optional[RetryConfig] = None,
                          circuit_breaker: bool = True):
        """Decorator for comprehensive error handling"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                config = retry_config or self.default_retry_config
                retry_count = 0
                last_exception = None
                
                while retry_count <= config.max_retries:
                    try:
                        # Apply circuit breaker if enabled
                        if circuit_breaker and operation_type in self.circuit_breakers:
                            cb = self.circuit_breakers[operation_type]
                            return await cb.call(func)(*args, **kwargs)
                        else:
                            return await func(*args, **kwargs)
                            
                    except Exception as e:
                        last_exception = e
                        error_info = await self._process_error(e, func.__name__, retry_count, args, kwargs)
                        
                        # Check if we should retry
                        if not await self._should_retry(e, retry_count, config):
                            await self._log_final_error(error_info)
                            raise e
                        
                        # Calculate delay and wait
                        if retry_count < config.max_retries:
                            delay = self._calculate_delay(retry_count, config)
                            logger.warning(f"Retrying {func.__name__} in {delay:.2f}s (attempt {retry_count + 1}/{config.max_retries})")
                            await asyncio.sleep(delay)
                        
                        retry_count += 1
                
                # All retries exhausted
                await self._log_final_error(self.error_history[-1] if self.error_history else None)
                raise last_exception
                
            return wrapper
        return decorator
    
    async def _process_error(self, 
                           exception: Exception, 
                           function_name: str, 
                           retry_count: int,
                           args: tuple,
                           kwargs: dict) -> ErrorInfo:
        """Process and categorize error"""
        error_type = type(exception).__name__
        severity = self._determine_severity(exception)
        
        error_info = ErrorInfo(
            error_type=error_type,
            message=str(exception),
            severity=severity,
            timestamp=time.time(),
            retry_count=retry_count,
            context={
                'function': function_name,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            },
            traceback=traceback.format_exc()
        )
        
        # Store error history
        self.error_history.append(error_info)
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        # Log error
        await self._log_error(error_info)
        
        # Notify callbacks
        await self._notify_error_callbacks(error_info)
        
        return error_info
    
    def _determine_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type"""
        severity_map = {
            KiteAuthenticationError: ErrorSeverity.CRITICAL,
            KiteTokenExpiredError: ErrorSeverity.HIGH,
            KiteOrderError: ErrorSeverity.HIGH,
            KiteRateLimitError: ErrorSeverity.MEDIUM,
            KiteWebSocketError: ErrorSeverity.MEDIUM,
            KiteNetworkError: ErrorSeverity.MEDIUM,
            KiteDataError: ErrorSeverity.LOW,
            ConnectionError: ErrorSeverity.MEDIUM,
            TimeoutError: ErrorSeverity.MEDIUM,
            ValueError: ErrorSeverity.LOW,
            KeyError: ErrorSeverity.LOW
        }
        
        return severity_map.get(type(exception), ErrorSeverity.MEDIUM)
    
    async def _should_retry(self, exception: Exception, retry_count: int, config: RetryConfig) -> bool:
        """Determine if operation should be retried"""
        if config.strategy == RetryStrategy.NO_RETRY:
            return False
        
        if retry_count >= config.max_retries:
            return False
        
        # Don't retry critical authentication errors
        if isinstance(exception, (KiteAuthenticationError, KiteTokenExpiredError)):
            return False
        
        # Don't retry validation errors
        if isinstance(exception, (ValueError, KeyError)) and not isinstance(exception, KiteException):
            return False
        
        # Retry network and temporary errors
        retryable_exceptions = (
            KiteRateLimitError,
            KiteNetworkError,
            KiteWebSocketError,
            ConnectionError,
            TimeoutError,
            KiteDataError
        )
        
        return isinstance(exception, retryable_exceptions)
    
    def _calculate_delay(self, retry_count: int, config: RetryConfig) -> float:
        """Calculate delay for next retry"""
        if config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (retry_count + 1)
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** retry_count)
        else:
            delay = config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Add jitter if enabled
        if config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    async def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_message = f"Error in operation: {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log detailed traceback for debugging
        if error_info.traceback:
            logger.debug(f"Traceback: {error_info.traceback}")
    
    async def _log_final_error(self, error_info: Optional[ErrorInfo]):
        """Log final error after all retries exhausted"""
        if error_info:
            logger.error(f"Operation failed permanently after {error_info.retry_count} retries: {error_info.message}")
    
    async def _notify_error_callbacks(self, error_info: ErrorInfo):
        """Notify registered error callbacks"""
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_info)
                else:
                    callback(error_info)
            except Exception as e:
                logger.error(f"Error in error callback: {str(e)}")
    
    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self.error_callbacks.append(callback)
    
    def remove_error_callback(self, callback: Callable):
        """Remove error callback"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {}
        
        total_errors = len(self.error_history)
        error_types = {}
        severity_counts = {}
        
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        return {
            'total_errors': total_errors,
            'recent_errors_1h': len(recent_errors),
            'error_types': error_types,
            'severity_distribution': severity_counts,
            'circuit_breaker_states': {
                name: cb.state for name, cb in self.circuit_breakers.items()
            }
        }
    
    def reset_circuit_breakers(self):
        """Reset all circuit breakers"""
        for name, cb in self.circuit_breakers.items():
            cb.reset()
            logger.info(f"Reset circuit breaker: {name}")
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        logger.info("Cleared error history")


class ReconnectionManager:
    """Manages reconnection logic for WebSocket and API connections"""
    
    def __init__(self, max_reconnect_attempts: int = 10, base_delay: float = 1.0):
        self.max_reconnect_attempts = max_reconnect_attempts
        self.base_delay = base_delay
        self.reconnect_attempts = 0
        self.is_reconnecting = False
        self.reconnect_callbacks: List[Callable] = []
        
    async def attempt_reconnection(self, reconnect_func: Callable) -> bool:
        """Attempt to reconnect with exponential backoff"""
        if self.is_reconnecting:
            logger.warning("Reconnection already in progress")
            return False
        
        self.is_reconnecting = True
        
        try:
            while self.reconnect_attempts < self.max_reconnect_attempts:
                try:
                    logger.info(f"Attempting reconnection {self.reconnect_attempts + 1}/{self.max_reconnect_attempts}")
                    
                    # Attempt reconnection
                    success = await reconnect_func()
                    
                    if success:
                        logger.info("Reconnection successful")
                        self.reconnect_attempts = 0
                        await self._notify_reconnect_success()
                        return True
                    
                except Exception as e:
                    logger.error(f"Reconnection attempt failed: {str(e)}")
                
                self.reconnect_attempts += 1
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    delay = self.base_delay * (2 ** self.reconnect_attempts)
                    delay = min(delay, 300)  # Max 5 minutes
                    logger.info(f"Waiting {delay}s before next reconnection attempt")
                    await asyncio.sleep(delay)
            
            logger.error("All reconnection attempts failed")
            await self._notify_reconnect_failure()
            return False
            
        finally:
            self.is_reconnecting = False
    
    def reset_reconnection_state(self):
        """Reset reconnection state"""
        self.reconnect_attempts = 0
        self.is_reconnecting = False
    
    def add_reconnect_callback(self, callback: Callable):
        """Add reconnection callback"""
        self.reconnect_callbacks.append(callback)
    
    async def _notify_reconnect_success(self):
        """Notify successful reconnection"""
        for callback in self.reconnect_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(True)
                else:
                    callback(True)
            except Exception as e:
                logger.error(f"Error in reconnect callback: {str(e)}")
    
    async def _notify_reconnect_failure(self):
        """Notify failed reconnection"""
        for callback in self.reconnect_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(False)
                else:
                    callback(False)
            except Exception as e:
                logger.error(f"Error in reconnect callback: {str(e)}")


# Global error handler instance
global_error_handler = KiteErrorHandler()