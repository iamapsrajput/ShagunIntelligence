"""
Enhanced Error Handling Middleware for Production
Comprehensive error handling, logging, and recovery mechanisms
"""

import time
import traceback
from typing import Any

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logging_config import structured_logger


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Enhanced error handling middleware with comprehensive logging and recovery"""

    def __init__(self, app, enable_debug: bool = False):
        super().__init__(app)
        self.settings = get_settings()
        self.enable_debug = enable_debug or self.settings.ENVIRONMENT == "development"

        # Error tracking
        self.error_counts: dict[str, int] = {}
        self.last_error_time: dict[str, float] = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with comprehensive error handling"""
        start_time = time.time()
        request_id = self._generate_request_id()

        # Add request ID to context
        request.state.request_id = request_id

        try:
            # Log incoming request
            await self._log_request(request, request_id)

            # Process request
            response = await call_next(request)

            # Log successful response
            process_time = time.time() - start_time
            await self._log_response(request, response, process_time, request_id)

            return response

        except HTTPException as e:
            # Handle HTTP exceptions
            return await self._handle_http_exception(request, e, request_id)

        except Exception as e:
            # Handle unexpected exceptions
            return await self._handle_unexpected_exception(
                request, e, request_id, start_time
            )

    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details"""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")

        structured_logger.with_context(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            query_params=str(request.query_params),
            client_ip=client_ip,
            user_agent=user_agent,
        ).info("Incoming request")

    async def _log_response(
        self, request: Request, response: Response, process_time: float, request_id: str
    ):
        """Log response details"""
        structured_logger.with_context(
            request_id=request_id,
            status_code=response.status_code,
            process_time=round(process_time * 1000, 2),  # Convert to milliseconds
        ).info("Request completed")

    async def _handle_http_exception(
        self, request: Request, exc: HTTPException, request_id: str
    ) -> JSONResponse:
        """Handle HTTP exceptions with proper logging"""
        error_details = {
            "error": "HTTP Exception",
            "status_code": exc.status_code,
            "detail": exc.detail,
            "request_id": request_id,
            "path": str(request.url.path),
            "method": request.method,
        }

        # Log based on severity
        if exc.status_code >= 500:
            logger.error(f"Server error: {error_details}")
        elif exc.status_code >= 400:
            logger.warning(f"Client error: {error_details}")

        # Return appropriate response
        response_data = {
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
        }

        if self.enable_debug:
            response_data["debug"] = error_details

        return JSONResponse(status_code=exc.status_code, content=response_data)

    async def _handle_unexpected_exception(
        self, request: Request, exc: Exception, request_id: str, start_time: float
    ) -> JSONResponse:
        """Handle unexpected exceptions with comprehensive logging and recovery"""

        process_time = time.time() - start_time
        error_type = type(exc).__name__
        error_message = str(exc)

        # Track error frequency
        self._track_error(error_type)

        # Comprehensive error logging
        error_context = {
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message,
            "path": str(request.url.path),
            "method": request.method,
            "process_time": round(process_time * 1000, 2),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", "unknown"),
        }

        if self.enable_debug:
            error_context["traceback"] = traceback.format_exc()

        logger.error(f"Unexpected error: {error_context}")

        # Determine if this is a critical error
        is_critical = self._is_critical_error(exc, error_type)

        if is_critical:
            await self._handle_critical_error(error_context)

        # Return user-friendly error response
        response_data = {
            "error": True,
            "message": "An internal server error occurred",
            "status_code": 500,
            "request_id": request_id,
        }

        if self.enable_debug:
            response_data["debug"] = {
                "error_type": error_type,
                "error_message": error_message,
                "traceback": traceback.format_exc(),
            }

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=response_data
        )

    def _track_error(self, error_type: str):
        """Track error frequency for monitoring"""
        current_time = time.time()

        # Increment error count
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_error_time[error_type] = current_time

        # Check for error spikes
        if self.error_counts[error_type] > 10:  # More than 10 errors of same type
            time_window = current_time - self.last_error_time.get(
                f"{error_type}_first", current_time
            )
            if time_window < 300:  # Within 5 minutes
                logger.critical(
                    f"Error spike detected: {error_type} occurred {self.error_counts[error_type]} times in {time_window:.1f}s"
                )

    def _is_critical_error(self, exc: Exception, error_type: str) -> bool:
        """Determine if error is critical and requires immediate attention"""
        critical_errors = [
            "DatabaseError",
            "ConnectionError",
            "AuthenticationError",
            "SecurityError",
            "KiteAuthenticationError",
            "KiteTokenExpiredError",
        ]

        return error_type in critical_errors or "trading" in str(exc).lower()

    async def _handle_critical_error(self, error_context: dict[str, Any]):
        """Handle critical errors with immediate notifications"""
        logger.critical(f"CRITICAL ERROR DETECTED: {error_context}")

        # Here you could add:
        # - Send email/SMS alerts
        # - Trigger emergency procedures
        # - Disable trading if necessary
        # - Scale up resources

        # For now, just log with high priority
        structured_logger.security_logger("system", "critical_error").critical(
            f"Critical system error: {error_context['error_type']} - {error_context['error_message']}"
        )

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers first (for load balancers/proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracking"""
        import uuid

        return str(uuid.uuid4())[:8]


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Circuit breaker pattern for external service calls"""

    def __init__(self, app, failure_threshold: int = 5, recovery_timeout: int = 60):
        super().__init__(app)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply circuit breaker pattern"""

        # Check if circuit should be closed
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")

        # If circuit is open, return error immediately
        if self.state == "OPEN":
            return JSONResponse(
                status_code=503,
                content={
                    "error": True,
                    "message": "Service temporarily unavailable",
                    "circuit_breaker": "OPEN",
                },
            )

        try:
            response = await call_next(request)

            # Success - reset failure count if in HALF_OPEN
            if self.state == "HALF_OPEN":
                self.failure_count = 0
                self.state = "CLOSED"
                logger.info("Circuit breaker reset to CLOSED")

            return response

        except Exception as e:
            # Failure - increment count and potentially open circuit
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(
                    f"Circuit breaker OPENED after {self.failure_count} failures"
                )

            raise e


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for API protection"""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply rate limiting"""
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time
                for req_time in self.requests[client_ip]
                if current_time - req_time < 60  # Keep requests from last minute
            ]
        else:
            self.requests[client_ip] = []

        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "message": "Rate limit exceeded",
                    "retry_after": 60,
                },
            )

        # Add current request
        self.requests[client_ip].append(current_time)

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        if request.client:
            return request.client.host
        return "unknown"
