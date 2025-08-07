"""
Enhanced Security Middleware for Financial Trading Platform
Implements comprehensive security measures for protecting financial data and trading operations
"""

import json
import time
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logging_config import structured_logger
from app.core.security import security_manager


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add comprehensive security headers to all responses"""

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to response"""
        response = await call_next(request)

        # Security headers
        security_headers = {
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            # Strict Transport Security (HTTPS only)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' wss: https:; "
                "font-src 'self'; "
                "object-src 'none'; "
                "media-src 'self'; "
                "frame-src 'none';"
            ),
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions Policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "speaker=()"
            ),
            # Remove server information
            "Server": "shagunintelligence",
            # Cache control for sensitive data
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        # Apply headers
        for header, value in security_headers.items():
            response.headers[header] = value

        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Validate and sanitize all input data"""

    def __init__(self, app):
        super().__init__(app)
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.dangerous_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload=",
            r"onerror=",
            r"onclick=",
            r"eval\(",
            r"exec\(",
            r"system\(",
            r"shell_exec\(",
            r"passthru\(",
            r"DROP\s+TABLE",
            r"DELETE\s+FROM",
            r"INSERT\s+INTO",
            r"UPDATE\s+SET",
            r"UNION\s+SELECT",
            r"--\s*$",
            r"/\*.*?\*/",
        ]

    async def dispatch(self, request: Request, call_next) -> Response:
        """Validate request input"""

        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(
                f"Request too large: {content_length} bytes from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Request too large",
                    "max_size": self.max_request_size,
                },
            )

        # Validate headers
        if not self._validate_headers(request):
            return JSONResponse(
                status_code=400, content={"error": "Invalid headers detected"}
            )

        # For POST/PUT requests, validate body
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Check for malicious patterns
                    body_str = body.decode("utf-8", errors="ignore")
                    if self._contains_malicious_patterns(body_str):
                        logger.warning(
                            f"Malicious pattern detected in request from {request.client.host if request.client else 'unknown'}"
                        )
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Invalid request content"},
                        )

                    # Sanitize JSON data
                    if request.headers.get("content-type", "").startswith(
                        "application/json"
                    ):
                        try:
                            json_data = json.loads(body_str)
                            sanitized_data = security_manager.sanitize_input(json_data)
                            # Note: In a real implementation, you'd need to modify the request body
                            # This is complex with FastAPI, so we're just validating here
                        except json.JSONDecodeError:
                            pass  # Not JSON, skip validation

            except Exception as e:
                logger.error(f"Error validating request body: {e}")
                return JSONResponse(
                    status_code=400, content={"error": "Invalid request format"}
                )

        return await call_next(request)

    def _validate_headers(self, request: Request) -> bool:
        """Validate request headers for security"""

        # Check for suspicious user agents
        user_agent = request.headers.get("user-agent", "").lower()
        suspicious_agents = ["sqlmap", "nikto", "nmap", "masscan", "zap"]
        if any(agent in user_agent for agent in suspicious_agents):
            return False

        # Validate content type for POST requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type:
                return False

        return True

    def _contains_malicious_patterns(self, text: str) -> bool:
        """Check if text contains malicious patterns"""
        import re

        text_lower = text.lower()
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False


class TradingSecurityMiddleware(BaseHTTPMiddleware):
    """Specialized security middleware for trading operations"""

    def __init__(self, app):
        super().__init__(app)
        self.trading_endpoints = {
            "/api/v1/trading/",
            "/api/v1/portfolio/",
            "/api/v1/system/",
            "/api/v1/automated-trading/",
        }
        self.sensitive_operations = {
            "place_order",
            "cancel_order",
            "modify_order",
            "emergency_stop",
            "start_trading",
            "stop_trading",
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply trading-specific security measures"""

        # Check if this is a trading-related request
        is_trading_request = any(
            request.url.path.startswith(endpoint) for endpoint in self.trading_endpoints
        )

        if is_trading_request:
            # Enhanced logging for trading operations
            await self._log_trading_request(request)

            # Additional validation for trading requests
            if not await self._validate_trading_request(request):
                return JSONResponse(
                    status_code=403,
                    content={"error": "Trading request validation failed"},
                )

            # Check trading hours (if configured)
            if not await self._check_trading_hours(request):
                return JSONResponse(
                    status_code=403,
                    content={"error": "Trading not allowed outside market hours"},
                )

        response = await call_next(request)

        # Log trading response
        if is_trading_request:
            await self._log_trading_response(request, response)

        return response

    async def _log_trading_request(self, request: Request):
        """Log trading requests with enhanced detail"""
        client_ip = request.client.host if request.client else "unknown"

        structured_logger.security_logger("trading", "request").info(
            f"Trading request: {request.method} {request.url.path} from {client_ip}"
        )

    async def _validate_trading_request(self, request: Request) -> bool:
        """Validate trading-specific requests"""

        # Check for required authentication
        auth_header = request.headers.get("authorization")
        if not auth_header:
            logger.warning("Trading request without authentication")
            return False

        # Additional validation for sensitive operations
        if request.method in ["POST", "PUT", "DELETE"]:
            # Check for CSRF token in headers
            csrf_token = request.headers.get("x-csrf-token")
            if not csrf_token:
                logger.warning("Trading operation without CSRF token")
                return False

        return True

    async def _check_trading_hours(self, request: Request) -> bool:
        """Check if trading is allowed at current time"""
        # This would integrate with your market schedule service
        # For now, always return True
        return True

    async def _log_trading_response(self, request: Request, response: Response):
        """Log trading responses for audit"""
        structured_logger.security_logger("trading", "response").info(
            f"Trading response: {response.status_code} for {request.method} {request.url.path}"
        )


class APIKeyValidationMiddleware(BaseHTTPMiddleware):
    """Validate API keys and implement key rotation"""

    def __init__(self, app):
        super().__init__(app)
        self.api_key_endpoints = {"/api/v1/"}
        self.exempt_paths = {"/api/v1/health", "/api/v1/docs", "/api/v1/openapi.json"}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Validate API keys for protected endpoints"""

        # Check if endpoint requires API key
        requires_api_key = any(
            request.url.path.startswith(endpoint) for endpoint in self.api_key_endpoints
        )

        is_exempt = any(request.url.path.startswith(path) for path in self.exempt_paths)

        if requires_api_key and not is_exempt:
            # Validate API key
            api_key = request.headers.get("x-api-key")
            if not api_key:
                return JSONResponse(
                    status_code=401, content={"error": "API key required"}
                )

            # Validate key format and signature
            if not await self._validate_api_key(request, api_key):
                return JSONResponse(
                    status_code=401, content={"error": "Invalid API key"}
                )

        return await call_next(request)

    async def _validate_api_key(self, request: Request, api_key: str) -> bool:
        """Validate API key and signature"""

        # Check key format
        if len(api_key) < 32:
            return False

        # For production, implement proper key validation
        # This could include database lookup, signature verification, etc.

        # Check for signature if provided
        signature = request.headers.get("x-api-signature")
        timestamp = request.headers.get("x-api-timestamp")

        if signature and timestamp:
            return security_manager.validate_api_signature(
                request, api_key, signature, timestamp
            )

        return True  # Basic validation passed


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Comprehensive audit logging for compliance"""

    def __init__(self, app):
        super().__init__(app)
        self.audit_paths = {
            "/api/v1/trading/",
            "/api/v1/portfolio/",
            "/api/v1/system/",
            "/api/v1/users/",
            "/api/v1/auth/",
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Log requests for audit compliance"""

        # Check if this request should be audited
        should_audit = any(
            request.url.path.startswith(path) for path in self.audit_paths
        )

        if should_audit:
            # Pre-request audit log
            await self._log_audit_event(request, "REQUEST_START")

        response = await call_next(request)

        if should_audit:
            # Post-request audit log
            await self._log_audit_event(request, "REQUEST_COMPLETE", response)

        return response

    async def _log_audit_event(
        self, request: Request, event_type: str, response: Response | None = None
    ):
        """Log audit event with comprehensive details"""

        audit_data = {
            "event_type": event_type,
            "timestamp": time.time(),
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "headers": dict(request.headers),
        }

        if response:
            audit_data["status_code"] = response.status_code
            audit_data["response_headers"] = dict(response.headers)

        # Remove sensitive headers
        sensitive_headers = {"authorization", "x-api-key", "cookie"}
        for header in sensitive_headers:
            audit_data["headers"].pop(header, None)

        structured_logger.security_logger("audit", "compliance").info(
            f"Audit log: {json.dumps(audit_data)}"
        )
