"""
Production Security Configuration
Implements comprehensive security measures for the trading platform
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext

from app.core.config import get_settings


class SecurityManager:
    """Comprehensive security manager for production environment"""

    def __init__(self):
        self.settings = get_settings()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()

        # Rate limiting storage
        self._rate_limit_storage: dict[str, list[float]] = {}

        # Failed login attempts tracking
        self._failed_attempts: dict[str, dict[str, Any]] = {}

        # API key validation
        self._api_key_cache: dict[str, dict[str, Any]] = {}

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.JWT_EXPIRE_MINUTES
            )

        to_encode.update({"exp": expire, "iat": datetime.utcnow()})

        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.JWT_SECRET_KEY,
            algorithm=self.settings.JWT_ALGORITHM,
        )

        return encoded_jwt

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.settings.JWT_SECRET_KEY,
                algorithms=[self.settings.JWT_ALGORITHM],
            )

            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )

            return payload

        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

    def check_rate_limit(
        self, identifier: str, max_requests: int = 100, window_seconds: int = 60
    ) -> bool:
        """Check if request is within rate limits"""
        if not self.settings.RATE_LIMIT_ENABLED:
            return True

        current_time = time.time()
        window_start = current_time - window_seconds

        # Clean old entries
        if identifier in self._rate_limit_storage:
            self._rate_limit_storage[identifier] = [
                timestamp
                for timestamp in self._rate_limit_storage[identifier]
                if timestamp > window_start
            ]
        else:
            self._rate_limit_storage[identifier] = []

        # Check current count
        current_count = len(self._rate_limit_storage[identifier])

        if current_count >= max_requests:
            logger.warning(
                f"Rate limit exceeded for {identifier}: {current_count}/{max_requests}"
            )
            return False

        # Add current request
        self._rate_limit_storage[identifier].append(current_time)
        return True

    def track_failed_login(self, identifier: str) -> bool:
        """Track failed login attempts and implement lockout"""
        current_time = time.time()

        if identifier not in self._failed_attempts:
            self._failed_attempts[identifier] = {
                "count": 0,
                "last_attempt": current_time,
                "locked_until": None,
            }

        attempt_data = self._failed_attempts[identifier]

        # Check if account is locked
        if attempt_data["locked_until"] and current_time < attempt_data["locked_until"]:
            remaining_time = int(attempt_data["locked_until"] - current_time)
            logger.warning(
                f"Account {identifier} is locked for {remaining_time} seconds"
            )
            return False

        # Reset count if last attempt was more than 15 minutes ago
        if current_time - attempt_data["last_attempt"] > 900:  # 15 minutes
            attempt_data["count"] = 0

        # Increment failed attempts
        attempt_data["count"] += 1
        attempt_data["last_attempt"] = current_time

        # Lock account after 5 failed attempts
        if attempt_data["count"] >= 5:
            lockout_duration = min(
                300 * (2 ** (attempt_data["count"] - 5)), 3600
            )  # Max 1 hour
            attempt_data["locked_until"] = current_time + lockout_duration

            logger.warning(
                f"Account {identifier} locked for {lockout_duration} seconds after {attempt_data['count']} failed attempts"
            )
            return False

        return True

    def reset_failed_login(self, identifier: str):
        """Reset failed login attempts after successful login"""
        if identifier in self._failed_attempts:
            del self._failed_attempts[identifier]

    def validate_api_signature(
        self, request: Request, api_key: str, signature: str, timestamp: str
    ) -> bool:
        """Validate API request signature"""
        try:
            # Check timestamp (prevent replay attacks)
            request_time = float(timestamp)
            current_time = time.time()

            if abs(current_time - request_time) > 300:  # 5 minutes tolerance
                logger.warning(f"API request timestamp too old: {timestamp}")
                return False

            # Get API secret for the key
            api_secret = self._get_api_secret(api_key)
            if not api_secret:
                logger.warning(f"Invalid API key: {api_key}")
                return False

            # Create expected signature
            body = request.body() if hasattr(request, "body") else b""
            message = f"{request.method}{request.url.path}{timestamp}{body.decode()}"

            expected_signature = hmac.new(
                api_secret.encode(), message.encode(), hashlib.sha256
            ).hexdigest()

            # Compare signatures
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning(f"Invalid API signature for key: {api_key}")
                return False

            return True

        except Exception as e:
            logger.error(f"API signature validation error: {e}")
            return False

    def _get_api_secret(self, api_key: str) -> str | None:
        """Get API secret for given key (implement based on your storage)"""
        # This should be implemented to fetch from your secure storage
        # For now, return None to indicate invalid key
        return None

    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data to prevent injection attacks"""
        if isinstance(data, str):
            # Remove potentially dangerous characters
            dangerous_chars = ["<", ">", '"', "'", "&", "\x00"]
            for char in dangerous_chars:
                data = data.replace(char, "")

            # Limit length
            if len(data) > 1000:
                data = data[:1000]

        elif isinstance(data, dict):
            return {key: self.sanitize_input(value) for key, value in data.items()}

        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]

        return data

    def validate_trading_request(self, request_data: dict[str, Any]) -> bool:
        """Validate trading request for security"""
        try:
            # Check required fields
            required_fields = ["symbol", "quantity", "order_type"]
            for field in required_fields:
                if field not in request_data:
                    logger.warning(
                        f"Missing required field in trading request: {field}"
                    )
                    return False

            # Validate symbol format
            symbol = request_data["symbol"]
            if not isinstance(symbol, str) or len(symbol) > 20:
                logger.warning(f"Invalid symbol format: {symbol}")
                return False

            # Validate quantity
            quantity = request_data["quantity"]
            if (
                not isinstance(quantity, (int, float))
                or quantity <= 0
                or quantity > 10000
            ):
                logger.warning(f"Invalid quantity: {quantity}")
                return False

            # Validate order type
            valid_order_types = ["MARKET", "LIMIT", "SL", "SL-M"]
            if request_data["order_type"] not in valid_order_types:
                logger.warning(f"Invalid order type: {request_data['order_type']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Trading request validation error: {e}")
            return False

    def log_security_event(
        self, event_type: str, details: dict[str, Any], severity: str = "INFO"
    ):
        """Log security events for audit"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "details": details,
        }

        if severity == "CRITICAL":
            logger.critical(f"SECURITY EVENT: {log_entry}")
        elif severity == "WARNING":
            logger.warning(f"SECURITY EVENT: {log_entry}")
        else:
            logger.info(f"SECURITY EVENT: {log_entry}")

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data (implement based on your needs)"""
        # This should use proper encryption like AES
        # For now, return base64 encoded (NOT SECURE - implement proper encryption)
        import base64

        return base64.b64encode(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data (implement based on your needs)"""
        # This should use proper decryption like AES
        # For now, return base64 decoded (NOT SECURE - implement proper decryption)
        import base64

        return base64.b64decode(encrypted_data.encode()).decode()


# Global security manager instance
security_manager = SecurityManager()
