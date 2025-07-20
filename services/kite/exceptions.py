"""Custom exceptions for Kite Connect service"""

class KiteException(Exception):
    """Base exception for Kite Connect related errors"""
    def __init__(self, message: str, error_code: str = None, status_code: int = None):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

class KiteAuthenticationError(KiteException):
    """Raised when authentication fails"""
    pass

class KiteTokenExpiredError(KiteException):
    """Raised when access token has expired"""
    pass

class KiteRateLimitError(KiteException):
    """Raised when rate limit is exceeded"""
    pass

class KiteNetworkError(KiteException):
    """Raised when network connection fails"""
    pass

class KiteOrderError(KiteException):
    """Raised when order placement fails"""
    pass

class KiteDataError(KiteException):
    """Raised when data retrieval fails"""
    pass

class KiteWebSocketError(KiteException):
    """Raised when WebSocket connection fails"""
    pass

class KiteValidationError(KiteException):
    """Raised when data validation fails"""
    pass