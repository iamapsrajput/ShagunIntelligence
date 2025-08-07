"""
Production Logging Configuration
Comprehensive logging setup for trading platform
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.config import get_settings


class ProductionLogger:
    """Production-ready logging configuration"""

    def __init__(self):
        self.settings = get_settings()
        self.setup_logging()

    def setup_logging(self):
        """Configure comprehensive logging for production"""

        # Remove default loguru handler
        logger.remove()

        # Create logs directory
        log_dir = Path(self.settings.LOG_FILE_PATH).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Console logging (for development and debugging)
        if self.settings.ENVIRONMENT != "production":
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=self.settings.LOG_LEVEL,
                colorize=True,
            )

        # Main application log
        logger.add(
            self.settings.LOG_FILE_PATH,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=self.settings.LOG_LEVEL,
            rotation=self.settings.LOG_ROTATION_SIZE,
            retention=f"{self.settings.LOG_RETENTION_DAYS} days",
            compression="gz",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

        # Trading-specific logs
        if self.settings.ENABLE_TRADE_LOGGING:
            logger.add(
                "logs/trading_activity.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[trade_type]} | {extra[symbol]} | {message}",
                level="INFO",
                rotation="100MB",
                retention="90 days",
                compression="gz",
                filter=lambda record: "trade_type" in record["extra"],
            )

        # Decision logging for AI agents
        if self.settings.ENABLE_DECISION_LOGGING:
            logger.add(
                "logs/agent_decisions.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {extra[agent]} | {extra[symbol]} | {extra[decision]} | {extra[confidence]} | {message}",
                level="INFO",
                rotation="50MB",
                retention="60 days",
                compression="gz",
                filter=lambda record: "agent" in record["extra"],
            )

        # Security and audit logs
        if self.settings.ENABLE_AUDIT_LOGGING:
            logger.add(
                "logs/security_audit.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[user_id]} | {extra[action]} | {extra[ip_address]} | {message}",
                level="INFO",
                rotation="50MB",
                retention="365 days",  # Keep security logs for 1 year
                compression="gz",
                filter=lambda record: "audit" in record["extra"],
            )

        # Error logs
        logger.add(
            "logs/errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {extra[request_id]} | {message}",
            level="ERROR",
            rotation="50MB",
            retention="180 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
        )

        # Performance logs
        logger.add(
            "logs/performance.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {extra[operation]} | {extra[duration]} | {extra[status]} | {message}",
            level="INFO",
            rotation="50MB",
            retention="30 days",
            compression="gz",
            filter=lambda record: "operation" in record["extra"],
        )

        # API access logs
        logger.add(
            "logs/api_access.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {extra[method]} | {extra[path]} | {extra[status_code]} | {extra[response_time]} | {extra[ip_address]} | {extra[user_agent]}",
            level="INFO",
            rotation="100MB",
            retention="90 days",
            compression="gz",
            filter=lambda record: "method" in record["extra"],
        )

    def log_trade_execution(self, trade_data: dict[str, Any]):
        """Log trade execution with structured data"""
        logger.bind(
            trade_type="EXECUTION",
            symbol=trade_data.get("symbol"),
            quantity=trade_data.get("quantity"),
            price=trade_data.get("price"),
            order_id=trade_data.get("order_id"),
        ).info(f"Trade executed: {trade_data}")

    def log_agent_decision(
        self,
        agent_name: str,
        symbol: str,
        decision: str,
        confidence: float,
        reasoning: str,
    ):
        """Log AI agent decision with structured data"""
        logger.bind(
            agent=agent_name, symbol=symbol, decision=decision, confidence=confidence
        ).info(f"Agent decision: {reasoning}")

    def log_security_event(
        self,
        user_id: str,
        action: str,
        ip_address: str,
        details: str,
        success: bool = True,
    ):
        """Log security event for audit"""
        logger.bind(
            audit=True,
            user_id=user_id,
            action=action,
            ip_address=ip_address,
            success=success,
        ).info(f"Security event: {details}")

    def log_api_access(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time: float,
        ip_address: str,
        user_agent: str,
    ):
        """Log API access for monitoring"""
        logger.bind(
            method=method,
            path=path,
            status_code=status_code,
            response_time=response_time,
            ip_address=ip_address,
            user_agent=user_agent,
        ).info("API access")

    def log_performance(
        self, operation: str, duration: float, status: str, details: str = ""
    ):
        """Log performance metrics"""
        logger.bind(operation=operation, duration=duration, status=status).info(
            f"Performance: {details}"
        )

    def log_system_health(self, component: str, status: str, metrics: dict[str, Any]):
        """Log system health metrics"""
        logger.bind(component=component, status=status, **metrics).info(
            f"System health: {component} is {status}"
        )


class RequestLogger:
    """Middleware for logging HTTP requests"""

    def __init__(self):
        self.production_logger = ProductionLogger()

    async def log_request(self, request, response, process_time: float):
        """Log HTTP request details"""
        self.production_logger.log_api_access(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            response_time=process_time,
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
        )


class StructuredLogger:
    """Structured logging utilities"""

    @staticmethod
    def with_context(**kwargs):
        """Create logger with context"""
        return logger.bind(**kwargs)

    @staticmethod
    def trade_logger(symbol: str, order_type: str):
        """Create trade-specific logger"""
        return logger.bind(
            trade_type=order_type,
            symbol=symbol,
            request_id=f"trade_{symbol}_{order_type}",
        )

    @staticmethod
    def agent_logger(agent_name: str):
        """Create agent-specific logger"""
        return logger.bind(agent=agent_name, request_id=f"agent_{agent_name}")

    @staticmethod
    def security_logger(user_id: str, action: str):
        """Create security-specific logger"""
        return logger.bind(
            audit=True,
            user_id=user_id,
            action=action,
            request_id=f"security_{user_id}_{action}",
        )


# Global instances
production_logger = ProductionLogger()
request_logger = RequestLogger()
structured_logger = StructuredLogger()


def setup_production_logging():
    """Setup production logging configuration"""
    production_logger.setup_logging()
    logger.info("Production logging configured successfully")


def get_logger(name: str = None):
    """Get logger instance with optional name"""
    if name:
        return logger.bind(name=name)
    return logger
