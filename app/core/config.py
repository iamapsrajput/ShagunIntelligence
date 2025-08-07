from functools import lru_cache

from pydantic_settings import BaseSettings

from .api_config import APIProvider, get_api_config


class Settings(BaseSettings):
    # App Configuration
    ENVIRONMENT: str = "development"
    APP_NAME: str = "AI Trading Platform"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"  # Changed from 0.0.0.0 for security
    PORT: int = 8000
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1"]  # Restricted for security
    API_V1_STR: str = "/api/v1"

    # Zerodha Kite Connect API (Primary Source)
    KITE_API_KEY: str = ""
    KITE_API_SECRET: str = ""
    KITE_ACCESS_TOKEN: str = ""
    KITE_REQUEST_TOKEN: str = ""

    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4"

    # Database Configuration
    DATABASE_URL: str = (
        "sqlite:///./dev_trading.db"  # Default to SQLite for development
    )
    REDIS_URL: str = "redis://localhost:6379"

    # Security
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    REFRESH_SECRET_KEY: str = "dev-refresh-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Trading Configuration
    TRADING_MODE: str = "paper"  # paper/live/maintenance
    LIVE_TRADING_ENABLED: bool = False
    PAPER_TRADING_ENABLED: bool = True
    AUTOMATED_TRADING_ENABLED: bool = False
    MANUAL_APPROVAL_REQUIRED: bool = True

    # Auto-enable live trading if environment variables are set

    # Risk Management
    MAX_RISK_PER_TRADE: float = 0.02  # 2% max risk per trade
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily loss
    DEFAULT_POSITION_SIZE: float = 10000  # Default position size in INR
    MAX_POSITION_VALUE: float = 200  # Maximum per position
    MIN_POSITION_SIZE: int = 1  # Minimum shares
    MAX_CONCURRENT_POSITIONS: int = 3  # Maximum concurrent positions

    # Circuit Breaker
    CIRCUIT_BREAKER_ENABLED: bool = True
    CIRCUIT_BREAKER_LOSS_PERCENT: float = 0.08  # 8% emergency stop
    EMERGENCY_STOP_LOSS_AMOUNT: float = 80  # ₹80 emergency stop

    # Trading Hours
    ENFORCE_TRADING_HOURS: bool = True
    TRADING_START_TIME: str = "09:15"
    TRADING_END_TIME: str = "15:30"
    TRADING_TIMEZONE: str = "Asia/Kolkata"

    # Automated Position Management
    AUTO_STOP_LOSS: bool = True
    AUTO_STOP_LOSS_PERCENT: float = 0.05  # 5% stop loss
    AUTO_TAKE_PROFIT: bool = True
    AUTO_TAKE_PROFIT_PERCENT: float = 0.10  # 10% take profit

    # Market Data
    MARKET_DATA_REFRESH_INTERVAL: int = 1  # seconds

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "logs/trading.log"
    LOG_ROTATION_SIZE: str = "100MB"
    LOG_RETENTION_DAYS: int = 30
    ENABLE_AUDIT_LOGGING: bool = True
    ENABLE_TRADE_LOGGING: bool = True
    ENABLE_DECISION_LOGGING: bool = True
    LOG_FILE: str = "logs/trading.log"

    # Multi-Source Data Configuration
    DATA_SOURCE_FAILOVER_ENABLED: bool = True
    DATA_SOURCE_HEALTH_CHECK_INTERVAL: int = 30  # seconds
    DATA_SOURCE_TIMEOUT: int = 10  # seconds
    DATA_SOURCE_MAX_RETRIES: int = 3
    DATA_SOURCE_RETRY_DELAY: int = 5  # seconds

    # Alternative Data Sources (for failover)
    # Alpha Vantage
    ALPHA_VANTAGE_API_KEY: str | None = None
    ALPHA_VANTAGE_ENABLED: bool = True  # Enable if API key is provided
    ALPHA_VANTAGE_PRIORITY: int = 2  # Lower = higher priority

    # Yahoo Finance
    YAHOO_FINANCE_ENABLED: bool = True  # Enable as backup
    YAHOO_FINANCE_PRIORITY: int = 3

    # NSE Official API
    NSE_API_ENABLED: bool = True  # Enable as backup
    NSE_API_PRIORITY: int = 4
    NSE_API_BASE_URL: str = "https://www.nseindia.com"

    # News and Sentiment Sources
    NEWS_API_KEY: str | None = None
    NEWS_API_ENABLED: bool = True  # Enable if API key is provided

    TWITTER_API_KEY: str | None = None
    TWITTER_API_SECRET: str | None = None
    TWITTER_BEARER_TOKEN: str | None = None
    TWITTER_API_ENABLED: bool = True  # Enable if API keys are provided

    # Finnhub API
    FINNHUB_API_KEY: str | None = None
    FINNHUB_ENABLED: bool = True  # Enable if API key is provided

    # Connection Pool Settings
    CONNECTION_POOL_SIZE: int = 10
    CONNECTION_POOL_OVERFLOW: int = 5

    # Rate Limiting (requests per minute)
    ZERODHA_RATE_LIMIT: int = 180  # 3 per second
    ALPHA_VANTAGE_RATE_LIMIT: int = 5  # Free tier limit
    YAHOO_FINANCE_RATE_LIMIT: int = 100
    NSE_API_RATE_LIMIT: int = 60

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Auto-enable live trading if Kite credentials are provided
        if self.KITE_API_KEY and self.KITE_ACCESS_TOKEN and self.TRADING_MODE == "live":
            object.__setattr__(self, "LIVE_TRADING_ENABLED", True)
            object.__setattr__(self, "AUTOMATED_TRADING_ENABLED", True)
            object.__setattr__(self, "PAPER_TRADING_ENABLED", False)
            object.__setattr__(self, "MANUAL_APPROVAL_REQUIRED", False)

        # Auto-enable APIs when keys are provided
        if self.ALPHA_VANTAGE_API_KEY:
            object.__setattr__(self, "ALPHA_VANTAGE_ENABLED", True)
        if self.NEWS_API_KEY:
            object.__setattr__(self, "NEWS_API_ENABLED", True)
        if self.FINNHUB_API_KEY:
            object.__setattr__(self, "FINNHUB_ENABLED", True)
        if self.TWITTER_API_KEY and self.TWITTER_API_SECRET:
            object.__setattr__(self, "TWITTER_API_ENABLED", True)

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Helper function to get API configuration
def get_api_settings():
    """Get API configuration settings."""
    return get_api_config()


# Helper function to check if an API is enabled
def is_api_enabled(provider: APIProvider) -> bool:
    """Check if an API provider is enabled."""
    api_config = get_api_config()
    config = api_config.get_api_config(provider)
    return config.enabled if config else False
