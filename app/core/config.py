from pydantic_settings import BaseSettings
from typing import List, Dict, Optional
import os
from functools import lru_cache
from .api_config import get_api_config, APIProvider

class Settings(BaseSettings):
    # App Configuration
    APP_NAME: str = "AI Trading Platform"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Zerodha Kite Connect API (Primary Source)
    KITE_API_KEY: str = ""
    KITE_API_SECRET: str = ""
    KITE_ACCESS_TOKEN: str = ""
    KITE_REQUEST_TOKEN: str = ""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4"
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost/shagunintelligence"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Trading Configuration
    MAX_RISK_PER_TRADE: float = 0.02  # 2% max risk per trade
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily loss
    DEFAULT_POSITION_SIZE: float = 10000  # Default position size in INR
    
    # Market Data
    MARKET_DATA_REFRESH_INTERVAL: int = 1  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading.log"
    
    # Multi-Source Data Configuration
    DATA_SOURCE_FAILOVER_ENABLED: bool = True
    DATA_SOURCE_HEALTH_CHECK_INTERVAL: int = 30  # seconds
    DATA_SOURCE_TIMEOUT: int = 10  # seconds
    DATA_SOURCE_MAX_RETRIES: int = 3
    DATA_SOURCE_RETRY_DELAY: int = 5  # seconds
    
    # Alternative Data Sources (for failover)
    # Alpha Vantage
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_ENABLED: bool = False
    ALPHA_VANTAGE_PRIORITY: int = 2  # Lower = higher priority
    
    # Yahoo Finance
    YAHOO_FINANCE_ENABLED: bool = False
    YAHOO_FINANCE_PRIORITY: int = 3
    
    # NSE Official API
    NSE_API_ENABLED: bool = False
    NSE_API_PRIORITY: int = 4
    NSE_API_BASE_URL: str = "https://www.nseindia.com"
    
    # News and Sentiment Sources
    NEWS_API_KEY: Optional[str] = None
    NEWS_API_ENABLED: bool = False
    
    TWITTER_API_KEY: Optional[str] = None
    TWITTER_API_SECRET: Optional[str] = None
    TWITTER_API_ENABLED: bool = False
    
    # Connection Pool Settings
    CONNECTION_POOL_SIZE: int = 10
    CONNECTION_POOL_OVERFLOW: int = 5
    
    # Rate Limiting (requests per minute)
    ZERODHA_RATE_LIMIT: int = 180  # 3 per second
    ALPHA_VANTAGE_RATE_LIMIT: int = 5  # Free tier limit
    YAHOO_FINANCE_RATE_LIMIT: int = 100
    NSE_API_RATE_LIMIT: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
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