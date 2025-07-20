from pydantic_settings import BaseSettings
from typing import List
import os
from functools import lru_cache

class Settings(BaseSettings):
    # App Configuration
    APP_NAME: str = "AI Trading Platform"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Zerodha Kite Connect API
    KITE_API_KEY: str = ""
    KITE_API_SECRET: str = ""
    KITE_ACCESS_TOKEN: str = ""
    KITE_REQUEST_TOKEN: str = ""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4"
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost/aitrading"
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
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()