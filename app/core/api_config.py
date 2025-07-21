"""
Comprehensive API configuration management for AlgoHive.

This module provides centralized configuration for all API integrations
with security, rate limiting, and monitoring capabilities.
"""

from pydantic import BaseSettings, Field, SecretStr, validator
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import os
from functools import lru_cache
import json
from cryptography.fernet import Fernet
from loguru import logger


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class APIProvider(str, Enum):
    """Supported API providers."""
    KITE_CONNECT = "kite_connect"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    TWITTER = "twitter"
    GLOBAL_DATAFEEDS = "global_datafeeds"
    EODHD = "eodhd"
    MARKETAUX = "marketaux"
    GROK = "grok"
    NEWSAPI = "newsapi"
    YAHOO_FINANCE = "yahoo_finance"
    NSE_OFFICIAL = "nse_official"


class APITier(str, Enum):
    """API subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class APIConfig(BaseSettings):
    """Configuration for a single API provider."""
    
    # Basic settings
    enabled: bool = True
    priority: int = 5  # 1-10, higher number = higher priority
    tier: APITier = APITier.FREE
    
    # Authentication
    api_key: Optional[SecretStr] = None
    api_secret: Optional[SecretStr] = None
    access_token: Optional[SecretStr] = None
    bearer_token: Optional[SecretStr] = None
    
    # Endpoints
    base_url: str
    websocket_url: Optional[str] = None
    sandbox_url: Optional[str] = None
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: Optional[int] = None
    rate_limit_per_day: Optional[int] = None
    burst_limit: int = 10
    
    # Cost management
    cost_per_request: float = 0.0
    monthly_budget: Optional[float] = None
    free_requests_per_month: int = 0
    
    # Quality settings
    min_quality_score: float = 0.7
    latency_threshold_ms: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Features
    supports_websocket: bool = False
    supports_historical: bool = True
    supports_realtime: bool = True
    supports_news: bool = False
    supports_sentiment: bool = False
    
    # Health check
    health_check_endpoint: Optional[str] = None
    health_check_interval_seconds: int = 300


class APIConfigurations(BaseSettings):
    """Master configuration for all APIs."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    
    # Security
    encryption_key: SecretStr = Field(default_factory=lambda: SecretStr(Fernet.generate_key().decode()))
    api_key_rotation_days: int = 90
    
    # Global settings
    global_timeout_seconds: int = 30
    global_max_retries: int = 3
    enable_cost_monitoring: bool = True
    enable_usage_monitoring: bool = True
    
    # Kite Connect (Primary Indian market data)
    kite_connect: APIConfig = APIConfig(
        base_url="https://api.kite.trade",
        websocket_url="wss://ws.kite.trade",
        rate_limit_per_minute=180,
        priority=10,
        tier=APITier.PREMIUM,
        supports_websocket=True,
        min_quality_score=0.9,
        latency_threshold_ms=50
    )
    
    # Alpha Vantage (Global market data)
    alpha_vantage: APIConfig = APIConfig(
        base_url="https://www.alphavantage.co/query",
        rate_limit_per_minute=5,  # Free tier
        rate_limit_per_day=500,
        priority=7,
        tier=APITier.FREE,
        free_requests_per_month=500,
        cost_per_request=0.0012,  # Premium tier
        monthly_budget=30.0,
        latency_threshold_ms=500
    )
    
    # Finnhub (Real-time market data & news)
    finnhub: APIConfig = APIConfig(
        base_url="https://finnhub.io/api/v1",
        websocket_url="wss://ws.finnhub.io",
        rate_limit_per_minute=60,  # Free tier
        priority=8,
        tier=APITier.FREE,
        supports_websocket=True,
        supports_news=True,
        supports_sentiment=True,
        cost_per_request=0.001,
        monthly_budget=50.0
    )
    
    # Polygon.io (Institutional-grade data)
    polygon: APIConfig = APIConfig(
        base_url="https://api.polygon.io",
        websocket_url="wss://socket.polygon.io",
        rate_limit_per_minute=5,  # Free tier
        priority=9,
        tier=APITier.FREE,
        supports_websocket=True,
        cost_per_request=0.0015,
        monthly_budget=100.0,
        min_quality_score=0.85
    )
    
    # Twitter API v2 (Social sentiment)
    twitter: APIConfig = APIConfig(
        base_url="https://api.twitter.com/2",
        rate_limit_per_minute=300,  # v2 limits
        rate_limit_per_hour=1500,
        priority=5,
        tier=APITier.BASIC,
        supports_sentiment=True,
        cost_per_request=0.0001,
        monthly_budget=100.0
    )
    
    # Global Datafeeds (NSE/BSE data)
    global_datafeeds: APIConfig = APIConfig(
        base_url="https://api.globaldatafeeds.in/v1",
        websocket_url="wss://ws.globaldatafeeds.in",
        rate_limit_per_minute=100,
        priority=9,
        tier=APITier.PREMIUM,
        supports_websocket=True,
        min_quality_score=0.85,
        cost_per_request=0.0008,
        monthly_budget=200.0
    )
    
    # EODHD (End of Day Historical Data)
    eodhd: APIConfig = APIConfig(
        base_url="https://eodhistoricaldata.com/api",
        rate_limit_per_minute=20,  # Free tier
        rate_limit_per_day=1000,
        priority=6,
        tier=APITier.FREE,
        supports_historical=True,
        supports_realtime=False,
        cost_per_request=0.0005,
        monthly_budget=50.0
    )
    
    # Marketaux (Financial news)
    marketaux: APIConfig = APIConfig(
        base_url="https://api.marketaux.com/v1",
        rate_limit_per_minute=10,  # Free tier
        rate_limit_per_day=100,
        priority=6,
        tier=APITier.FREE,
        supports_news=True,
        supports_sentiment=True,
        free_requests_per_month=100,
        cost_per_request=0.01,
        monthly_budget=50.0
    )
    
    # Grok API (xAI - Advanced AI analysis)
    grok: APIConfig = APIConfig(
        base_url="https://api.x.ai/v1",
        rate_limit_per_minute=20,
        priority=7,
        tier=APITier.PREMIUM,
        supports_sentiment=True,
        cost_per_request=0.01,
        monthly_budget=500.0,
        timeout_seconds=60
    )
    
    # NewsAPI (General news)
    newsapi: APIConfig = APIConfig(
        base_url="https://newsapi.org/v2",
        rate_limit_per_minute=100,
        rate_limit_per_day=1000,
        priority=5,
        tier=APITier.FREE,
        supports_news=True,
        free_requests_per_month=1000,
        cost_per_request=0.001,
        monthly_budget=20.0
    )
    
    # Yahoo Finance (Backup free data)
    yahoo_finance: APIConfig = APIConfig(
        base_url="https://query1.finance.yahoo.com",
        rate_limit_per_minute=100,
        priority=3,
        tier=APITier.FREE,
        min_quality_score=0.6,
        supports_historical=True
    )
    
    # NSE Official (Direct NSE data)
    nse_official: APIConfig = APIConfig(
        base_url="https://www.nseindia.com",
        rate_limit_per_minute=30,
        priority=4,
        tier=APITier.FREE,
        min_quality_score=0.7,
        timeout_seconds=10
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Custom env prefix for each API
        env_prefix_map = {
            "kite_connect": "KITE_",
            "alpha_vantage": "ALPHA_VANTAGE_",
            "finnhub": "FINNHUB_",
            "polygon": "POLYGON_",
            "twitter": "TWITTER_",
            "global_datafeeds": "GLOBAL_DATAFEEDS_",
            "eodhd": "EODHD_",
            "marketaux": "MARKETAUX_",
            "grok": "GROK_",
            "newsapi": "NEWSAPI_",
            "yahoo_finance": "YAHOO_",
            "nse_official": "NSE_"
        }
    
    @validator("*", pre=True)
    def load_api_config(cls, v, field):
        """Load API configuration from environment variables."""
        if isinstance(v, APIConfig):
            return v
            
        # Try to load from environment
        if field.name in cls.Config.env_prefix_map:
            prefix = cls.Config.env_prefix_map[field.name]
            config_dict = {}
            
            # Load all environment variables with the prefix
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    # Convert to lowercase and remove prefix
                    config_key = key[len(prefix):].lower()
                    config_dict[config_key] = value
            
            if config_dict:
                return APIConfig(**config_dict)
        
        return v
    
    def get_api_config(self, provider: APIProvider) -> Optional[APIConfig]:
        """Get configuration for a specific API provider."""
        provider_map = {
            APIProvider.KITE_CONNECT: self.kite_connect,
            APIProvider.ALPHA_VANTAGE: self.alpha_vantage,
            APIProvider.FINNHUB: self.finnhub,
            APIProvider.POLYGON: self.polygon,
            APIProvider.TWITTER: self.twitter,
            APIProvider.GLOBAL_DATAFEEDS: self.global_datafeeds,
            APIProvider.EODHD: self.eodhd,
            APIProvider.MARKETAUX: self.marketaux,
            APIProvider.GROK: self.grok,
            APIProvider.NEWSAPI: self.newsapi,
            APIProvider.YAHOO_FINANCE: self.yahoo_finance,
            APIProvider.NSE_OFFICIAL: self.nse_official
        }
        
        return provider_map.get(provider)
    
    def get_enabled_apis(self) -> Dict[APIProvider, APIConfig]:
        """Get all enabled API configurations."""
        enabled = {}
        
        for provider in APIProvider:
            config = self.get_api_config(provider)
            if config and config.enabled:
                enabled[provider] = config
        
        return enabled
    
    def get_apis_by_priority(self) -> List[tuple[APIProvider, APIConfig]]:
        """Get APIs sorted by priority (highest first)."""
        enabled_apis = self.get_enabled_apis()
        
        sorted_apis = sorted(
            enabled_apis.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        return sorted_apis
    
    def get_apis_for_feature(self, feature: str) -> List[tuple[APIProvider, APIConfig]]:
        """Get APIs that support a specific feature."""
        feature_map = {
            'websocket': lambda c: c.supports_websocket,
            'historical': lambda c: c.supports_historical,
            'realtime': lambda c: c.supports_realtime,
            'news': lambda c: c.supports_news,
            'sentiment': lambda c: c.supports_sentiment
        }
        
        if feature not in feature_map:
            return []
        
        check_func = feature_map[feature]
        enabled_apis = self.get_enabled_apis()
        
        supporting_apis = [
            (provider, config)
            for provider, config in enabled_apis.items()
            if check_func(config)
        ]
        
        # Sort by priority
        return sorted(supporting_apis, key=lambda x: x[1].priority, reverse=True)
    
    def estimate_monthly_cost(self) -> Dict[str, float]:
        """Estimate monthly API costs."""
        costs = {}
        total = 0.0
        
        for provider in APIProvider:
            config = self.get_api_config(provider)
            if config and config.enabled and config.cost_per_request > 0:
                # Estimate based on rate limits
                requests_per_month = min(
                    config.rate_limit_per_minute * 60 * 24 * 30,
                    config.rate_limit_per_day * 30 if config.rate_limit_per_day else float('inf')
                )
                
                # Subtract free requests
                billable_requests = max(0, requests_per_month - config.free_requests_per_month)
                estimated_cost = billable_requests * config.cost_per_request
                
                # Cap at monthly budget
                if config.monthly_budget:
                    estimated_cost = min(estimated_cost, config.monthly_budget)
                
                costs[provider.value] = estimated_cost
                total += estimated_cost
        
        costs['total'] = total
        return costs
    
    def to_secure_dict(self) -> Dict[str, Any]:
        """Export configuration with masked sensitive data."""
        config_dict = {}
        
        for provider in APIProvider:
            config = self.get_api_config(provider)
            if config:
                provider_dict = config.dict()
                
                # Mask sensitive fields
                for field in ['api_key', 'api_secret', 'access_token', 'bearer_token']:
                    if field in provider_dict and provider_dict[field]:
                        provider_dict[field] = "***MASKED***"
                
                config_dict[provider.value] = provider_dict
        
        return {
            'environment': self.environment.value,
            'apis': config_dict,
            'estimated_monthly_cost': self.estimate_monthly_cost()
        }


# Environment-specific configurations
class DevelopmentConfig(APIConfigurations):
    """Development environment configuration."""
    environment: Environment = Environment.DEVELOPMENT
    
    # Use sandbox/demo endpoints where available
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override with sandbox URLs
        if self.kite_connect.sandbox_url:
            self.kite_connect.base_url = self.kite_connect.sandbox_url
        
        # Reduce rate limits for testing
        for provider in APIProvider:
            config = self.get_api_config(provider)
            if config:
                config.rate_limit_per_minute = min(config.rate_limit_per_minute, 10)


class ProductionConfig(APIConfigurations):
    """Production environment configuration."""
    environment: Environment = Environment.PRODUCTION
    
    # Stricter quality requirements
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Increase quality thresholds
        for provider in APIProvider:
            config = self.get_api_config(provider)
            if config:
                config.min_quality_score = max(config.min_quality_score, 0.8)
                config.timeout_seconds = min(config.timeout_seconds, 10)


# Singleton pattern for configuration
_config_instance: Optional[APIConfigurations] = None

@lru_cache()
def get_api_config() -> APIConfigurations:
    """Get the API configuration instance."""
    global _config_instance
    
    if _config_instance is None:
        env = os.getenv("ENVIRONMENT", "development").lower()
        
        if env == "production":
            _config_instance = ProductionConfig()
        elif env == "staging":
            _config_instance = APIConfigurations()
        else:
            _config_instance = DevelopmentConfig()
        
        logger.info(f"Loaded API configuration for environment: {env}")
    
    return _config_instance