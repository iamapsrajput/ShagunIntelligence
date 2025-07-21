"""
Base classes for all data sources in the multi-source data management system.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import asyncio
import logging
from dataclasses import dataclass, field
import time


class DataSourceStatus(Enum):
    """Status of a data source."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"


class DataSourceType(Enum):
    """Type of data source."""
    MARKET_DATA = "market_data"
    SENTIMENT = "sentiment"
    NEWS = "news"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


@dataclass
class DataSourceHealth:
    """Health information for a data source."""
    status: DataSourceStatus
    last_check: datetime
    last_success: Optional[datetime] = None
    error_count: int = 0
    recent_errors: List[str] = field(default_factory=list)
    latency_ms: Optional[float] = None
    success_rate: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    priority: int = 0  # Lower number = higher priority
    enabled: bool = True
    health_check_interval: int = 30  # seconds
    timeout: int = 10  # seconds
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    rate_limit: Optional[int] = None  # requests per minute
    connection_pool_size: int = 10
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)


class BaseDataSource(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.health = DataSourceHealth(
            status=DataSourceStatus.DISCONNECTED,
            last_check=datetime.now()
        )
        self._rate_limiter = None
        self._connection_pool = []
        self._is_connected = False
        self._health_check_task = None
        self._performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency': 0.0
        }
        
    @property
    @abstractmethod
    def source_type(self) -> DataSourceType:
        """Return the type of this data source."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    async def health_check(self) -> DataSourceHealth:
        """Check the health of the data source."""
        pass
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate API credentials."""
        pass
    
    async def start_health_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self._health_check_task:
            return
            
        async def monitor():
            while self._is_connected:
                try:
                    await self.health_check()
                except Exception as e:
                    self.logger.error(f"Health check failed: {e}")
                await asyncio.sleep(self.config.health_check_interval)
        
        self._health_check_task = asyncio.create_task(monitor())
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
    
    def update_health_status(self, status: DataSourceStatus, error: Optional[str] = None) -> None:
        """Update the health status of the data source."""
        self.health.status = status
        self.health.last_check = datetime.now()
        
        if status == DataSourceStatus.HEALTHY:
            self.health.last_success = datetime.now()
            self.health.error_count = 0
            self.health.recent_errors.clear()
        elif error:
            self.health.error_count += 1
            self.health.recent_errors.append(f"{datetime.now()}: {error}")
            # Keep only last 10 errors
            self.health.recent_errors = self.health.recent_errors[-10:]
        
        # Calculate success rate
        if self._performance_metrics['total_requests'] > 0:
            self.health.success_rate = (
                self._performance_metrics['successful_requests'] / 
                self._performance_metrics['total_requests'] * 100
            )
    
    def record_request_metrics(self, success: bool, latency_ms: float) -> None:
        """Record request performance metrics."""
        self._performance_metrics['total_requests'] += 1
        self._performance_metrics['total_latency'] += latency_ms
        
        if success:
            self._performance_metrics['successful_requests'] += 1
        else:
            self._performance_metrics['failed_requests'] += 1
        
        # Update average latency
        self.health.latency_ms = (
            self._performance_metrics['total_latency'] / 
            self._performance_metrics['total_requests']
        )
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                latency = (time.time() - start_time) * 1000
                
                self.record_request_metrics(True, latency)
                return result
                
            except Exception as e:
                latency = (time.time() - start_time) * 1000
                self.record_request_metrics(False, latency)
                last_error = e
                
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    self.logger.error(f"Request failed after {self.config.max_retries} attempts: {e}")
        
        raise last_error


class MarketDataSource(BaseDataSource):
    """Base class for market data sources."""
    
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.MARKET_DATA
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol."""
        pass
    
    @abstractmethod
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: datetime,
        to_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get historical data for a symbol."""
        pass
    
    @abstractmethod
    async def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """Get market depth (order book) for a symbol."""
        pass
    
    @abstractmethod
    async def subscribe_live_data(
        self,
        symbols: List[str],
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Subscribe to live market data."""
        pass
    
    @abstractmethod
    async def unsubscribe_live_data(self, symbols: List[str]) -> None:
        """Unsubscribe from live market data."""
        pass


class SentimentDataSource(BaseDataSource):
    """Base class for sentiment data sources."""
    
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.SENTIMENT
    
    @abstractmethod
    async def get_sentiment_score(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment score for a symbol."""
        pass
    
    @abstractmethod
    async def get_news_sentiment(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get news sentiment for a symbol."""
        pass
    
    @abstractmethod
    async def get_social_sentiment(
        self,
        symbol: str,
        platforms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get social media sentiment for a symbol."""
        pass
    
    @abstractmethod
    async def get_trending_topics(
        self,
        sector: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trending topics in the market."""
        pass