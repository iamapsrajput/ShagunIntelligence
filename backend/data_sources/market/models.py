from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class MarketDataQuality(Enum):
    """Quality level of market data"""
    INSTITUTIONAL = "institutional"  # Highest quality, real-time
    PROFESSIONAL = "professional"    # High quality, minimal delay
    STANDARD = "standard"           # Good quality, some delay
    BASIC = "basic"                # Basic quality, delayed


class DataCostTier(Enum):
    """Cost tier for data sources"""
    FREE = "free"
    LOW = "low"          # < $50/month
    MEDIUM = "medium"    # $50-$500/month
    HIGH = "high"        # $500-$2000/month
    PREMIUM = "premium"  # > $2000/month


@dataclass
class MarketData:
    """Standardized market data across all sources"""
    
    # Basic identifiers
    symbol: str
    exchange: str
    timestamp: datetime
    source: str
    
    # Price data
    current_price: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    previous_close: Optional[float] = None
    
    # Volume data
    volume: Optional[int] = None
    average_volume: Optional[int] = None
    day_volume: Optional[int] = None
    
    # Bid/Ask
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    
    # Change metrics
    change: Optional[float] = None
    change_percent: Optional[float] = None
    
    # Additional data
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    
    # Quality metrics
    data_quality: MarketDataQuality = MarketDataQuality.STANDARD
    latency_ms: Optional[int] = None
    is_delayed: bool = False
    delay_minutes: int = 0
    
    # Additional metrics
    vwap: Optional[float] = None
    sentiment_score: Optional[float] = None
    sentiment_buzz: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "current_price": self.current_price,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "previous_close": self.previous_close,
            "volume": self.volume,
            "average_volume": self.average_volume,
            "day_volume": self.day_volume,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "change": self.change,
            "change_percent": self.change_percent,
            "market_cap": self.market_cap,
            "pe_ratio": self.pe_ratio,
            "week_52_high": self.week_52_high,
            "week_52_low": self.week_52_low,
            "data_quality": self.data_quality.value,
            "latency_ms": self.latency_ms,
            "is_delayed": self.is_delayed,
            "delay_minutes": self.delay_minutes,
            "vwap": self.vwap,
            "sentiment_score": self.sentiment_score,
            "sentiment_buzz": self.sentiment_buzz
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from dictionary"""
        # Handle timestamp
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.utcnow()
        
        # Handle quality
        quality_str = data.get("data_quality", "standard")
        try:
            quality = MarketDataQuality(quality_str)
        except ValueError:
            quality = MarketDataQuality.STANDARD
        
        return cls(
            symbol=data.get("symbol", ""),
            exchange=data.get("exchange", ""),
            timestamp=timestamp,
            source=data.get("source", ""),
            current_price=float(data.get("current_price", 0)),
            open=data.get("open"),
            high=data.get("high"),
            low=data.get("low"),
            close=data.get("close"),
            previous_close=data.get("previous_close"),
            volume=data.get("volume"),
            average_volume=data.get("average_volume"),
            day_volume=data.get("day_volume"),
            bid=data.get("bid"),
            ask=data.get("ask"),
            bid_size=data.get("bid_size"),
            ask_size=data.get("ask_size"),
            change=data.get("change"),
            change_percent=data.get("change_percent"),
            market_cap=data.get("market_cap"),
            pe_ratio=data.get("pe_ratio"),
            week_52_high=data.get("week_52_high"),
            week_52_low=data.get("week_52_low"),
            data_quality=quality,
            latency_ms=data.get("latency_ms"),
            is_delayed=data.get("is_delayed", False),
            delay_minutes=data.get("delay_minutes", 0),
            vwap=data.get("vwap"),
            sentiment_score=data.get("sentiment_score"),
            sentiment_buzz=data.get("sentiment_buzz")
        )


@dataclass
class MarketDepth:
    """Market depth/order book data"""
    
    symbol: str
    timestamp: datetime
    source: str
    
    bids: List[Dict[str, float]] = field(default_factory=list)  # [{"price": x, "size": y}, ...]
    asks: List[Dict[str, float]] = field(default_factory=list)
    
    total_bid_volume: Optional[int] = None
    total_ask_volume: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "bids": self.bids,
            "asks": self.asks,
            "total_bid_volume": self.total_bid_volume,
            "total_ask_volume": self.total_ask_volume
        }


@dataclass
class HistoricalBar:
    """Historical price bar data"""
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Optional fields
    vwap: Optional[float] = None
    trades: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trades": self.trades
        }


@dataclass
class DataSourceCost:
    """Cost information for a data source"""
    
    tier: DataCostTier
    monthly_cost: float
    per_request_cost: float = 0.0
    free_requests: int = 0
    
    # Usage limits
    requests_per_minute: Optional[int] = None
    requests_per_day: Optional[int] = None
    requests_per_month: Optional[int] = None
    
    # Data coverage
    includes_realtime: bool = True
    includes_historical: bool = True
    includes_options: bool = False
    includes_forex: bool = False
    includes_crypto: bool = False
    
    def get_request_cost(self, requests: int) -> float:
        """Calculate cost for number of requests"""
        if requests <= self.free_requests:
            return 0.0
        
        billable_requests = requests - self.free_requests
        return billable_requests * self.per_request_cost