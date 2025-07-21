"""Market data sources module"""

from .models import (
    MarketData,
    MarketDepth,
    HistoricalBar,
    MarketDataQuality,
    DataCostTier,
    DataSourceCost
)

from .global_datafeeds import GlobalDatafeedsSource
from .alpha_vantage_market import AlphaVantageMarketSource
from .finnhub import FinnhubSource
from .polygon import PolygonSource
from .market_source_manager import MarketSourceManager, SourceSelectionStrategy

__all__ = [
    "MarketData",
    "MarketDepth",
    "HistoricalBar",
    "MarketDataQuality",
    "DataCostTier",
    "DataSourceCost",
    "GlobalDatafeedsSource",
    "AlphaVantageMarketSource",
    "FinnhubSource",
    "PolygonSource",
    "MarketSourceManager",
    "SourceSelectionStrategy"
]