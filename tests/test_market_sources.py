from unittest.mock import AsyncMock, patch

import pytest

from backend.data_sources.base import DataSourceConfig, DataSourceStatus, DataSourceType
from backend.data_sources.market.alpha_vantage_market import AlphaVantageMarketSource
from backend.data_sources.market.finnhub import FinnhubSource
from backend.data_sources.market.global_datafeeds import GlobalDatafeedsSource
from backend.data_sources.market.market_source_manager import (
    MarketSourceManager,
    SourceSelectionStrategy,
)
from backend.data_sources.market.models import (
    DataCostTier,
    DataSourceCost,
    MarketDataQuality,
)
from backend.data_sources.market.polygon import PolygonSource


class TestMarketDataSources:
    """Test all market data sources"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return DataSourceConfig(
            name="test_source",
            source_type=DataSourceType.MARKET_DATA,
            enabled=True,
            credentials={"api_key": "test_key", "user_id": "test_user"},
        )

    @pytest.mark.asyncio
    async def test_global_datafeeds_source(self, config):
        """Test GlobalDatafeeds source"""
        source = GlobalDatafeedsSource(config)

        # Test cost info
        cost_info = source.get_cost_info()
        assert cost_info.tier == DataCostTier.MEDIUM
        assert cost_info.monthly_cost == 299.0
        assert cost_info.includes_realtime is True

        # Test market support
        assert source.supports_market("NSE") is True
        assert source.supports_market("BSE") is True
        assert source.supports_market("NYSE") is False

        # Mock API response
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "success": True,
                "data": {
                    "ltp": 1850.50,
                    "open": 1840.0,
                    "high": 1855.0,
                    "low": 1835.0,
                    "close": 1845.0,
                    "prev_close": 1842.0,
                    "volume": 1000000,
                    "bid": 1850.25,
                    "ask": 1850.75,
                    "bid_qty": 100,
                    "ask_qty": 150,
                    "change": 8.50,
                    "change_percent": 0.46,
                },
            }

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = (
                mock_response
            )

            # Test connection
            source.session = mock_session.return_value.__aenter__.return_value
            connected = await source.connect()
            assert connected is True

            # Test quote fetching
            quote = await source.get_quote("RELIANCE.NSE")
            assert quote["symbol"] == "RELIANCE.NSE"
            assert quote["current_price"] == 1850.50
            assert quote["data_quality"] == MarketDataQuality.PROFESSIONAL.value
            assert quote["latency_ms"] == 50

    @pytest.mark.asyncio
    async def test_alpha_vantage_source(self, config):
        """Test Alpha Vantage source"""
        source = AlphaVantageMarketSource(config)

        # Test cost info
        cost_info = source.get_cost_info()
        assert cost_info.tier == DataCostTier.FREE
        assert cost_info.monthly_cost == 0.0
        assert cost_info.free_requests == 500
        assert cost_info.requests_per_minute == 5

        # Test market support
        assert source.supports_market("US") is True
        assert source.supports_market("FOREX") is True
        assert source.supports_market("CRYPTO") is True

        # Test premium tiers
        premium_tier = source.upgrade_tier("premium")
        assert premium_tier.monthly_cost == 49.99
        assert premium_tier.requests_per_minute == 75

    @pytest.mark.asyncio
    async def test_finnhub_source(self, config):
        """Test Finnhub source"""
        source = FinnhubSource(config)

        # Test unique features
        assert source.supports_sentiment() is True

        # Mock sentiment response
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "sentiment": {"score": 0.65},
                "buzz": {"buzz": 0.8, "articlesInLastWeek": 120},
                "companyNewsScore": 0.75,
            }

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )
            source.session = mock_session.return_value.__aenter__.return_value

            # Test sentiment analysis
            sentiment = await source.get_sentiment("AAPL")
            assert sentiment["score"] == 0.65
            assert sentiment["buzz"] == 0.8
            assert sentiment["articles_in_last_week"] == 120

    @pytest.mark.asyncio
    async def test_polygon_source(self, config):
        """Test Polygon source"""
        source = PolygonSource(config)

        # Test institutional features
        assert source.supports_institutional_features() is True

        # Test cost tiers
        cost_info = source.get_cost_info()
        assert cost_info.tier == DataCostTier.FREE

        advanced_tier = source.upgrade_tier("stocks_advanced")
        assert advanced_tier.tier == DataCostTier.PREMIUM
        assert advanced_tier.monthly_cost == 1999.0
        assert advanced_tier.requests_per_minute == 10000


class TestMarketSourceManager:
    """Test market source manager"""

    @pytest.fixture
    def manager(self):
        """Create test manager"""
        return MarketSourceManager(SourceSelectionStrategy.BALANCED)

    @pytest.mark.asyncio
    async def test_source_selection_strategies(self, manager):
        """Test different source selection strategies"""
        # Test cost optimized strategy
        manager.set_strategy(SourceSelectionStrategy.COST_OPTIMIZED)
        sources = manager._select_sources_for_request("US", "quote")
        # Alpha Vantage (free) should be prioritized
        assert "alpha_vantage" in sources

        # Test quality first strategy
        manager.set_strategy(SourceSelectionStrategy.QUALITY_FIRST)
        sources = manager._select_sources_for_request("US", "quote")
        # Polygon (highest quality) should be prioritized if available

        # Test latency optimized strategy
        manager.set_strategy(SourceSelectionStrategy.LATENCY_OPTIMIZED)
        sources = manager._select_sources_for_request("US", "quote")
        # Sources with lowest latency should be prioritized

    def test_market_detection(self, manager):
        """Test market detection from symbols"""
        assert manager._detect_market("RELIANCE.NSE") == "NSE"
        assert manager._detect_market("TCS.BSE") == "BSE"
        assert manager._detect_market("AAPL") == "US"
        assert manager._detect_market("NYSE:IBM") == "NYSE"
        assert manager._detect_market("EURUSD") == "US"

    def test_symbol_grouping(self, manager):
        """Test symbol grouping by market"""
        symbols = ["RELIANCE.NSE", "TCS.NSE", "HDFC.BSE", "AAPL", "GOOGL", "NYSE:IBM"]

        groups = manager._group_symbols_by_market(symbols)

        assert len(groups["NSE"]) == 2
        assert len(groups["BSE"]) == 1
        assert len(groups["US"]) == 2
        assert len(groups["NYSE"]) == 1

    @pytest.mark.asyncio
    async def test_failover_mechanism(self, manager):
        """Test failover between sources"""
        # Mock sources
        mock_source1 = AsyncMock()
        mock_source1.get_quote.side_effect = Exception("Source 1 failed")
        mock_source1.get_cost_info.return_value = DataSourceCost(
            tier=DataCostTier.FREE, monthly_cost=0.0
        )

        mock_source2 = AsyncMock()
        mock_source2.get_quote.return_value = {"symbol": "AAPL", "current_price": 150.0}
        mock_source2.get_cost_info.return_value = DataSourceCost(
            tier=DataCostTier.LOW, monthly_cost=49.99
        )

        manager.sources = {"source1": mock_source1, "source2": mock_source2}
        manager.source_health = {
            "source1": DataSourceStatus.HEALTHY,
            "source2": DataSourceStatus.HEALTHY,
        }
        manager.market_coverage = {"source1": ["US"], "source2": ["US"]}

        # Test failover
        quote = await manager.get_quote("AAPL")
        assert quote is not None
        assert quote["current_price"] == 150.0
        assert quote["data_source"] == "source2"

        # Check failure tracking
        assert manager.failure_counts["source1"] == 1

    def test_cost_tracking(self, manager):
        """Test cost tracking and optimization"""
        # Track requests
        manager._track_request_success("polygon", 100)
        manager._track_request_success("alpha_vantage", 50)

        # Check request counts
        assert manager.request_counts["polygon"] == 100
        assert manager.request_counts["alpha_vantage"] == 50

        # Get status
        status = manager.get_source_status()
        assert status["total_requests"] == 150


@pytest.mark.asyncio
class TestIntegration:
    """Test integration scenarios"""

    async def test_multi_market_quote_fetching(self):
        """Test fetching quotes from multiple markets"""
        manager = MarketSourceManager(SourceSelectionStrategy.BALANCED)

        # Mock sources for different markets
        indian_source = AsyncMock()
        indian_source.get_quotes.return_value = {
            "RELIANCE.NSE": {"current_price": 1850.0},
            "TCS.NSE": {"current_price": 3200.0},
        }
        indian_source.get_cost_info.return_value = DataSourceCost(
            tier=DataCostTier.MEDIUM, monthly_cost=299.0
        )

        us_source = AsyncMock()
        us_source.get_quotes.return_value = {
            "AAPL": {"current_price": 150.0},
            "GOOGL": {"current_price": 2800.0},
        }
        us_source.get_cost_info.return_value = DataSourceCost(
            tier=DataCostTier.LOW, monthly_cost=49.99
        )

        manager.sources = {"global_datafeeds": indian_source, "polygon": us_source}
        manager.source_health = {
            "global_datafeeds": DataSourceStatus.HEALTHY,
            "polygon": DataSourceStatus.HEALTHY,
        }
        manager.market_coverage = {
            "global_datafeeds": ["NSE", "BSE"],
            "polygon": ["US", "NYSE", "NASDAQ"],
        }

        # Test multi-market quote fetching
        symbols = ["RELIANCE.NSE", "TCS.NSE", "AAPL", "GOOGL"]
        quotes = await manager.get_quotes(symbols)

        assert len(quotes) == 4
        assert quotes["RELIANCE.NSE"]["current_price"] == 1850.0
        assert quotes["AAPL"]["current_price"] == 150.0

    async def test_sentiment_enhanced_quotes(self):
        """Test quotes enhanced with sentiment data"""
        manager = MarketSourceManager()

        # Mock Finnhub source with sentiment
        finnhub_source = AsyncMock()
        finnhub_source.get_quote.return_value = {
            "symbol": "AAPL",
            "current_price": 150.0,
            "sentiment_score": 0.65,
            "sentiment_buzz": 0.8,
        }
        finnhub_source.get_sentiment.return_value = {"score": 0.65, "buzz": 0.8}
        finnhub_source.supports_sentiment.return_value = True
        finnhub_source.get_cost_info.return_value = DataSourceCost(
            tier=DataCostTier.FREE, monthly_cost=0.0
        )

        manager.sources = {"finnhub": finnhub_source}
        manager.source_health = {"finnhub": DataSourceStatus.HEALTHY}
        manager.market_coverage = {"finnhub": ["US"]}
        manager.feature_support = {"finnhub": {"sentiment": True}}

        # Get sentiment
        sentiment = await manager.get_sentiment("AAPL")
        assert sentiment["score"] == 0.65
        assert sentiment["data_source"] == "finnhub"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
