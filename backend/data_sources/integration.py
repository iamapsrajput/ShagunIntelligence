"""
Integration module to connect the multi-source data manager with existing Shagun Intelligence components.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from .base import DataSourceType
from .factory import get_data_manager
from .market.market_source_manager import MarketSourceManager, SourceSelectionStrategy
from .news.news_aggregator import NewsAggregator
from .sentiment.grok_api import GrokSentimentSource
from .sentiment.sentiment_fusion import MultiSourceSentimentFusion
from .sentiment.twitter_api import TwitterSentimentSource

logger = logging.getLogger(__name__)


class DataSourceIntegration:
    """
    Integration layer between multi-source data manager and Shagun Intelligence components.
    Provides backward compatibility with existing code.
    """

    def __init__(self):
        self.manager = get_data_manager()
        self.sentiment_fusion = MultiSourceSentimentFusion()
        self._twitter_source = None
        self._grok_source = None
        self.news_aggregator = NewsAggregator()
        self.market_source_manager = MarketSourceManager(
            SourceSelectionStrategy.BALANCED
        )

    async def initialize(self) -> None:
        """Initialize the data manager."""
        await self.manager.start()

        # Initialize Twitter sentiment if configured
        await self._initialize_twitter_sentiment()

        # Initialize Grok sentiment if configured
        await self._initialize_grok_sentiment()

        # Initialize news sources
        await self.news_aggregator.initialize_sources()

        # Initialize market source manager
        await self._initialize_market_sources()

        logger.info("Data source integration initialized")

    async def shutdown(self) -> None:
        """Shutdown the data manager."""
        await self.manager.stop()

        # Shutdown Twitter sentiment
        if self._twitter_source:
            await self._twitter_source.disconnect()

        # Shutdown Grok sentiment
        if self._grok_source:
            await self._grok_source.disconnect()

        # Shutdown news aggregator
        await self.news_aggregator.shutdown()

        # Shutdown market source manager
        await self.market_source_manager.close()

        logger.info("Data source integration shutdown")

    async def _initialize_twitter_sentiment(self) -> None:
        """Initialize Twitter sentiment source if configured."""
        try:
            # Check if Twitter credentials are available
            import os

            bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

            if bearer_token:
                from .base import DataSourceConfig

                config = DataSourceConfig(
                    name="twitter_sentiment",
                    source_type=DataSourceType.SENTIMENT,
                    enabled=True,
                    credentials={
                        "bearer_token": bearer_token,
                        "api_key": os.getenv("TWITTER_API_KEY"),
                        "api_secret": os.getenv("TWITTER_API_SECRET"),
                    },
                )

                self._twitter_source = TwitterSentimentSource(config)
                success = await self._twitter_source.connect()

                if success:
                    # Add to sentiment fusion
                    self.sentiment_fusion.add_source(
                        "twitter", self._twitter_source, weight=1.0
                    )

                    # Also add to main manager as sentiment source
                    self.manager.add_source(self._twitter_source)

                    logger.info("Twitter sentiment source initialized successfully")
                else:
                    logger.warning("Failed to connect to Twitter sentiment source")

        except Exception as e:
            logger.error(f"Error initializing Twitter sentiment: {e}")

    async def _initialize_grok_sentiment(self) -> None:
        """Initialize Grok sentiment source if configured."""
        try:
            # Check if Grok credentials are available
            import os

            grok_api_key = os.getenv("GROK_API_KEY")

            if grok_api_key:
                from .base import DataSourceConfig

                config = DataSourceConfig(
                    name="grok_sentiment",
                    source_type=DataSourceType.SENTIMENT,
                    enabled=True,
                    credentials={
                        "api_key": grok_api_key,
                        "api_url": os.getenv(
                            "GROK_API_URL", "https://api.x.ai/v1/chat/completions"
                        ),
                        "daily_budget": float(os.getenv("GROK_DAILY_BUDGET", "100.0")),
                    },
                    priority=1,  # Higher priority due to advanced capabilities
                )

                self._grok_source = GrokSentimentSource(config)
                success = await self._grok_source.connect()

                if success:
                    # Add to sentiment fusion with higher weight due to advanced AI
                    self.sentiment_fusion.add_source(
                        "grok", self._grok_source, weight=1.5
                    )

                    # Also add to main manager as sentiment source
                    self.manager.add_source(self._grok_source)

                    logger.info("Grok sentiment source initialized successfully")
                else:
                    logger.warning("Failed to connect to Grok sentiment source")

        except Exception as e:
            logger.error(f"Error initializing Grok sentiment: {e}")

    async def _initialize_market_sources(self) -> None:
        """Initialize market data sources."""
        try:
            import os

            from .base import DataSourceConfig

            configs = {}

            # Global Datafeeds (NSE/BSE)
            if os.getenv("GLOBAL_DATAFEEDS_API_KEY"):
                configs["global_datafeeds"] = DataSourceConfig(
                    name="global_datafeeds",
                    source_type=DataSourceType.MARKET_DATA,
                    enabled=True,
                    credentials={
                        "api_key": os.getenv("GLOBAL_DATAFEEDS_API_KEY"),
                        "user_id": os.getenv("GLOBAL_DATAFEEDS_USER_ID"),
                    },
                    priority=1,  # High priority for Indian markets
                )

            # Alpha Vantage
            if os.getenv("ALPHA_VANTAGE_API_KEY"):
                configs["alpha_vantage"] = DataSourceConfig(
                    name="alpha_vantage",
                    source_type=DataSourceType.MARKET_DATA,
                    enabled=True,
                    credentials={"api_key": os.getenv("ALPHA_VANTAGE_API_KEY")},
                    priority=3,  # Lower priority due to rate limits
                )

            # Finnhub
            if os.getenv("FINNHUB_API_KEY"):
                configs["finnhub"] = DataSourceConfig(
                    name="finnhub",
                    source_type=DataSourceType.MARKET_DATA,
                    enabled=True,
                    credentials={"api_key": os.getenv("FINNHUB_API_KEY")},
                    priority=2,  # Good balance of features
                )

            # Polygon
            if os.getenv("POLYGON_API_KEY"):
                configs["polygon"] = DataSourceConfig(
                    name="polygon",
                    source_type=DataSourceType.MARKET_DATA,
                    enabled=True,
                    credentials={"api_key": os.getenv("POLYGON_API_KEY")},
                    priority=1,  # High priority for US markets
                )

            if configs:
                await self.market_source_manager.initialize(configs)
                logger.info(f"Initialized {len(configs)} market data sources")
            else:
                logger.warning("No market data sources configured")

        except Exception as e:
            logger.error(f"Error initializing market sources: {e}")

    # Market Data Methods (compatible with existing Zerodha interface)

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get quote for a symbol with automatic failover."""
        try:
            # Try market source manager first (supports more sources)
            if self.market_source_manager.sources:
                quote = await self.market_source_manager.get_quote(symbol)
                if quote:
                    return quote

            # Fallback to original manager
            return await self.manager.get_quote(symbol)
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get quotes for multiple symbols with automatic failover."""
        try:
            # Try market source manager first
            if self.market_source_manager.sources:
                quotes = await self.market_source_manager.get_quotes(symbols)
                if quotes:
                    return quotes

            # Fallback to original manager
            return await self.manager.get_quotes(symbols)
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            raise

    async def get_ohlc(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get OHLC data - backward compatibility method."""
        quotes = await self.get_quotes(symbols)

        # Extract OHLC from quotes
        ohlc_data = {}
        for symbol, quote in quotes.items():
            ohlc_data[symbol] = {
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "last_price": quote.get("last_price"),
            }

        return ohlc_data

    async def get_historical_data(
        self, symbol: str, interval: str, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """Get historical data with automatic failover."""
        try:
            # Try market source manager first
            if self.market_source_manager.sources:
                data = await self.market_source_manager.get_historical_data(
                    symbol, interval, from_date, to_date
                )
                if data:
                    return data

            # Fallback to original manager
            return await self.manager.get_historical_data(
                symbol, interval, from_date, to_date
            )
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise

    async def get_market_depth(self, symbol: str) -> dict[str, Any]:
        """Get market depth with automatic failover."""
        try:
            # Try market source manager first
            if self.market_source_manager.sources:
                # Global Datafeeds and Polygon support market depth
                sources = ["global_datafeeds", "polygon"]
                for source_name in sources:
                    if source_name in self.market_source_manager.sources:
                        try:
                            source = self.market_source_manager.sources[source_name]
                            depth = await source.get_market_depth(symbol)
                            if depth and (depth.get("bids") or depth.get("asks")):
                                return depth
                        except:
                            continue

            # Fallback to original manager
            return await self.manager.get_market_depth(symbol)
        except Exception as e:
            logger.error(f"Failed to get market depth for {symbol}: {e}")
            # Return empty depth if not available
            return {"bids": [], "asks": []}

    # Sentiment Data Methods

    async def get_sentiment_score(self, symbol: str) -> dict[str, Any]:
        """Get sentiment score using fusion from all available sources."""
        try:
            # Use sentiment fusion if sources are available
            if self.sentiment_fusion.sources:
                result = await self.sentiment_fusion.get_fused_sentiment(symbol)
                return {
                    "symbol": symbol,
                    "sentiment": result["fused_sentiment_score"],
                    "sentiment_label": result["fused_sentiment_label"],
                    "confidence": result["confidence"],
                    "agreement_score": result["agreement_score"],
                    "source_count": result["source_count"],
                    "sources": result["sources"],
                    "timestamp": result["timestamp"],
                }
            else:
                # Try direct from manager as fallback
                source = await self.manager.get_healthy_source(DataSourceType.SENTIMENT)
                if source:
                    return await self.manager.get_sentiment_score(symbol)
                else:
                    # Return neutral sentiment if no source available
                    return {
                        "symbol": symbol,
                        "sentiment": 0.0,
                        "sentiment_label": "neutral",
                        "confidence": 0.0,
                        "source": "none",
                    }
        except Exception as e:
            logger.error(f"Failed to get sentiment for {symbol}: {e}")
            return {
                "symbol": symbol,
                "sentiment": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "source": "error",
            }

    async def get_sentiment_trends(
        self, symbol: str, hours: int = 24
    ) -> dict[str, Any]:
        """Get sentiment trends over time."""
        try:
            if self.sentiment_fusion.sources:
                return await self.sentiment_fusion.get_sentiment_trends(symbol, hours)
            else:
                return {
                    "symbol": symbol,
                    "trends": [],
                    "period_hours": hours,
                    "sources": [],
                }
        except Exception as e:
            logger.error(f"Failed to get sentiment trends for {symbol}: {e}")
            return {
                "symbol": symbol,
                "trends": [],
                "period_hours": hours,
                "error": str(e),
            }

    async def get_sentiment_alerts(
        self, symbol: str, threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        """Get real-time sentiment alerts."""
        try:
            if self.sentiment_fusion.sources:
                return await self.sentiment_fusion.get_real_time_alerts(
                    symbol, threshold
                )
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get sentiment alerts for {symbol}: {e}")
            return []

    def get_sentiment_sources_stats(self) -> dict[str, Any]:
        """Get statistics about sentiment sources."""
        return self.sentiment_fusion.get_source_statistics()

    def track_symbols_for_sentiment(self, symbols: list[str]) -> None:
        """Update symbols to track for sentiment analysis."""
        if self._twitter_source:
            self._twitter_source.track_symbols(symbols)

    async def get_grok_analysis(self, symbol: str) -> dict[str, Any]:
        """Get detailed Grok AI analysis for a symbol."""
        if not self._grok_source:
            return {"error": "Grok sentiment source not initialized"}

        try:
            response = await self._grok_source.analyze_sentiment(symbol)
            if response:
                return response.to_dict()
            else:
                return {"error": "Failed to get Grok analysis"}
        except Exception as e:
            logger.error(f"Error getting Grok analysis for {symbol}: {e}")
            return {"error": str(e)}

    async def get_batch_grok_analysis(self, symbols: list[str]) -> dict[str, Any]:
        """Get Grok analysis for multiple symbols."""
        if not self._grok_source:
            return {"error": "Grok sentiment source not initialized"}

        try:
            responses = await self._grok_source.analyze_batch(symbols)
            return {
                symbol: response.to_dict() for symbol, response in responses.items()
            }
        except Exception as e:
            logger.error(f"Error getting batch Grok analysis: {e}")
            return {"error": str(e)}

    def get_grok_cost_stats(self) -> dict[str, Any]:
        """Get Grok API usage and cost statistics."""
        if not self._grok_source:
            return {"error": "Grok sentiment source not initialized"}

        return self._grok_source.get_cost_stats()

    # News Methods

    async def get_news(
        self,
        symbols: list[str] | None = None,
        categories: list[str] | None = None,
        hours: int = 24,
        min_relevance: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Get aggregated news from all sources."""
        from .news.base import NewsCategory

        # Convert string categories to enums
        category_enums = None
        if categories:
            category_enums = []
            for cat in categories:
                try:
                    category_enums.append(NewsCategory(cat))
                except ValueError:
                    logger.warning(f"Invalid news category: {cat}")

        # Set date range
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(hours=hours)

        # Fetch news
        articles = await self.news_aggregator.fetch_aggregated_news(
            symbols=symbols,
            from_date=from_date,
            to_date=to_date,
            categories=category_enums,
            min_relevance=min_relevance,
        )

        # Convert to dict format
        return [article.to_dict() for article in articles]

    async def get_breaking_news(
        self, symbols: list[str] | None = None, minutes: int = 15
    ) -> list[dict[str, Any]]:
        """Get breaking news from all sources."""
        articles = await self.news_aggregator.fetch_breaking_news(
            symbols=symbols, minutes=minutes
        )

        return [article.to_dict() for article in articles]

    async def monitor_news_for_symbols(self, symbols: list[str]) -> None:
        """Start monitoring news for specific symbols."""
        await self.news_aggregator.monitor_symbols(symbols)

    def register_news_alert_callback(self, callback: Callable) -> None:
        """Register callback for news alerts."""
        self.news_aggregator.register_alert_callback(callback)

    def get_news_sources_stats(self) -> dict[str, Any]:
        """Get statistics about news sources."""
        return self.news_aggregator.get_source_stats()

    async def get_news_sentiment_summary(
        self, symbol: str, hours: int = 24
    ) -> dict[str, Any]:
        """Get news sentiment summary for a symbol."""
        # Get recent news
        articles = await self.news_aggregator.fetch_aggregated_news(
            symbols=[symbol], from_date=datetime.utcnow() - timedelta(hours=hours)
        )

        if not articles:
            return {
                "symbol": symbol,
                "article_count": 0,
                "average_sentiment": 0.0,
                "sentiment_distribution": {},
                "top_categories": [],
                "market_impact": "minimal",
            }

        # Calculate sentiment statistics
        sentiment_scores = [article.sentiment_score for article in articles]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

        # Sentiment distribution
        from .news.base import NewsSentiment

        sentiment_dist = {sentiment.value: 0 for sentiment in NewsSentiment}
        for article in articles:
            sentiment_dist[article.sentiment.value] += 1

        # Top categories
        category_counts = defaultdict(int)
        for article in articles:
            for category in article.categories:
                category_counts[category.value] += 1

        top_categories = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Overall market impact
        from .news.base import MarketImpact

        impact_scores = {
            MarketImpact.HIGH: 4,
            MarketImpact.MEDIUM: 3,
            MarketImpact.LOW: 2,
            MarketImpact.MINIMAL: 1,
        }

        avg_impact = sum(
            impact_scores.get(article.market_impact, 1) for article in articles
        ) / len(articles)

        if avg_impact >= 3.5:
            overall_impact = "high"
        elif avg_impact >= 2.5:
            overall_impact = "medium"
        elif avg_impact >= 1.5:
            overall_impact = "low"
        else:
            overall_impact = "minimal"

        return {
            "symbol": symbol,
            "article_count": len(articles),
            "average_sentiment": avg_sentiment,
            "sentiment_distribution": sentiment_dist,
            "top_categories": [
                {"category": cat, "count": count} for cat, count in top_categories
            ],
            "market_impact": overall_impact,
            "period_hours": hours,
        }

    # Health and Monitoring Methods

    def get_data_sources_health(self) -> dict[str, Any]:
        """Get health status of all data sources."""
        health_data = self.manager.get_all_sources_health()

        # Format for dashboard display
        formatted_health = {
            "sources": [],
            "summary": {"total": 0, "healthy": 0, "degraded": 0, "unhealthy": 0},
        }

        for name, health in health_data.items():
            formatted_health["sources"].append(
                {
                    "name": name,
                    "status": health.status.value,
                    "last_check": health.last_check.isoformat(),
                    "latency_ms": health.latency_ms,
                    "success_rate": health.success_rate,
                    "error_count": health.error_count,
                }
            )

            formatted_health["summary"]["total"] += 1
            if health.status.value == "healthy":
                formatted_health["summary"]["healthy"] += 1
            elif health.status.value == "degraded":
                formatted_health["summary"]["degraded"] += 1
            else:
                formatted_health["summary"]["unhealthy"] += 1

        return formatted_health

    async def force_health_check(
        self, source_name: str | None = None
    ) -> dict[str, Any]:
        """Force health check on data sources."""
        results = await self.manager.force_health_check(source_name)

        # Format results
        formatted_results = {}
        for name, health in results.items():
            formatted_results[name] = {
                "status": health.status.value,
                "last_check": health.last_check.isoformat(),
                "latency_ms": health.latency_ms,
                "success_rate": health.success_rate,
            }

        return formatted_results

    def get_active_source(self, source_type: str = "market_data") -> str | None:
        """Get the currently active source for a given type."""
        try:
            type_enum = DataSourceType(source_type)
            source = self.manager.get_healthy_source(type_enum)
            return source.config.name if source else None
        except Exception:
            return None

    # Market Source Manager Methods

    async def stream_market_data(
        self, symbols: list[str], callback: Any
    ) -> asyncio.Task | None:
        """Stream real-time market data."""
        if self.market_source_manager.sources:
            return await self.market_source_manager.stream_quotes(symbols, callback)
        return None

    def set_market_source_strategy(self, strategy: str) -> None:
        """Set market source selection strategy."""
        try:
            strategy_enum = SourceSelectionStrategy(strategy)
            self.market_source_manager.set_strategy(strategy_enum)
            logger.info(f"Market source strategy set to: {strategy}")
        except ValueError:
            logger.error(f"Invalid strategy: {strategy}")

    def get_market_sources_status(self) -> dict[str, Any]:
        """Get status of all market data sources."""
        return self.market_source_manager.get_source_status()

    async def get_market_sentiment(self, symbol: str) -> dict[str, Any] | None:
        """Get market sentiment from Finnhub."""
        return await self.market_source_manager.get_sentiment(symbol)

    async def get_technical_indicators(
        self, symbol: str, indicator: str, **kwargs
    ) -> dict[str, Any] | None:
        """Get technical indicators from Alpha Vantage."""
        return await self.market_source_manager.get_technical_indicators(
            symbol, indicator, **kwargs
        )


# Singleton instance
_integration: DataSourceIntegration | None = None


def get_data_source_integration() -> DataSourceIntegration:
    """Get or create the singleton integration instance."""
    global _integration

    if _integration is None:
        _integration = DataSourceIntegration()

    return _integration


# Convenience functions for backward compatibility
async def get_market_quote(symbol: str) -> dict[str, Any]:
    """Get market quote - backward compatible function."""
    integration = get_data_source_integration()
    return await integration.get_quote(symbol)


async def get_market_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Get market quotes - backward compatible function."""
    integration = get_data_source_integration()
    return await integration.get_quotes(symbols)


async def get_historical_prices(
    symbol: str, interval: str, from_date: datetime, to_date: datetime
) -> list[dict[str, Any]]:
    """Get historical prices - backward compatible function."""
    integration = get_data_source_integration()
    return await integration.get_historical_data(symbol, interval, from_date, to_date)
