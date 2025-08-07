import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from loguru import logger

from backend.data_sources.base import DataSourceConfig, DataSourceStatus

from .alpha_vantage_market import AlphaVantageMarketSource
from .finnhub import FinnhubSource
from .global_datafeeds import GlobalDatafeedsSource
from .polygon import PolygonSource


class SourceSelectionStrategy(Enum):
    """Strategies for selecting data sources"""

    COST_OPTIMIZED = "cost_optimized"  # Minimize cost
    QUALITY_FIRST = "quality_first"  # Best quality regardless of cost
    LATENCY_OPTIMIZED = "latency_optimized"  # Lowest latency
    BALANCED = "balanced"  # Balance of all factors


class MarketSourceManager:
    """Intelligent market data source manager with failover and optimization"""

    def __init__(
        self, strategy: SourceSelectionStrategy = SourceSelectionStrategy.BALANCED
    ):
        self.strategy = strategy
        self.sources: dict[str, Any] = {}
        self.source_configs: dict[str, DataSourceConfig] = {}

        # Source health tracking
        self.source_health: dict[str, DataSourceStatus] = {}
        self.failure_counts: dict[str, int] = defaultdict(int)
        self.last_failure_time: dict[str, datetime] = {}

        # Usage tracking for cost optimization
        self.request_counts: dict[str, int] = defaultdict(int)
        self.monthly_costs: dict[str, float] = defaultdict(float)

        # Market coverage mapping
        self.market_coverage = {
            "global_datafeeds": ["NSE", "BSE", "NFO", "MCX"],
            "alpha_vantage": ["US", "NYSE", "NASDAQ", "FOREX", "CRYPTO", "TSX", "LSE"],
            "finnhub": ["US", "NYSE", "NASDAQ", "FOREX", "CRYPTO", "EU", "UK", "JP"],
            "polygon": ["US", "NYSE", "NASDAQ", "FOREX", "CRYPTO", "OPTIONS"],
        }

        # Feature support mapping
        self.feature_support = {
            "global_datafeeds": {
                "realtime": True,
                "historical": True,
                "depth": True,
                "streaming": True,
                "sentiment": False,
                "technical": False,
            },
            "alpha_vantage": {
                "realtime": True,
                "historical": True,
                "depth": False,
                "streaming": False,
                "sentiment": False,
                "technical": True,  # Unique feature
            },
            "finnhub": {
                "realtime": True,
                "historical": True,
                "depth": False,
                "streaming": True,
                "sentiment": True,  # Unique feature
                "technical": False,
            },
            "polygon": {
                "realtime": True,
                "historical": True,
                "depth": True,
                "streaming": True,
                "sentiment": False,
                "technical": False,
                "institutional": True,  # Premium feature
            },
        }

        # Quality rankings (1-10 scale)
        self.quality_scores = {
            "global_datafeeds": 9,  # Professional NSE/BSE data
            "alpha_vantage": 7,  # Good quality, some latency
            "finnhub": 8,  # Professional with sentiment
            "polygon": 10,  # Institutional grade
        }

        # Latency estimates (milliseconds)
        self.latency_estimates = {
            "global_datafeeds": 10,  # Ultra-low for Indian markets
            "alpha_vantage": 1000,  # API latency
            "finnhub": 100,  # Good latency
            "polygon": 5,  # Ultra-low institutional
        }

        logger.info(f"Initialized MarketSourceManager with {strategy.value} strategy")

    async def initialize(self, configs: dict[str, DataSourceConfig]):
        """Initialize all configured data sources"""
        self.source_configs = configs

        # Create source instances
        source_classes = {
            "global_datafeeds": GlobalDatafeedsSource,
            "alpha_vantage": AlphaVantageMarketSource,
            "finnhub": FinnhubSource,
            "polygon": PolygonSource,
        }

        for source_name, config in configs.items():
            if source_name in source_classes:
                try:
                    source = source_classes[source_name](config)
                    self.sources[source_name] = source

                    # Try to connect
                    connected = await source.connect()
                    if connected:
                        self.source_health[source_name] = DataSourceStatus.HEALTHY
                        logger.info(f"Connected to {source_name}")
                    else:
                        self.source_health[source_name] = DataSourceStatus.UNHEALTHY
                        logger.warning(f"Failed to connect to {source_name}")

                except Exception as e:
                    logger.error(f"Error initializing {source_name}: {e}")
                    self.source_health[source_name] = DataSourceStatus.UNHEALTHY

    async def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Get quote with intelligent source selection and failover"""
        # Detect market from symbol
        market = self._detect_market(symbol)

        # Get prioritized sources for this request
        sources = self._select_sources_for_request(market, "quote")

        # Try sources in priority order
        for source_name in sources:
            if self._is_source_available(source_name):
                try:
                    source = self.sources[source_name]
                    quote = await source.get_quote(symbol)

                    # Track successful request
                    self._track_request_success(source_name)

                    # Add source metadata
                    quote["data_source"] = source_name
                    quote["source_cost_tier"] = source.get_cost_info().tier.value

                    return quote

                except Exception as e:
                    logger.error(f"Error fetching quote from {source_name}: {e}")
                    self._track_request_failure(source_name)
                    continue

        logger.error(f"All sources failed for quote: {symbol}")
        return None

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get multiple quotes with optimal source selection"""
        # Group symbols by market
        market_groups = self._group_symbols_by_market(symbols)
        quotes = {}

        # Process each market group with best source
        for market, market_symbols in market_groups.items():
            sources = self._select_sources_for_request(market, "quotes")

            for source_name in sources:
                if self._is_source_available(source_name):
                    try:
                        source = self.sources[source_name]
                        market_quotes = await source.get_quotes(market_symbols)

                        # Add source metadata
                        for symbol, quote in market_quotes.items():
                            if quote:
                                quote["data_source"] = source_name
                                quotes[symbol] = quote

                        self._track_request_success(source_name, len(market_symbols))
                        break

                    except Exception as e:
                        logger.error(f"Error fetching quotes from {source_name}: {e}")
                        self._track_request_failure(source_name)
                        continue

        return quotes

    async def get_historical_data(
        self, symbol: str, interval: str, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """Get historical data with source optimization"""
        market = self._detect_market(symbol)
        sources = self._select_sources_for_request(market, "historical")

        for source_name in sources:
            if self._is_source_available(source_name):
                try:
                    source = self.sources[source_name]
                    data = await source.get_historical_data(
                        symbol, interval, from_date, to_date
                    )

                    if data:
                        self._track_request_success(source_name)
                        # Add source metadata to first bar
                        if data:
                            data[0]["data_source"] = source_name
                        return data

                except Exception as e:
                    logger.error(
                        f"Error fetching historical data from {source_name}: {e}"
                    )
                    self._track_request_failure(source_name)
                    continue

        return []

    async def stream_quotes(
        self, symbols: list[str], callback: Any
    ) -> asyncio.Task | None:
        """Stream quotes from best available source"""
        # Group by market and find best streaming source
        market_groups = self._group_symbols_by_market(symbols)

        streaming_tasks = []
        for market, market_symbols in market_groups.items():
            sources = self._select_sources_for_request(market, "streaming")

            for source_name in sources:
                if self._is_source_available(source_name) and self.feature_support[
                    source_name
                ].get("streaming", False):
                    try:
                        source = self.sources[source_name]

                        # Wrap callback to add source info
                        async def wrapped_callback(symbol, data):
                            data["data_source"] = source_name
                            await callback(symbol, data)

                        task = asyncio.create_task(
                            source.stream_quotes(market_symbols, wrapped_callback)
                        )
                        streaming_tasks.append(task)

                        logger.info(
                            f"Started streaming {len(market_symbols)} symbols from {source_name}"
                        )
                        break

                    except Exception as e:
                        logger.error(f"Error starting stream from {source_name}: {e}")
                        continue

        if streaming_tasks:
            # Return a combined task
            return asyncio.create_task(asyncio.gather(*streaming_tasks))

        return None

    async def get_sentiment(self, symbol: str) -> dict[str, Any] | None:
        """Get sentiment analysis from capable sources"""
        # Only Finnhub supports sentiment currently
        if "finnhub" in self.sources and self._is_source_available("finnhub"):
            try:
                source = self.sources["finnhub"]
                sentiment = await source.get_sentiment(symbol)

                if sentiment:
                    self._track_request_success("finnhub")
                    sentiment["data_source"] = "finnhub"
                    return sentiment

            except Exception as e:
                logger.error(f"Error fetching sentiment: {e}")
                self._track_request_failure("finnhub")

        return None

    async def get_technical_indicators(
        self, symbol: str, indicator: str, **kwargs
    ) -> dict[str, Any] | None:
        """Get technical indicators from capable sources"""
        # Only Alpha Vantage supports technical indicators
        if "alpha_vantage" in self.sources and self._is_source_available(
            "alpha_vantage"
        ):
            try:
                source = self.sources["alpha_vantage"]
                result = await source.get_technical_indicators(
                    symbol, indicator, **kwargs
                )

                if result:
                    self._track_request_success("alpha_vantage")
                    result["data_source"] = "alpha_vantage"
                    return result

            except Exception as e:
                logger.error(f"Error fetching technical indicators: {e}")
                self._track_request_failure("alpha_vantage")

        return None

    def _select_sources_for_request(self, market: str, request_type: str) -> list[str]:
        """Select and prioritize sources based on strategy"""
        # Get sources that support this market
        available_sources = [
            source
            for source, markets in self.market_coverage.items()
            if market in markets and source in self.sources
        ]

        if not available_sources:
            # Fallback to any US market source for unknown markets
            available_sources = [
                source
                for source, markets in self.market_coverage.items()
                if "US" in markets and source in self.sources
            ]

        # Apply strategy-based sorting
        if self.strategy == SourceSelectionStrategy.COST_OPTIMIZED:
            # Sort by cost (free first, then by monthly cost)
            available_sources.sort(
                key=lambda s: (
                    self.sources[s].get_cost_info().monthly_cost,
                    -self.quality_scores.get(s, 0),
                )
            )

        elif self.strategy == SourceSelectionStrategy.QUALITY_FIRST:
            # Sort by quality score
            available_sources.sort(key=lambda s: -self.quality_scores.get(s, 0))

        elif self.strategy == SourceSelectionStrategy.LATENCY_OPTIMIZED:
            # Sort by latency
            available_sources.sort(key=lambda s: self.latency_estimates.get(s, 999))

        else:  # BALANCED
            # Weighted scoring
            def balanced_score(source):
                quality = self.quality_scores.get(source, 5) / 10
                latency = 1 - (self.latency_estimates.get(source, 500) / 1000)
                cost = (
                    1 if self.sources[source].get_cost_info().monthly_cost == 0 else 0.5
                )

                # Weights: quality=40%, latency=30%, cost=30%
                return (quality * 0.4) + (latency * 0.3) + (cost * 0.3)

            available_sources.sort(key=balanced_score, reverse=True)

        return available_sources

    def _is_source_available(self, source_name: str) -> bool:
        """Check if source is available for use"""
        # Check health status
        if self.source_health.get(source_name) != DataSourceStatus.HEALTHY:
            return False

        # Check failure rate (circuit breaker)
        if self.failure_counts[source_name] >= 5:
            # Check if enough time has passed to retry
            last_failure = self.last_failure_time.get(source_name)
            if last_failure and (datetime.utcnow() - last_failure) < timedelta(
                minutes=5
            ):
                return False
            else:
                # Reset failure count after cooldown
                self.failure_counts[source_name] = 0

        # Check cost limits if configured
        # TODO: Implement monthly cost tracking and limits

        return True

    def _track_request_success(self, source_name: str, count: int = 1):
        """Track successful request"""
        self.request_counts[source_name] += count

        # Update cost tracking
        source = self.sources[source_name]
        cost_info = source.get_cost_info()

        if cost_info.per_request_cost > 0:
            request_cost = cost_info.get_request_cost(self.request_counts[source_name])
            self.monthly_costs[source_name] = cost_info.monthly_cost + request_cost
        else:
            self.monthly_costs[source_name] = cost_info.monthly_cost

    def _track_request_failure(self, source_name: str):
        """Track failed request"""
        self.failure_counts[source_name] += 1
        self.last_failure_time[source_name] = datetime.utcnow()

        # Update health status if too many failures
        if self.failure_counts[source_name] >= 3:
            self.source_health[source_name] = DataSourceStatus.DEGRADED
        if self.failure_counts[source_name] >= 5:
            self.source_health[source_name] = DataSourceStatus.UNHEALTHY

    def _detect_market(self, symbol: str) -> str:
        """Detect market from symbol format"""
        # Indian markets
        if symbol.endswith(".NS") or symbol.endswith(".NSE"):
            return "NSE"
        elif symbol.endswith(".BO") or symbol.endswith(".BSE"):
            return "BSE"
        elif symbol.endswith(".NFO"):
            return "NFO"
        elif symbol.endswith(".MCX"):
            return "MCX"

        # International markets
        elif ":" in symbol:
            exchange = symbol.split(":")[0]
            return exchange.upper()
        elif "." in symbol:
            parts = symbol.split(".")
            if len(parts) > 1:
                return parts[-1].upper()

        # Default to US market
        return "US"

    def _group_symbols_by_market(self, symbols: list[str]) -> dict[str, list[str]]:
        """Group symbols by their market"""
        market_groups = defaultdict(list)

        for symbol in symbols:
            market = self._detect_market(symbol)
            market_groups[market].append(symbol)

        return dict(market_groups)

    def get_source_status(self) -> dict[str, Any]:
        """Get status of all data sources"""
        status = {}

        for source_name, source in self.sources.items():
            cost_info = source.get_cost_info()

            status[source_name] = {
                "health": self.source_health.get(
                    source_name, DataSourceStatus.UNKNOWN
                ).value,
                "failure_count": self.failure_counts[source_name],
                "request_count": self.request_counts[source_name],
                "monthly_cost": self.monthly_costs[source_name],
                "cost_tier": cost_info.tier.value,
                "markets": self.market_coverage.get(source_name, []),
                "features": self.feature_support.get(source_name, {}),
                "quality_score": self.quality_scores.get(source_name, 0),
                "latency_ms": self.latency_estimates.get(source_name, 0),
            }

        return {
            "strategy": self.strategy.value,
            "sources": status,
            "total_monthly_cost": sum(self.monthly_costs.values()),
            "total_requests": sum(self.request_counts.values()),
        }

    def set_strategy(self, strategy: SourceSelectionStrategy):
        """Change source selection strategy"""
        self.strategy = strategy
        logger.info(f"Changed source selection strategy to {strategy.value}")

    async def close(self):
        """Close all data source connections"""
        for source_name, source in self.sources.items():
            try:
                await source.disconnect()
                logger.info(f"Disconnected from {source_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {source_name}: {e}")

        self.sources.clear()
        self.source_health.clear()
