"""
Factory for creating and configuring the multi-source data manager.
"""

import logging

from app.core.config import get_settings

from .adapters import ZerodhaMarketDataSource
from .base import DataSourceConfig
from .multi_source_manager import MultiSourceDataManager

logger = logging.getLogger(__name__)


def create_data_source_manager() -> MultiSourceDataManager:
    """
    Create and configure the multi-source data manager based on settings.
    """
    settings = get_settings()
    manager = MultiSourceDataManager()

    # Configure Zerodha as primary source
    if settings.KITE_API_KEY and settings.KITE_ACCESS_TOKEN:
        zerodha_config = DataSourceConfig(
            name="zerodha",
            priority=1,  # Highest priority
            enabled=True,
            health_check_interval=settings.DATA_SOURCE_HEALTH_CHECK_INTERVAL,
            timeout=settings.DATA_SOURCE_TIMEOUT,
            max_retries=settings.DATA_SOURCE_MAX_RETRIES,
            retry_delay=settings.DATA_SOURCE_RETRY_DELAY,
            rate_limit=settings.ZERODHA_RATE_LIMIT,
            connection_pool_size=settings.CONNECTION_POOL_SIZE,
            api_key=settings.KITE_API_KEY,
            api_secret=settings.KITE_API_SECRET,
            extra_config={
                "access_token": settings.KITE_ACCESS_TOKEN,
                "enable_websocket": True,
            },
        )

        zerodha_source = ZerodhaMarketDataSource(zerodha_config)
        manager.add_source(zerodha_source)
        logger.info("Added Zerodha as primary market data source")

    # Configure Alpha Vantage as backup
    if settings.ALPHA_VANTAGE_ENABLED and settings.ALPHA_VANTAGE_API_KEY:
        # Note: AlphaVantageMarketDataSource would need to be implemented
        logger.info(
            "Alpha Vantage configuration detected but adapter not implemented yet"
        )

    # Configure Yahoo Finance as additional backup
    if settings.YAHOO_FINANCE_ENABLED:
        # Note: YahooFinanceMarketDataSource would need to be implemented
        logger.info(
            "Yahoo Finance configuration detected but adapter not implemented yet"
        )

    # Configure NSE API as additional backup
    if settings.NSE_API_ENABLED:
        # Note: NSEMarketDataSource would need to be implemented
        logger.info("NSE API configuration detected but adapter not implemented yet")

    # Configure news sources
    if settings.NEWS_API_ENABLED and settings.NEWS_API_KEY:
        # Note: NewsAPISentimentSource would need to be implemented
        logger.info("News API configuration detected but adapter not implemented yet")

    # Configure Twitter for sentiment
    if settings.TWITTER_API_ENABLED and settings.TWITTER_API_KEY:
        # Note: TwitterSentimentSource would need to be implemented
        logger.info(
            "Twitter API configuration detected but adapter not implemented yet"
        )

    # Add failover callback for notifications
    async def failover_notification(source_type, from_source, to_source):
        logger.warning(
            f"Data source failover: {from_source.config.name} -> {to_source.config.name} "
            f"for {source_type.value}"
        )
        # Here you could add notifications (email, Slack, etc.)

    manager.add_failover_callback(failover_notification)

    return manager


# Singleton instance
_data_manager: MultiSourceDataManager | None = None


def get_data_manager() -> MultiSourceDataManager:
    """Get or create the singleton data manager instance."""
    global _data_manager

    if _data_manager is None:
        _data_manager = create_data_source_manager()

    return _data_manager


async def initialize_data_manager() -> MultiSourceDataManager:
    """Initialize and start the data manager."""
    manager = get_data_manager()
    await manager.start()
    return manager


async def shutdown_data_manager() -> None:
    """Shutdown the data manager."""
    global _data_manager

    if _data_manager:
        await _data_manager.stop()
        _data_manager = None
