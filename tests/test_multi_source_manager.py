"""
Tests for the multi-source data manager.
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest
from backend.data_sources import (
    DataSourceConfig,
    DataSourceHealth,
    DataSourceStatus,
    DataSourceType,
    MarketDataSource,
    MultiSourceDataManager,
)
from backend.data_sources.integration import DataSourceIntegration


class MockMarketDataSource(MarketDataSource):
    """Mock market data source for testing."""

    def __init__(self, config: DataSourceConfig, fail_after: int = -1):
        super().__init__(config)
        self.call_count = 0
        self.fail_after = fail_after
        self.connect_called = False
        self.disconnect_called = False

    async def connect(self) -> bool:
        self.connect_called = True
        self._is_connected = True
        self.update_health_status(DataSourceStatus.HEALTHY)
        return True

    async def disconnect(self) -> None:
        self.disconnect_called = True
        self._is_connected = False
        self.update_health_status(DataSourceStatus.DISCONNECTED)

    async def health_check(self) -> DataSourceHealth:
        if self._is_connected:
            self.update_health_status(DataSourceStatus.HEALTHY)
        return self.health

    async def validate_credentials(self) -> bool:
        return True

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        self.call_count += 1

        if self.fail_after >= 0 and self.call_count > self.fail_after:
            raise Exception(f"Mock failure after {self.fail_after} calls")

        return {"symbol": symbol, "last_price": 100.0, "source": self.config.name}

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        result = {}
        for symbol in symbols:
            result[symbol] = await self.get_quote(symbol)
        return result

    async def get_historical_data(
        self, symbol: str, interval: str, from_date: datetime, to_date: datetime
    ) -> List[Dict[str, Any]]:
        return [{"timestamp": from_date, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}]

    async def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        return {"bids": [{"price": 99.5, "quantity": 100}], "asks": [{"price": 100.5, "quantity": 100}]}

    async def subscribe_live_data(self, symbols: List[str], callback: Callable) -> None:
        pass

    async def unsubscribe_live_data(self, symbols: List[str]) -> None:
        pass


@pytest.mark.asyncio
class TestMultiSourceDataManager:
    """Test cases for MultiSourceDataManager."""

    async def test_add_remove_sources(self):
        """Test adding and removing data sources."""
        manager = MultiSourceDataManager()

        # Create test sources
        config1 = DataSourceConfig(name="source1", priority=1)
        source1 = MockMarketDataSource(config1)

        config2 = DataSourceConfig(name="source2", priority=2)
        source2 = MockMarketDataSource(config2)

        # Add sources
        manager.add_source(source1)
        manager.add_source(source2)

        # Check sources are added
        assert len(manager._sources[DataSourceType.MARKET_DATA]) == 2

        # Check priority ordering
        sources = manager._sources[DataSourceType.MARKET_DATA]
        assert sources[0].config.name == "source1"
        assert sources[1].config.name == "source2"

        # Remove source
        manager.remove_source("source1")
        assert len(manager._sources[DataSourceType.MARKET_DATA]) == 1
        assert manager._sources[DataSourceType.MARKET_DATA][0].config.name == "source2"

    async def test_start_stop_manager(self):
        """Test starting and stopping the manager."""
        manager = MultiSourceDataManager()

        # Add test source
        config = DataSourceConfig(name="test", priority=1)
        source = MockMarketDataSource(config)
        manager.add_source(source)

        # Start manager
        await manager.start()
        assert manager._is_running
        assert source.connect_called

        # Stop manager
        await manager.stop()
        assert not manager._is_running
        assert source.disconnect_called

    async def test_automatic_failover(self):
        """Test automatic failover between sources."""
        manager = MultiSourceDataManager()

        # Create primary source that fails after 2 calls
        config1 = DataSourceConfig(name="primary", priority=1)
        primary = MockMarketDataSource(config1, fail_after=2)

        # Create backup source
        config2 = DataSourceConfig(name="backup", priority=2)
        backup = MockMarketDataSource(config2)

        manager.add_source(primary)
        manager.add_source(backup)

        await manager.start()

        # First two calls should succeed with primary
        quote1 = await manager.get_quote("RELIANCE")
        assert quote1["source"] == "primary"

        quote2 = await manager.get_quote("RELIANCE")
        assert quote2["source"] == "primary"

        # Third call should fail over to backup
        quote3 = await manager.get_quote("RELIANCE")
        assert quote3["source"] == "backup"

        await manager.stop()

    async def test_health_monitoring(self):
        """Test health monitoring functionality."""
        manager = MultiSourceDataManager()

        config = DataSourceConfig(
            name="test",
            priority=1,
            health_check_interval=1,  # 1 second for testing
        )
        source = MockMarketDataSource(config)
        manager.add_source(source)

        await manager.start()

        # Check initial health
        health = manager.get_all_sources_health()
        assert "test" in health
        assert health["test"].status == DataSourceStatus.HEALTHY

        # Force health check
        results = await manager.force_health_check("test")
        assert "test" in results
        assert results["test"].status == DataSourceStatus.HEALTHY

        await manager.stop()

    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        manager = MultiSourceDataManager()

        # Create source with rate limit
        config = DataSourceConfig(
            name="limited",
            priority=1,
            rate_limit=2,  # 2 requests per minute
        )
        source = MockMarketDataSource(config)
        manager.add_source(source)

        await manager.start()

        # Make rapid requests
        start_time = asyncio.get_event_loop().time()

        await manager.get_quote("RELIANCE")
        await manager.get_quote("RELIANCE")

        # Third request should be delayed
        await manager.get_quote("RELIANCE")

        elapsed = asyncio.get_event_loop().time() - start_time

        # Should have been delayed (at least somewhat)
        assert elapsed > 0.5  # Some delay expected

        await manager.stop()

    async def test_execute_with_failover(self):
        """Test execute_with_failover method."""
        manager = MultiSourceDataManager()

        # Create three sources with different health states
        config1 = DataSourceConfig(name="unhealthy", priority=1)
        unhealthy = MockMarketDataSource(config1)
        unhealthy.health.status = DataSourceStatus.UNHEALTHY

        config2 = DataSourceConfig(name="degraded", priority=2)
        degraded = MockMarketDataSource(config2)
        degraded.health.status = DataSourceStatus.DEGRADED

        config3 = DataSourceConfig(name="healthy", priority=3)
        healthy = MockMarketDataSource(config3)

        manager.add_source(unhealthy)
        manager.add_source(degraded)
        manager.add_source(healthy)

        # Should skip unhealthy and use degraded
        result = await manager.execute_with_failover(DataSourceType.MARKET_DATA, "get_quote", "RELIANCE")

        assert result["source"] == "degraded"


@pytest.mark.asyncio
class TestDataSourceIntegration:
    """Test cases for DataSourceIntegration."""

    async def test_integration_initialization(self):
        """Test integration initialization."""
        integration = DataSourceIntegration()

        # Mock the manager
        mock_manager = AsyncMock()
        integration.manager = mock_manager

        await integration.initialize()
        mock_manager.start.assert_called_once()

        await integration.shutdown()
        mock_manager.stop.assert_called_once()

    async def test_backward_compatibility(self):
        """Test backward compatibility methods."""
        integration = DataSourceIntegration()

        # Mock the manager
        mock_manager = AsyncMock()
        mock_manager.get_quotes.return_value = {
            "RELIANCE": {
                "symbol": "RELIANCE",
                "last_price": 2500,
                "open": 2480,
                "high": 2520,
                "low": 2470,
                "close": 2500,
            }
        }
        integration.manager = mock_manager

        # Test get_ohlc
        ohlc = await integration.get_ohlc(["RELIANCE"])
        assert "RELIANCE" in ohlc
        assert ohlc["RELIANCE"]["open"] == 2480
        assert ohlc["RELIANCE"]["last_price"] == 2500

    async def test_health_status_formatting(self):
        """Test health status formatting."""
        integration = DataSourceIntegration()

        # Mock health data
        mock_health = {
            "zerodha": Mock(
                status=DataSourceStatus.HEALTHY,
                last_check=datetime.now(),
                latency_ms=50.0,
                success_rate=99.5,
                error_count=1,
            ),
            "backup": Mock(
                status=DataSourceStatus.DEGRADED,
                last_check=datetime.now(),
                latency_ms=150.0,
                success_rate=85.0,
                error_count=15,
            ),
        }

        integration.manager = Mock()
        integration.manager.get_all_sources_health.return_value = mock_health

        health_data = integration.get_data_sources_health()

        assert health_data["summary"]["total"] == 2
        assert health_data["summary"]["healthy"] == 1
        assert health_data["summary"]["degraded"] == 1
        assert len(health_data["sources"]) == 2
