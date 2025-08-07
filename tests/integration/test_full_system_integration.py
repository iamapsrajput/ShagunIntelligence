"""
Full System Integration Tests
Tests the complete trading system end-to-end
"""

import asyncio
from datetime import datetime

import pandas as pd
import pytest

from app.services.advanced_order_management import AdvancedOrderManager
from app.services.broker_integration import (
    BrokerManager,
    MockBrokerAPI,
    OrderRequest,
    OrderType,
    TransactionType,
)
from app.services.enhanced_risk_management import EnhancedRiskManager
from app.services.market_data_integration import (
    DataProvider,
    DataType,
    HistoricalDataService,
    RealTimeDataFeed,
)
from app.services.multi_timeframe_analysis import MultiTimeframeAnalyzer


class TestFullSystemIntegration:
    """Integration tests for the complete trading system"""

    @pytest.fixture
    async def trading_system(self):
        """Set up a complete trading system for testing"""
        # Initialize market data services
        real_time_feed = RealTimeDataFeed(DataProvider.ALPHA_VANTAGE)
        historical_service = HistoricalDataService(DataProvider.YAHOO_FINANCE)

        # Initialize broker services
        broker_manager = BrokerManager()
        mock_broker = MockBrokerAPI("test_api_key", "test_access_token")
        broker_manager.add_broker("test_broker", mock_broker)

        # Initialize advanced services
        order_manager = AdvancedOrderManager()
        risk_manager = EnhancedRiskManager()
        analyzer = MultiTimeframeAnalyzer()

        return {
            "real_time_feed": real_time_feed,
            "historical_service": historical_service,
            "broker_manager": broker_manager,
            "order_manager": order_manager,
            "risk_manager": risk_manager,
            "analyzer": analyzer,
        }

    @pytest.mark.asyncio
    async def test_market_data_to_broker_flow(self, trading_system):
        """Test complete flow from market data to order execution"""

        # Step 1: Connect to market data feed
        await trading_system["real_time_feed"].connect()
        assert trading_system["real_time_feed"].is_connected

        # Step 2: Subscribe to market data
        symbols = ["RELIANCE", "TCS"]
        await trading_system["real_time_feed"].subscribe(symbols, [DataType.QUOTE])

        # Step 3: Wait for mock data generation
        await asyncio.sleep(2)

        # Step 4: Get market data
        quote = await trading_system["real_time_feed"].get_cached_quote("RELIANCE")
        assert quote is not None
        assert quote.last_price > 0

        # Step 5: Authenticate broker
        auth_results = await trading_system["broker_manager"].authenticate_all()
        assert auth_results["test_broker"] is True

        # Step 6: Place order based on market data
        order = OrderRequest(
            symbol="RELIANCE",
            exchange="NSE",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            price=quote.last_price,
        )

        response = await trading_system["broker_manager"].place_order(order)
        assert response.order_id is not None
        assert response.status.value in ["OPEN", "COMPLETE"]

        # Step 7: Verify order execution
        positions = await trading_system["broker_manager"].get_consolidated_positions()
        assert "test_broker" in positions

        # Cleanup
        await trading_system["real_time_feed"].disconnect()

    @pytest.mark.asyncio
    async def test_historical_data_analysis_flow(self, trading_system):
        """Test historical data analysis and signal generation"""

        # Step 1: Get historical data
        async with trading_system["historical_service"]:
            data = await trading_system["historical_service"].get_historical_data(
                "RELIANCE", "1d", limit=50
            )

        assert not data.empty
        assert len(data) > 0

        # Step 2: Perform multi-timeframe analysis
        analyzer = trading_system["analyzer"]

        # Add mock data to analyzer
        analyzer.add_data("RELIANCE", "1d", data)

        # Calculate technical indicators
        indicators = analyzer.calculate_indicators("RELIANCE", "1d")
        assert "sma_20" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators

        # Step 3: Generate trading signals
        signals = analyzer.generate_signals("RELIANCE")
        assert isinstance(signals, list)

        # Step 4: Test risk management
        risk_manager = trading_system["risk_manager"]

        # Mock portfolio data
        portfolio_data = {
            "total_value": 1000000,
            "positions": [
                {"symbol": "RELIANCE", "quantity": 100, "price": 2500, "value": 250000}
            ],
        }

        # Calculate position sizing
        position_size = risk_manager.calculate_position_size(
            portfolio_value=portfolio_data["total_value"],
            risk_per_trade=0.02,  # 2% risk
            entry_price=2500,
            stop_loss_price=2400,
        )

        assert position_size > 0
        assert position_size <= 200  # Should be reasonable size

    @pytest.mark.asyncio
    async def test_advanced_order_management_flow(self, trading_system):
        """Test advanced order management features"""

        order_manager = trading_system["order_manager"]
        broker_manager = trading_system["broker_manager"]

        # Step 1: Test bracket order
        bracket_params = {
            "symbol": "TCS",
            "quantity": 50,
            "entry_price": 3600,
            "target_price": 3700,
            "stop_loss_price": 3500,
            "trailing_stop_percent": 2.0,
        }

        bracket_order = await order_manager.create_bracket_order(**bracket_params)
        assert bracket_order is not None
        assert bracket_order["main_order"] is not None
        assert bracket_order["target_order"] is not None
        assert bracket_order["stop_loss_order"] is not None

        # Step 2: Test TWAP order
        twap_params = {
            "symbol": "HDFCBANK",
            "total_quantity": 200,
            "duration_minutes": 60,
            "price_limit": 1600,
        }

        twap_order = await order_manager.create_twap_order(**twap_params)
        assert twap_order is not None
        assert twap_order["total_quantity"] == 200
        assert twap_order["slice_size"] > 0

        # Step 3: Test order modification
        # Place a simple order first
        order = OrderRequest(
            symbol="INFY",
            exchange="NSE",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.LIMIT,
            quantity=75,
            price=2500,
        )

        response = await broker_manager.place_order(order)
        assert response.order_id is not None

        # Modify the order
        broker = broker_manager.get_active_broker()
        async with broker:
            modify_response = await broker.modify_order(
                response.order_id, price=2520, quantity=80
            )

        assert modify_response.status.value in ["MODIFIED", "OPEN"]

    @pytest.mark.asyncio
    async def test_risk_management_integration(self, trading_system):
        """Test risk management integration with trading system"""

        risk_manager = trading_system["risk_manager"]
        broker_manager = trading_system["broker_manager"]

        # Step 1: Set up portfolio for risk analysis
        mock_positions = [
            {"symbol": "RELIANCE", "quantity": 100, "price": 2500, "last_price": 2520},
            {"symbol": "TCS", "quantity": 50, "price": 3600, "last_price": 3580},
            {"symbol": "HDFCBANK", "quantity": 75, "price": 1550, "last_price": 1570},
        ]

        # Step 2: Calculate portfolio metrics
        total_value = sum(pos["quantity"] * pos["last_price"] for pos in mock_positions)
        total_pnl = sum(
            pos["quantity"] * (pos["last_price"] - pos["price"])
            for pos in mock_positions
        )

        # Step 3: Calculate VaR
        returns_data = pd.Series(
            [0.01, -0.02, 0.015, -0.01, 0.005] * 50
        )  # Mock returns
        var_result = risk_manager.calculate_var(returns_data, confidence_level=0.95)

        assert var_result["var_absolute"] > 0
        assert 0 < var_result["confidence_level"] <= 1

        # Step 4: Test position size limits
        max_position_size = risk_manager.calculate_max_position_size(
            portfolio_value=total_value,
            symbol="NEWSTOCK",
            price=1000,
            max_position_percent=5.0,
        )

        expected_max_size = int((total_value * 0.05) / 1000)
        assert max_position_size <= expected_max_size

        # Step 5: Test risk alerts
        risk_alerts = risk_manager.check_risk_limits(
            {
                "total_exposure": total_value * 1.5,  # 150% exposure
                "max_single_position": total_value * 0.15,  # 15% in single position
                "sector_concentration": {"Technology": 0.4},  # 40% in tech
            }
        )

        assert isinstance(risk_alerts, list)

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, trading_system):
        """Test system performance monitoring"""

        # Step 1: Test market data performance
        start_time = datetime.now()

        async with trading_system["historical_service"]:
            data = await trading_system["historical_service"].get_historical_data(
                "RELIANCE", "1d", limit=100
            )

        data_fetch_time = (datetime.now() - start_time).total_seconds()
        assert data_fetch_time < 5.0  # Should complete within 5 seconds

        # Step 2: Test order execution performance
        start_time = datetime.now()

        order = OrderRequest(
            symbol="TESTSTOCK",
            exchange="NSE",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
        )

        response = await trading_system["broker_manager"].place_order(order)
        order_execution_time = (datetime.now() - start_time).total_seconds()

        assert order_execution_time < 2.0  # Should complete within 2 seconds
        assert response.order_id is not None

        # Step 3: Test concurrent operations
        start_time = datetime.now()

        # Run multiple operations concurrently
        tasks = [
            trading_system["historical_service"].get_historical_data(
                "RELIANCE", "1d", limit=10
            ),
            trading_system["historical_service"].get_historical_data(
                "TCS", "1d", limit=10
            ),
            trading_system["historical_service"].get_historical_data(
                "HDFCBANK", "1d", limit=10
            ),
        ]

        async with trading_system["historical_service"]:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        concurrent_time = (datetime.now() - start_time).total_seconds()

        # Should be faster than sequential execution
        assert concurrent_time < 10.0
        assert len(results) == 3

        # Check that all operations completed successfully
        for result in results:
            assert not isinstance(result, Exception)

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, trading_system):
        """Test system error handling and recovery"""

        # Step 1: Test invalid symbol handling
        async with trading_system["historical_service"]:
            data = await trading_system["historical_service"].get_historical_data(
                "INVALID_SYMBOL", "1d", limit=10
            )

        # Should return empty DataFrame, not raise exception
        assert isinstance(data, pd.DataFrame)

        # Step 2: Test invalid order handling
        invalid_order = OrderRequest(
            symbol="TESTSTOCK",
            exchange="NSE",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.LIMIT,
            quantity=-100,  # Invalid negative quantity
            price=2500,
        )

        response = await trading_system["broker_manager"].place_order(invalid_order)
        assert response.status.value == "REJECTED"

        # Step 3: Test connection recovery
        # Disconnect and reconnect market data feed
        await trading_system["real_time_feed"].disconnect()
        assert not trading_system["real_time_feed"].is_connected

        await trading_system["real_time_feed"].connect()
        # Connection should be restored (or attempt made)

        # Step 4: Test risk limit violations
        risk_manager = trading_system["risk_manager"]

        # Test with extreme position size
        position_size = risk_manager.calculate_position_size(
            portfolio_value=100000,
            risk_per_trade=0.5,  # 50% risk (extreme)
            entry_price=2500,
            stop_loss_price=2400,
        )

        # Should be capped at reasonable limits
        assert position_size <= 400  # Should not allow extreme position sizes

    def test_system_configuration(self, trading_system):
        """Test system configuration and settings"""

        # Test that all components are properly initialized
        assert trading_system["real_time_feed"] is not None
        assert trading_system["historical_service"] is not None
        assert trading_system["broker_manager"] is not None
        assert trading_system["order_manager"] is not None
        assert trading_system["risk_manager"] is not None
        assert trading_system["analyzer"] is not None

        # Test broker manager configuration
        broker_status = trading_system["broker_manager"].get_broker_status()
        assert broker_status["total_brokers"] > 0
        assert broker_status["active_broker"] is not None

        # Test risk manager configuration
        risk_config = trading_system["risk_manager"].get_risk_config()
        assert "max_portfolio_risk" in risk_config
        assert "max_position_size" in risk_config
        assert "var_confidence_level" in risk_config

        # Test order manager configuration
        order_config = trading_system["order_manager"].get_configuration()
        assert "max_order_size" in order_config
        assert "default_validity" in order_config


@pytest.mark.asyncio
async def test_end_to_end_trading_scenario():
    """Test a complete end-to-end trading scenario"""

    # Initialize system components
    broker_manager = BrokerManager()
    mock_broker = MockBrokerAPI("test_api", "test_token")
    broker_manager.add_broker("test", mock_broker)

    order_manager = AdvancedOrderManager()
    risk_manager = EnhancedRiskManager()

    # Authenticate broker
    auth_results = await broker_manager.authenticate_all()
    assert auth_results["test"] is True

    # Scenario: Buy RELIANCE with risk management
    symbol = "RELIANCE"
    entry_price = 2500.0
    portfolio_value = 1000000.0

    # Step 1: Calculate position size based on risk
    position_size = risk_manager.calculate_position_size(
        portfolio_value=portfolio_value,
        risk_per_trade=0.02,  # 2% risk
        entry_price=entry_price,
        stop_loss_price=2400.0,
    )

    # Step 2: Place bracket order
    bracket_order = await order_manager.create_bracket_order(
        symbol=symbol,
        quantity=position_size,
        entry_price=entry_price,
        target_price=2600.0,
        stop_loss_price=2400.0,
        trailing_stop_percent=1.5,
    )

    assert bracket_order is not None
    assert bracket_order["main_order"]["quantity"] == position_size

    # Step 3: Execute main order
    main_order = OrderRequest(
        symbol=symbol,
        exchange="NSE",
        transaction_type=TransactionType.BUY,
        order_type=OrderType.LIMIT,
        quantity=position_size,
        price=entry_price,
    )

    response = await broker_manager.place_order(main_order)
    assert response.order_id is not None

    # Step 4: Monitor position
    positions = await broker_manager.get_consolidated_positions()
    assert "test" in positions

    # Step 5: Check risk metrics
    portfolio_positions = positions["test"]
    total_exposure = sum(
        abs(pos.quantity * pos.last_price) for pos in portfolio_positions
    )

    # Ensure exposure is within limits
    max_exposure = portfolio_value * 1.5  # 150% max exposure
    assert total_exposure <= max_exposure

    print("✅ End-to-end scenario completed successfully")
    print(f"   Position size: {position_size}")
    print(f"   Total exposure: ₹{total_exposure:,.2f}")
    print(f"   Order ID: {response.order_id}")


if __name__ == "__main__":
    # Run a simple test
    asyncio.run(test_end_to_end_trading_scenario())
