"""
Unit tests for AutomatedTradingService
Tests the core automated trading functionality with comprehensive coverage
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.services.automated_trading import AutomatedTradingService


class TestAutomatedTradingService:
    """Test suite for AutomatedTradingService"""

    @pytest.fixture
    def mock_kite_client(self):
        """Mock Kite client"""
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.get_quote.return_value = {
            "RELIANCE": {
                "last_price": 2500.0,
                "volume": 1000000,
                "ohlc": {
                    "high": 2520.0,
                    "low": 2480.0,
                    "open": 2490.0,
                    "close": 2500.0,
                },
            }
        }
        mock_client.get_positions.return_value = {"net": [], "day": []}
        mock_client.place_order.return_value = {"order_id": "test_order_123"}
        return mock_client

    @pytest.fixture
    def mock_crew_manager(self):
        """Mock CrewManager"""
        mock_manager = AsyncMock()
        mock_manager.analyze_market.return_value = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "confidence": 0.8,
            "target_price": 2550.0,
            "stop_loss": 2450.0,
            "quantity": 1,
            "reasoning": "Strong technical indicators",
        }
        return mock_manager

    @pytest.fixture
    def mock_market_schedule(self):
        """Mock market schedule service"""
        mock_schedule = MagicMock()
        mock_schedule.get_market_status.return_value = {
            "is_open": True,
            "status": "OPEN",
            "message": "Market is open",
        }
        return mock_schedule

    @pytest.fixture
    def trading_service(
        self, mock_kite_client, mock_crew_manager, mock_market_schedule
    ):
        """Create AutomatedTradingService instance with mocks"""
        with patch(
            "app.services.automated_trading.market_schedule", mock_market_schedule
        ):
            service = AutomatedTradingService()
            service.kite_client = mock_kite_client
            service.crew_manager = mock_crew_manager
            return service

    def test_initialization(self):
        """Test service initialization"""
        service = AutomatedTradingService()

        assert service.is_running is False
        assert service.trading_enabled is True
        assert service.max_daily_trades == 3
        assert service.max_position_value == 300
        assert service.daily_loss_limit == 100
        assert len(service.target_symbols) > 0

    @pytest.mark.asyncio
    async def test_start_trading_success(self, trading_service, mock_kite_client):
        """Test successful trading start"""
        mock_kite_client.is_connected.return_value = True

        result = await trading_service.start_trading()

        assert result["status"] == "started"
        assert trading_service.is_running is True
        assert "Trading started successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_start_trading_already_running(self, trading_service):
        """Test starting trading when already running"""
        trading_service.is_running = True

        result = await trading_service.start_trading()

        assert result["status"] == "already_running"
        assert "already running" in result["message"]

    @pytest.mark.asyncio
    async def test_start_trading_kite_not_connected(
        self, trading_service, mock_kite_client
    ):
        """Test starting trading when Kite is not connected"""
        mock_kite_client.is_connected.return_value = False

        result = await trading_service.start_trading()

        assert result["status"] == "error"
        assert "Kite client not connected" in result["message"]

    @pytest.mark.asyncio
    async def test_stop_trading(self, trading_service):
        """Test stopping trading"""
        trading_service.is_running = True

        result = await trading_service.stop_trading()

        assert result["status"] == "stopped"
        assert trading_service.is_running is False

    @pytest.mark.asyncio
    async def test_analyze_and_trade_success(
        self, trading_service, mock_crew_manager, mock_kite_client
    ):
        """Test successful analysis and trade execution"""
        trading_service.is_running = True
        trading_service.daily_trades = 0
        trading_service.daily_pnl = 0

        # Mock successful order placement
        mock_kite_client.place_order.return_value = {"order_id": "test_order_123"}

        result = await trading_service.analyze_and_trade()

        assert result["status"] == "success"
        assert result["trades_executed"] >= 0
        mock_crew_manager.analyze_market.assert_called()

    @pytest.mark.asyncio
    async def test_analyze_and_trade_market_closed(
        self, trading_service, mock_market_schedule
    ):
        """Test analysis when market is closed"""
        mock_market_schedule.get_market_status.return_value = {
            "is_open": False,
            "status": "CLOSED",
            "message": "Market is closed",
        }

        result = await trading_service.analyze_and_trade()

        assert result["status"] == "market_closed"
        assert "Market is closed" in result["message"]

    @pytest.mark.asyncio
    async def test_analyze_and_trade_daily_limit_reached(self, trading_service):
        """Test analysis when daily trade limit is reached"""
        trading_service.daily_trades = trading_service.max_daily_trades

        result = await trading_service.analyze_and_trade()

        assert result["status"] == "daily_limit_reached"
        assert "Daily trade limit reached" in result["message"]

    @pytest.mark.asyncio
    async def test_analyze_and_trade_loss_limit_reached(self, trading_service):
        """Test analysis when daily loss limit is reached"""
        trading_service.daily_pnl = -trading_service.daily_loss_limit - 10

        result = await trading_service.analyze_and_trade()

        assert result["status"] == "loss_limit_reached"
        assert "Daily loss limit reached" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_trade_buy_order(self, trading_service, mock_kite_client):
        """Test executing a buy order"""
        decision = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "confidence": 0.8,
            "target_price": 2550.0,
            "stop_loss": 2450.0,
            "quantity": 1,
            "reasoning": "Strong technical indicators",
        }

        mock_kite_client.place_order.return_value = {"order_id": "test_order_123"}

        result = await trading_service._execute_trade(decision)

        assert result["status"] == "success"
        assert result["order_id"] == "test_order_123"
        assert result["symbol"] == "RELIANCE"
        assert result["action"] == "BUY"

    @pytest.mark.asyncio
    async def test_execute_trade_insufficient_confidence(self, trading_service):
        """Test skipping trade due to insufficient confidence"""
        decision = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "confidence": 0.5,  # Below threshold
            "target_price": 2550.0,
            "stop_loss": 2450.0,
            "quantity": 1,
            "reasoning": "Weak signals",
        }

        result = await trading_service._execute_trade(decision)

        assert result["status"] == "skipped"
        assert "Confidence too low" in result["reason"]

    @pytest.mark.asyncio
    async def test_execute_trade_position_limit_reached(
        self, trading_service, mock_kite_client
    ):
        """Test skipping trade when position limit is reached"""
        # Mock existing positions
        mock_kite_client.get_positions.return_value = {
            "net": [
                {"tradingsymbol": "RELIANCE", "quantity": 1},
                {"tradingsymbol": "TCS", "quantity": 1},
                {"tradingsymbol": "INFY", "quantity": 1},
            ]
        }

        decision = {
            "symbol": "HDFC",
            "action": "BUY",
            "confidence": 0.8,
            "target_price": 1500.0,
            "stop_loss": 1450.0,
            "quantity": 1,
            "reasoning": "Good opportunity",
        }

        result = await trading_service._execute_trade(decision)

        assert result["status"] == "skipped"
        assert "Maximum positions reached" in result["reason"]

    def test_calculate_position_size(self, trading_service):
        """Test position size calculation"""
        # Test normal case
        size = trading_service._calculate_position_size(2500.0, 0.8)
        assert size > 0
        assert size <= trading_service.max_position_value // 2500.0

        # Test high confidence
        high_conf_size = trading_service._calculate_position_size(2500.0, 0.9)
        normal_conf_size = trading_service._calculate_position_size(2500.0, 0.7)
        assert high_conf_size >= normal_conf_size

    def test_should_trade_symbol(self, trading_service):
        """Test symbol trading eligibility"""
        # Valid symbol
        assert trading_service._should_trade_symbol("RELIANCE") is True

        # Invalid symbol (not in target list)
        assert trading_service._should_trade_symbol("INVALID") is False

    @pytest.mark.asyncio
    async def test_check_risk_limits_normal(self, trading_service, mock_kite_client):
        """Test risk limits check under normal conditions"""
        mock_kite_client.get_positions.return_value = {"net": []}

        result = await trading_service._check_risk_limits()

        assert result["within_limits"] is True
        assert result["daily_trades"] == 0
        assert result["daily_pnl"] == 0

    @pytest.mark.asyncio
    async def test_check_risk_limits_exceeded(self, trading_service):
        """Test risk limits when exceeded"""
        trading_service.daily_trades = trading_service.max_daily_trades + 1
        trading_service.daily_pnl = -trading_service.daily_loss_limit - 10

        result = await trading_service._check_risk_limits()

        assert result["within_limits"] is False
        assert "Daily trade limit exceeded" in result["violations"]
        assert "Daily loss limit exceeded" in result["violations"]

    def test_get_status(self, trading_service):
        """Test getting service status"""
        status = trading_service.get_status()

        assert "is_running" in status
        assert "trading_enabled" in status
        assert "daily_trades" in status
        assert "daily_pnl" in status
        assert "max_daily_trades" in status
        assert "max_position_value" in status
        assert "daily_loss_limit" in status

    @pytest.mark.asyncio
    async def test_emergency_stop(self, trading_service, mock_kite_client):
        """Test emergency stop functionality"""
        # Mock positions to close
        mock_kite_client.get_positions.return_value = {
            "net": [
                {
                    "tradingsymbol": "RELIANCE",
                    "quantity": 1,
                    "exchange": "NSE",
                    "product": "MIS",
                }
            ]
        }

        mock_kite_client.place_order.return_value = {"order_id": "emergency_order_123"}

        result = await trading_service.emergency_stop("Test emergency")

        assert result["status"] == "emergency_stop_executed"
        assert result["reason"] == "Test emergency"
        assert trading_service.is_running is False
        assert trading_service.trading_enabled is False

    @pytest.mark.asyncio
    async def test_error_handling_in_analyze_and_trade(
        self, trading_service, mock_crew_manager
    ):
        """Test error handling in analyze_and_trade method"""
        # Mock an exception in crew manager
        mock_crew_manager.analyze_market.side_effect = Exception("Analysis failed")

        result = await trading_service.analyze_and_trade()

        assert result["status"] == "error"
        assert "Analysis failed" in result["message"]

    def test_reset_daily_counters(self, trading_service):
        """Test resetting daily counters"""
        trading_service.daily_trades = 5
        trading_service.daily_pnl = -50

        trading_service._reset_daily_counters()

        assert trading_service.daily_trades == 0
        assert trading_service.daily_pnl == 0
