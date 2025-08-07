import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.trade_executor.agent import TradeExecutorAgent
from app.schemas.trading import OrderType, TradeSignal


class TestTradeExecutorAgent:
    """Test suite for Trade Executor Agent"""

    @pytest.fixture
    def agent(self):
        """Create a Trade Executor agent instance"""
        return TradeExecutorAgent()

    @pytest.fixture
    def mock_kite_client(self, mock_kite_client):
        """Use the mock Kite client from conftest"""
        return mock_kite_client

    @pytest.fixture
    def sample_trade_signal(self, sample_trade_signal):
        """Use sample trade signal from conftest"""
        signal = TradeSignal(**sample_trade_signal)
        return signal

    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.name == "Trade Executor"
        assert agent.role == "Senior Trading Specialist"
        assert agent.goal.startswith("Execute trades efficiently")
        assert hasattr(agent, "kite_client")
        assert hasattr(agent, "order_manager")
        assert hasattr(agent, "position_monitor")

    @pytest.mark.asyncio
    async def test_execute_trade(self, agent, sample_trade_signal, mock_kite_client):
        """Test trade execution"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            # Test successful trade execution
            result = await agent.execute_trade(sample_trade_signal)

            assert result is not None
            assert "success" in result
            assert result["success"] is True
            assert "order_id" in result
            assert "execution_price" in result
            assert result["order_id"] is not None

    @pytest.mark.asyncio
    async def test_execute_paper_trade(self, agent, sample_trade_signal):
        """Test paper trading execution"""
        agent.paper_trading_mode = True

        result = await agent.execute_trade(sample_trade_signal)

        assert result is not None
        assert result["success"] is True
        assert "paper_trade" in result
        assert result["paper_trade"] is True
        assert result["order_id"].startswith("PAPER_")

    @pytest.mark.asyncio
    async def test_monitor_positions(self, agent, mock_kite_client):
        """Test position monitoring"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            positions = await agent.monitor_positions()

            assert isinstance(positions, list)
            # Check if monitoring returns position details
            if len(positions) > 0:
                position = positions[0]
                assert "symbol" in position
                assert "quantity" in position
                assert "pnl" in position
                assert "status" in position

    @pytest.mark.asyncio
    async def test_handle_partial_fills(self, agent, mock_kite_client):
        """Test handling of partial order fills"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            # Create a partially filled order
            order = {
                "order_id": str(uuid.uuid4()),
                "symbol": "RELIANCE",
                "quantity": 100,
                "filled_quantity": 50,
                "status": "PARTIAL",
                "order_type": "LIMIT",
                "price": 2500,
            }

            agent.order_manager._execute = AsyncMock(
                return_value={
                    "action": "modify_order",
                    "new_price": 2501,
                    "reason": "Improve fill probability",
                }
            )

            result = await agent.handle_partial_fill(order)

            assert result is not None
            assert "action" in result

    @pytest.mark.asyncio
    async def test_manage_stop_loss(self, agent, mock_kite_client):
        """Test stop loss management"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            position = {
                "symbol": "RELIANCE",
                "quantity": 100,
                "average_price": 2500,
                "current_price": 2550,
                "stop_loss_order_id": None,
            }

            # Test placing initial stop loss
            result = await agent.manage_stop_loss(position, stop_loss_price=2480)

            assert result is not None
            assert "order_id" in result
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_trailing_stop_update(self, agent, mock_kite_client):
        """Test trailing stop loss update"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            position = {
                "symbol": "RELIANCE",
                "quantity": 100,
                "average_price": 2500,
                "current_price": 2600,
                "stop_loss_order_id": "SL123",
                "current_stop_loss": 2480,
            }

            # Price moved up, trailing stop should update
            result = await agent.update_trailing_stop(position, trailing_percent=2.0)

            assert result is not None
            assert "new_stop_loss" in result
            assert result["new_stop_loss"] > position["current_stop_loss"]

    @pytest.mark.asyncio
    async def test_emergency_exit(self, agent, mock_kite_client):
        """Test emergency exit functionality"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            results = await agent.emergency_exit_all_positions(
                reason="Risk limit breach"
            )

            assert len(results) == 2
            for result in results:
                assert "symbol" in result
                assert "success" in result
                assert result["order_type"] == "MARKET"

    @pytest.mark.asyncio
    async def test_order_validation(self, agent):
        """Test order validation"""
        # Test valid order
        valid_order = {
            "symbol": "RELIANCE",
            "quantity": 100,
            "order_type": OrderType.BUY,
            "price": 2500,
        }

        is_valid, message = agent.validate_order(valid_order)
        assert is_valid is True

        # Test invalid order (negative quantity)
        invalid_order = {
            "symbol": "RELIANCE",
            "quantity": -100,
            "order_type": OrderType.BUY,
            "price": 2500,
        }

        is_valid, message = agent.validate_order(invalid_order)
        assert is_valid is False
        assert "quantity" in message.lower()

    @pytest.mark.asyncio
    async def test_execution_analytics(self, agent):
        """Test execution analytics tracking"""
        # Execute several trades
        agent.execution_history = [
            {
                "symbol": "RELIANCE",
                "requested_price": 2500,
                "executed_price": 2501,
                "slippage": -1,
                "execution_time": 0.5,
            },
            {
                "symbol": "TCS",
                "requested_price": 3500,
                "executed_price": 3498,
                "slippage": 2,
                "execution_time": 0.3,
            },
        ]

        analytics = agent.get_execution_analytics()

        assert "average_slippage" in analytics
        assert "average_execution_time" in analytics
        assert "total_trades" in analytics
        assert analytics["total_trades"] == 2

    @pytest.mark.asyncio
    async def test_smart_order_routing(
        self, agent, sample_trade_signal, mock_kite_client
    ):
        """Test smart order routing logic"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            # Large order should be split
            large_signal = sample_trade_signal
            large_signal.quantity = 10000

            agent._execute = AsyncMock(
                return_value={
                    "routing_strategy": "iceberg",
                    "split_orders": 5,
                    "order_ids": [f"ORDER_{i}" for i in range(5)],
                }
            )

            result = await agent.execute_large_order(large_signal)

            assert result is not None
            assert "routing_strategy" in result
            assert "split_orders" in result
            assert result["split_orders"] > 1

    @pytest.mark.asyncio
    async def test_order_retry_mechanism(
        self, agent, sample_trade_signal, mock_kite_client
    ):
        """Test order retry on failure"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            # First attempt fails
            mock_kite_client.place_order = AsyncMock(
                side_effect=[
                    Exception("Network error"),
                    {"order_id": "RETRY_SUCCESS", "status": "success"},
                ]
            )

            result = await agent.execute_trade_with_retry(
                sample_trade_signal, max_retries=3
            )

            assert result is not None
            assert result["success"] is True
            assert "retry_count" in result
            assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_position_exit_strategy(self, agent, mock_kite_client):
        """Test position exit strategy"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            position = {
                "symbol": "RELIANCE",
                "quantity": 100,
                "average_price": 2500,
                "current_price": 2600,
                "pnl": 10000,
                "pnl_percent": 4.0,
            }

            agent._execute = AsyncMock(
                return_value={
                    "exit_strategy": "scaled_exit",
                    "exit_levels": [
                        {"price": 2600, "quantity": 30},
                        {"price": 2620, "quantity": 40},
                        {"price": 2650, "quantity": 30},
                    ],
                }
            )

            result = await agent.plan_position_exit(position)

            assert result is not None
            assert "exit_strategy" in result
            assert "exit_levels" in result
            assert len(result["exit_levels"]) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, agent, mock_kite_client):
        """Test error handling in trade execution"""
        with patch("agents.trade_executor.agent.kite_client", mock_kite_client):
            # Test with API error
            mock_kite_client.place_order = AsyncMock(
                side_effect=Exception("Insufficient funds")
            )

            signal = TradeSignal(
                symbol="RELIANCE", action="BUY", quantity=1000, confidence=0.9
            )

            result = await agent.execute_trade(signal)

            assert result is not None
            assert result["success"] is False
            assert "error" in result
            assert "Insufficient funds" in result["error"]
