import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.risk_manager.agent import RiskManagerAgent
from agents.risk_manager.risk_metrics import RiskMetrics
from agents.risk_manager.position_sizing import PositionSizer
from agents.risk_manager.stop_loss_manager import StopLossManager


class TestRiskManagerAgent:
    """Test suite for Risk Manager Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create a Risk Manager agent instance"""
        return RiskManagerAgent()
        
    @pytest.fixture
    def mock_kite_client(self, mock_kite_client):
        """Use the mock Kite client from conftest"""
        return mock_kite_client
        
    @pytest.fixture
    def sample_position(self, sample_position):
        """Use sample position from conftest"""
        return sample_position
        
    @pytest.fixture
    def risk_parameters(self, risk_parameters):
        """Use risk parameters from conftest"""
        return risk_parameters
        
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.name == "Risk Manager"
        assert agent.role == "Chief Risk Officer"
        assert agent.goal.startswith("Monitor and manage")
        assert hasattr(agent, "risk_metrics")
        assert hasattr(agent, "position_sizer")
        assert hasattr(agent, "stop_loss_manager")
        
    @pytest.mark.asyncio
    async def test_evaluate_trade_risk(self, agent, sample_trade_signal, mock_kite_client):
        """Test trade risk evaluation"""
        with patch('agents.risk_manager.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Risk evaluation complete",
                "risk_assessment": {
                    "approved": True,
                    "risk_score": 0.3,
                    "position_size": 100,
                    "stop_loss": 2450.0,
                    "take_profit": 2600.0,
                    "max_loss_amount": 5000,
                    "risk_reward_ratio": 3.0
                }
            })
            
            result = await agent.evaluate_trade_risk(sample_trade_signal)
            
            assert result is not None
            assert "risk_assessment" in result
            assert "approved" in result["risk_assessment"]
            assert "risk_score" in result["risk_assessment"]
            assert result["risk_assessment"]["risk_score"] >= 0
            
    @pytest.mark.asyncio
    async def test_calculate_position_size(self, agent, mock_kite_client, risk_parameters):
        """Test position size calculation"""
        with patch('agents.risk_manager.agent.kite_client', mock_kite_client):
            # Test with valid parameters
            position_size = await agent.calculate_position_size(
                symbol="RELIANCE",
                entry_price=2500.0,
                stop_loss=2450.0,
                account_balance=1000000,
                risk_percent=2.0
            )
            
            assert position_size > 0
            # Max risk is 2% of 1000000 = 20000
            # Risk per share = 50
            # Position size should be 400 shares
            assert position_size == 400
            
    @pytest.mark.asyncio
    async def test_monitor_portfolio_risk(self, agent, mock_kite_client):
        """Test portfolio risk monitoring"""
        with patch('agents.risk_manager.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Portfolio risk analysis complete",
                "portfolio_metrics": {
                    "total_exposure": 250000,
                    "portfolio_var": 12500,
                    "beta": 1.2,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 8.5,
                    "risk_level": "moderate",
                    "diversification_score": 0.7
                }
            })
            
            result = await agent.monitor_portfolio_risk()
            
            assert result is not None
            assert "portfolio_metrics" in result
            assert "risk_level" in result["portfolio_metrics"]
            assert result["portfolio_metrics"]["risk_level"] in ["low", "moderate", "high", "critical"]
            
    @pytest.mark.asyncio
    async def test_stop_loss_management(self, agent, sample_position):
        """Test stop loss management"""
        stop_loss_manager = agent.stop_loss_manager
        
        # Test initial stop loss
        initial_stop = stop_loss_manager.calculate_stop_loss(
            entry_price=2500.0,
            position_type="long",
            volatility=0.02,
            support_level=2450.0
        )
        
        assert initial_stop < 2500.0
        assert initial_stop >= 2450.0
        
        # Test trailing stop
        trailing_stop = stop_loss_manager.update_trailing_stop(
            current_price=2600.0,
            entry_price=2500.0,
            current_stop=2450.0,
            position_type="long",
            trailing_percent=1.5
        )
        
        assert trailing_stop > 2450.0
        assert trailing_stop < 2600.0
        
    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, agent, mock_kite_client):
        """Test risk metrics calculation"""
        risk_metrics = agent.risk_metrics
        
        with patch('agents.risk_manager.agent.kite_client', mock_kite_client):
            # Test VaR calculation
            historical_returns = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.005]
            var_95 = risk_metrics.calculate_var(historical_returns, confidence=0.95)
            
            assert var_95 < 0  # VaR should be negative
            
            # Test Sharpe ratio
            returns = [0.01, 0.02, -0.01, 0.015, 0.005]
            risk_free_rate = 0.05 / 252  # Daily risk-free rate
            sharpe = risk_metrics.calculate_sharpe_ratio(returns, risk_free_rate)
            
            assert isinstance(sharpe, float)
            
    @pytest.mark.asyncio
    async def test_check_risk_limits(self, agent, mock_kite_client, risk_parameters):
        """Test risk limit checking"""
        with patch('agents.risk_manager.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Risk limits checked",
                "limits_status": {
                    "position_size_limit": {"current": 8.5, "limit": 10.0, "breached": False},
                    "portfolio_risk_limit": {"current": 15.0, "limit": 20.0, "breached": False},
                    "daily_loss_limit": {"current": 2.5, "limit": 5.0, "breached": False},
                    "correlation_limit": {"current": 0.6, "limit": 0.8, "breached": False}
                },
                "overall_status": "within_limits"
            })
            
            result = await agent.check_risk_limits(risk_parameters)
            
            assert result is not None
            assert "limits_status" in result
            assert "overall_status" in result
            assert result["overall_status"] in ["within_limits", "warning", "breached"]
            
    @pytest.mark.asyncio
    async def test_hedge_recommendations(self, agent, mock_kite_client):
        """Test hedge recommendations"""
        with patch('agents.risk_manager.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Hedge analysis complete",
                "hedge_recommendations": [
                    {
                        "instrument": "NIFTY_PUT_18000",
                        "hedge_type": "protective_put",
                        "cost": 5000,
                        "protection_level": 0.85,
                        "recommended": True
                    }
                ],
                "portfolio_beta": 1.3,
                "hedge_required": True
            })
            
            result = await agent.recommend_hedges()
            
            assert result is not None
            assert "hedge_recommendations" in result
            assert "hedge_required" in result
            
    @pytest.mark.asyncio
    async def test_correlation_analysis(self, agent):
        """Test portfolio correlation analysis"""
        agent._execute = AsyncMock(return_value={
            "output": "Correlation analysis complete",
            "correlation_matrix": {
                "RELIANCE": {"TCS": 0.7, "INFY": 0.65, "HDFC": 0.5},
                "TCS": {"RELIANCE": 0.7, "INFY": 0.85, "HDFC": 0.45},
                "INFY": {"RELIANCE": 0.65, "TCS": 0.85, "HDFC": 0.4},
                "HDFC": {"RELIANCE": 0.5, "TCS": 0.45, "INFY": 0.4}
            },
            "portfolio_correlation": 0.65,
            "diversification_benefit": 0.15
        })
        
        positions = ["RELIANCE", "TCS", "INFY", "HDFC"]
        result = await agent.analyze_correlation(positions)
        
        assert result is not None
        assert "correlation_matrix" in result
        assert "portfolio_correlation" in result
        
    @pytest.mark.asyncio
    async def test_stress_testing(self, agent, mock_kite_client):
        """Test portfolio stress testing"""
        with patch('agents.risk_manager.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Stress test complete",
                "scenarios": {
                    "market_crash_10": {"portfolio_loss": -85000, "loss_percent": -8.5},
                    "market_crash_20": {"portfolio_loss": -170000, "loss_percent": -17.0},
                    "volatility_spike": {"portfolio_loss": -45000, "loss_percent": -4.5},
                    "liquidity_crisis": {"portfolio_loss": -60000, "loss_percent": -6.0}
                },
                "worst_case_loss": -170000,
                "recommendations": ["Reduce leverage", "Add protective puts"]
            })
            
            result = await agent.run_stress_test()
            
            assert result is not None
            assert "scenarios" in result
            assert "worst_case_loss" in result
            assert result["worst_case_loss"] < 0
            
    def test_position_sizer(self, agent):
        """Test PositionSizer utility"""
        position_sizer = agent.position_sizer
        
        # Test Kelly Criterion
        win_rate = 0.6
        avg_win = 1.5
        avg_loss = 1.0
        kelly_percent = position_sizer.kelly_criterion(win_rate, avg_win, avg_loss)
        
        assert kelly_percent > 0
        assert kelly_percent < 1  # Should be fractional
        
        # Test with volatility adjustment
        base_size = 100
        current_vol = 0.03
        target_vol = 0.02
        adjusted_size = position_sizer.adjust_for_volatility(
            base_size, current_vol, target_vol
        )
        
        assert adjusted_size < base_size  # Should reduce size due to higher vol
        
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in risk manager"""
        # Test with invalid risk parameters
        agent._execute = AsyncMock(side_effect=ValueError("Invalid risk parameters"))
        
        with pytest.raises(ValueError):
            await agent.evaluate_trade_risk({"invalid": "signal"})