"""Test quality-aware agent functionality."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from agents.base_quality_aware_agent import (
    BaseQualityAwareAgent,
    DataQualityLevel,
    TradingMode,
)
from agents.coordinator.agent import AgentType, CoordinatorAgent, TradingOpportunity
from agents.market_analyst.agent import MarketAnalystAgent
from agents.risk_manager.enhanced_agent import EnhancedRiskManagerAgent
from agents.sentiment_analyst.agent import SentimentAnalystAgent
from agents.technical_indicator.agent import TechnicalIndicatorAgent
from agents.trade_executor.agent import TradeExecutorAgent, TradeSignal


class TestQualityAwareAgents:
    """Test suite for quality-aware agent functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = Mock()
        mock.generate.return_value = "Test response"
        return mock

    @pytest.fixture
    def mock_data_source(self):
        """Create a mock data source integration."""
        mock = AsyncMock()
        mock.aggregate_data.return_value = {
            "source": "test_source",
            "quality": 0.85,
            "data": {
                "current_price": 100.0,
                "bid": 99.5,
                "ask": 100.5,
                "volume": 10000,
            },
        }
        return mock

    def test_base_agent_quality_levels(self):
        """Test base agent quality level functionality."""
        agent = BaseQualityAwareAgent()

        # Test quality level determination
        assert agent.get_trading_mode(DataQualityLevel.HIGH) == TradingMode.AGGRESSIVE
        assert agent.get_trading_mode(DataQualityLevel.MEDIUM) == TradingMode.NORMAL
        assert agent.get_trading_mode(DataQualityLevel.LOW) == TradingMode.CONSERVATIVE
        assert (
            agent.get_trading_mode(DataQualityLevel.CRITICAL) == TradingMode.EXIT_ONLY
        )

    @pytest.mark.asyncio
    async def test_market_analyst_quality_aware_analysis(self, mock_llm):
        """Test market analyst with quality-aware analysis."""
        agent = MarketAnalystAgent(mock_llm)

        # Mock the quality-weighted data method
        with patch.object(agent, "get_quality_weighted_data") as mock_get_data:
            mock_get_data.return_value = (
                {"current_price": 100.0, "volume": 10000},
                0.9,  # High quality
                DataQualityLevel.HIGH,
            )

            # Test analyze_with_quality method
            result = await agent.analyze_with_quality("AAPL")

            assert "quality_level" in result
            assert result["quality_level"] == DataQualityLevel.HIGH.value
            assert result["quality_score"] == 0.9
            assert result["analysis_type"] == "full"

    @pytest.mark.asyncio
    async def test_sentiment_analyst_multi_source(self, mock_llm):
        """Test sentiment analyst with multi-source aggregation."""
        agent = SentimentAnalystAgent(mock_llm)

        # Mock multi-source consensus
        with patch.object(agent, "get_multi_source_consensus") as mock_consensus:
            mock_consensus.return_value = (
                {"sentiment": 0.7, "sources": ["news", "social"], "confidence": 0.85},
                0.85,  # Consensus confidence
            )

            result = await agent.analyze_sentiment_multi_source("AAPL")

            assert "multi_source_sentiment" in result
            assert result["consensus_confidence"] == 0.85
            assert result["sources_count"] == 2

    @pytest.mark.asyncio
    async def test_risk_manager_quality_adjustment(self, mock_llm):
        """Test risk manager with quality-based position sizing."""
        agent = EnhancedRiskManagerAgent(
            llm=mock_llm, capital=100000, max_risk_per_trade=0.02
        )

        # Mock quality data
        with patch.object(agent, "get_quality_weighted_data") as mock_get_data:
            mock_get_data.return_value = (
                {"current_price": 100.0},
                0.7,  # Medium quality
                DataQualityLevel.MEDIUM,
            )

            # Create mock market data
            market_data = pd.DataFrame(
                {
                    "close": [100, 101, 102, 101, 100],
                    "high": [102, 103, 104, 103, 102],
                    "low": [99, 100, 101, 100, 99],
                    "volume": [10000, 11000, 12000, 11000, 10000],
                }
            )

            result = await agent.evaluate_trade_risk(
                symbol="AAPL",
                entry_price=100.0,
                target_price=110.0,
                market_data=market_data,
                confidence=0.8,
            )

            assert "quality_adjusted_risk_score" in result
            assert result["quality_level"] == DataQualityLevel.MEDIUM.value
            assert result["trading_mode"] == TradingMode.NORMAL.value
            # Medium quality should reduce position size
            assert result["position_size"]["risk_percentage"] <= 0.01  # Half of max

    @pytest.mark.asyncio
    async def test_technical_indicator_quality_filtering(self):
        """Test technical indicator agent with quality-based indicator selection."""
        agent = TechnicalIndicatorAgent()

        # Create mock price data
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 101, 100],
                "high": [102, 103, 104, 103, 102],
                "low": [99, 100, 101, 100, 99],
                "close": [101, 102, 103, 102, 101],
                "volume": [10000, 11000, 12000, 11000, 10000],
            }
        )

        # Mock quality assessment
        with patch.object(agent, "_get_quality_weighted_ohlcv") as mock_quality:
            # Test with low quality data
            mock_quality.return_value = (
                data,
                {
                    "overall_score": 0.4,
                    "quality_level": DataQualityLevel.LOW,
                    "consensus_confidence": 0.3,
                    "data_source": "single_source",
                    "has_multi_source": False,
                },
            )

            result = await agent.analyze_symbol("AAPL", data)

            # Low quality should only use basic indicators
            assert len(result["indicators"]) <= 2  # Only SMA and TREND
            assert "warnings" in result
            assert len(result["warnings"]) > 0

    @pytest.mark.asyncio
    async def test_trade_executor_pre_execution_checks(self):
        """Test trade executor with pre-execution quality validation."""
        agent = TradeExecutorAgent(paper_trading=True)

        # Create a trade signal
        signal = TradeSignal(
            symbol="AAPL",
            action="BUY",
            quantity=100,
            order_type="MARKET",
            confidence=0.8,
        )

        # Mock quality check
        with patch.object(agent, "get_quality_weighted_data") as mock_get_data:
            # Test with low quality - should block market order
            mock_get_data.return_value = (
                {"current_price": 100.0, "bid": 99.5, "ask": 100.5},
                0.5,  # Low quality
                DataQualityLevel.LOW,
            )

            result = await agent.execute_trade_with_quality_check(signal)

            assert result["status"] == "rejected"
            assert "Market orders require high data quality" in result["reason"]

    @pytest.mark.asyncio
    async def test_coordinator_quality_orchestration(self, mock_llm):
        """Test coordinator agent orchestrating quality-aware analysis."""
        # Create mock agents
        agents = {
            AgentType.MARKET_ANALYST: Mock(),
            AgentType.TECHNICAL_INDICATOR: Mock(),
            AgentType.SENTIMENT_ANALYST: Mock(),
            AgentType.RISK_MANAGER: Mock(),
        }

        coordinator = CoordinatorAgent(agents)

        # Mock quality assessment
        with patch.object(coordinator, "get_quality_weighted_data") as mock_get_data:
            mock_get_data.return_value = (
                {"current_price": 100.0},
                0.8,  # High quality
                DataQualityLevel.HIGH,
            )

            with patch.object(
                coordinator, "get_multi_source_consensus"
            ) as mock_consensus:
                mock_consensus.return_value = ({"consensus": 0.8}, 0.8)

                # Mock task delegator
                with patch.object(
                    coordinator.task_delegator, "execute_parallel_tasks"
                ) as mock_execute:
                    mock_execute.return_value = [
                        {
                            "status": "success",
                            "agent_type": AgentType.MARKET_ANALYST,
                            "data": {
                                "symbol_analysis": {
                                    "AAPL": {
                                        "confidence": 0.8,
                                        "signal": "BUY",
                                        "expected_return": 0.05,
                                        "risk_score": 0.3,
                                    }
                                }
                            },
                        }
                    ]

                    opportunities = await coordinator.analyze_market_quality_aware(
                        ["AAPL"]
                    )

                    assert len(opportunities) > 0
                    assert opportunities[0].data_quality_score == 0.8
                    assert opportunities[0].quality_level == DataQualityLevel.HIGH.value

    def test_quality_aware_opportunity_ranking(self):
        """Test opportunity ranking with quality as primary factor."""
        coordinator = CoordinatorAgent({})

        # Create test opportunities with different quality levels
        opportunities = [
            TradingOpportunity(
                id="1",
                symbol="AAPL",
                action="BUY",
                confidence=0.7,
                expected_return=0.03,
                risk_score=0.4,
                priority=0.0,
                source_agents=["market", "technical"],
                analysis={},
                timestamp=datetime.now(),
                data_quality_score=0.9,  # High quality
                quality_level="high",
                multi_source_consensus=0.8,
            ),
            TradingOpportunity(
                id="2",
                symbol="GOOGL",
                action="BUY",
                confidence=0.8,
                expected_return=0.05,
                risk_score=0.3,
                priority=0.0,
                source_agents=["market", "technical"],
                analysis={},
                timestamp=datetime.now(),
                data_quality_score=0.5,  # Low quality
                quality_level="low",
                multi_source_consensus=0.4,
            ),
        ]

        ranked = coordinator._rank_quality_aware_opportunities(opportunities)

        # High quality opportunity should rank higher despite lower returns
        assert ranked[0].symbol == "AAPL"
        assert ranked[0].priority > ranked[1].priority


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
