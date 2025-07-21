import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.market_analyst.agent import MarketAnalystAgent
from agents.market_analyst.tools import MarketDataTool, MarketAnalysisTool


class TestMarketAnalystAgent:
    """Test suite for Market Analyst Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create a Market Analyst agent instance"""
        return MarketAnalystAgent()
        
    @pytest.fixture
    def mock_kite_client(self, mock_kite_client):
        """Use the mock Kite client from conftest"""
        return mock_kite_client
        
    @pytest.fixture
    def sample_market_data(self, sample_market_data):
        """Use sample market data from conftest"""
        return sample_market_data
        
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.name == "Market Analyst"
        assert agent.role == "Senior Market Analyst"
        assert agent.goal.startswith("Analyze market conditions")
        assert len(agent.tools) > 0
        assert agent.max_iter == 5
        
    @pytest.mark.asyncio
    async def test_analyze_market_conditions(self, agent, mock_kite_client, sample_market_data):
        """Test market condition analysis"""
        with patch('agents.market_analyst.agent.kite_client', mock_kite_client):
            # Mock the agent's task execution
            agent._execute = AsyncMock(return_value={
                "output": "Market analysis complete",
                "trend": "bullish",
                "confidence": 0.85,
                "key_levels": {
                    "support": 2450,
                    "resistance": 2550
                }
            })
            
            result = await agent.analyze_market_conditions("RELIANCE")
            
            assert result is not None
            assert "trend" in result
            assert "confidence" in result
            assert result["confidence"] >= 0 and result["confidence"] <= 1
            
    @pytest.mark.asyncio
    async def test_identify_trends(self, agent, mock_kite_client):
        """Test trend identification"""
        with patch('agents.market_analyst.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Trend analysis complete",
                "primary_trend": "uptrend",
                "trend_strength": "strong",
                "trend_duration": "short-term",
                "confirmation_indicators": ["volume", "momentum"]
            })
            
            result = await agent.identify_trends(["RELIANCE", "TCS", "INFY"])
            
            assert result is not None
            assert "primary_trend" in result
            assert result["primary_trend"] in ["uptrend", "downtrend", "sideways"]
            
    @pytest.mark.asyncio
    async def test_detect_patterns(self, agent, mock_kite_client):
        """Test pattern detection"""
        with patch('agents.market_analyst.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Pattern detection complete",
                "patterns": [
                    {
                        "name": "Head and Shoulders",
                        "reliability": 0.75,
                        "target_price": 2600
                    },
                    {
                        "name": "Support Break",
                        "reliability": 0.8,
                        "target_price": 2400
                    }
                ]
            })
            
            result = await agent.detect_patterns("RELIANCE", "5minute")
            
            assert result is not None
            assert "patterns" in result
            assert len(result["patterns"]) > 0
            assert all("name" in p and "reliability" in p for p in result["patterns"])
            
    @pytest.mark.asyncio
    async def test_volume_analysis(self, agent, mock_kite_client):
        """Test volume analysis"""
        with patch('agents.market_analyst.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Volume analysis complete",
                "volume_trend": "increasing",
                "volume_strength": "above_average",
                "unusual_activity": False,
                "institutional_interest": "moderate"
            })
            
            result = await agent.analyze_volume_profile("RELIANCE")
            
            assert result is not None
            assert "volume_trend" in result
            assert "volume_strength" in result
            assert result["volume_trend"] in ["increasing", "decreasing", "stable"]
            
    @pytest.mark.asyncio
    async def test_market_sentiment(self, agent):
        """Test market sentiment analysis"""
        agent._execute = AsyncMock(return_value={
            "output": "Sentiment analysis complete",
            "overall_sentiment": "bullish",
            "sentiment_score": 0.7,
            "market_phase": "accumulation",
            "risk_level": "moderate"
        })
        
        result = await agent.analyze_market_sentiment()
        
        assert result is not None
        assert "overall_sentiment" in result
        assert "sentiment_score" in result
        assert result["sentiment_score"] >= -1 and result["sentiment_score"] <= 1
        
    @pytest.mark.asyncio
    async def test_support_resistance_levels(self, agent, mock_kite_client):
        """Test support and resistance level calculation"""
        with patch('agents.market_analyst.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Support/Resistance analysis complete",
                "levels": {
                    "major_resistance": [2600, 2650],
                    "minor_resistance": [2550],
                    "major_support": [2400, 2350],
                    "minor_support": [2450],
                    "pivot_point": 2500
                }
            })
            
            result = await agent.calculate_support_resistance("RELIANCE")
            
            assert result is not None
            assert "levels" in result
            assert "major_support" in result["levels"]
            assert "major_resistance" in result["levels"]
            
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in agent"""
        # Test with invalid symbol
        agent._execute = AsyncMock(side_effect=Exception("API Error"))
        
        with pytest.raises(Exception):
            await agent.analyze_market_conditions("INVALID_SYMBOL")
            
    def test_market_data_tool(self, mock_kite_client):
        """Test MarketDataTool"""
        tool = MarketDataTool(kite_client=mock_kite_client)
        
        # Test tool properties
        assert tool.name == "market_data_fetcher"
        assert tool.description is not None
        
    def test_market_analysis_tool(self):
        """Test MarketAnalysisTool"""
        tool = MarketAnalysisTool()
        
        # Test tool properties
        assert tool.name == "market_analyzer"
        assert tool.description is not None
        
    @pytest.mark.asyncio
    async def test_multi_timeframe_analysis(self, agent, mock_kite_client):
        """Test multi-timeframe analysis"""
        with patch('agents.market_analyst.agent.kite_client', mock_kite_client):
            agent._execute = AsyncMock(return_value={
                "output": "Multi-timeframe analysis complete",
                "timeframes": {
                    "5min": {"trend": "up", "strength": 0.8},
                    "15min": {"trend": "up", "strength": 0.7},
                    "1hour": {"trend": "sideways", "strength": 0.5},
                    "daily": {"trend": "up", "strength": 0.9}
                },
                "alignment": "partial",
                "recommendation": "wait_for_better_alignment"
            })
            
            result = await agent.multi_timeframe_analysis("RELIANCE")
            
            assert result is not None
            assert "timeframes" in result
            assert "alignment" in result
            assert len(result["timeframes"]) > 0
            
    @pytest.mark.asyncio
    async def test_market_breadth_analysis(self, agent):
        """Test market breadth analysis"""
        agent._execute = AsyncMock(return_value={
            "output": "Market breadth analysis complete",
            "advancing_stocks": 150,
            "declining_stocks": 100,
            "advance_decline_ratio": 1.5,
            "new_highs": 25,
            "new_lows": 10,
            "market_strength": "positive"
        })
        
        result = await agent.analyze_market_breadth()
        
        assert result is not None
        assert "advance_decline_ratio" in result
        assert "market_strength" in result
        assert result["advance_decline_ratio"] > 0