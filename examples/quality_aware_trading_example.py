"""
Example demonstrating quality-aware trading with multi-source data.

This example shows how the AlgoHive agents use data quality as a primary
factor in their trading decisions.
"""

import asyncio
from datetime import datetime
import pandas as pd
from typing import Dict, List

# Import quality-aware agents
from agents.market_analyst.agent import MarketAnalystAgent
from agents.sentiment_analyst.agent import SentimentAnalysisAgent
from agents.risk_manager.enhanced_agent import EnhancedRiskManagerAgent
from agents.technical_indicator.agent import TechnicalIndicatorAgent
from agents.trade_executor.agent import TradeExecutorAgent, TradeSignal
from agents.coordinator.agent import CoordinatorAgent, AgentType
from agents.base_quality_aware_agent import DataQualityLevel, TradingMode

# Import multi-source data manager
from backend.data_sources.multi_source_manager import MultiSourceDataManager
from backend.data_sources.integration import get_data_source_integration


async def demonstrate_quality_aware_trading():
    """Demonstrate the quality-aware trading workflow."""
    
    print("=== AlgoHive Quality-Aware Trading Demonstration ===\n")
    
    # Initialize multi-source data manager
    data_manager = MultiSourceDataManager()
    
    # Example symbol
    symbol = "RELIANCE.NSE"
    
    # Step 1: Fetch multi-source data with quality assessment
    print("1. Fetching data from multiple sources...")
    aggregated_data = await data_manager.get_aggregated_data(symbol, "quote")
    
    print(f"   Symbol: {symbol}")
    print(f"   Primary Source: {aggregated_data['source']}")
    print(f"   Data Quality: {aggregated_data['quality']:.1%}")
    print(f"   Consensus Price: ₹{aggregated_data['data']['current_price']:.2f}")
    print(f"   Sources Used: {len(aggregated_data.get('all_sources', []))}")
    
    # Step 2: Determine trading mode based on quality
    quality_level = DataQualityLevel.HIGH if aggregated_data['quality'] > 0.8 else \
                   DataQualityLevel.MEDIUM if aggregated_data['quality'] > 0.6 else \
                   DataQualityLevel.LOW
    
    print(f"\n2. Data Quality Assessment:")
    print(f"   Quality Level: {quality_level.value}")
    print(f"   Trading Mode: {TradingMode.AGGRESSIVE.value if quality_level == DataQualityLevel.HIGH else TradingMode.NORMAL.value}")
    
    # Step 3: Quality-aware market analysis
    print("\n3. Quality-Aware Market Analysis:")
    
    # Mock LLM for demonstration
    class MockLLM:
        def generate(self, prompt):
            return "Bullish trend detected with strong momentum"
    
    mock_llm = MockLLM()
    
    # Initialize agents
    market_analyst = MarketAnalystAgent(mock_llm)
    
    # Demonstrate quality-based analysis adjustments
    if quality_level == DataQualityLevel.HIGH:
        print("   ✓ Full technical analysis with all indicators")
        print("   ✓ Aggressive position sizing allowed")
        print("   ✓ All order types available")
    elif quality_level == DataQualityLevel.MEDIUM:
        print("   ⚠ Conservative analysis with robust indicators only")
        print("   ⚠ Position size reduced by 50%")
        print("   ⚠ Limit orders only (no market orders)")
    else:
        print("   ❌ Basic trend analysis only")
        print("   ❌ Minimal position sizing (25% of normal)")
        print("   ❌ Defensive positioning only")
    
    # Step 4: Risk management with quality adjustment
    print("\n4. Quality-Adjusted Risk Management:")
    
    risk_manager = EnhancedRiskManagerAgent(
        llm=mock_llm,
        capital=1000000,  # 10 lakh capital
        max_risk_per_trade=0.02  # 2% max risk
    )
    
    # Calculate quality-adjusted position size
    base_position_size = 0.02  # 2% of capital
    quality_multiplier = {
        DataQualityLevel.HIGH: 1.0,
        DataQualityLevel.MEDIUM: 0.5,
        DataQualityLevel.LOW: 0.25,
        DataQualityLevel.CRITICAL: 0.0
    }[quality_level]
    
    adjusted_position_size = base_position_size * quality_multiplier
    position_value = 1000000 * adjusted_position_size
    
    print(f"   Base Risk: {base_position_size:.1%} of capital")
    print(f"   Quality Multiplier: {quality_multiplier:.1%}")
    print(f"   Adjusted Position Size: ₹{position_value:,.0f}")
    
    # Step 5: Trade execution with quality checks
    print("\n5. Pre-Execution Quality Validation:")
    
    trade_executor = TradeExecutorAgent(paper_trading=True)
    
    # Create a trade signal
    signal = TradeSignal(
        symbol=symbol,
        action="BUY",
        quantity=int(position_value / aggregated_data['data']['current_price']),
        order_type="MARKET" if quality_level == DataQualityLevel.HIGH else "LIMIT",
        price=aggregated_data['data']['current_price'] if quality_level != DataQualityLevel.HIGH else None,
        confidence=0.8 * quality_multiplier,
        data_quality_score=aggregated_data['quality'],
        quality_level=quality_level.value
    )
    
    print(f"   Order Type: {signal.order_type}")
    print(f"   Quantity: {signal.quantity} shares")
    print(f"   Confidence: {signal.confidence:.1%}")
    
    # Step 6: Multi-source consensus validation
    print("\n6. Multi-Source Consensus Check:")
    
    consensus_data = await data_manager.get_multi_source_consensus(symbol)
    
    if consensus_data:
        print(f"   Price Variance: {consensus_data.get('price_variance', 0):.2%}")
        print(f"   Source Agreement: {consensus_data.get('consensus_score', 0):.1%}")
        print(f"   Reliable Sources: {consensus_data.get('reliable_sources', 0)}/{consensus_data.get('total_sources', 0)}")
    
    # Step 7: Quality-aware decision summary
    print("\n7. Quality-Aware Trading Decision:")
    print(f"   Symbol: {symbol}")
    print(f"   Action: {signal.action}")
    print(f"   Data Quality: {quality_level.value} ({aggregated_data['quality']:.1%})")
    print(f"   Position Size: ₹{position_value:,.0f} ({adjusted_position_size:.1%} of capital)")
    print(f"   Order Type: {signal.order_type}")
    
    if quality_level == DataQualityLevel.HIGH:
        print("   Decision: ✅ EXECUTE - High quality data supports full position")
    elif quality_level == DataQualityLevel.MEDIUM:
        print("   Decision: ⚠️ EXECUTE WITH CAUTION - Reduced size due to medium quality")
    else:
        print("   Decision: ❌ HOLD - Data quality too low for new positions")
    
    print("\n" + "="*50)
    print("Quality-aware trading ensures better risk management by:")
    print("• Adjusting position sizes based on data reliability")
    print("• Restricting order types when quality is uncertain")
    print("• Requiring multi-source consensus for large trades")
    print("• Blocking trades entirely when data quality is critical")


async def demonstrate_coordinator_orchestration():
    """Demonstrate the coordinator orchestrating quality-aware agents."""
    
    print("\n\n=== Coordinator Quality-Aware Orchestration ===\n")
    
    # Mock agents for demonstration
    class MockAgent:
        def __init__(self, name):
            self.name = name
    
    agents = {
        AgentType.MARKET_ANALYST: MockAgent("Market Analyst"),
        AgentType.TECHNICAL_INDICATOR: MockAgent("Technical Indicator"),
        AgentType.SENTIMENT_ANALYST: MockAgent("Sentiment Analyst"),
        AgentType.RISK_MANAGER: MockAgent("Risk Manager")
    }
    
    coordinator = CoordinatorAgent(agents)
    
    symbols = ["RELIANCE.NSE", "TCS.NSE", "INFY.NSE"]
    
    print("Coordinator analyzing multiple symbols with quality awareness:")
    print(f"Symbols: {', '.join(symbols)}")
    print("\nQuality-based task delegation:")
    print("• High quality symbols → Full analysis with all agents")
    print("• Medium quality → Skip sentiment, use conservative indicators")
    print("• Low quality → Basic analysis only, defensive mode")
    print("\nThe coordinator automatically:")
    print("1. Assesses data quality for each symbol")
    print("2. Groups symbols by quality level")
    print("3. Delegates appropriate tasks to specialist agents")
    print("4. Weights results based on data reliability")
    print("5. Ranks opportunities with quality as primary factor")


if __name__ == "__main__":
    # Run the demonstrations
    asyncio.run(demonstrate_quality_aware_trading())
    asyncio.run(demonstrate_coordinator_orchestration())