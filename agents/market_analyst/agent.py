from crewai import Agent
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain.llms.base import BaseLLM
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import asyncio
from datetime import datetime

from ..base_quality_aware_agent import BaseQualityAwareAgent, DataQualityLevel, TradingMode
from backend.data_sources.integration import get_data_source_integration


class MarketAnalystAgent(BaseQualityAwareAgent):
    def __init__(self, llm: BaseLLM):
        super().__init__()
        self.llm = llm
        self.agent = self._create_agent()
        
        # Market analyst specific quality thresholds
        self.quality_thresholds = {
            DataQualityLevel.HIGH: 0.85,    # Higher threshold for market analysis
            DataQualityLevel.MEDIUM: 0.65,
            DataQualityLevel.LOW: 0.4
        }
    
    def _create_agent(self) -> Agent:
        return Agent(
            role='Quality-Aware Market Analyst',
            goal='Analyze market conditions with data quality awareness, providing confidence-weighted technical analysis',
            backstory="""You are an experienced market analyst with expertise in multi-source data analysis.
            You understand that data quality directly impacts trading decisions. You analyze markets using
            technical indicators, patterns, and volume, while always considering data reliability.
            You adjust your analysis confidence based on data quality:
            - High quality (>85%): Full technical analysis with high confidence predictions
            - Medium quality (65-85%): Conservative analysis with hedged recommendations  
            - Low quality (40-65%): Basic trend analysis only, no specific price targets
            - Critical quality (<40%): Recommend holding current positions only""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[
                # Add tools for market analysis
                # SerperDevTool(),  # For web search if needed
                # WebsiteSearchTool()  # For specific website search
            ]
        )
    
    async def analyze_with_quality(self, symbol: str) -> Dict[str, Any]:
        """Analyze market with quality-aware data fetching"""
        try:
            # Get quality-weighted data
            quote_data, quality_score, quality_level = await self.get_quality_weighted_data(
                symbol, "quote"
            )
            
            if not quote_data:
                return {
                    "symbol": symbol,
                    "analysis": "Unable to fetch market data",
                    "quality_level": DataQualityLevel.CRITICAL.value,
                    "trading_mode": TradingMode.EMERGENCY.value,
                    "confidence": 0.0
                }
            
            # Get multi-source consensus for important decisions
            consensus_data, consensus_confidence = await self.get_multi_source_consensus(symbol)
            
            # Determine trading mode
            trading_mode = self.get_trading_mode(quality_level)
            
            # Get historical data with quality check
            historical_quality = await self._get_historical_data_quality(symbol)
            
            # Calculate overall confidence
            confidence = self.get_confidence_score(
                quality_score,
                {
                    "consensus": consensus_confidence,
                    "historical": historical_quality
                }
            )
            
            # Perform analysis based on quality level
            analysis = await self._perform_quality_based_analysis(
                symbol, quote_data, consensus_data, quality_level, trading_mode
            )
            
            return {
                "symbol": symbol,
                "analysis": analysis,
                "quote": quote_data,
                "quality_score": quality_score,
                "quality_level": quality_level.value,
                "trading_mode": trading_mode.value,
                "confidence": confidence,
                "data_sources": self._get_data_sources_used(quote_data, consensus_data),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in quality-aware analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "quality_level": DataQualityLevel.CRITICAL.value,
                "trading_mode": TradingMode.EMERGENCY.value
            }
    
    async def _perform_quality_based_analysis(
        self,
        symbol: str,
        quote_data: Dict[str, Any],
        consensus_data: Optional[Dict[str, Any]],
        quality_level: DataQualityLevel,
        trading_mode: TradingMode
    ) -> str:
        """Perform analysis adjusted for data quality"""
        
        if quality_level == DataQualityLevel.HIGH:
            # Full analysis with specific targets
            return await self._perform_full_analysis(symbol, quote_data, consensus_data)
        
        elif quality_level == DataQualityLevel.MEDIUM:
            # Conservative analysis
            return await self._perform_conservative_analysis(symbol, quote_data)
        
        elif quality_level == DataQualityLevel.LOW:
            # Basic trend analysis only
            return await self._perform_basic_analysis(symbol, quote_data)
        
        else:  # CRITICAL
            # Emergency mode
            return self._generate_emergency_analysis(symbol, quote_data)
    
    async def _perform_full_analysis(self, symbol: str, quote_data: Dict, consensus_data: Optional[Dict]) -> str:
        """Full technical analysis with high confidence"""
        analysis = f"HIGH QUALITY ANALYSIS for {symbol}\n\n"
        
        current_price = quote_data.get('current_price', 0)
        change_percent = quote_data.get('change_percent', 0)
        volume = quote_data.get('volume', 0)
        
        analysis += f"Current Price: {current_price:.2f} ({change_percent:+.2f}%)\n"
        analysis += f"Volume: {volume:,}\n\n"
        
        # Price action analysis
        if consensus_data:
            consensus_price = consensus_data.get('_consensus_price', current_price)
            price_variance = consensus_data.get('_price_variance', 0)
            
            analysis += f"Multi-Source Consensus: {consensus_price:.2f}\n"
            analysis += f"Price Agreement: {'Strong' if price_variance < 0.5 else 'Moderate'}\n\n"
        
        # Technical levels
        analysis += "Technical Levels:\n"
        analysis += f"- Resistance: {current_price * 1.02:.2f}\n"
        analysis += f"- Support: {current_price * 0.98:.2f}\n"
        analysis += f"- Stop Loss: {current_price * 0.97:.2f}\n\n"
        
        # Trading recommendation
        if change_percent > 1:
            analysis += "RECOMMENDATION: BUY with momentum\n"
            analysis += f"Entry: {current_price:.2f}, Target: {current_price * 1.015:.2f}"
        elif change_percent < -1:
            analysis += "RECOMMENDATION: SELL on weakness\n"
            analysis += f"Entry: {current_price:.2f}, Target: {current_price * 0.985:.2f}"
        else:
            analysis += "RECOMMENDATION: WAIT for clear direction\n"
        
        return analysis
    
    async def _perform_conservative_analysis(self, symbol: str, quote_data: Dict) -> str:
        """Conservative analysis for medium quality data"""
        analysis = f"CONSERVATIVE ANALYSIS for {symbol} (Medium Data Quality)\n\n"
        
        current_price = quote_data.get('current_price', 0)
        change_percent = quote_data.get('change_percent', 0)
        
        analysis += f"Current Price: {current_price:.2f} ({change_percent:+.2f}%)\n\n"
        
        analysis += "⚠️ Data Quality Warning: Operating in conservative mode\n"
        analysis += "- Reduced position sizes recommended (50% of normal)\n"
        analysis += "- Wider stop losses suggested\n"
        analysis += "- Avoid aggressive entries\n\n"
        
        # Basic trend assessment
        if abs(change_percent) > 2:
            trend = "Strong" if abs(change_percent) > 3 else "Moderate"
            direction = "Bullish" if change_percent > 0 else "Bearish"
            analysis += f"Trend: {trend} {direction}\n"
            analysis += "RECOMMENDATION: Trade with trend but reduce size\n"
        else:
            analysis += "Trend: Neutral/Unclear\n"
            analysis += "RECOMMENDATION: Stay on sidelines\n"
        
        return analysis
    
    async def _perform_basic_analysis(self, symbol: str, quote_data: Dict) -> str:
        """Basic analysis for low quality data"""
        analysis = f"BASIC ANALYSIS for {symbol} (Low Data Quality)\n\n"
        
        current_price = quote_data.get('current_price', 0)
        data_source = quote_data.get('data_source', 'unknown')
        
        analysis += f"Last Known Price: {current_price:.2f}\n"
        analysis += f"Data Source: {data_source}\n\n"
        
        analysis += "⚠️ LOW DATA QUALITY - DEFENSIVE MODE\n"
        analysis += "- No new positions recommended\n"
        analysis += "- Hold existing positions with tight stops\n"
        analysis += "- Wait for data quality improvement\n\n"
        
        analysis += "RECOMMENDATION: HOLD only, no new trades\n"
        
        return analysis
    
    def _generate_emergency_analysis(self, symbol: str, quote_data: Optional[Dict]) -> str:
        """Emergency analysis for critical data quality"""
        analysis = f"EMERGENCY ALERT for {symbol}\n\n"
        
        analysis += "🚨 CRITICAL DATA QUALITY ISSUE\n"
        analysis += "- Data reliability below acceptable threshold\n"
        analysis += "- Unable to perform meaningful analysis\n\n"
        
        analysis += "IMMEDIATE ACTIONS REQUIRED:\n"
        analysis += "1. Exit all speculative positions\n"
        analysis += "2. Hold only core positions\n"
        analysis += "3. Do not initiate new trades\n"
        analysis += "4. Monitor data source recovery\n\n"
        
        if quote_data:
            analysis += f"Last known price: {quote_data.get('current_price', 'N/A')}\n"
        
        analysis += "\nRECOMMENDATION: EMERGENCY EXIT MODE\n"
        
        return analysis
    
    async def _get_historical_data_quality(self, symbol: str) -> float:
        """Assess quality of historical data"""
        try:
            # This would fetch and assess historical data quality
            # For now, return a placeholder
            return 0.8
        except:
            return 0.0
    
    def _get_data_sources_used(self, quote_data: Dict, consensus_data: Optional[Dict]) -> List[str]:
        """Get list of data sources used"""
        sources = []
        
        if quote_data and 'data_source' in quote_data:
            sources.append(quote_data['data_source'])
        
        if consensus_data and '_source_count' in consensus_data:
            sources.append(f"consensus ({consensus_data['_source_count']} sources)")
        
        return sources
    
    async def analyze_multiple_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze multiple symbols with quality awareness"""
        analyses = {}
        
        # Analyze symbols concurrently
        tasks = [self.analyze_with_quality(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                analyses[symbol] = {
                    "error": str(result),
                    "quality_level": DataQualityLevel.CRITICAL.value
                }
            else:
                analyses[symbol] = result
        
        # Summary statistics
        quality_distribution = {
            level.value: sum(1 for a in analyses.values() 
                           if a.get('quality_level') == level.value)
            for level in DataQualityLevel
        }
        
        avg_confidence = sum(
            a.get('confidence', 0) for a in analyses.values()
        ) / len(analyses) if analyses else 0
        
        return {
            "analyses": analyses,
            "summary": {
                "total_symbols": len(symbols),
                "quality_distribution": quality_distribution,
                "average_confidence": avg_confidence,
                "high_quality_symbols": [
                    s for s, a in analyses.items() 
                    if a.get('quality_level') == DataQualityLevel.HIGH.value
                ]
            }
        }