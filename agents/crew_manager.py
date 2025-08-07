import asyncio
from typing import Any

from crewai import Crew, Process, Task
from langchain_openai import ChatOpenAI
from loguru import logger

from app.core.config import get_settings

from .data_processor.agent import DataProcessorAgent
from .market_analyst.agent import MarketAnalystAgent
from .risk_manager.agent import RiskManagerAgent
from .sentiment_analyst.agent import SentimentAnalystAgent
from .technical_indicator.agent import TechnicalIndicatorAgent
from .trader.agent import TraderAgent


class CrewManager:
    def __init__(self):
        self.settings = get_settings()

        self.llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL,
            api_key=self.settings.OPENAI_API_KEY,
            temperature=0.1,
        )

        # Initialize agents
        self.market_analyst = MarketAnalystAgent(self.llm)
        self.risk_manager = RiskManagerAgent(
            self.llm, capital=100000
        )  # Default 100k capital
        self.trader = TraderAgent(self.llm)
        self.data_processor = DataProcessorAgent(self.llm)
        self.technical_indicator = TechnicalIndicatorAgent()
        self.sentiment_analyst = SentimentAnalystAgent(self.settings.OPENAI_API_KEY)

        logger.info("CrewManager initialized with all agents")

    async def analyze_trade_opportunity(self, symbol: str) -> dict[str, Any]:
        """Analyze a trading opportunity using the crew of agents"""
        try:
            # Create tasks for the crew
            data_task = Task(
                description=f"Collect and process market data for {symbol}",
                agent=self.data_processor.agent,
                expected_output="Processed market data with technical indicators",
            )

            analysis_task = Task(
                description=f"Analyze market conditions and trends for {symbol}",
                agent=self.market_analyst.agent,
                expected_output="Market analysis with trade recommendations",
                context=[data_task],
            )

            risk_task = Task(
                description=f"Assess risk factors for trading {symbol}",
                agent=self.risk_manager.agent,
                expected_output="Risk assessment with position sizing recommendations",
                context=[data_task, analysis_task],
            )

            trade_task = Task(
                description=f"Make final trading decision for {symbol}",
                agent=self.trader.agent,
                expected_output="Final trade decision with entry/exit points",
                context=[data_task, analysis_task, risk_task],
            )

            # Create and run the crew
            crew = Crew(
                agents=[
                    self.data_processor.agent,
                    self.market_analyst.agent,
                    self.risk_manager.agent,
                    self.trader.agent,
                ],
                tasks=[data_task, analysis_task, risk_task, trade_task],
                process=Process.sequential,
                verbose=True,
            )

            # Execute the crew
            result = await asyncio.to_thread(crew.kickoff)

            return {
                "symbol": symbol,
                "recommended": "BUY" in str(result).upper()
                or "LONG" in str(result).upper(),
                "analysis": str(result),
                "timestamp": asyncio.get_event_loop().time(),
            }

        except Exception as e:
            logger.error(f"Error in crew analysis: {str(e)}")
            return {
                "symbol": symbol,
                "recommended": False,
                "analysis": f"Error in analysis: {str(e)}",
                "timestamp": asyncio.get_event_loop().time(),
            }

    async def monitor_positions(self, positions: list) -> dict[str, Any]:
        """Monitor existing positions and provide recommendations"""
        try:
            monitoring_tasks = []

            for position in positions:
                task = Task(
                    description=f"Monitor position for {position['symbol']} and provide recommendations",
                    agent=self.risk_manager.agent,
                    expected_output="Position monitoring recommendations",
                )
                monitoring_tasks.append(task)

            if monitoring_tasks:
                crew = Crew(
                    agents=[self.risk_manager.agent, self.market_analyst.agent],
                    tasks=monitoring_tasks,
                    process=Process.parallel,
                    verbose=True,
                )

                result = await asyncio.to_thread(crew.kickoff)
                return {"recommendations": str(result)}

            return {"recommendations": "No positions to monitor"}

        except Exception as e:
            logger.error(f"Error in position monitoring: {str(e)}")
            return {"recommendations": f"Error: {str(e)}"}

    async def analyze_with_indicators(
        self, symbol: str, data: Any, timeframe: str = "5min"
    ) -> dict[str, Any]:
        """Analyze a symbol using technical indicators and generate signals"""
        try:
            # Get technical analysis from indicator agent
            indicator_analysis = await self.technical_indicator.analyze_symbol(
                symbol=symbol, data=data, timeframe=timeframe
            )

            # Create task for market analyst to review indicator signals
            indicator_task = Task(
                description=f"Review technical indicator analysis for {symbol}: {indicator_analysis['signals']}",
                agent=self.market_analyst.agent,
                expected_output="Market opinion on technical signals",
            )

            # Create task for risk manager to assess indicator-based risks
            risk_task = Task(
                description=f"Assess risk based on technical indicators for {symbol}",
                agent=self.risk_manager.agent,
                expected_output="Risk assessment for indicator-based trading",
                context=[indicator_task],
            )

            # Create task for trader to make decision
            trade_task = Task(
                description=f"Make trading decision for {symbol} based on technical indicators",
                agent=self.trader.agent,
                expected_output="Trading decision with entry/exit points",
                context=[indicator_task, risk_task],
            )

            # Create crew for technical analysis
            crew = Crew(
                agents=[
                    self.market_analyst.agent,
                    self.risk_manager.agent,
                    self.trader.agent,
                ],
                tasks=[indicator_task, risk_task, trade_task],
                process=Process.sequential,
                verbose=True,
            )

            # Execute crew analysis
            crew_result = await asyncio.to_thread(crew.kickoff)

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "technical_indicators": indicator_analysis,
                "crew_analysis": str(crew_result),
                "recommended_action": self._extract_action(str(crew_result)),
                "timestamp": asyncio.get_event_loop().time(),
            }

        except Exception as e:
            logger.error(f"Error in indicator analysis: {str(e)}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time(),
            }

    async def get_real_time_signals(
        self, symbols: list, timeframe: str = "5min"
    ) -> dict[str, Any]:
        """Get real-time signals for multiple symbols"""
        try:
            signals = {}

            # Get signals for each symbol
            for symbol in symbols:
                signal = await self.technical_indicator.get_real_time_signals(
                    symbol=symbol, timeframe=timeframe
                )
                signals[symbol] = signal

            # Filter high-confidence signals
            high_confidence_signals = {
                symbol: signal
                for symbol, signal in signals.items()
                if signal.get("confidence", 0) > 0.7
            }

            return {
                "all_signals": signals,
                "high_confidence": high_confidence_signals,
                "timestamp": asyncio.get_event_loop().time(),
            }

        except Exception as e:
            logger.error(f"Error getting real-time signals: {str(e)}")
            return {"error": str(e)}

    def _extract_action(self, analysis: str) -> str:
        """Extract recommended action from analysis text"""
        analysis_upper = analysis.upper()

        if "STRONG BUY" in analysis_upper:
            return "STRONG_BUY"
        elif "BUY" in analysis_upper or "LONG" in analysis_upper:
            return "BUY"
        elif "STRONG SELL" in analysis_upper:
            return "STRONG_SELL"
        elif "SELL" in analysis_upper or "SHORT" in analysis_upper:
            return "SELL"
        else:
            return "HOLD"

    async def analyze_with_sentiment(
        self, symbol: str, include_social: bool = True
    ) -> dict[str, Any]:
        """Analyze a symbol combining technical indicators and sentiment analysis."""
        try:
            # Get sentiment analysis
            sentiment_analysis = await self.sentiment_analyst.analyze_symbol_sentiment(
                symbol=symbol, include_social=include_social, lookback_hours=24
            )

            # Create task for market analyst to review sentiment
            sentiment_task = Task(
                description=f"Review sentiment analysis for {symbol}: Overall score={sentiment_analysis['sentiment_scores']['overall_score']:.2f}",
                agent=self.market_analyst.agent,
                expected_output="Market opinion on sentiment signals",
            )

            # Create task for risk assessment based on sentiment
            risk_task = Task(
                description=f"Assess risk based on sentiment score {sentiment_analysis['sentiment_scores']['overall_score']} for {symbol}",
                agent=self.risk_manager.agent,
                expected_output="Risk assessment incorporating sentiment",
                context=[sentiment_task],
            )

            # Create trading decision task
            trade_task = Task(
                description=f"Make trading decision for {symbol} considering sentiment analysis",
                agent=self.trader.agent,
                expected_output="Trading decision with sentiment consideration",
                context=[sentiment_task, risk_task],
            )

            # Create crew
            crew = Crew(
                agents=[
                    self.market_analyst.agent,
                    self.risk_manager.agent,
                    self.trader.agent,
                ],
                tasks=[sentiment_task, risk_task, trade_task],
                process=Process.sequential,
                verbose=True,
            )

            # Execute crew analysis
            crew_result = await asyncio.to_thread(crew.kickoff)

            return {
                "symbol": symbol,
                "sentiment_analysis": sentiment_analysis,
                "crew_analysis": str(crew_result),
                "recommended_action": self._extract_action(str(crew_result)),
                "sentiment_score": sentiment_analysis["sentiment_scores"][
                    "overall_score"
                ],
                "sentiment_confidence": sentiment_analysis["sentiment_scores"][
                    "confidence"
                ],
                "timestamp": asyncio.get_event_loop().time(),
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time(),
            }

    async def monitor_market_sentiment(
        self, sectors: list | None = None, top_symbols: list | None = None
    ) -> dict[str, Any]:
        """Monitor overall market sentiment across sectors and symbols."""
        try:
            # Get market sentiment analysis
            market_sentiment = await self.sentiment_analyst.analyze_market_sentiment(
                sectors=sectors, top_symbols=top_symbols
            )

            # Create task for market interpretation
            market_task = Task(
                description=f"Interpret market sentiment: Score={market_sentiment['market_sentiment']['score']:.2f}, Mood={market_sentiment['market_mood']}",
                agent=self.market_analyst.agent,
                expected_output="Market outlook based on sentiment",
            )

            # Create risk assessment task
            risk_task = Task(
                description=f"Assess market risk with sentiment score {market_sentiment['market_sentiment']['score']}",
                agent=self.risk_manager.agent,
                expected_output="Market risk assessment",
            )

            # Execute tasks
            crew = Crew(
                agents=[self.market_analyst.agent, self.risk_manager.agent],
                tasks=[market_task, risk_task],
                process=Process.sequential,
                verbose=True,
            )

            crew_result = await asyncio.to_thread(crew.kickoff)

            return {
                "market_sentiment": market_sentiment,
                "crew_interpretation": str(crew_result),
                "market_mood": market_sentiment["market_mood"],
                "recommendations": self._generate_market_recommendations(
                    market_sentiment
                ),
                "timestamp": asyncio.get_event_loop().time(),
            }

        except Exception as e:
            logger.error(f"Error monitoring market sentiment: {str(e)}")
            return {"error": str(e)}

    async def generate_sentiment_report(self, symbols: list[str]) -> dict[str, Any]:
        """Generate comprehensive sentiment report for given symbols."""
        try:
            # Generate daily report
            report = await self.sentiment_analyst.generate_daily_report(symbols)

            # Have crew review and add insights
            review_task = Task(
                description="Review sentiment report and provide trading insights",
                agent=self.trader.agent,
                expected_output="Key trading opportunities from sentiment report",
            )

            crew = Crew(
                agents=[self.trader.agent],
                tasks=[review_task],
                process=Process.sequential,
                verbose=True,
            )

            insights = await asyncio.to_thread(crew.kickoff)

            report["crew_insights"] = str(insights)
            return report

        except Exception as e:
            logger.error(f"Error generating sentiment report: {str(e)}")
            return {"error": str(e)}

    def _generate_market_recommendations(
        self, market_sentiment: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on market sentiment."""
        recommendations = []

        market_score = market_sentiment["market_sentiment"]["score"]
        market_mood = market_sentiment["market_mood"]

        if market_score > 0.5:
            recommendations.append("Consider increasing long exposure")
            recommendations.append("Look for momentum plays")
        elif market_score < -0.5:
            recommendations.append("Consider defensive positions")
            recommendations.append("Tighten stop losses")

        if market_mood == "risk_on":
            recommendations.append("Focus on growth sectors")
        elif market_mood == "risk_off":
            recommendations.append("Focus on defensive sectors")

        return recommendations
