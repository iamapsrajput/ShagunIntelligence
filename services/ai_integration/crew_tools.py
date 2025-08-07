import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from services.ai_integration.ai_service_manager import AIProvider, AIServiceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketSentimentInput(BaseModel):
    """Input for market sentiment analysis"""

    text: str = Field(description="Text to analyze for market sentiment")
    context: dict[str, Any] | None = Field(
        default=None, description="Additional context for analysis"
    )
    provider: str | None = Field(
        default="openai", description="AI provider to use (openai/anthropic)"
    )


class TradeAnalysisInput(BaseModel):
    """Input for trade analysis"""

    market_data: dict[str, Any] = Field(
        description="Market data including price, volume, indicators"
    )
    provider: str | None = Field(default="openai", description="AI provider to use")


class NewsAnalysisInput(BaseModel):
    """Input for news analysis"""

    articles: list[str] = Field(description="List of news articles to analyze")
    max_words: int | None = Field(default=500, description="Maximum words for summary")
    provider: str | None = Field(default="anthropic", description="AI provider to use")


class RiskAssessmentInput(BaseModel):
    """Input for risk assessment"""

    trade_data: dict[str, Any] = Field(
        description="Trade details including symbol, action, quantity, prices"
    )
    provider: str | None = Field(default="openai", description="AI provider to use")


class MarketSentimentTool(BaseTool):
    """AI-powered market sentiment analysis tool for CrewAI agents"""

    name: str = "market_sentiment_analyzer"
    description: str = """Analyzes text for market sentiment using advanced AI.
    Returns sentiment score, confidence, key factors, and market implications.
    Useful for understanding market mood from news, reports, or social media."""

    args_schema: type[BaseModel] = MarketSentimentInput
    ai_manager: AIServiceManager | None = None

    def __init__(self, ai_manager: AIServiceManager, **data):
        super().__init__(**data)
        self.ai_manager = ai_manager

    def _run(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        provider: str = "openai",
    ) -> str:
        """Synchronous wrapper for async sentiment analysis"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            provider_enum = AIProvider[provider.upper()]
            result = loop.run_until_complete(
                self.ai_manager.analyze_market_sentiment(
                    text=text, provider=provider_enum, context=context
                )
            )

            # Format result for agent consumption
            formatted_result = {
                "sentiment": result.get("sentiment", "neutral"),
                "score": result.get("score", 0.0),
                "confidence": result.get("confidence", 0.0),
                "key_factors": result.get("key_factors", []),
                "market_implications": result.get("market_implications", ""),
                "recommendations": self._generate_recommendations(result),
            }

            return json.dumps(formatted_result, indent=2)

        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {str(e)}")
            return json.dumps(
                {
                    "error": str(e),
                    "sentiment": "neutral",
                    "score": 0.0,
                    "confidence": 0.0,
                }
            )
        finally:
            loop.close()

    def _generate_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations from sentiment analysis"""
        recommendations = []

        score = analysis.get("score", 0)
        confidence = analysis.get("confidence", 0)

        if confidence > 0.7:
            if score > 0.5:
                recommendations.append(
                    "Consider bullish positions with proper risk management"
                )
                recommendations.append("Look for entry points on minor pullbacks")
            elif score < -0.5:
                recommendations.append("Consider defensive positions or hedging")
                recommendations.append("Wait for sentiment reversal signals")
            else:
                recommendations.append(
                    "Market sentiment is neutral - wait for clearer signals"
                )
        else:
            recommendations.append("Low confidence in sentiment - gather more data")

        return recommendations


class TradeAnalysisTool(BaseTool):
    """AI-powered trade analysis tool for CrewAI agents"""

    name: str = "trade_analyzer"
    description: str = """Analyzes market data to provide trading recommendations.
    Returns entry/exit points, technical signals, risk-reward ratios, and market conditions.
    Essential for making informed trading decisions."""

    args_schema: type[BaseModel] = TradeAnalysisInput
    ai_manager: AIServiceManager | None = None

    def __init__(self, ai_manager: AIServiceManager, **data):
        super().__init__(**data)
        self.ai_manager = ai_manager

    def _run(self, market_data: dict[str, Any], provider: str = "openai") -> str:
        """Synchronous wrapper for async trade analysis"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            provider_enum = AIProvider[provider.upper()]
            result = loop.run_until_complete(
                self.ai_manager.analyze_trade_opportunity(
                    market_data=market_data, provider=provider_enum
                )
            )

            # Enhance with additional insights
            result["trade_score"] = self._calculate_trade_score(result)
            result["execution_tips"] = self._generate_execution_tips(result)

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Trade analysis failed: {str(e)}")
            return json.dumps(
                {"error": str(e), "recommendation": "hold", "confidence": 0.0}
            )
        finally:
            loop.close()

    def _calculate_trade_score(self, analysis: dict[str, Any]) -> float:
        """Calculate overall trade score based on multiple factors"""
        score = 0.0

        # Confidence weight
        confidence = analysis.get("confidence", 0.5)
        score += confidence * 30

        # Risk-reward weight
        rr_ratio = analysis.get("risk_reward_ratio", 1.0)
        if rr_ratio >= 3:
            score += 30
        elif rr_ratio >= 2:
            score += 20
        elif rr_ratio >= 1.5:
            score += 10

        # Technical signals weight
        signals = analysis.get("technical_signals", [])
        bullish_signals = sum(1 for s in signals if s.get("signal") == "bullish")
        bearish_signals = sum(1 for s in signals if s.get("signal") == "bearish")

        if analysis.get("recommendation") == "buy":
            score += (bullish_signals - bearish_signals) * 10
        else:
            score += (bearish_signals - bullish_signals) * 10

        # Market conditions weight
        conditions = analysis.get("market_conditions", {})
        if (
            conditions.get("trend") == "uptrend"
            and analysis.get("recommendation") == "buy"
        ):
            score += 15
        elif (
            conditions.get("trend") == "downtrend"
            and analysis.get("recommendation") == "sell"
        ):
            score += 15

        return max(0, min(100, score))

    def _generate_execution_tips(self, analysis: dict[str, Any]) -> list[str]:
        """Generate execution tips based on analysis"""
        tips = []

        conditions = analysis.get("market_conditions", {})

        if conditions.get("volatility") == "high":
            tips.append("Use wider stops due to high volatility")
            tips.append("Consider scaling into position")

        if conditions.get("volume") == "decreasing":
            tips.append("Watch for volume confirmation before entry")

        if analysis.get("recommendation") == "buy":
            tips.append("Consider using limit orders near support levels")
        elif analysis.get("recommendation") == "sell":
            tips.append("Consider using limit orders near resistance levels")

        return tips


class NewsAnalysisTool(BaseTool):
    """AI-powered news analysis tool for CrewAI agents"""

    name: str = "news_analyzer"
    description: str = """Analyzes and summarizes multiple news articles.
    Identifies key themes, market implications, and trading opportunities.
    Helps agents stay informed about market-moving events."""

    args_schema: type[BaseModel] = NewsAnalysisInput
    ai_manager: AIServiceManager | None = None

    def __init__(self, ai_manager: AIServiceManager, **data):
        super().__init__(**data)
        self.ai_manager = ai_manager

    def _run(
        self, articles: list[str], max_words: int = 500, provider: str = "anthropic"
    ) -> str:
        """Synchronous wrapper for async news analysis"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            provider_enum = AIProvider[provider.upper()]
            result = loop.run_until_complete(
                self.ai_manager.summarize_news(
                    articles=articles, max_words=max_words, provider=provider_enum
                )
            )

            # Extract and structure key information
            structured_result = {
                "summary": result.get("summary", ""),
                "key_themes": self._extract_themes(result.get("summary", "")),
                "market_impact": self._assess_market_impact(result.get("summary", "")),
                "affected_sectors": self._identify_sectors(result.get("summary", "")),
                "trading_signals": self._extract_trading_signals(
                    result.get("summary", "")
                ),
                "metadata": {
                    "article_count": len(articles),
                    "summary_length": result.get("summary_length", 0),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            return json.dumps(structured_result, indent=2)

        except Exception as e:
            logger.error(f"News analysis failed: {str(e)}")
            return json.dumps(
                {"error": str(e), "summary": "Failed to analyze news articles"}
            )
        finally:
            loop.close()

    def _extract_themes(self, summary: str) -> list[str]:
        """Extract key themes from news summary"""
        # Simple keyword-based extraction (could be enhanced with NLP)
        themes = []

        theme_keywords = {
            "earnings": ["earnings", "revenue", "profit", "guidance"],
            "regulation": ["regulation", "regulatory", "SEC", "compliance"],
            "merger": ["merger", "acquisition", "buyout", "takeover"],
            "economic": ["GDP", "inflation", "interest rate", "Fed", "economy"],
            "technology": ["AI", "technology", "innovation", "disruption"],
            "geopolitical": ["war", "sanctions", "trade", "tariff", "political"],
        }

        summary_lower = summary.lower()

        for theme, keywords in theme_keywords.items():
            if any(keyword in summary_lower for keyword in keywords):
                themes.append(theme)

        return themes

    def _assess_market_impact(self, summary: str) -> str:
        """Assess potential market impact of news"""
        # Simplified impact assessment
        negative_words = ["decline", "fall", "drop", "concern", "risk", "warning"]
        positive_words = ["growth", "rise", "increase", "opportunity", "strong", "beat"]

        summary_lower = summary.lower()

        negative_count = sum(1 for word in negative_words if word in summary_lower)
        positive_count = sum(1 for word in positive_words if word in summary_lower)

        if negative_count > positive_count * 1.5:
            return "negative"
        elif positive_count > negative_count * 1.5:
            return "positive"
        else:
            return "neutral"

    def _identify_sectors(self, summary: str) -> list[str]:
        """Identify affected market sectors"""
        sectors = []

        sector_keywords = {
            "technology": ["tech", "software", "hardware", "semiconductor"],
            "finance": ["bank", "financial", "insurance", "investment"],
            "healthcare": ["health", "pharma", "biotech", "medical"],
            "energy": ["oil", "gas", "energy", "renewable"],
            "consumer": ["retail", "consumer", "shopping", "e-commerce"],
            "industrial": ["manufacturing", "industrial", "aerospace", "defense"],
        }

        summary_lower = summary.lower()

        for sector, keywords in sector_keywords.items():
            if any(keyword in summary_lower for keyword in keywords):
                sectors.append(sector)

        return sectors

    def _extract_trading_signals(self, summary: str) -> list[dict[str, str]]:
        """Extract potential trading signals from news"""
        signals = []

        # Simple pattern matching for trading signals
        if "beat expectations" in summary.lower():
            signals.append(
                {"signal": "bullish", "reason": "Earnings beat expectations"}
            )

        if "miss expectations" in summary.lower():
            signals.append(
                {"signal": "bearish", "reason": "Earnings miss expectations"}
            )

        if "upgrade" in summary.lower():
            signals.append({"signal": "bullish", "reason": "Analyst upgrade"})

        if "downgrade" in summary.lower():
            signals.append({"signal": "bearish", "reason": "Analyst downgrade"})

        return signals


class RiskAssessmentTool(BaseTool):
    """AI-powered risk assessment tool for CrewAI agents"""

    name: str = "risk_assessor"
    description: str = """Performs comprehensive risk assessment for trades.
    Evaluates risk factors, suggests position sizing, and provides mitigation strategies.
    Critical for protecting capital and optimizing risk-reward ratios."""

    args_schema: type[BaseModel] = RiskAssessmentInput
    ai_manager: AIServiceManager | None = None

    def __init__(self, ai_manager: AIServiceManager, **data):
        super().__init__(**data)
        self.ai_manager = ai_manager

    def _run(self, trade_data: dict[str, Any], provider: str = "openai") -> str:
        """Synchronous wrapper for async risk assessment"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            provider_enum = AIProvider[provider.upper()]
            result = loop.run_until_complete(
                self.ai_manager.assess_trade_risk(
                    trade_data=trade_data, provider=provider_enum
                )
            )

            # Add risk management recommendations
            result["risk_management"] = self._generate_risk_management(
                result, trade_data
            )
            result["position_sizing_calculator"] = self._calculate_position_size(
                result, trade_data
            )

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return json.dumps(
                {
                    "error": str(e),
                    "risk_level": "high",
                    "risk_score": 8,
                    "recommendation": "Avoid trade due to assessment failure",
                }
            )
        finally:
            loop.close()

    def _generate_risk_management(
        self, assessment: dict[str, Any], trade_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate comprehensive risk management plan"""
        risk_level = assessment.get("risk_level", "high")
        risk_score = assessment.get("risk_score", 8)

        management = {
            "max_position_size": 0,
            "stop_loss_type": "",
            "risk_per_trade": 0,
            "scaling_strategy": "",
            "hedging_required": False,
            "monitoring_frequency": "",
        }

        # Position size based on risk
        if risk_score <= 3:
            management["max_position_size"] = 0.10  # 10% of portfolio
            management["risk_per_trade"] = 0.02  # 2% risk
            management["monitoring_frequency"] = "daily"
        elif risk_score <= 6:
            management["max_position_size"] = 0.05  # 5% of portfolio
            management["risk_per_trade"] = 0.01  # 1% risk
            management["monitoring_frequency"] = "twice daily"
        else:
            management["max_position_size"] = 0.02  # 2% of portfolio
            management["risk_per_trade"] = 0.005  # 0.5% risk
            management["monitoring_frequency"] = "hourly"

        # Stop loss strategy
        if risk_level == "low":
            management["stop_loss_type"] = "mental stop"
        elif risk_level == "medium":
            management["stop_loss_type"] = "soft stop with alerts"
        else:
            management["stop_loss_type"] = "hard stop loss order"

        # Scaling strategy
        volatility = trade_data.get("market_conditions", {}).get("volatility", "medium")
        if volatility == "high":
            management["scaling_strategy"] = "Scale in 3 tranches: 40%, 40%, 20%"
        else:
            management["scaling_strategy"] = "Scale in 2 tranches: 60%, 40%"

        # Hedging
        if risk_score >= 7:
            management["hedging_required"] = True

        return management

    def _calculate_position_size(
        self, assessment: dict[str, Any], trade_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate optimal position size based on risk assessment"""
        portfolio_value = trade_data.get("portfolio_value", 100000)
        entry_price = trade_data.get("entry_price", 100)
        stop_loss = assessment.get("stop_loss_analysis", {}).get("recommended_stop", 95)

        # Risk per trade from risk management
        risk_pct = assessment.get("risk_management", {}).get("risk_per_trade", 0.01)
        risk_amount = portfolio_value * risk_pct

        # Calculate position size
        price_risk = abs(entry_price - stop_loss)
        if price_risk > 0:
            shares = int(risk_amount / price_risk)
            position_value = shares * entry_price
            position_pct = (position_value / portfolio_value) * 100
        else:
            shares = 0
            position_value = 0
            position_pct = 0

        return {
            "recommended_shares": shares,
            "position_value": position_value,
            "position_percentage": position_pct,
            "risk_amount": risk_amount,
            "stop_loss_distance": price_risk,
            "r_multiple_target": 2.0,  # Target 2R profit
        }


def create_ai_tools(ai_manager: AIServiceManager) -> list[BaseTool]:
    """Create all AI-powered tools for CrewAI agents"""
    tools = [
        MarketSentimentTool(ai_manager=ai_manager),
        TradeAnalysisTool(ai_manager=ai_manager),
        NewsAnalysisTool(ai_manager=ai_manager),
        RiskAssessmentTool(ai_manager=ai_manager),
    ]

    logger.info(f"Created {len(tools)} AI-powered tools for CrewAI")
    return tools
