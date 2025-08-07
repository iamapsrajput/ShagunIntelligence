import json
import logging
from datetime import datetime
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplates:
    """Manages prompt templates for different AI use cases"""

    def __init__(self):
        self.templates = self._initialize_templates()
        logger.info("PromptTemplates initialized")

    def _initialize_templates(self) -> dict[str, str]:
        """Initialize all prompt templates"""
        return {
            "market_sentiment": self._market_sentiment_template(),
            "trade_analysis": self._trade_analysis_template(),
            "news_summary": self._news_summary_template(),
            "risk_assessment": self._risk_assessment_template(),
            "metric_extraction": self._metric_extraction_template(),
            "technical_analysis": self._technical_analysis_template(),
            "earnings_analysis": self._earnings_analysis_template(),
            "portfolio_review": self._portfolio_review_template(),
        }

    def get_market_sentiment_prompt(
        self, text: str, context: dict[str, Any] | None = None
    ) -> str:
        """Generate market sentiment analysis prompt"""
        base_prompt = self.templates["market_sentiment"]

        context_str = ""
        if context:
            context_str = f"\nAdditional Context:\n{json.dumps(context, indent=2)}"

        return base_prompt.format(
            text=text,
            context=context_str,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def get_trade_analysis_prompt(self, market_data: dict[str, Any]) -> str:
        """Generate trade analysis prompt"""
        base_prompt = self.templates["trade_analysis"]

        # Format market data
        formatted_data = self._format_market_data(market_data)

        return base_prompt.format(
            market_data=formatted_data,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def get_news_summary_prompt(self, articles: list[str], max_words: int = 500) -> str:
        """Generate news summary prompt"""
        base_prompt = self.templates["news_summary"]

        # Combine articles
        combined_text = "\n\n---\n\n".join(
            f"Article {i+1}:\n{article}" for i, article in enumerate(articles)
        )

        return base_prompt.format(
            articles=combined_text, max_words=max_words, article_count=len(articles)
        )

    def get_risk_assessment_prompt(self, trade_data: dict[str, Any]) -> str:
        """Generate risk assessment prompt"""
        base_prompt = self.templates["risk_assessment"]

        # Extract key trade information
        trade_info = {
            "symbol": trade_data.get("symbol", "UNKNOWN"),
            "action": trade_data.get("action", "BUY"),
            "quantity": trade_data.get("quantity", 0),
            "entry_price": trade_data.get("entry_price", 0),
            "stop_loss": trade_data.get("stop_loss"),
            "target": trade_data.get("target"),
            "portfolio_value": trade_data.get("portfolio_value", 100000),
            "market_conditions": trade_data.get("market_conditions", {}),
        }

        return base_prompt.format(
            trade_data=json.dumps(trade_info, indent=2),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def get_metric_extraction_prompt(self, financial_text: str) -> str:
        """Generate metric extraction prompt"""
        base_prompt = self.templates["metric_extraction"]

        return base_prompt.format(
            text=financial_text, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def _market_sentiment_template(self) -> str:
        """Template for market sentiment analysis"""
        return """Analyze the following text for market sentiment and provide a comprehensive assessment:

Text to analyze:
{text}
{context}

Timestamp: {timestamp}

Please provide your analysis in the following JSON format:
{{
    "sentiment": "bullish/bearish/neutral",
    "score": -1.0 to 1.0,
    "confidence": 0.0 to 1.0,
    "key_factors": [
        {{
            "factor": "description",
            "impact": "positive/negative",
            "weight": 0.0 to 1.0
        }}
    ],
    "market_implications": "detailed explanation",
    "time_horizon": "short/medium/long",
    "sectors_affected": ["sector1", "sector2"],
    "risk_factors": ["risk1", "risk2"],
    "opportunities": ["opportunity1", "opportunity2"]
}}

Consider both explicit statements and implicit signals. Focus on actionable insights."""

    def _trade_analysis_template(self) -> str:
        """Template for trade analysis"""
        return """Analyze the following market data and provide comprehensive trading recommendations:

Market Data:
{market_data}

Timestamp: {timestamp}

Please provide your analysis in the following JSON format:
{{
    "recommendation": "buy/sell/hold",
    "confidence": 0.0 to 1.0,
    "entry_points": [
        {{
            "price": 0.0,
            "condition": "description",
            "probability": 0.0 to 1.0
        }}
    ],
    "exit_points": [
        {{
            "price": 0.0,
            "type": "target/stop_loss",
            "reasoning": "explanation"
        }}
    ],
    "technical_signals": [
        {{
            "indicator": "name",
            "signal": "bullish/bearish",
            "strength": 0.0 to 1.0
        }}
    ],
    "risk_reward_ratio": 0.0,
    "expected_holding_period": "hours/days/weeks",
    "market_conditions": {{
        "trend": "uptrend/downtrend/sideways",
        "volatility": "low/medium/high",
        "volume": "increasing/decreasing/stable"
    }},
    "key_levels": {{
        "support": [0.0],
        "resistance": [0.0],
        "pivot_points": [0.0]
    }},
    "alternative_scenarios": [
        {{
            "scenario": "description",
            "probability": 0.0 to 1.0,
            "action": "recommended action"
        }}
    ]
}}

Provide detailed, actionable insights based on the data."""

    def _news_summary_template(self) -> str:
        """Template for news summarization"""
        return """Summarize the following {article_count} news articles into a cohesive summary of no more than {max_words} words:

{articles}

Requirements:
1. Focus on the most important and actionable information
2. Identify common themes across articles
3. Highlight any conflicting information
4. Emphasize market-moving insights
5. Maintain objectivity and accuracy

Structure your summary as follows:
- Key Headlines (2-3 bullet points)
- Main Themes and Trends
- Market Implications
- Notable Quotes or Data Points
- Potential Trading Opportunities or Risks

Make the summary concise, clear, and valuable for trading decisions."""

    def _risk_assessment_template(self) -> str:
        """Template for risk assessment"""
        return """Perform a comprehensive risk assessment for the following trade:

Trade Details:
{trade_data}

Timestamp: {timestamp}

Please provide your assessment in the following JSON format:
{{
    "risk_level": "low/medium/high/extreme",
    "risk_score": 1 to 10,
    "confidence": 0.0 to 1.0,
    "risk_factors": [
        {{
            "factor": "description",
            "severity": "low/medium/high",
            "probability": 0.0 to 1.0,
            "potential_impact": "percentage or description"
        }}
    ],
    "position_sizing": {{
        "recommended_size": 0.0,
        "max_size": 0.0,
        "reasoning": "explanation"
    }},
    "stop_loss_analysis": {{
        "recommended_stop": 0.0,
        "risk_amount": 0.0,
        "risk_percentage": 0.0,
        "technical_level": "description"
    }},
    "mitigation_strategies": [
        {{
            "strategy": "description",
            "effectiveness": "low/medium/high",
            "implementation": "how to implement"
        }}
    ],
    "correlation_risks": [
        {{
            "asset": "symbol",
            "correlation": -1.0 to 1.0,
            "risk": "description"
        }}
    ],
    "market_risk_factors": {{
        "volatility_risk": "assessment",
        "liquidity_risk": "assessment",
        "event_risk": "assessment",
        "systemic_risk": "assessment"
    }},
    "scenario_analysis": [
        {{
            "scenario": "description",
            "probability": 0.0 to 1.0,
            "impact": "percentage loss/gain",
            "action_plan": "what to do"
        }}
    ],
    "overall_recommendation": "detailed recommendation"
}}

Consider all aspects of risk including market, position-specific, and systemic factors."""

    def _metric_extraction_template(self) -> str:
        """Template for metric extraction"""
        return """Extract all key financial metrics and data points from the following text:

Text:
{text}

Timestamp: {timestamp}

Please provide the extracted information in the following JSON format:
{{
    "metrics": {{
        "revenue": {{
            "value": 0.0,
            "period": "Q1/Q2/Q3/Q4/FY",
            "year": 2024,
            "growth_yoy": 0.0,
            "unit": "currency"
        }},
        "earnings": {{
            "eps": 0.0,
            "total": 0.0,
            "growth_yoy": 0.0
        }},
        "margins": {{
            "gross": 0.0,
            "operating": 0.0,
            "net": 0.0
        }},
        "guidance": {{
            "revenue": "range or value",
            "eps": "range or value",
            "period": "description"
        }},
        "operational_metrics": {{
            "metric_name": "value"
        }},
        "ratios": {{
            "pe": 0.0,
            "pb": 0.0,
            "debt_to_equity": 0.0
        }}
    }},
    "key_developments": [
        "development 1",
        "development 2"
    ],
    "management_commentary": [
        "quote or summary 1",
        "quote or summary 2"
    ],
    "analyst_estimates": {{
        "consensus": "description",
        "price_targets": []
    }},
    "dates_mentioned": [
        {{
            "date": "YYYY-MM-DD",
            "event": "description"
        }}
    ]
}}

Extract only factual information present in the text. If a metric is not mentioned, omit it from the response."""

    def _technical_analysis_template(self) -> str:
        """Template for technical analysis"""
        return """Perform technical analysis on the provided market data:

{market_data}

Provide comprehensive technical analysis including:
1. Trend analysis (primary, secondary, minor trends)
2. Support and resistance levels
3. Chart patterns (if any)
4. Key technical indicators and their signals
5. Volume analysis
6. Market structure analysis
7. Potential entry and exit points

Format your response as actionable trading insights."""

    def _earnings_analysis_template(self) -> str:
        """Template for earnings analysis"""
        return """Analyze the following earnings information:

{earnings_data}

Provide analysis covering:
1. Earnings performance vs. expectations
2. Revenue trends and drivers
3. Margin analysis
4. Guidance assessment
5. Key business metrics
6. Management commentary highlights
7. Market reaction potential
8. Trading opportunities

Focus on actionable insights for traders."""

    def _portfolio_review_template(self) -> str:
        """Template for portfolio review"""
        return """Review the following portfolio data:

{portfolio_data}

Provide comprehensive analysis including:
1. Performance assessment
2. Risk analysis (concentration, correlation, etc.)
3. Rebalancing recommendations
4. Optimization opportunities
5. Hedging suggestions
6. Tax considerations
7. Market outlook alignment

Structure recommendations by priority and potential impact."""

    def _format_market_data(self, data: dict[str, Any]) -> str:
        """Format market data for prompts"""
        formatted = []

        # Price data
        if "price" in data:
            formatted.append(f"Current Price: {data['price']}")

        if "ohlc" in data:
            ohlc = data["ohlc"]
            formatted.append(
                f"OHLC: O={ohlc.get('open')}, H={ohlc.get('high')}, "
                f"L={ohlc.get('low')}, C={ohlc.get('close')}"
            )

        # Volume
        if "volume" in data:
            formatted.append(f"Volume: {data['volume']:,}")

        # Technical indicators
        if "indicators" in data:
            formatted.append("\nTechnical Indicators:")
            for name, value in data["indicators"].items():
                formatted.append(f"  {name}: {value}")

        # Additional data
        if "news" in data:
            formatted.append(f"\nRecent News: {len(data['news'])} articles")

        if "sentiment" in data:
            formatted.append(f"Sentiment Score: {data['sentiment']}")

        return "\n".join(formatted)

    def create_custom_prompt(self, template_name: str, **kwargs) -> str:
        """Create a custom prompt from a template"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")

        template = self.templates[template_name]

        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(
                f"Missing required parameter for template '{template_name}': {e}"
            )
            raise

    def add_custom_template(self, name: str, template: str) -> None:
        """Add a custom template"""
        self.templates[name] = template
        logger.info(f"Added custom template: {name}")
