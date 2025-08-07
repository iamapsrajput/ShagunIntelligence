"""Sentiment scoring module using OpenAI GPT-4 and other NLP models."""

import asyncio
import re
from datetime import datetime
from typing import Any

import nltk
import openai
from loguru import logger
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


class SentimentScorer:
    """Score sentiment using multiple NLP models including GPT-4."""

    def __init__(self, openai_api_key: str | None = None):
        """Initialize the sentiment scorer."""
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key

        # Initialize NLTK components
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            logger.info("Downloading NLTK vader_lexicon...")
            nltk.download("vader_lexicon", quiet=True)
            self.sia = SentimentIntensityAnalyzer()

        # Sentiment keywords for financial context
        self.positive_keywords = {
            "strong": [
                "surge",
                "soar",
                "rally",
                "breakout",
                "bullish",
                "upgrade",
                "beat",
                "exceed",
            ],
            "moderate": [
                "gain",
                "rise",
                "increase",
                "improve",
                "positive",
                "growth",
                "expand",
            ],
            "mild": ["steady", "stable", "maintain", "support", "optimistic"],
        }

        self.negative_keywords = {
            "strong": [
                "crash",
                "plunge",
                "collapse",
                "bearish",
                "downgrade",
                "miss",
                "fail",
            ],
            "moderate": [
                "fall",
                "drop",
                "decline",
                "decrease",
                "negative",
                "loss",
                "cut",
            ],
            "mild": ["concern", "worry", "caution", "risk", "uncertain"],
        }

        # Market event patterns
        self.event_patterns = {
            "earnings_beat": r"beat\s+(?:earnings|expectations|estimates)",
            "earnings_miss": r"miss\s+(?:earnings|expectations|estimates)",
            "upgrade": r"(?:upgrade|raised|boost)\s+(?:to|from|rating|target)",
            "downgrade": r"(?:downgrade|cut|lower)\s+(?:to|from|rating|target)",
            "merger": r"(?:merger|acquisition|acquire|takeover|deal)",
            "lawsuit": r"(?:lawsuit|sue|legal|court|investigation)",
            "product_launch": r"(?:launch|unveil|introduce|announce)\s+(?:new|product|service)",
            "guidance": r"(?:guidance|forecast|outlook|projection)",
        }

        self.cache = {}
        logger.info("SentimentScorer initialized")

    async def analyze_text(
        self, text: str, context: str | None = None, use_gpt: bool = True
    ) -> dict[str, Any]:
        """
        Analyze sentiment of text using multiple methods.

        Args:
            text: Text to analyze
            context: Additional context (e.g., stock symbol)
            use_gpt: Whether to use GPT-4 for analysis

        Returns:
            Sentiment analysis results
        """
        try:
            # Check cache
            cache_key = f"{hash(text)}_{context}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Basic sentiment analysis
            basic_sentiment = self._analyze_basic_sentiment(text)

            # Financial keyword analysis
            keyword_sentiment = self._analyze_keywords(text)

            # Event detection
            events = self._detect_events(text)

            # GPT-4 analysis if available
            gpt_sentiment = None
            if use_gpt and self.openai_api_key:
                gpt_sentiment = await self._analyze_with_gpt(text, context)

            # Combine scores
            final_score = self._combine_scores(
                basic_sentiment, keyword_sentiment, gpt_sentiment
            )

            result = {
                "score": final_score,
                "confidence": self._calculate_confidence(
                    basic_sentiment, keyword_sentiment, gpt_sentiment
                ),
                "interpretation": self._interpret_score(final_score),
                "basic_sentiment": basic_sentiment,
                "keyword_sentiment": keyword_sentiment,
                "gpt_sentiment": gpt_sentiment,
                "events_detected": events,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            self.cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "score": 0.0,
                "confidence": 0.0,
                "interpretation": "neutral",
                "error": str(e),
            }

    def _analyze_basic_sentiment(self, text: str) -> dict[str, float]:
        """Perform basic sentiment analysis using NLTK and TextBlob."""
        # NLTK VADER sentiment
        vader_scores = self.sia.polarity_scores(text)
        vader_compound = vader_scores["compound"]

        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
        except:
            textblob_polarity = 0.0
            textblob_subjectivity = 0.5

        return {
            "vader_compound": vader_compound,
            "vader_positive": vader_scores["pos"],
            "vader_negative": vader_scores["neg"],
            "vader_neutral": vader_scores["neu"],
            "textblob_polarity": textblob_polarity,
            "textblob_subjectivity": textblob_subjectivity,
            "combined_score": (vader_compound + textblob_polarity) / 2,
        }

    def _analyze_keywords(self, text: str) -> dict[str, Any]:
        """Analyze financial keywords in text."""
        text_lower = text.lower()

        positive_score = 0
        negative_score = 0
        keywords_found = []

        # Check positive keywords
        for strength, keywords in self.positive_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if strength == "strong":
                        positive_score += 0.3
                    elif strength == "moderate":
                        positive_score += 0.2
                    else:
                        positive_score += 0.1
                    keywords_found.append((keyword, "positive", strength))

        # Check negative keywords
        for strength, keywords in self.negative_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if strength == "strong":
                        negative_score += 0.3
                    elif strength == "moderate":
                        negative_score += 0.2
                    else:
                        negative_score += 0.1
                    keywords_found.append((keyword, "negative", strength))

        # Calculate net score (-1 to 1)
        net_score = min(positive_score, 1.0) - min(negative_score, 1.0)

        return {
            "score": net_score,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "keywords_found": keywords_found,
        }

    def _detect_events(self, text: str) -> list[dict[str, Any]]:
        """Detect market-relevant events in text."""
        events = []
        text_lower = text.lower()

        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text_lower):
                # Determine sentiment impact of event
                if event_type in ["earnings_beat", "upgrade", "product_launch"]:
                    impact = "positive"
                    score_modifier = 0.2
                elif event_type in ["earnings_miss", "downgrade", "lawsuit"]:
                    impact = "negative"
                    score_modifier = -0.2
                else:
                    impact = "neutral"
                    score_modifier = 0.0

                events.append(
                    {
                        "type": event_type,
                        "impact": impact,
                        "score_modifier": score_modifier,
                    }
                )

        return events

    async def _analyze_with_gpt(
        self, text: str, context: str | None = None
    ) -> dict[str, Any] | None:
        """Analyze sentiment using GPT-4."""
        try:
            prompt = f"""Analyze the sentiment of the following financial text on a scale from -1 (very negative) to 1 (very positive).

Context: {context or 'General financial news'}
Text: {text}

Provide your analysis in the following format:
1. Sentiment Score: [number between -1 and 1]
2. Key Factors: [list main factors affecting sentiment]
3. Market Impact: [low/medium/high]
4. Trading Implication: [bullish/bearish/neutral]

Be precise and consider the financial context."""

            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analysis expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )

            # Parse GPT response
            gpt_text = response.choices[0].message.content

            # Extract sentiment score
            score_match = re.search(r"Sentiment Score:\s*([-\d.]+)", gpt_text)
            score = float(score_match.group(1)) if score_match else 0.0

            # Extract other information
            impact_match = re.search(r"Market Impact:\s*(\w+)", gpt_text)
            impact = impact_match.group(1).lower() if impact_match else "medium"

            implication_match = re.search(r"Trading Implication:\s*(\w+)", gpt_text)
            implication = (
                implication_match.group(1).lower() if implication_match else "neutral"
            )

            return {
                "score": max(-1, min(1, score)),  # Ensure within bounds
                "market_impact": impact,
                "trading_implication": implication,
                "raw_response": gpt_text,
            }

        except Exception as e:
            logger.error(f"Error with GPT analysis: {str(e)}")
            return None

    def _combine_scores(
        self,
        basic: dict[str, float],
        keyword: dict[str, Any],
        gpt: dict[str, Any] | None,
    ) -> float:
        """Combine sentiment scores from different methods."""
        scores = []
        weights = []

        # Basic sentiment (NLTK + TextBlob)
        scores.append(basic["combined_score"])
        weights.append(0.3)

        # Keyword sentiment
        scores.append(keyword["score"])
        weights.append(0.3)

        # GPT sentiment (if available)
        if gpt and "score" in gpt:
            scores.append(gpt["score"])
            weights.append(0.4)
        else:
            # Redistribute weight
            weights[0] = 0.5
            weights[1] = 0.5

        # Weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights, strict=False))

        # Clamp to [-1, 1]
        return max(-1, min(1, weighted_score))

    def _calculate_confidence(
        self,
        basic: dict[str, float],
        keyword: dict[str, Any],
        gpt: dict[str, Any] | None,
    ) -> float:
        """Calculate confidence in sentiment analysis."""
        # Base confidence on agreement between methods
        scores = [basic["combined_score"], keyword["score"]]
        if gpt and "score" in gpt:
            scores.append(gpt["score"])

        # Calculate standard deviation
        if len(scores) > 1:
            avg = sum(scores) / len(scores)
            variance = sum((s - avg) ** 2 for s in scores) / len(scores)
            std_dev = variance**0.5

            # Lower std dev = higher confidence
            confidence = max(0, 1 - (std_dev * 2))
        else:
            confidence = 0.5

        # Adjust for subjectivity (from TextBlob)
        subjectivity = basic.get("textblob_subjectivity", 0.5)
        confidence *= 1 - subjectivity * 0.3

        return min(max(confidence, 0), 1)

    def _interpret_score(self, score: float) -> str:
        """Interpret sentiment score."""
        if score > 0.6:
            return "very_positive"
        elif score > 0.2:
            return "positive"
        elif score > -0.2:
            return "neutral"
        elif score > -0.6:
            return "negative"
        else:
            return "very_negative"

    async def batch_analyze(
        self, texts: list[str], contexts: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Analyze multiple texts in batch."""
        if contexts and len(contexts) != len(texts):
            contexts = [None] * len(texts)
        elif not contexts:
            contexts = [None] * len(texts)

        tasks = [
            self.analyze_text(text, context)
            for text, context in zip(texts, contexts, strict=False)
        ]

        results = await asyncio.gather(*tasks)
        return results

    def get_market_sentiment_summary(
        self, analyses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate summary of market sentiment from multiple analyses."""
        if not analyses:
            return {
                "overall_sentiment": 0.0,
                "sentiment_distribution": {},
                "confidence": 0.0,
            }

        scores = [a["score"] for a in analyses if "score" in a]
        confidences = [a["confidence"] for a in analyses if "confidence" in a]

        # Calculate distribution
        distribution = {
            "very_positive": sum(1 for s in scores if s > 0.6),
            "positive": sum(1 for s in scores if 0.2 < s <= 0.6),
            "neutral": sum(1 for s in scores if -0.2 <= s <= 0.2),
            "negative": sum(1 for s in scores if -0.6 <= s < -0.2),
            "very_negative": sum(1 for s in scores if s < -0.6),
        }

        # Weight by confidence
        weighted_scores = (
            [s * c for s, c in zip(scores, confidences, strict=False)]
            if confidences
            else scores
        )

        overall_sentiment = (
            sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        )
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": distribution,
            "confidence": avg_confidence,
            "total_analyzed": len(analyses),
            "interpretation": self._interpret_score(overall_sentiment),
        }

    def get_status(self) -> dict[str, Any]:
        """Get the status of the sentiment scorer."""
        return {
            "status": "active",
            "gpt_enabled": bool(self.openai_api_key),
            "cache_size": len(self.cache),
            "models": ["vader", "textblob"]
            + (["gpt-4"] if self.openai_api_key else []),
        }
