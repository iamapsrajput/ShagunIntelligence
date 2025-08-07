from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger


class SentimentSource:
    """Represents a sentiment data source with its weight and reliability"""

    def __init__(self, name: str, source: Any, weight: float = 1.0):
        self.name = name
        self.source = source
        self.weight = weight
        self.reliability_score = 1.0
        self.success_count = 0
        self.error_count = 0
        self.last_update = datetime.utcnow()

    def update_reliability(self, success: bool):
        """Update reliability score based on success/failure"""
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        total = self.success_count + self.error_count
        if total > 0:
            self.reliability_score = self.success_count / total

        self.last_update = datetime.utcnow()


class MultiSourceSentimentFusion:
    """Fuses sentiment data from multiple sources with weighted aggregation"""

    def __init__(self):
        self.sources: dict[str, SentimentSource] = {}
        self.fusion_history = defaultdict(lambda: defaultdict(list))
        self.cache_ttl = 300  # 5 minutes
        self.cache = {}
        logger.info("Initialized MultiSourceSentimentFusion")

    def add_source(self, name: str, source: Any, weight: float = 1.0):
        """Add a sentiment source"""
        self.sources[name] = SentimentSource(name, source, weight)
        logger.info(f"Added sentiment source: {name} with weight {weight}")

    def remove_source(self, name: str):
        """Remove a sentiment source"""
        if name in self.sources:
            del self.sources[name]
            logger.info(f"Removed sentiment source: {name}")

    async def get_fused_sentiment(
        self, symbol: str, use_cache: bool = True
    ) -> dict[str, Any]:
        """Get fused sentiment from all sources"""
        # Check cache
        cache_key = f"{symbol}_{int(datetime.utcnow().timestamp() / self.cache_ttl)}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        sentiments = []
        source_results = {}

        # Collect sentiment from each source
        for name, sentiment_source in self.sources.items():
            try:
                result = await sentiment_source.source.get_sentiment_score(symbol)
                if result and "sentiment_score" in result:
                    sentiments.append(
                        {
                            "source": name,
                            "score": result["sentiment_score"],
                            "confidence": result.get("confidence", 0.5),
                            "weight": sentiment_source.weight,
                            "reliability": sentiment_source.reliability_score,
                            "raw_data": result,
                        }
                    )
                    source_results[name] = result
                    sentiment_source.update_reliability(True)
                else:
                    sentiment_source.update_reliability(False)

            except Exception as e:
                logger.error(f"Error getting sentiment from {name}: {e}")
                sentiment_source.update_reliability(False)

        # Fuse sentiments
        if not sentiments:
            return {
                "symbol": symbol,
                "fused_sentiment_score": 0.0,
                "fused_sentiment_label": "neutral",
                "confidence": 0.0,
                "source_count": 0,
                "sources": {},
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Calculate weighted average
        fused_score, confidence = self._calculate_weighted_sentiment(sentiments)

        # Determine label
        if fused_score > 0.1:
            label = "positive"
        elif fused_score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        # Calculate agreement score
        agreement = self._calculate_source_agreement(sentiments)

        result = {
            "symbol": symbol,
            "fused_sentiment_score": fused_score,
            "fused_sentiment_label": label,
            "confidence": confidence,
            "agreement_score": agreement,
            "source_count": len(sentiments),
            "sources": source_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Cache result
        self.cache[cache_key] = result

        # Store in history
        self._update_history(symbol, result)

        return result

    def _calculate_weighted_sentiment(
        self, sentiments: list[dict[str, Any]]
    ) -> tuple[float, float]:
        """Calculate weighted sentiment score and confidence"""
        if not sentiments:
            return 0.0, 0.0

        total_weight = 0
        weighted_sum = 0
        confidence_sum = 0

        for sentiment in sentiments:
            # Combined weight: source weight * reliability * confidence
            combined_weight = (
                sentiment["weight"] * sentiment["reliability"] * sentiment["confidence"]
            )

            weighted_sum += sentiment["score"] * combined_weight
            confidence_sum += sentiment["confidence"] * sentiment["weight"]
            total_weight += combined_weight

        if total_weight == 0:
            return 0.0, 0.0

        fused_score = weighted_sum / total_weight
        avg_confidence = confidence_sum / sum(s["weight"] for s in sentiments)

        # Adjust confidence based on source agreement
        agreement_boost = self._calculate_source_agreement(sentiments) * 0.2
        final_confidence = min(avg_confidence + agreement_boost, 1.0)

        return fused_score, final_confidence

    def _calculate_source_agreement(self, sentiments: list[dict[str, Any]]) -> float:
        """Calculate how much sources agree (0-1)"""
        if len(sentiments) < 2:
            return 1.0

        scores = [s["score"] for s in sentiments]
        std_dev = np.std(scores)

        # Convert standard deviation to agreement score
        # Low std dev = high agreement
        agreement = max(0, 1 - (std_dev * 2))

        return agreement

    def _update_history(self, symbol: str, result: dict[str, Any]):
        """Update fusion history for analysis"""
        timestamp = datetime.utcnow()
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)

        self.fusion_history[symbol][hour_key].append(
            {
                "timestamp": timestamp,
                "score": result["fused_sentiment_score"],
                "confidence": result["confidence"],
                "source_count": result["source_count"],
            }
        )

        # Keep only last 24 hours
        cutoff = datetime.utcnow() - timedelta(hours=24)
        old_keys = [k for k in self.fusion_history[symbol] if k < cutoff]
        for key in old_keys:
            del self.fusion_history[symbol][key]

    async def get_sentiment_trends(
        self, symbol: str, hours: int = 24
    ) -> dict[str, Any]:
        """Get sentiment trends from fused data"""
        trends = []

        # Get trends from each source
        source_trends = {}
        for name, sentiment_source in self.sources.items():
            try:
                if hasattr(sentiment_source.source, "get_sentiment_trends"):
                    result = await sentiment_source.source.get_sentiment_trends(
                        symbol, hours
                    )
                    source_trends[name] = result.get("trends", [])
            except Exception as e:
                logger.error(f"Error getting trends from {name}: {e}")

        # Aggregate by hour
        hourly_data = defaultdict(lambda: {"scores": [], "counts": []})

        for source_name, trend_list in source_trends.items():
            source = self.sources[source_name]
            for trend in trend_list:
                timestamp = datetime.fromisoformat(trend["timestamp"])
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)

                hourly_data[hour_key]["scores"].append(
                    {
                        "score": trend.get("sentiment_score", 0),
                        "weight": source.weight * source.reliability_score,
                        "count": trend.get("tweet_count", 0),
                    }
                )

        # Calculate fused trends
        for hour, data in sorted(hourly_data.items()):
            if data["scores"]:
                # Weighted average
                total_weight = sum(s["weight"] for s in data["scores"])
                if total_weight > 0:
                    weighted_score = (
                        sum(s["score"] * s["weight"] for s in data["scores"])
                        / total_weight
                    )

                    total_count = sum(s["count"] for s in data["scores"])

                    trends.append(
                        {
                            "timestamp": hour.isoformat(),
                            "sentiment_score": weighted_score,
                            "data_points": total_count,
                            "source_count": len(data["scores"]),
                        }
                    )

        return {
            "symbol": symbol,
            "trends": trends,
            "period_hours": hours,
            "sources": list(self.sources.keys()),
        }

    async def get_real_time_alerts(
        self, symbol: str, threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        """Get real-time sentiment alerts when threshold is exceeded"""
        alerts = []

        # Get current sentiment
        sentiment = await self.get_fused_sentiment(symbol)

        if abs(sentiment["fused_sentiment_score"]) > threshold:
            alert_type = (
                "bullish" if sentiment["fused_sentiment_score"] > 0 else "bearish"
            )

            alerts.append(
                {
                    "symbol": symbol,
                    "alert_type": alert_type,
                    "sentiment_score": sentiment["fused_sentiment_score"],
                    "confidence": sentiment["confidence"],
                    "threshold": threshold,
                    "sources": sentiment["source_count"],
                    "timestamp": sentiment["timestamp"],
                }
            )

        return alerts

    def get_source_statistics(self) -> dict[str, Any]:
        """Get statistics about sentiment sources"""
        stats = {}

        for name, source in self.sources.items():
            total_requests = source.success_count + source.error_count

            stats[name] = {
                "weight": source.weight,
                "reliability_score": source.reliability_score,
                "success_rate": (
                    source.success_count / total_requests if total_requests > 0 else 0
                ),
                "total_requests": total_requests,
                "last_update": source.last_update.isoformat(),
            }

        return stats

    async def backtest_fusion_accuracy(
        self, symbol: str, price_data: list[dict[str, Any]], hours: int = 24
    ) -> dict[str, Any]:
        """Backtest sentiment fusion accuracy against price movements"""
        # Get historical sentiment trends
        trends = await self.get_sentiment_trends(symbol, hours)

        if not trends["trends"] or not price_data:
            return {
                "symbol": symbol,
                "accuracy": 0.0,
                "correlation": 0.0,
                "sample_size": 0,
            }

        # Align sentiment and price data by timestamp
        aligned_data = []

        for sentiment in trends["trends"]:
            sentiment_time = datetime.fromisoformat(sentiment["timestamp"])

            # Find corresponding price data
            for i, price in enumerate(price_data):
                price_time = datetime.fromisoformat(price["timestamp"])

                if (
                    abs((price_time - sentiment_time).total_seconds()) < 3600
                ):  # Within 1 hour
                    # Calculate price change to next period
                    if i < len(price_data) - 1:
                        price_change = (
                            price_data[i + 1]["close"] - price["close"]
                        ) / price["close"]

                        aligned_data.append(
                            {
                                "sentiment": sentiment["sentiment_score"],
                                "price_change": price_change,
                            }
                        )
                    break

        if not aligned_data:
            return {
                "symbol": symbol,
                "accuracy": 0.0,
                "correlation": 0.0,
                "sample_size": 0,
            }

        # Calculate accuracy (correct direction predictions)
        correct_predictions = sum(
            1
            for d in aligned_data
            if (d["sentiment"] > 0 and d["price_change"] > 0)
            or (d["sentiment"] < 0 and d["price_change"] < 0)
        )

        accuracy = correct_predictions / len(aligned_data)

        # Calculate correlation
        sentiments = [d["sentiment"] for d in aligned_data]
        price_changes = [d["price_change"] for d in aligned_data]

        if len(sentiments) > 1:
            correlation = np.corrcoef(sentiments, price_changes)[0, 1]
        else:
            correlation = 0.0

        return {
            "symbol": symbol,
            "accuracy": accuracy,
            "correlation": correlation,
            "sample_size": len(aligned_data),
            "period_hours": hours,
        }
