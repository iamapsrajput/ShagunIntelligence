"""Alert manager for significant sentiment changes and market events."""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from loguru import logger


class AlertType(Enum):
    """Types of sentiment alerts."""

    SENTIMENT_SPIKE = "sentiment_spike"
    SENTIMENT_CRASH = "sentiment_crash"
    TREND_REVERSAL = "trend_reversal"
    HIGH_VOLUME = "high_volume"
    BREAKING_NEWS = "breaking_news"
    SOCIAL_BUZZ = "social_buzz"
    MARKET_EVENT = "market_event"


class AlertPriority(Enum):
    """Alert priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertManager:
    """Manage and distribute sentiment-based alerts."""

    def __init__(self):
        """Initialize the alert manager."""
        self.alerts: list[dict[str, Any]] = []
        self.alert_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.alert_thresholds = {
            "sentiment_change": 0.3,  # 30% change triggers alert
            "volume_spike": 2.0,  # 2x normal volume
            "confidence_minimum": 0.6,  # Minimum confidence for alerts
        }

        # Alert cooldown to prevent spam
        self.cooldown_periods = {
            AlertType.SENTIMENT_SPIKE: timedelta(hours=1),
            AlertType.SENTIMENT_CRASH: timedelta(hours=1),
            AlertType.TREND_REVERSAL: timedelta(hours=2),
            AlertType.HIGH_VOLUME: timedelta(minutes=30),
            AlertType.BREAKING_NEWS: timedelta(minutes=15),
            AlertType.SOCIAL_BUZZ: timedelta(minutes=30),
            AlertType.MARKET_EVENT: timedelta(hours=1),
        }

        # Notification channels
        self.notification_channels: list[Callable] = []

        logger.info("AlertManager initialized")

    async def create_sentiment_alert(
        self,
        symbol: str,
        current_score: float,
        change: float,
        top_stories: list[dict[str, Any]],
        confidence: float = 1.0,
    ) -> dict[str, Any] | None:
        """
        Create an alert for significant sentiment changes.

        Args:
            symbol: Stock symbol
            current_score: Current sentiment score
            change: Sentiment change amount
            top_stories: Related news stories
            confidence: Confidence in the alert

        Returns:
            Created alert or None if conditions not met
        """
        try:
            # Check if alert should be created
            if abs(change) < self.alert_thresholds["sentiment_change"]:
                return None

            if confidence < self.alert_thresholds["confidence_minimum"]:
                return None

            # Determine alert type
            if change > 0:
                alert_type = AlertType.SENTIMENT_SPIKE
            else:
                alert_type = AlertType.SENTIMENT_CRASH

            # Check cooldown
            if self._is_in_cooldown(symbol, alert_type):
                logger.debug(f"Alert for {symbol} in cooldown period")
                return None

            # Determine priority
            priority = self._calculate_priority(abs(change), confidence)

            # Create alert
            alert = {
                "id": self._generate_alert_id(),
                "type": alert_type.value,
                "symbol": symbol,
                "timestamp": datetime.now(),
                "priority": priority.value,
                "data": {
                    "current_score": current_score,
                    "change": change,
                    "confidence": confidence,
                    "direction": "positive" if change > 0 else "negative",
                    "magnitude": abs(change),
                },
                "context": {
                    "top_stories": top_stories[:3],
                    "story_count": len(top_stories),
                },
                "message": self._generate_alert_message(
                    symbol, alert_type, change, current_score
                ),
                "actions": self._suggest_actions(alert_type, change, confidence),
            }

            # Store alert
            self.alerts.append(alert)
            self.alert_history[symbol].append(alert)

            # Send notifications
            await self._send_notifications(alert)

            logger.info(
                f"Created {alert_type.value} alert for {symbol}: change={change:.2f}"
            )
            return alert

        except Exception as e:
            logger.error(f"Error creating sentiment alert: {str(e)}")
            return None

    async def create_volume_alert(
        self, symbol: str, volume_ratio: float, sentiment_score: float
    ) -> dict[str, Any] | None:
        """Create alert for unusual volume with sentiment context."""
        try:
            if volume_ratio < self.alert_thresholds["volume_spike"]:
                return None

            alert_type = AlertType.HIGH_VOLUME

            if self._is_in_cooldown(symbol, alert_type):
                return None

            alert = {
                "id": self._generate_alert_id(),
                "type": alert_type.value,
                "symbol": symbol,
                "timestamp": datetime.now(),
                "priority": (
                    AlertPriority.HIGH.value
                    if volume_ratio > 3
                    else AlertPriority.MEDIUM.value
                ),
                "data": {
                    "volume_ratio": volume_ratio,
                    "sentiment_score": sentiment_score,
                    "sentiment_context": self._interpret_volume_sentiment(
                        volume_ratio, sentiment_score
                    ),
                },
                "message": f"High volume alert for {symbol}: {volume_ratio:.1f}x normal volume",
                "actions": ["monitor_closely", "check_news", "review_positions"],
            }

            self.alerts.append(alert)
            await self._send_notifications(alert)

            return alert

        except Exception as e:
            logger.error(f"Error creating volume alert: {str(e)}")
            return None

    async def create_event_alert(
        self, symbol: str, event_type: str, event_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Create alert for specific market events."""
        try:
            alert_type = AlertType.MARKET_EVENT

            if self._is_in_cooldown(f"{symbol}_{event_type}", alert_type):
                return None

            # Determine priority based on event type
            priority_map = {
                "earnings_beat": AlertPriority.HIGH,
                "earnings_miss": AlertPriority.HIGH,
                "upgrade": AlertPriority.MEDIUM,
                "downgrade": AlertPriority.MEDIUM,
                "merger": AlertPriority.CRITICAL,
                "lawsuit": AlertPriority.HIGH,
            }

            priority = priority_map.get(event_type, AlertPriority.MEDIUM)

            alert = {
                "id": self._generate_alert_id(),
                "type": alert_type.value,
                "symbol": symbol,
                "timestamp": datetime.now(),
                "priority": priority.value,
                "data": {"event_type": event_type, "event_details": event_data},
                "message": f"Market event for {symbol}: {event_type.replace('_', ' ').title()}",
                "actions": self._suggest_event_actions(event_type),
            }

            self.alerts.append(alert)
            await self._send_notifications(alert)

            return alert

        except Exception as e:
            logger.error(f"Error creating event alert: {str(e)}")
            return None

    def get_active_alerts(
        self,
        symbol: str | None = None,
        priority: AlertPriority | None = None,
        hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Get active alerts based on filters."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_alerts = []
        for alert in self.alerts:
            # Time filter
            if alert["timestamp"] < cutoff_time:
                continue

            # Symbol filter
            if symbol and alert.get("symbol") != symbol:
                continue

            # Priority filter
            if priority and alert.get("priority") != priority.value:
                continue

            filtered_alerts.append(alert)

        # Sort by timestamp (newest first)
        filtered_alerts.sort(key=lambda x: x["timestamp"], reverse=True)

        return filtered_alerts

    def get_alert_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get summary of recent alerts."""
        recent_alerts = self.get_active_alerts(hours=hours)

        # Count by type
        type_counts = defaultdict(int)
        for alert in recent_alerts:
            type_counts[alert["type"]] += 1

        # Count by priority
        priority_counts = defaultdict(int)
        for alert in recent_alerts:
            priority_counts[alert["priority"]] += 1

        # Most alerted symbols
        symbol_counts = defaultdict(int)
        for alert in recent_alerts:
            if "symbol" in alert:
                symbol_counts[alert["symbol"]] += 1

        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return {
            "total_alerts": len(recent_alerts),
            "alerts_by_type": dict(type_counts),
            "alerts_by_priority": dict(priority_counts),
            "top_alerted_symbols": top_symbols,
            "time_range_hours": hours,
        }

    def add_notification_channel(self, channel: Callable):
        """Add a notification channel (webhook, email, etc.)."""
        self.notification_channels.append(channel)

    async def _send_notifications(self, alert: dict[str, Any]):
        """Send alert through all configured channels."""
        for channel in self.notification_channels:
            try:
                if asyncio.iscoroutinefunction(channel):
                    await channel(alert)
                else:
                    await asyncio.to_thread(channel, alert)
            except Exception as e:
                logger.error(f"Error sending notification: {str(e)}")

    def _is_in_cooldown(self, identifier: str, alert_type: AlertType) -> bool:
        """Check if an alert is in cooldown period."""
        if identifier not in self.alert_history:
            return False

        cooldown_period = self.cooldown_periods.get(alert_type, timedelta(hours=1))
        cutoff_time = datetime.now() - cooldown_period

        # Check recent alerts of same type
        recent_alerts = [
            alert
            for alert in self.alert_history[identifier]
            if alert["type"] == alert_type.value and alert["timestamp"] > cutoff_time
        ]

        return len(recent_alerts) > 0

    def _calculate_priority(
        self, change_magnitude: float, confidence: float
    ) -> AlertPriority:
        """Calculate alert priority based on magnitude and confidence."""
        score = change_magnitude * confidence

        if score > 0.6:
            return AlertPriority.CRITICAL
        elif score > 0.4:
            return AlertPriority.HIGH
        elif score > 0.2:
            return AlertPriority.MEDIUM
        else:
            return AlertPriority.LOW

    def _generate_alert_message(
        self, symbol: str, alert_type: AlertType, change: float, current_score: float
    ) -> str:
        """Generate human-readable alert message."""
        if alert_type == AlertType.SENTIMENT_SPIKE:
            return (
                f"🚀 Positive sentiment spike for {symbol}: "
                f"+{abs(change)*100:.1f}% change to {current_score:.2f}"
            )
        elif alert_type == AlertType.SENTIMENT_CRASH:
            return (
                f"📉 Negative sentiment crash for {symbol}: "
                f"-{abs(change)*100:.1f}% change to {current_score:.2f}"
            )
        elif alert_type == AlertType.TREND_REVERSAL:
            direction = "bullish" if change > 0 else "bearish"
            return f"🔄 Sentiment trend reversal for {symbol}: turning {direction}"
        else:
            return f"Alert for {symbol}: {alert_type.value}"

    def _suggest_actions(
        self, alert_type: AlertType, change: float, confidence: float
    ) -> list[str]:
        """Suggest actions based on alert type."""
        actions = ["review_position", "check_news"]

        if alert_type == AlertType.SENTIMENT_SPIKE:
            if confidence > 0.8:
                actions.extend(["consider_entry", "set_alerts"])
            else:
                actions.append("wait_confirmation")
        elif alert_type == AlertType.SENTIMENT_CRASH:
            if confidence > 0.8:
                actions.extend(["consider_exit", "tighten_stops"])
            else:
                actions.append("monitor_closely")

        return actions

    def _suggest_event_actions(self, event_type: str) -> list[str]:
        """Suggest actions for specific events."""
        action_map = {
            "earnings_beat": ["review_targets", "consider_adding"],
            "earnings_miss": ["review_position", "consider_reducing"],
            "upgrade": ["check_analyst_reputation", "review_targets"],
            "downgrade": ["review_thesis", "check_stops"],
            "merger": ["await_details", "check_arbitrage"],
            "lawsuit": ["assess_impact", "monitor_developments"],
        }

        return action_map.get(event_type, ["monitor_closely", "check_news"])

    def _interpret_volume_sentiment(self, volume_ratio: float, sentiment: float) -> str:
        """Interpret the combination of volume and sentiment."""
        if volume_ratio > 2 and sentiment > 0.3:
            return "bullish_accumulation"
        elif volume_ratio > 2 and sentiment < -0.3:
            return "bearish_distribution"
        elif volume_ratio > 3:
            return "high_volatility_expected"
        else:
            return "increased_interest"

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"ALERT_{timestamp}"

    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days)

        # Clear from main list
        self.alerts = [
            alert for alert in self.alerts if alert["timestamp"] > cutoff_time
        ]

        # Clear from history
        for symbol in list(self.alert_history.keys()):
            self.alert_history[symbol] = [
                alert
                for alert in self.alert_history[symbol]
                if alert["timestamp"] > cutoff_time
            ]

            if not self.alert_history[symbol]:
                del self.alert_history[symbol]

        logger.info(f"Cleared alerts older than {days} days")

    def get_status(self) -> dict[str, Any]:
        """Get the status of the alert manager."""
        return {
            "status": "active",
            "active_alerts": len(self.alerts),
            "symbols_monitored": len(self.alert_history),
            "notification_channels": len(self.notification_channels),
            "thresholds": self.alert_thresholds,
        }
