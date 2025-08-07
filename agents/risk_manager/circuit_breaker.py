"""Circuit breaker for extreme market conditions and risk management."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class CircuitBreakerStatus(Enum):
    """Circuit breaker status levels."""

    NORMAL = "normal"
    WARNING = "warning"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


class CircuitBreaker:
    """Implement circuit breakers for extreme market conditions."""

    def __init__(self):
        """Initialize circuit breaker."""
        # Trigger thresholds
        self.thresholds = {
            "market_drop": -0.03,  # 3% market drop
            "portfolio_loss": -0.05,  # 5% portfolio loss
            "volatility_spike": 2.5,  # 2.5x normal volatility
            "correlation_spike": 0.9,  # 90% correlation threshold
            "drawdown_limit": -0.10,  # 10% drawdown
            "loss_streak": 5,  # 5 consecutive losses
            "daily_loss_limit": -0.02,  # 2% daily loss
            "position_loss_limit": -0.08,  # 8% single position loss
        }

        # Circuit breaker state
        self.status = CircuitBreakerStatus.NORMAL
        self.triggered_conditions = []
        self.trigger_history = []
        self.cooldown_period = timedelta(minutes=30)
        self.last_trigger_time = None

        # Tracking metrics
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.volatility_baseline = None

        logger.info("CircuitBreaker initialized")

    def check_conditions(
        self,
        market_data: pd.DataFrame,
        positions: dict[str, Any],
        portfolio_value: float | None = None,
        initial_capital: float | None = None,
    ) -> dict[str, Any]:
        """
        Check all circuit breaker conditions.

        Args:
            market_data: Current market data
            positions: Current portfolio positions
            portfolio_value: Current portfolio value
            initial_capital: Initial capital for drawdown calculation

        Returns:
            Circuit breaker status and conditions
        """
        try:
            conditions = {}
            triggered = []

            # Check if in cooldown
            if self._is_in_cooldown():
                return {
                    "status": CircuitBreakerStatus.COOLDOWN.value,
                    "conditions": {},
                    "triggered": [],
                    "time_until_reset": self._time_until_reset(),
                    "message": "Circuit breaker in cooldown period",
                }

            # 1. Market drop check
            market_drop = self._check_market_drop(market_data)
            conditions["market_drop"] = market_drop
            if market_drop["triggered"]:
                triggered.append("market_drop")

            # 2. Volatility spike check
            volatility_spike = self._check_volatility_spike(market_data)
            conditions["volatility_spike"] = volatility_spike
            if volatility_spike["triggered"]:
                triggered.append("volatility_spike")

            # 3. Portfolio loss check
            if portfolio_value and initial_capital:
                portfolio_loss = self._check_portfolio_loss(
                    portfolio_value, initial_capital
                )
                conditions["portfolio_loss"] = portfolio_loss
                if portfolio_loss["triggered"]:
                    triggered.append("portfolio_loss")

                # 4. Drawdown check
                drawdown = self._check_drawdown(portfolio_value, initial_capital)
                conditions["drawdown"] = drawdown
                if drawdown["triggered"]:
                    triggered.append("drawdown")

            # 5. Loss streak check
            loss_streak = self._check_loss_streak()
            conditions["loss_streak"] = loss_streak
            if loss_streak["triggered"]:
                triggered.append("loss_streak")

            # 6. Daily loss limit check
            daily_loss = self._check_daily_loss_limit()
            conditions["daily_loss"] = daily_loss
            if daily_loss["triggered"]:
                triggered.append("daily_loss")

            # 7. Position-specific checks
            if positions:
                position_losses = self._check_position_losses(positions)
                conditions["position_losses"] = position_losses
                if position_losses["triggered"]:
                    triggered.append("position_losses")

            # Determine overall status
            if triggered:
                self._trigger_circuit_breaker(triggered)
                status = CircuitBreakerStatus.TRIGGERED
            elif self._approaching_limits(conditions):
                status = CircuitBreakerStatus.WARNING
            else:
                status = CircuitBreakerStatus.NORMAL

            self.status = status

            return {
                "status": status.value,
                "conditions": conditions,
                "triggered": triggered,
                "approaching_trigger": self._approaching_limits(conditions),
                "message": self._generate_status_message(status, triggered),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error checking circuit breaker conditions: {str(e)}")
            return {"status": CircuitBreakerStatus.NORMAL.value, "error": str(e)}

    def check_portfolio_circuit(
        self,
        total_exposure: dict[str, Any],
        portfolio_var: dict[str, Any],
        current_drawdown: float,
    ) -> dict[str, Any]:
        """
        Check portfolio-level circuit breaker conditions.

        Args:
            total_exposure: Portfolio exposure metrics
            portfolio_var: Value at Risk metrics
            current_drawdown: Current drawdown percentage

        Returns:
            Portfolio circuit breaker status
        """
        try:
            triggered = []

            # Check exposure limits
            if total_exposure.get("leverage", 0) > 2.0:
                triggered.append("excessive_leverage")

            # Check VaR limits
            if portfolio_var.get("var_percentage", 0) > 0.15:  # 15% VaR
                triggered.append("high_var")

            # Check drawdown
            if current_drawdown < self.thresholds["drawdown_limit"]:
                triggered.append("drawdown_breach")

            status = (
                CircuitBreakerStatus.TRIGGERED
                if triggered
                else CircuitBreakerStatus.NORMAL
            )

            return {
                "status": status.value,
                "triggered": triggered,
                "exposure_check": total_exposure.get("leverage", 0) <= 2.0,
                "var_check": portfolio_var.get("var_percentage", 0) <= 0.15,
                "drawdown_check": current_drawdown >= self.thresholds["drawdown_limit"],
            }

        except Exception as e:
            logger.error(f"Error checking portfolio circuit: {str(e)}")
            return {"status": CircuitBreakerStatus.NORMAL.value, "error": str(e)}

    def update_daily_pnl(self, pnl: float):
        """Update daily P&L for tracking."""
        self.daily_pnl = pnl

        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def reset_daily_metrics(self):
        """Reset daily tracking metrics."""
        self.daily_pnl = 0
        logger.info("Daily circuit breaker metrics reset")

    def manual_trigger(self, reason: str):
        """Manually trigger circuit breaker."""
        self._trigger_circuit_breaker([f"manual: {reason}"])
        logger.warning(f"Circuit breaker manually triggered: {reason}")

    def reset(self):
        """Reset circuit breaker to normal state."""
        self.status = CircuitBreakerStatus.NORMAL
        self.triggered_conditions = []
        self.last_trigger_time = None
        logger.info("Circuit breaker reset to normal")

    def is_triggered(self) -> bool:
        """Check if circuit breaker is currently triggered."""
        return self.status == CircuitBreakerStatus.TRIGGERED

    def _check_market_drop(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Check for significant market drop."""
        try:
            if len(market_data) < 2:
                return {"triggered": False, "value": 0}

            # Calculate market return
            current_price = market_data["close"].iloc[-1]
            prev_price = market_data["close"].iloc[-2]
            market_return = (current_price - prev_price) / prev_price

            triggered = market_return < self.thresholds["market_drop"]

            return {
                "triggered": triggered,
                "value": market_return,
                "threshold": self.thresholds["market_drop"],
                "severity": (
                    abs(market_return / self.thresholds["market_drop"])
                    if triggered
                    else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error checking market drop: {str(e)}")
            return {"triggered": False, "error": str(e)}

    def _check_volatility_spike(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Check for volatility spike."""
        try:
            if len(market_data) < 20:
                return {"triggered": False, "value": 0}

            # Calculate current volatility
            returns = market_data["close"].pct_change().dropna()
            current_vol = returns.tail(5).std() * np.sqrt(252)  # 5-day volatility

            # Calculate baseline if not set
            if self.volatility_baseline is None:
                self.volatility_baseline = returns.tail(20).std() * np.sqrt(252)

            # Check for spike
            if self.volatility_baseline > 0:
                vol_ratio = current_vol / self.volatility_baseline
                triggered = vol_ratio > self.thresholds["volatility_spike"]
            else:
                vol_ratio = 0
                triggered = False

            return {
                "triggered": triggered,
                "value": vol_ratio,
                "current_volatility": current_vol,
                "baseline_volatility": self.volatility_baseline,
                "threshold": self.thresholds["volatility_spike"],
            }

        except Exception as e:
            logger.error(f"Error checking volatility spike: {str(e)}")
            return {"triggered": False, "error": str(e)}

    def _check_portfolio_loss(
        self, current_value: float, initial_value: float
    ) -> dict[str, Any]:
        """Check for portfolio loss threshold."""
        try:
            portfolio_return = (current_value - initial_value) / initial_value
            triggered = portfolio_return < self.thresholds["portfolio_loss"]

            return {
                "triggered": triggered,
                "value": portfolio_return,
                "threshold": self.thresholds["portfolio_loss"],
                "current_value": current_value,
                "initial_value": initial_value,
            }

        except Exception as e:
            logger.error(f"Error checking portfolio loss: {str(e)}")
            return {"triggered": False, "error": str(e)}

    def _check_drawdown(
        self, current_value: float, peak_value: float
    ) -> dict[str, Any]:
        """Check for maximum drawdown breach."""
        try:
            drawdown = (current_value - peak_value) / peak_value
            triggered = drawdown < self.thresholds["drawdown_limit"]

            return {
                "triggered": triggered,
                "value": drawdown,
                "threshold": self.thresholds["drawdown_limit"],
                "drawdown_percentage": abs(drawdown) * 100,
            }

        except Exception as e:
            logger.error(f"Error checking drawdown: {str(e)}")
            return {"triggered": False, "error": str(e)}

    def _check_loss_streak(self) -> dict[str, Any]:
        """Check for consecutive loss streak."""
        triggered = self.consecutive_losses >= self.thresholds["loss_streak"]

        return {
            "triggered": triggered,
            "value": self.consecutive_losses,
            "threshold": self.thresholds["loss_streak"],
        }

    def _check_daily_loss_limit(self) -> dict[str, Any]:
        """Check daily loss limit."""
        triggered = self.daily_pnl < self.thresholds["daily_loss_limit"]

        return {
            "triggered": triggered,
            "value": self.daily_pnl,
            "threshold": self.thresholds["daily_loss_limit"],
        }

    def _check_position_losses(self, positions: dict[str, Any]) -> dict[str, Any]:
        """Check for excessive single position losses."""
        try:
            max_loss = 0
            worst_position = None
            positions_at_risk = []

            for symbol, position in positions.items():
                if "unrealized_pnl_percentage" in position:
                    pnl_pct = position["unrealized_pnl_percentage"]

                    if pnl_pct < max_loss:
                        max_loss = pnl_pct
                        worst_position = symbol

                    if pnl_pct < self.thresholds["position_loss_limit"]:
                        positions_at_risk.append({"symbol": symbol, "loss": pnl_pct})

            triggered = len(positions_at_risk) > 0

            return {
                "triggered": triggered,
                "worst_position": worst_position,
                "max_loss": max_loss,
                "positions_at_risk": positions_at_risk,
                "threshold": self.thresholds["position_loss_limit"],
            }

        except Exception as e:
            logger.error(f"Error checking position losses: {str(e)}")
            return {"triggered": False, "error": str(e)}

    def _trigger_circuit_breaker(self, conditions: list[str]):
        """Trigger the circuit breaker."""
        self.status = CircuitBreakerStatus.TRIGGERED
        self.triggered_conditions = conditions
        self.last_trigger_time = datetime.now()

        trigger_record = {
            "timestamp": self.last_trigger_time,
            "conditions": conditions,
            "status": self.status.value,
        }
        self.trigger_history.append(trigger_record)

        logger.warning(f"Circuit breaker TRIGGERED! Conditions: {conditions}")

    def _is_in_cooldown(self) -> bool:
        """Check if circuit breaker is in cooldown period."""
        if self.last_trigger_time is None:
            return False

        return datetime.now() - self.last_trigger_time < self.cooldown_period

    def _time_until_reset(self) -> str | None:
        """Get time until circuit breaker resets."""
        if self.last_trigger_time is None:
            return None

        time_elapsed = datetime.now() - self.last_trigger_time
        time_remaining = self.cooldown_period - time_elapsed

        if time_remaining.total_seconds() > 0:
            return str(time_remaining).split(".")[0]  # Remove microseconds

        return "Ready to reset"

    def _approaching_limits(self, conditions: dict[str, Any]) -> bool:
        """Check if approaching circuit breaker limits."""
        warning_threshold = 0.8  # 80% of limit

        for _key, condition in conditions.items():
            if (
                isinstance(condition, dict)
                and "value" in condition
                and "threshold" in condition
            ):
                threshold = condition["threshold"]
                value = condition["value"]

                # For negative thresholds (losses)
                if threshold < 0:
                    if value < 0 and value > threshold * warning_threshold:
                        return True
                # For positive thresholds
                else:
                    if value > threshold * warning_threshold:
                        return True

        return False

    def _generate_status_message(
        self, status: CircuitBreakerStatus, triggered_conditions: list[str]
    ) -> str:
        """Generate status message."""
        if status == CircuitBreakerStatus.TRIGGERED:
            return f"Trading halted due to: {', '.join(triggered_conditions)}"
        elif status == CircuitBreakerStatus.WARNING:
            return "Warning: Approaching circuit breaker limits"
        elif status == CircuitBreakerStatus.COOLDOWN:
            return f"In cooldown period. Reset in {self._time_until_reset()}"
        else:
            return "All systems normal"

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "status": self.status.value,
            "is_triggered": self.is_triggered(),
            "triggered_conditions": self.triggered_conditions,
            "last_trigger": (
                self.last_trigger_time.isoformat() if self.last_trigger_time else None
            ),
            "cooldown_remaining": self._time_until_reset(),
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "trigger_count": len(self.trigger_history),
            "thresholds": self.thresholds,
        }

    def update_thresholds(self, **kwargs):
        """Update circuit breaker thresholds."""
        for key, value in kwargs.items():
            if key in self.thresholds:
                self.thresholds[key] = value
                logger.info(f"Updated circuit breaker threshold {key} to {value}")

    def get_trigger_history(self, days: int = 7) -> list[dict[str, Any]]:
        """Get circuit breaker trigger history."""
        cutoff = datetime.now() - timedelta(days=days)

        return [
            trigger for trigger in self.trigger_history if trigger["timestamp"] > cutoff
        ]
