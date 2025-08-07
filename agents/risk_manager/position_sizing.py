"""Position sizing calculator with multiple algorithms."""

from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger


class PositionSizer:
    """Calculate optimal position sizes using various algorithms."""

    def __init__(self, capital: float, max_risk_per_trade: float = 0.02):
        """
        Initialize position sizer.

        Args:
            capital: Total trading capital
            max_risk_per_trade: Maximum risk per trade (default 2%)
        """
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.position_history = []

        logger.info(f"PositionSizer initialized with capital: ${capital:,.2f}")

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        volatility: float,
        confidence: float = 1.0,
        method: str = "fixed_risk",
    ) -> dict[str, Any]:
        """
        Calculate position size using specified method.

        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            volatility: Asset volatility (annualized)
            confidence: Confidence level in the trade (0-1)
            method: Sizing method to use

        Returns:
            Position sizing details
        """
        try:
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)

            if risk_per_share == 0:
                logger.warning("Risk per share is zero, returning minimum position")
                return self._minimum_position()

            # Base position size calculation
            if method == "fixed_risk":
                shares = self._fixed_risk_method(risk_per_share, confidence)
            elif method == "percent_volatility":
                shares = self._percent_volatility_method(entry_price, volatility)
            elif method == "fixed_ratio":
                shares = self._fixed_ratio_method(entry_price)
            else:
                shares = self._fixed_risk_method(risk_per_share, confidence)

            # Apply position limits
            shares = self._apply_position_limits(shares, entry_price)

            # Calculate position metrics
            position_value = shares * entry_price
            risk_amount = shares * risk_per_share
            risk_percentage = risk_amount / self.capital

            position_details = {
                "shares": int(shares),
                "position_value": position_value,
                "risk_amount": risk_amount,
                "risk_percentage": risk_percentage,
                "method_used": method,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "risk_per_share": risk_per_share,
                "capital_allocated_pct": (position_value / self.capital) * 100,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in history
            self.position_history.append(position_details)

            return position_details

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return self._minimum_position()

    def _fixed_risk_method(self, risk_per_share: float, confidence: float) -> float:
        """Fixed percentage risk per trade method."""
        # Adjust risk based on confidence
        adjusted_risk = self.max_risk_per_trade * confidence
        risk_amount = self.capital * adjusted_risk

        return risk_amount / risk_per_share

    def _percent_volatility_method(self, price: float, volatility: float) -> float:
        """Size position based on volatility."""
        # Target 1% daily volatility impact
        daily_volatility = volatility / np.sqrt(252)
        target_volatility = 0.01

        # Position size to achieve target volatility
        position_value = (self.capital * target_volatility) / daily_volatility

        return position_value / price

    def _fixed_ratio_method(self, price: float) -> float:
        """Fixed ratio of capital per position."""
        # Use 5% of capital per position
        position_value = self.capital * 0.05

        return position_value / price

    def kelly_criterion_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        entry_price: float,
        stop_loss: float,
        kelly_fraction: float = 0.25,
    ) -> dict[str, Any]:
        """
        Calculate position size using Kelly Criterion.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            entry_price: Entry price
            stop_loss: Stop loss price
            kelly_fraction: Fraction of Kelly to use (default 25%)

        Returns:
            Position sizing details
        """
        try:
            # Kelly formula: f = (p * b - q) / b
            # where p = win rate, q = loss rate, b = win/loss ratio

            if avg_loss == 0:
                logger.warning("Average loss is zero, using minimum position")
                return self._minimum_position()

            loss_rate = 1 - win_rate
            win_loss_ratio = avg_win / avg_loss

            # Calculate Kelly percentage
            kelly_percentage = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

            # Apply Kelly fraction (never use full Kelly)
            kelly_percentage = max(0, kelly_percentage * kelly_fraction)

            # Limit to maximum risk
            kelly_percentage = min(kelly_percentage, self.max_risk_per_trade)

            # Calculate position size
            risk_amount = self.capital * kelly_percentage
            risk_per_share = abs(entry_price - stop_loss)
            shares = risk_amount / risk_per_share if risk_per_share > 0 else 0

            # Apply limits
            shares = self._apply_position_limits(shares, entry_price)

            return {
                "shares": int(shares),
                "position_value": shares * entry_price,
                "risk_amount": shares * risk_per_share,
                "risk_percentage": kelly_percentage,
                "kelly_percentage": kelly_percentage,
                "kelly_fraction_used": kelly_fraction,
                "method_used": "kelly_criterion",
                "win_rate": win_rate,
                "win_loss_ratio": win_loss_ratio,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in Kelly criterion calculation: {str(e)}")
            return self._minimum_position()

    def volatility_based_size(
        self, entry_price: float, volatility: float, volatility_target: float = 0.15
    ) -> dict[str, Any]:
        """
        Size position based on volatility targeting.

        Args:
            entry_price: Entry price
            volatility: Asset volatility (annualized)
            volatility_target: Target portfolio volatility

        Returns:
            Position sizing details
        """
        try:
            # Calculate position weight to achieve target volatility
            if volatility == 0:
                logger.warning("Zero volatility, using minimum position")
                return self._minimum_position()

            position_weight = volatility_target / volatility

            # Limit position weight
            position_weight = min(position_weight, 0.25)  # Max 25% per position

            # Calculate shares
            position_value = self.capital * position_weight
            shares = position_value / entry_price

            # Apply limits
            shares = self._apply_position_limits(shares, entry_price)

            return {
                "shares": int(shares),
                "position_value": shares * entry_price,
                "position_weight": position_weight,
                "volatility": volatility,
                "volatility_target": volatility_target,
                "method_used": "volatility_based",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in volatility-based sizing: {str(e)}")
            return self._minimum_position()

    def risk_parity_size(
        self,
        entry_price: float,
        volatility: float,
        correlation_with_portfolio: float,
        num_positions: int,
    ) -> dict[str, Any]:
        """
        Calculate position size using risk parity approach.

        Args:
            entry_price: Entry price
            volatility: Asset volatility
            correlation_with_portfolio: Correlation with existing portfolio
            num_positions: Number of positions in portfolio

        Returns:
            Position sizing details
        """
        try:
            # Risk parity: each position contributes equally to portfolio risk
            target_risk_contribution = 1.0 / (num_positions + 1)

            # Adjust for correlation
            correlation_adjustment = 1.0 - abs(correlation_with_portfolio) * 0.5

            # Calculate position weight
            base_weight = target_risk_contribution / (
                volatility * correlation_adjustment
            )

            # Apply constraints
            position_weight = min(base_weight, 0.2)  # Max 20% per position

            # Calculate shares
            position_value = self.capital * position_weight
            shares = position_value / entry_price

            # Apply limits
            shares = self._apply_position_limits(shares, entry_price)

            return {
                "shares": int(shares),
                "position_value": shares * entry_price,
                "position_weight": position_weight,
                "risk_contribution": target_risk_contribution,
                "correlation_adjustment": correlation_adjustment,
                "method_used": "risk_parity",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in risk parity sizing: {str(e)}")
            return self._minimum_position()

    def update_capital(self, new_capital: float):
        """Update available capital."""
        self.capital = new_capital
        logger.info(f"Capital updated to: ${new_capital:,.2f}")

    def _apply_position_limits(self, shares: float, price: float) -> float:
        """Apply position sizing limits."""
        # Maximum position value (10% of capital)
        max_position_value = self.capital * 0.10
        max_shares_by_value = max_position_value / price

        # Minimum position (0.5% of capital)
        min_position_value = self.capital * 0.005
        min_shares = max(1, min_position_value / price)

        # Apply limits
        shares = max(min_shares, min(shares, max_shares_by_value))

        return shares

    def _minimum_position(self) -> dict[str, Any]:
        """Return minimum position details."""
        return {
            "shares": 0,
            "position_value": 0,
            "risk_amount": 0,
            "risk_percentage": 0,
            "method_used": "minimum",
            "error": "Unable to calculate position size",
            "timestamp": datetime.now().isoformat(),
        }

    def get_position_stats(self) -> dict[str, Any]:
        """Get statistics on position sizing history."""
        if not self.position_history:
            return {"message": "No position history available"}

        risk_percentages = [p["risk_percentage"] for p in self.position_history]
        position_values = [p["position_value"] for p in self.position_history]

        return {
            "total_positions": len(self.position_history),
            "average_risk_percentage": np.mean(risk_percentages),
            "max_risk_percentage": max(risk_percentages),
            "average_position_value": np.mean(position_values),
            "total_capital_allocated": sum(position_values),
            "methods_used": list({p["method_used"] for p in self.position_history}),
        }
