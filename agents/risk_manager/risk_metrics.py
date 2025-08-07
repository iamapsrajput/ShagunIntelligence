"""Risk metrics calculator for VaR, drawdown, and other portfolio metrics."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class RiskMetricsCalculator:
    """Calculate various risk metrics for portfolio management."""

    def __init__(self):
        """Initialize risk metrics calculator."""
        self.metrics_history = []
        self.peak_values = {}

        logger.info("RiskMetricsCalculator initialized")

    def calculate_portfolio_var(
        self,
        positions: list[dict[str, Any]],
        market_data: dict[str, pd.DataFrame],
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "historical",
    ) -> dict[str, Any]:
        """
        Calculate Value at Risk (VaR) for the portfolio.

        Args:
            positions: List of current positions
            market_data: Historical price data for each position
            confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            VaR analysis results
        """
        try:
            if not positions:
                return {
                    "var_amount": 0,
                    "var_percentage": 0,
                    "confidence_level": confidence_level,
                    "method": method,
                }

            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(
                positions, market_data
            )

            if portfolio_returns is None or len(portfolio_returns) < 20:
                return {
                    "var_amount": 0,
                    "var_percentage": 0,
                    "error": "Insufficient data for VaR calculation",
                }

            # Calculate VaR based on method
            if method == "historical":
                var_return = self._historical_var(portfolio_returns, confidence_level)
            elif method == "parametric":
                var_return = self._parametric_var(portfolio_returns, confidence_level)
            elif method == "monte_carlo":
                var_return = self._monte_carlo_var(
                    portfolio_returns, confidence_level, time_horizon
                )
            else:
                var_return = self._historical_var(portfolio_returns, confidence_level)

            # Scale to time horizon
            var_return_scaled = var_return * np.sqrt(time_horizon)

            # Calculate portfolio value
            portfolio_value = sum(
                abs(p["shares"] * p["current_price"]) for p in positions
            )

            # Calculate VaR amount
            var_amount = abs(var_return_scaled * portfolio_value)

            return {
                "var_amount": var_amount,
                "var_percentage": abs(var_return_scaled),
                "var_return": var_return,
                "confidence_level": confidence_level,
                "time_horizon": time_horizon,
                "method": method,
                "portfolio_value": portfolio_value,
                "interpretation": self._interpret_var(var_amount, portfolio_value),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return {"var_amount": 0, "var_percentage": 0, "error": str(e)}

    def calculate_cvar(
        self,
        positions: list[dict[str, Any]],
        market_data: dict[str, pd.DataFrame],
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            positions: List of positions
            market_data: Historical price data
            confidence_level: Confidence level

        Returns:
            CVaR analysis
        """
        try:
            # Get portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(
                positions, market_data
            )

            if portfolio_returns is None or len(portfolio_returns) < 20:
                return {"cvar_amount": 0, "cvar_percentage": 0}

            # Calculate VaR threshold
            var_threshold = np.percentile(
                portfolio_returns, (1 - confidence_level) * 100
            )

            # Calculate expected value of returns below VaR
            tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
            cvar_return = (
                np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
            )

            # Calculate portfolio value
            portfolio_value = sum(
                abs(p["shares"] * p["current_price"]) for p in positions
            )

            cvar_amount = abs(cvar_return * portfolio_value)

            return {
                "cvar_amount": cvar_amount,
                "cvar_percentage": abs(cvar_return),
                "var_threshold": var_threshold,
                "tail_observations": len(tail_returns),
                "confidence_level": confidence_level,
                "interpretation": self._interpret_cvar(cvar_amount, portfolio_value),
            }

        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            return {"cvar_amount": 0, "cvar_percentage": 0, "error": str(e)}

    def calculate_max_drawdown(
        self,
        capital: float,
        positions: list[dict[str, Any]],
        market_data: dict[str, pd.DataFrame],
        lookback_days: int = 252,
    ) -> dict[str, Any]:
        """
        Calculate maximum drawdown metrics.

        Args:
            capital: Total capital
            positions: Current positions
            market_data: Historical price data
            lookback_days: Days to look back for max drawdown

        Returns:
            Drawdown analysis
        """
        try:
            # Calculate historical portfolio values
            portfolio_values = self._calculate_historical_portfolio_values(
                positions, market_data, lookback_days
            )

            if portfolio_values is None or len(portfolio_values) < 2:
                current_value = (
                    sum(abs(p["shares"] * p["current_price"]) for p in positions)
                    + capital
                )

                # Simple drawdown from capital
                return {
                    "max_drawdown": 0,
                    "current_drawdown": 0,
                    "peak_value": current_value,
                    "trough_value": current_value,
                    "recovery_time": None,
                }

            # Calculate drawdown series
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak

            # Find maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_idx = drawdown.idxmin()

            # Find peak before max drawdown
            peak_idx = portfolio_values[:max_drawdown_idx].idxmax()

            # Calculate recovery time if recovered
            recovery_idx = None
            if max_drawdown < 0:
                post_trough = portfolio_values[max_drawdown_idx:]
                recovery_mask = post_trough >= portfolio_values[peak_idx]
                if recovery_mask.any():
                    recovery_idx = recovery_mask.idxmax()

            # Current drawdown
            current_value = portfolio_values.iloc[-1]
            current_peak = peak.iloc[-1]
            current_drawdown = (
                (current_value - current_peak) / current_peak if current_peak > 0 else 0
            )

            return {
                "max_drawdown": max_drawdown,
                "max_drawdown_percentage": abs(max_drawdown) * 100,
                "current_drawdown": current_drawdown,
                "current_drawdown_percentage": abs(current_drawdown) * 100,
                "peak_value": portfolio_values[peak_idx] if peak_idx else current_peak,
                "peak_date": peak_idx,
                "trough_value": (
                    portfolio_values[max_drawdown_idx]
                    if max_drawdown_idx
                    else current_value
                ),
                "trough_date": max_drawdown_idx,
                "recovery_date": recovery_idx,
                "recovery_time": (
                    (recovery_idx - max_drawdown_idx).days if recovery_idx else None
                ),
                "drawdown_duration": (
                    (max_drawdown_idx - peak_idx).days
                    if peak_idx and max_drawdown_idx
                    else None
                ),
                "in_drawdown": current_drawdown < -0.01,  # More than 1% drawdown
                "interpretation": self._interpret_drawdown(
                    max_drawdown, current_drawdown
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            return {"max_drawdown": 0, "current_drawdown": 0, "error": str(e)}

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            # Calculate excess returns
            daily_rf_rate = risk_free_rate / periods_per_year
            excess_returns = returns - daily_rf_rate

            # Calculate Sharpe ratio
            mean_excess = excess_returns.mean()
            std_excess = excess_returns.std()

            if std_excess == 0:
                return 0.0

            sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)

            return sharpe

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Sortino ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            # Calculate excess returns
            daily_rf_rate = risk_free_rate / periods_per_year
            excess_returns = returns - daily_rf_rate

            # Calculate downside deviation
            negative_returns = excess_returns[excess_returns < 0]

            if len(negative_returns) == 0:
                return float("inf")  # No negative returns

            downside_std = np.sqrt(np.mean(negative_returns**2))

            if downside_std == 0:
                return 0.0

            mean_excess = excess_returns.mean()
            sortino = mean_excess / downside_std * np.sqrt(periods_per_year)

            return sortino

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0

    def calculate_calmar_ratio(
        self, returns: pd.Series, max_drawdown: float, periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).

        Args:
            returns: Return series
            max_drawdown: Maximum drawdown (as negative number)
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        try:
            if len(returns) < periods_per_year or max_drawdown >= 0:
                return 0.0

            # Calculate annualized return
            total_return = (1 + returns).prod() - 1
            years = len(returns) / periods_per_year
            annual_return = (1 + total_return) ** (1 / years) - 1

            # Calculate Calmar ratio
            calmar = annual_return / abs(max_drawdown)

            return calmar

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return 0.0

    def calculate_risk_adjusted_metrics(
        self,
        positions: list[dict[str, Any]],
        market_data: dict[str, pd.DataFrame],
        risk_free_rate: float = 0.02,
    ) -> dict[str, Any]:
        """
        Calculate comprehensive risk-adjusted performance metrics.

        Args:
            positions: Portfolio positions
            market_data: Historical price data
            risk_free_rate: Risk-free rate

        Returns:
            Risk-adjusted metrics
        """
        try:
            # Get portfolio returns
            returns = self._calculate_portfolio_returns(positions, market_data)

            if returns is None or len(returns) < 20:
                return {
                    "sharpe_ratio": 0,
                    "sortino_ratio": 0,
                    "calmar_ratio": 0,
                    "error": "Insufficient data",
                }

            # Calculate metrics
            sharpe = self.calculate_sharpe_ratio(returns, risk_free_rate)
            sortino = self.calculate_sortino_ratio(returns, risk_free_rate)

            # Get max drawdown for Calmar
            drawdown_info = self.calculate_max_drawdown(0, positions, market_data)
            max_dd = drawdown_info.get("max_drawdown", 0)

            calmar = self.calculate_calmar_ratio(returns, max_dd)

            # Calculate additional metrics
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)

            return {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "return_per_unit_risk": (
                    annual_return / annual_vol if annual_vol > 0 else 0
                ),
                "interpretation": self._interpret_risk_metrics(sharpe, sortino, calmar),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {str(e)}")
            return {"error": str(e)}

    def _calculate_portfolio_returns(
        self, positions: list[dict[str, Any]], market_data: dict[str, pd.DataFrame]
    ) -> pd.Series | None:
        """Calculate historical portfolio returns."""
        try:
            if not positions:
                return None

            # Get returns for each position
            weighted_returns = []
            total_value = sum(abs(p["shares"] * p["current_price"]) for p in positions)

            for position in positions:
                symbol = position["symbol"]
                if symbol not in market_data:
                    continue

                # Calculate position weight
                position_value = abs(position["shares"] * position["current_price"])
                weight = position_value / total_value if total_value > 0 else 0

                # Get returns
                prices = market_data[symbol]["close"]
                returns = prices.pct_change().dropna()

                # Apply weight
                weighted_returns.append(returns * weight)

            if not weighted_returns:
                return None

            # Sum weighted returns
            portfolio_returns = pd.concat(weighted_returns, axis=1).sum(axis=1)

            return portfolio_returns

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {str(e)}")
            return None

    def _calculate_historical_portfolio_values(
        self,
        positions: list[dict[str, Any]],
        market_data: dict[str, pd.DataFrame],
        lookback_days: int,
    ) -> pd.Series | None:
        """Calculate historical portfolio values."""
        try:
            if not positions:
                return None

            portfolio_values = None

            for position in positions:
                symbol = position["symbol"]
                if symbol not in market_data:
                    continue

                # Get historical prices
                prices = market_data[symbol]["close"].tail(lookback_days)

                # Calculate position values
                position_values = prices * position["shares"]

                if portfolio_values is None:
                    portfolio_values = position_values
                else:
                    portfolio_values = portfolio_values.add(
                        position_values, fill_value=0
                    )

            return portfolio_values

        except Exception as e:
            logger.error(f"Error calculating historical values: {str(e)}")
            return None

    def _historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate historical VaR."""
        return np.percentile(returns, (1 - confidence_level) * 100)

    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate parametric VaR (assumes normal distribution)."""
        mean_return = returns.mean()
        std_return = returns.std()

        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)

        return mean_return + z_score * std_return

    def _monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_level: float,
        time_horizon: int,
        num_simulations: int = 10000,
    ) -> float:
        """Calculate Monte Carlo VaR."""
        mean_return = returns.mean()
        std_return = returns.std()

        # Generate simulations
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            num_simulations,
        )

        return np.percentile(simulated_returns, (1 - confidence_level) * 100)

    def _interpret_var(self, var_amount: float, portfolio_value: float) -> str:
        """Interpret VaR results."""
        var_pct = (var_amount / portfolio_value) * 100 if portfolio_value > 0 else 0

        if var_pct > 20:
            return "Critical risk - potential for severe losses"
        elif var_pct > 10:
            return "High risk - significant loss potential"
        elif var_pct > 5:
            return "Moderate risk - normal market conditions"
        else:
            return "Low risk - well-controlled exposure"

    def _interpret_cvar(self, cvar_amount: float, portfolio_value: float) -> str:
        """Interpret CVaR results."""
        cvar_pct = (cvar_amount / portfolio_value) * 100 if portfolio_value > 0 else 0

        if cvar_pct > 25:
            return "Extreme tail risk - potential for catastrophic losses"
        elif cvar_pct > 15:
            return "High tail risk - significant extreme event exposure"
        elif cvar_pct > 8:
            return "Moderate tail risk - manageable extreme scenarios"
        else:
            return "Low tail risk - limited extreme loss potential"

    def _interpret_drawdown(self, max_dd: float, current_dd: float) -> str:
        """Interpret drawdown results."""
        if current_dd < -0.10:
            return "Currently in significant drawdown - risk management critical"
        elif current_dd < -0.05:
            return "In moderate drawdown - monitor closely"
        elif abs(max_dd) > 0.20:
            return "Historical evidence of severe drawdowns"
        elif abs(max_dd) > 0.10:
            return "Moderate historical drawdowns"
        else:
            return "Limited drawdown history - good risk control"

    def _interpret_risk_metrics(
        self, sharpe: float, sortino: float, calmar: float
    ) -> str:
        """Interpret risk-adjusted metrics."""
        if sharpe > 2:
            return "Excellent risk-adjusted returns"
        elif sharpe > 1:
            return "Good risk-adjusted returns"
        elif sharpe > 0.5:
            return "Acceptable risk-adjusted returns"
        elif sharpe > 0:
            return "Poor risk-adjusted returns"
        else:
            return "Negative risk-adjusted returns - review strategy"

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of calculated metrics."""
        if not self.metrics_history:
            return {"message": "No metrics history available"}

        recent_metrics = self.metrics_history[-10:]

        return {
            "total_calculations": len(self.metrics_history),
            "recent_metrics": recent_metrics,
            "timestamp": datetime.now().isoformat(),
        }
