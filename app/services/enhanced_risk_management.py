"""
Enhanced Risk Management Framework
Comprehensive portfolio-level risk management with advanced metrics and controls
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from loguru import logger

from app.core.resilience import with_circuit_breaker, with_retry


class RiskLevel(Enum):
    """Risk level classifications"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskMetricType(Enum):
    """Types of risk metrics"""

    VAR = "VAR"  # Value at Risk
    CVAR = "CVAR"  # Conditional Value at Risk
    SHARPE = "SHARPE"  # Sharpe Ratio
    SORTINO = "SORTINO"  # Sortino Ratio
    CALMAR = "CALMAR"  # Calmar Ratio
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    BETA = "BETA"  # Market Beta
    CORRELATION = "CORRELATION"
    VOLATILITY = "VOLATILITY"


@dataclass
class RiskLimits:
    """Portfolio risk limits configuration"""

    max_portfolio_var: float = 0.05  # 5% daily VaR
    max_sector_exposure: float = 0.30  # 30% max per sector
    max_single_position: float = 0.10  # 10% max per position
    max_correlation: float = 0.70  # 70% max correlation
    max_leverage: float = 2.0  # 2x max leverage
    max_drawdown: float = 0.15  # 15% max drawdown
    min_sharpe_ratio: float = 0.5  # Minimum Sharpe ratio
    max_beta: float = 1.5  # Maximum portfolio beta

    # Dynamic limits based on market conditions
    volatility_adjustment: bool = True
    market_stress_multiplier: float = 0.5  # Reduce limits by 50% in stress


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio risk metrics"""

    total_value: float
    total_exposure: float
    leverage: float
    var_1d: float
    var_5d: float
    cvar_1d: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    alpha: float
    volatility: float
    correlation_matrix: dict[str, dict[str, float]]
    sector_exposures: dict[str, float]
    risk_level: RiskLevel
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedVaRCalculator:
    """Advanced Value at Risk calculation methods"""

    def __init__(self):
        self.confidence_levels = [0.95, 0.99, 0.999]
        self.time_horizons = [1, 5, 10, 22]  # Days

    def calculate_historical_var(
        self, returns: pd.Series, confidence: float = 0.95, time_horizon: int = 1
    ) -> dict[str, float]:
        """Calculate Historical VaR"""
        if len(returns) < 30:
            return {"var": 0.0, "method": "insufficient_data"}

        # Scale returns to time horizon
        scaled_returns = returns * np.sqrt(time_horizon)

        # Calculate percentile
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(scaled_returns, var_percentile)

        # Calculate Conditional VaR (Expected Shortfall)
        cvar_value = scaled_returns[scaled_returns <= var_value].mean()

        return {
            "var": abs(var_value),
            "cvar": abs(cvar_value),
            "confidence": confidence,
            "time_horizon": time_horizon,
            "method": "historical",
            "observations": len(returns),
        }

    def calculate_parametric_var(
        self, returns: pd.Series, confidence: float = 0.95, time_horizon: int = 1
    ) -> dict[str, float]:
        """Calculate Parametric VaR assuming normal distribution"""
        if len(returns) < 10:
            return {"var": 0.0, "method": "insufficient_data"}

        # Calculate mean and std
        mean_return = returns.mean()
        std_return = returns.std()

        # Scale to time horizon
        scaled_mean = mean_return * time_horizon
        scaled_std = std_return * np.sqrt(time_horizon)

        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence)
        var_value = scaled_mean + z_score * scaled_std

        # Calculate CVaR for normal distribution
        cvar_value = scaled_mean + scaled_std * stats.norm.pdf(z_score) / (
            1 - confidence
        )

        return {
            "var": abs(var_value),
            "cvar": abs(cvar_value),
            "confidence": confidence,
            "time_horizon": time_horizon,
            "method": "parametric",
            "mean": scaled_mean,
            "std": scaled_std,
        }

    def calculate_monte_carlo_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        time_horizon: int = 1,
        simulations: int = 10000,
    ) -> dict[str, float]:
        """Calculate Monte Carlo VaR"""
        if len(returns) < 20:
            return {"var": 0.0, "method": "insufficient_data"}

        # Fit distribution parameters
        mean_return = returns.mean()
        std_return = returns.std()

        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.normal(
            mean_return * time_horizon, std_return * np.sqrt(time_horizon), simulations
        )

        # Calculate VaR and CVaR
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(random_returns, var_percentile)
        cvar_value = random_returns[random_returns <= var_value].mean()

        return {
            "var": abs(var_value),
            "cvar": abs(cvar_value),
            "confidence": confidence,
            "time_horizon": time_horizon,
            "method": "monte_carlo",
            "simulations": simulations,
        }


class CorrelationAnalyzer:
    """Advanced correlation and dependency analysis"""

    def __init__(self):
        self.correlation_threshold = 0.7
        self.rolling_window = 60  # Days

    def calculate_correlation_matrix(
        self, returns_data: dict[str, pd.Series]
    ) -> dict[str, dict[str, float]]:
        """Calculate correlation matrix for portfolio assets"""
        if not returns_data or len(returns_data) < 2:
            return {}

        # Create DataFrame from returns
        df = pd.DataFrame(returns_data)

        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Convert to nested dict
        correlation_dict = {}
        for asset1 in corr_matrix.index:
            correlation_dict[asset1] = {}
            for asset2 in corr_matrix.columns:
                correlation_dict[asset1][asset2] = corr_matrix.loc[asset1, asset2]

        return correlation_dict

    def detect_high_correlations(
        self, correlation_matrix: dict[str, dict[str, float]]
    ) -> list[tuple[str, str, float]]:
        """Detect pairs with high correlation"""
        high_correlations = []

        for asset1, correlations in correlation_matrix.items():
            for asset2, corr_value in correlations.items():
                if asset1 != asset2 and abs(corr_value) > self.correlation_threshold:
                    # Avoid duplicates
                    pair = tuple(sorted([asset1, asset2]))
                    if (pair[0], pair[1], corr_value) not in high_correlations:
                        high_correlations.append((pair[0], pair[1], corr_value))

        return high_correlations

    def calculate_portfolio_diversification_ratio(
        self, weights: np.ndarray, correlation_matrix: np.ndarray
    ) -> float:
        """Calculate portfolio diversification ratio"""
        if len(weights) != correlation_matrix.shape[0]:
            return 0.0

        # Weighted average of individual volatilities
        individual_vol_weighted = np.sum(weights)  # Assuming unit volatilities

        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlation_matrix, weights)))

        # Diversification ratio
        if portfolio_vol > 0:
            return individual_vol_weighted / portfolio_vol
        return 0.0


class SectorExposureAnalyzer:
    """Sector exposure and concentration analysis"""

    def __init__(self):
        # Indian market sectors
        self.sector_mapping = {
            "RELIANCE": "Energy",
            "TCS": "IT",
            "HDFCBANK": "Banking",
            "INFY": "IT",
            "ICICIBANK": "Banking",
            "HINDUNILVR": "FMCG",
            "ITC": "FMCG",
            "SBIN": "Banking",
            "BHARTIARTL": "Telecom",
            "KOTAKBANK": "Banking",
            "LT": "Infrastructure",
            "ASIANPAINT": "Paints",
            "MARUTI": "Auto",
            "TITAN": "Consumer Discretionary",
            "ULTRACEMCO": "Cement",
        }

    def calculate_sector_exposures(
        self, positions: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate sector-wise exposure"""
        sector_exposures = {}
        total_exposure = 0

        for symbol, position in positions.items():
            sector = self.sector_mapping.get(symbol, "Others")
            exposure = abs(position.get("market_value", 0))

            if sector not in sector_exposures:
                sector_exposures[sector] = 0

            sector_exposures[sector] += exposure
            total_exposure += exposure

        # Convert to percentages
        if total_exposure > 0:
            for sector in sector_exposures:
                sector_exposures[sector] = sector_exposures[sector] / total_exposure

        return sector_exposures

    def check_sector_concentration(
        self, sector_exposures: dict[str, float], max_sector_exposure: float = 0.30
    ) -> dict[str, Any]:
        """Check for sector concentration risk"""
        violations = []
        max_exposure = 0
        max_sector = ""

        for sector, exposure in sector_exposures.items():
            if exposure > max_exposure:
                max_exposure = exposure
                max_sector = sector

            if exposure > max_sector_exposure:
                violations.append(
                    {
                        "sector": sector,
                        "exposure": exposure,
                        "limit": max_sector_exposure,
                        "excess": exposure - max_sector_exposure,
                    }
                )

        return {
            "violations": violations,
            "max_sector": max_sector,
            "max_exposure": max_exposure,
            "concentration_risk": len(violations) > 0,
            "herfindahl_index": sum(exp**2 for exp in sector_exposures.values()),
        }


class DynamicPositionSizer:
    """Dynamic position sizing based on market conditions and portfolio state"""

    def __init__(self, base_risk_per_trade: float = 0.02):
        self.base_risk_per_trade = base_risk_per_trade
        self.volatility_lookback = 20
        self.correlation_lookback = 60

    def calculate_dynamic_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        portfolio_state: dict[str, Any],
        market_conditions: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate position size with dynamic adjustments"""

        # Base position size
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return {"shares": 0, "reason": "No risk defined"}

        # Get portfolio metrics
        portfolio_value = portfolio_state.get("total_value", 100000)
        current_volatility = market_conditions.get("volatility", 0.02)
        market_stress = market_conditions.get("stress_level", 0.0)

        # Dynamic risk adjustment
        risk_multiplier = self._calculate_risk_multiplier(
            portfolio_state, market_conditions, current_volatility
        )

        adjusted_risk = self.base_risk_per_trade * risk_multiplier

        # Calculate position size
        risk_amount = portfolio_value * adjusted_risk
        shares = risk_amount / risk_per_share

        # Apply portfolio-level constraints
        max_position_value = portfolio_value * 0.10  # 10% max per position
        if shares * entry_price > max_position_value:
            shares = max_position_value / entry_price

        return {
            "shares": int(shares),
            "position_value": shares * entry_price,
            "risk_amount": risk_amount,
            "risk_percentage": adjusted_risk,
            "risk_multiplier": risk_multiplier,
            "base_risk": self.base_risk_per_trade,
            "adjustments": {
                "volatility_adj": market_conditions.get("volatility_adjustment", 1.0),
                "correlation_adj": portfolio_state.get("correlation_adjustment", 1.0),
                "stress_adj": 1.0 - market_stress * 0.5,
            },
        }

    def _calculate_risk_multiplier(
        self,
        portfolio_state: dict[str, Any],
        market_conditions: dict[str, Any],
        current_volatility: float,
    ) -> float:
        """Calculate dynamic risk multiplier"""

        multiplier = 1.0

        # Volatility adjustment
        baseline_volatility = 0.02  # 2% baseline
        vol_ratio = current_volatility / baseline_volatility
        if vol_ratio > 1.5:  # High volatility
            multiplier *= 0.7
        elif vol_ratio < 0.5:  # Low volatility
            multiplier *= 1.2

        # Portfolio drawdown adjustment
        current_drawdown = portfolio_state.get("current_drawdown", 0.0)
        if current_drawdown < -0.05:  # 5% drawdown
            multiplier *= 0.8
        elif current_drawdown < -0.10:  # 10% drawdown
            multiplier *= 0.6

        # Market stress adjustment
        stress_level = market_conditions.get("stress_level", 0.0)
        multiplier *= 1.0 - stress_level * 0.4

        # Ensure reasonable bounds
        return max(0.2, min(2.0, multiplier))


class EnhancedRiskManager:
    """Comprehensive Enhanced Risk Management Framework"""

    def __init__(self, risk_limits: RiskLimits | None = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.var_calculator = AdvancedVaRCalculator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.sector_analyzer = SectorExposureAnalyzer()
        self.position_sizer = DynamicPositionSizer()

        # Risk monitoring state
        self.risk_alerts = []
        self.risk_history = []
        self.last_risk_check = None

        # Performance tracking
        self.risk_metrics_cache = {}
        self.cache_expiry = timedelta(minutes=5)

        logger.info("Enhanced Risk Management Framework initialized")

    @with_circuit_breaker("risk_management")
    @with_retry(max_retries=2, delay=1.0)
    async def calculate_portfolio_metrics(
        self,
        positions: dict[str, dict[str, Any]],
        market_data: dict[str, pd.DataFrame],
        benchmark_data: pd.Series | None = None,
    ) -> PortfolioMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Calculate basic portfolio values
            total_value = sum(pos.get("market_value", 0) for pos in positions.values())
            total_exposure = sum(
                abs(pos.get("market_value", 0)) for pos in positions.values()
            )
            leverage = total_exposure / total_value if total_value > 0 else 0

            # Get portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(
                positions, market_data
            )

            # Calculate VaR metrics
            var_metrics = await self._calculate_var_metrics(portfolio_returns)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                portfolio_returns, benchmark_data
            )

            # Calculate correlation matrix
            returns_data = {
                symbol: data.pct_change().dropna()
                for symbol, data in market_data.items()
                if len(data) > 0
            }
            correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(
                returns_data
            )

            # Calculate sector exposures
            sector_exposures = self.sector_analyzer.calculate_sector_exposures(
                positions
            )

            # Determine risk level
            risk_level = self._determine_risk_level(
                var_metrics, performance_metrics, sector_exposures
            )

            return PortfolioMetrics(
                total_value=total_value,
                total_exposure=total_exposure,
                leverage=leverage,
                var_1d=var_metrics.get("var_1d", 0.0),
                var_5d=var_metrics.get("var_5d", 0.0),
                cvar_1d=var_metrics.get("cvar_1d", 0.0),
                expected_shortfall=var_metrics.get("expected_shortfall", 0.0),
                max_drawdown=performance_metrics.get("max_drawdown", 0.0),
                current_drawdown=performance_metrics.get("current_drawdown", 0.0),
                sharpe_ratio=performance_metrics.get("sharpe_ratio", 0.0),
                sortino_ratio=performance_metrics.get("sortino_ratio", 0.0),
                calmar_ratio=performance_metrics.get("calmar_ratio", 0.0),
                beta=performance_metrics.get("beta", 1.0),
                alpha=performance_metrics.get("alpha", 0.0),
                volatility=performance_metrics.get("volatility", 0.0),
                correlation_matrix=correlation_matrix,
                sector_exposures=sector_exposures,
                risk_level=risk_level,
            )

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            # Return default metrics
            return PortfolioMetrics(
                total_value=0,
                total_exposure=0,
                leverage=0,
                var_1d=0,
                var_5d=0,
                cvar_1d=0,
                expected_shortfall=0,
                max_drawdown=0,
                current_drawdown=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                beta=1,
                alpha=0,
                volatility=0,
                correlation_matrix={},
                sector_exposures={},
                risk_level=RiskLevel.CRITICAL,
            )

    async def _calculate_var_metrics(
        self, portfolio_returns: pd.Series
    ) -> dict[str, float]:
        """Calculate comprehensive VaR metrics"""
        if len(portfolio_returns) < 30:
            return {
                "var_1d": 0.0,
                "var_5d": 0.0,
                "cvar_1d": 0.0,
                "expected_shortfall": 0.0,
            }

        # Calculate VaR for different time horizons
        var_1d = self.var_calculator.calculate_historical_var(
            portfolio_returns, 0.95, 1
        )
        var_5d = self.var_calculator.calculate_historical_var(
            portfolio_returns, 0.95, 5
        )

        # Calculate Expected Shortfall (CVaR)
        expected_shortfall = var_1d.get("cvar", 0.0)

        return {
            "var_1d": var_1d.get("var", 0.0),
            "var_5d": var_5d.get("var", 0.0),
            "cvar_1d": var_1d.get("cvar", 0.0),
            "expected_shortfall": expected_shortfall,
        }

    def _calculate_performance_metrics(
        self, portfolio_returns: pd.Series, benchmark_data: pd.Series | None = None
    ) -> dict[str, float]:
        """Calculate performance and risk-adjusted metrics"""
        if len(portfolio_returns) < 20:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "beta": 1.0,
                "alpha": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
            }

        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (
            (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        )

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(252)
            if len(downside_returns) > 0
            else volatility
        )
        sortino_ratio = (
            (annual_return - risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else 0
        )

        # Drawdown calculations
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Beta and Alpha (if benchmark provided)
        beta = 1.0
        alpha = 0.0
        if benchmark_data is not None and len(benchmark_data) > 20:
            # Align data
            aligned_data = pd.concat(
                [portfolio_returns, benchmark_data], axis=1
            ).dropna()
            if len(aligned_data) > 20:
                portfolio_aligned = aligned_data.iloc[:, 0]
                benchmark_aligned = aligned_data.iloc[:, 1]

                # Calculate beta
                covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
                benchmark_variance = np.var(benchmark_aligned)
                beta = (
                    covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                )

                # Calculate alpha
                benchmark_return = benchmark_aligned.mean() * 252
                alpha = annual_return - (
                    risk_free_rate + beta * (benchmark_return - risk_free_rate)
                )

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "beta": beta,
            "alpha": alpha,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "annual_return": annual_return,
        }

    def _calculate_portfolio_returns(
        self, positions: dict[str, dict[str, Any]], market_data: dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Calculate portfolio returns from positions and market data"""
        try:
            portfolio_values = []
            dates = None

            # Get common date range
            for symbol, data in market_data.items():
                if symbol in positions and len(data) > 0:
                    if dates is None:
                        dates = data.index
                    else:
                        dates = dates.intersection(data.index)

            if dates is None or len(dates) < 2:
                return pd.Series(dtype=float)

            # Calculate portfolio value for each date
            for date in dates:
                portfolio_value = 0
                for symbol, position in positions.items():
                    if symbol in market_data and date in market_data[symbol].index:
                        price = (
                            market_data[symbol].loc[date, "close"]
                            if "close" in market_data[symbol].columns
                            else market_data[symbol].loc[date].iloc[0]
                        )
                        shares = position.get("shares", 0)
                        portfolio_value += shares * price

                portfolio_values.append(portfolio_value)

            # Calculate returns
            portfolio_series = pd.Series(portfolio_values, index=dates)
            returns = portfolio_series.pct_change().dropna()

            return returns

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {str(e)}")
            return pd.Series(dtype=float)

    def _determine_risk_level(
        self,
        var_metrics: dict[str, float],
        performance_metrics: dict[str, float],
        sector_exposures: dict[str, float],
    ) -> RiskLevel:
        """Determine overall portfolio risk level"""

        risk_score = 0

        # VaR-based scoring
        var_1d = var_metrics.get("var_1d", 0.0)
        if var_1d > 0.08:  # 8% daily VaR
            risk_score += 3
        elif var_1d > 0.05:  # 5% daily VaR
            risk_score += 2
        elif var_1d > 0.03:  # 3% daily VaR
            risk_score += 1

        # Drawdown-based scoring
        current_drawdown = performance_metrics.get("current_drawdown", 0.0)
        if current_drawdown < -0.15:  # 15% drawdown
            risk_score += 3
        elif current_drawdown < -0.10:  # 10% drawdown
            risk_score += 2
        elif current_drawdown < -0.05:  # 5% drawdown
            risk_score += 1

        # Concentration risk scoring
        max_sector_exposure = max(sector_exposures.values()) if sector_exposures else 0
        if max_sector_exposure > 0.5:  # 50% in one sector
            risk_score += 3
        elif max_sector_exposure > 0.4:  # 40% in one sector
            risk_score += 2
        elif max_sector_exposure > 0.3:  # 30% in one sector
            risk_score += 1

        # Determine risk level
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def check_risk_limits(
        self, portfolio_metrics: PortfolioMetrics
    ) -> dict[str, Any]:
        """Check portfolio against risk limits"""
        violations = []
        warnings = []

        # Check VaR limits
        if portfolio_metrics.var_1d > self.risk_limits.max_portfolio_var:
            violations.append(
                {
                    "type": "var_limit",
                    "current": portfolio_metrics.var_1d,
                    "limit": self.risk_limits.max_portfolio_var,
                    "severity": "high",
                }
            )

        # Check leverage limits
        if portfolio_metrics.leverage > self.risk_limits.max_leverage:
            violations.append(
                {
                    "type": "leverage_limit",
                    "current": portfolio_metrics.leverage,
                    "limit": self.risk_limits.max_leverage,
                    "severity": "high",
                }
            )

        # Check sector concentration
        for sector, exposure in portfolio_metrics.sector_exposures.items():
            if exposure > self.risk_limits.max_sector_exposure:
                violations.append(
                    {
                        "type": "sector_concentration",
                        "sector": sector,
                        "current": exposure,
                        "limit": self.risk_limits.max_sector_exposure,
                        "severity": "medium",
                    }
                )

        # Check drawdown limits
        if portfolio_metrics.current_drawdown < -self.risk_limits.max_drawdown:
            violations.append(
                {
                    "type": "drawdown_limit",
                    "current": abs(portfolio_metrics.current_drawdown),
                    "limit": self.risk_limits.max_drawdown,
                    "severity": "critical",
                }
            )

        # Check performance metrics
        if portfolio_metrics.sharpe_ratio < self.risk_limits.min_sharpe_ratio:
            warnings.append(
                {
                    "type": "low_sharpe",
                    "current": portfolio_metrics.sharpe_ratio,
                    "minimum": self.risk_limits.min_sharpe_ratio,
                    "severity": "low",
                }
            )

        # Check beta limits
        if portfolio_metrics.beta > self.risk_limits.max_beta:
            warnings.append(
                {
                    "type": "high_beta",
                    "current": portfolio_metrics.beta,
                    "limit": self.risk_limits.max_beta,
                    "severity": "medium",
                }
            )

        return {
            "violations": violations,
            "warnings": warnings,
            "risk_level": portfolio_metrics.risk_level.value,
            "overall_status": (
                "VIOLATION" if violations else "WARNING" if warnings else "OK"
            ),
            "timestamp": datetime.now(),
        }

    async def generate_risk_recommendations(
        self, portfolio_metrics: PortfolioMetrics, risk_check: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate actionable risk management recommendations"""
        recommendations = []

        # High VaR recommendations
        if portfolio_metrics.var_1d > self.risk_limits.max_portfolio_var:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "risk_reduction",
                    "action": "reduce_position_sizes",
                    "description": f"Daily VaR ({portfolio_metrics.var_1d:.2%}) exceeds limit ({self.risk_limits.max_portfolio_var:.2%})",
                    "suggested_reduction": (
                        portfolio_metrics.var_1d - self.risk_limits.max_portfolio_var
                    )
                    / portfolio_metrics.var_1d,
                    "timeline": "immediate",
                }
            )

        # Sector concentration recommendations
        for sector, exposure in portfolio_metrics.sector_exposures.items():
            if exposure > self.risk_limits.max_sector_exposure:
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": "diversification",
                        "action": "reduce_sector_exposure",
                        "description": f"Reduce {sector} exposure from {exposure:.1%} to below {self.risk_limits.max_sector_exposure:.1%}",
                        "sector": sector,
                        "target_reduction": exposure
                        - self.risk_limits.max_sector_exposure,
                        "timeline": "within_week",
                    }
                )

        # High correlation recommendations
        high_correlations = self.correlation_analyzer.detect_high_correlations(
            portfolio_metrics.correlation_matrix
        )
        if high_correlations:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "diversification",
                    "action": "reduce_correlation",
                    "description": f"Found {len(high_correlations)} highly correlated pairs",
                    "correlated_pairs": high_correlations[:3],  # Top 3
                    "timeline": "within_month",
                }
            )

        # Drawdown recommendations
        if portfolio_metrics.current_drawdown < -0.10:  # 10% drawdown
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "capital_preservation",
                    "action": "reduce_risk",
                    "description": f"Current drawdown ({abs(portfolio_metrics.current_drawdown):.1%}) requires risk reduction",
                    "suggested_actions": [
                        "reduce_position_sizes",
                        "tighten_stop_losses",
                        "increase_cash_allocation",
                    ],
                    "timeline": "immediate",
                }
            )

        # Performance improvement recommendations
        if portfolio_metrics.sharpe_ratio < 0.5:
            recommendations.append(
                {
                    "priority": "LOW",
                    "category": "performance",
                    "action": "improve_risk_adjusted_returns",
                    "description": f"Low Sharpe ratio ({portfolio_metrics.sharpe_ratio:.2f}) suggests poor risk-adjusted returns",
                    "suggested_actions": [
                        "review_strategy",
                        "improve_entry_timing",
                        "optimize_position_sizing",
                    ],
                    "timeline": "within_month",
                }
            )

        return recommendations

    async def calculate_optimal_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        portfolio_state: dict[str, Any],
        market_conditions: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate optimal position size using enhanced risk management"""

        # Use dynamic position sizer
        base_sizing = self.position_sizer.calculate_dynamic_position_size(
            symbol, entry_price, stop_loss, portfolio_state, market_conditions
        )

        # Apply portfolio-level constraints
        portfolio_value = portfolio_state.get("total_value", 100000)
        current_var = portfolio_state.get("var_1d", 0.0)

        # VaR-based position sizing adjustment
        if current_var > self.risk_limits.max_portfolio_var * 0.8:  # 80% of limit
            var_adjustment = 0.7  # Reduce position size by 30%
            base_sizing["shares"] = int(base_sizing["shares"] * var_adjustment)
            base_sizing["var_adjustment"] = var_adjustment

        # Sector concentration adjustment
        sector = self.sector_analyzer.sector_mapping.get(symbol, "Others")
        current_sector_exposure = portfolio_state.get("sector_exposures", {}).get(
            sector, 0.0
        )

        if (
            current_sector_exposure > self.risk_limits.max_sector_exposure * 0.8
        ):  # 80% of limit
            sector_adjustment = max(
                0.5,
                1.0
                - (
                    current_sector_exposure - self.risk_limits.max_sector_exposure * 0.8
                ),
            )
            base_sizing["shares"] = int(base_sizing["shares"] * sector_adjustment)
            base_sizing["sector_adjustment"] = sector_adjustment

        # Final validation
        position_value = base_sizing["shares"] * entry_price
        max_position_value = portfolio_value * self.risk_limits.max_single_position

        if position_value > max_position_value:
            base_sizing["shares"] = int(max_position_value / entry_price)
            base_sizing["position_limit_applied"] = True

        return base_sizing

    def get_risk_dashboard_data(
        self, portfolio_metrics: PortfolioMetrics
    ) -> dict[str, Any]:
        """Get formatted data for risk management dashboard"""

        # Risk level color coding
        risk_colors = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "orange",
            RiskLevel.CRITICAL: "red",
        }

        # Key metrics summary
        key_metrics = {
            "portfolio_value": portfolio_metrics.total_value,
            "daily_var": portfolio_metrics.var_1d,
            "current_drawdown": portfolio_metrics.current_drawdown,
            "sharpe_ratio": portfolio_metrics.sharpe_ratio,
            "risk_level": portfolio_metrics.risk_level.value,
            "risk_color": risk_colors[portfolio_metrics.risk_level],
        }

        # Risk gauges (0-100 scale)
        risk_gauges = {
            "var_gauge": min(
                100,
                (portfolio_metrics.var_1d / self.risk_limits.max_portfolio_var) * 100,
            ),
            "leverage_gauge": min(
                100, (portfolio_metrics.leverage / self.risk_limits.max_leverage) * 100
            ),
            "drawdown_gauge": min(
                100,
                (
                    abs(portfolio_metrics.current_drawdown)
                    / self.risk_limits.max_drawdown
                )
                * 100,
            ),
            "concentration_gauge": min(
                100,
                (
                    max(portfolio_metrics.sector_exposures.values(), default=0)
                    / self.risk_limits.max_sector_exposure
                )
                * 100,
            ),
        }

        # Sector breakdown
        sector_breakdown = [
            {
                "sector": sector,
                "exposure": exposure,
                "limit": self.risk_limits.max_sector_exposure,
            }
            for sector, exposure in portfolio_metrics.sector_exposures.items()
        ]

        return {
            "key_metrics": key_metrics,
            "risk_gauges": risk_gauges,
            "sector_breakdown": sector_breakdown,
            "performance_metrics": {
                "sharpe_ratio": portfolio_metrics.sharpe_ratio,
                "sortino_ratio": portfolio_metrics.sortino_ratio,
                "calmar_ratio": portfolio_metrics.calmar_ratio,
                "max_drawdown": portfolio_metrics.max_drawdown,
                "volatility": portfolio_metrics.volatility,
                "beta": portfolio_metrics.beta,
                "alpha": portfolio_metrics.alpha,
            },
            "timestamp": portfolio_metrics.timestamp,
        }
