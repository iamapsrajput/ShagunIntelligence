"""
Enhanced Risk Management API Routes
Provides endpoints for comprehensive portfolio risk management
"""

from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.services.enhanced_risk_management import EnhancedRiskManager, RiskLimits

router = APIRouter(prefix="/risk", tags=["enhanced-risk"])

# Global instance - in production, this would be dependency injected
enhanced_risk_manager = EnhancedRiskManager()


class RiskLimitsModel(BaseModel):
    """API model for risk limits configuration"""

    max_portfolio_var: float = Field(
        default=0.05, ge=0.01, le=0.20, description="Maximum portfolio VaR (1-20%)"
    )
    max_sector_exposure: float = Field(
        default=0.30, ge=0.10, le=0.50, description="Maximum sector exposure (10-50%)"
    )
    max_single_position: float = Field(
        default=0.10, ge=0.02, le=0.25, description="Maximum single position (2-25%)"
    )
    max_correlation: float = Field(
        default=0.70, ge=0.30, le=0.95, description="Maximum correlation (30-95%)"
    )
    max_leverage: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Maximum leverage (1-5x)"
    )
    max_drawdown: float = Field(
        default=0.15, ge=0.05, le=0.30, description="Maximum drawdown (5-30%)"
    )
    min_sharpe_ratio: float = Field(
        default=0.5, ge=0.0, le=2.0, description="Minimum Sharpe ratio"
    )
    max_beta: float = Field(
        default=1.5, ge=0.5, le=3.0, description="Maximum portfolio beta"
    )


class PositionSizeRequest(BaseModel):
    """Request model for position sizing"""

    symbol: str = Field(..., description="Trading symbol")
    entry_price: float = Field(..., gt=0, description="Entry price")
    stop_loss: float = Field(..., gt=0, description="Stop loss price")
    portfolio_value: float = Field(..., gt=0, description="Current portfolio value")
    current_var: float | None = Field(None, description="Current portfolio VaR")
    sector_exposures: dict[str, float] = Field(
        default_factory=dict, description="Current sector exposures"
    )
    market_volatility: float | None = Field(
        None, description="Current market volatility"
    )
    market_stress: float | None = Field(
        None, description="Market stress level (0-1)"
    )


class PortfolioMetricsResponse(BaseModel):
    """Response model for portfolio metrics"""

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
    sector_exposures: dict[str, float]
    risk_level: str
    timestamp: datetime


class RiskRecommendation(BaseModel):
    """Risk management recommendation"""

    priority: str
    category: str
    action: str
    description: str
    timeline: str


@router.get("/metrics", response_model=PortfolioMetricsResponse)
async def get_portfolio_risk_metrics(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Get comprehensive portfolio risk metrics

    Returns detailed risk analysis including:
    - Value at Risk (VaR) calculations
    - Performance metrics (Sharpe, Sortino, Calmar ratios)
    - Drawdown analysis
    - Sector exposure analysis
    - Correlation analysis
    """
    try:
        # In a real implementation, this would fetch actual portfolio data
        # For now, we'll use mock data
        mock_positions = {
            "RELIANCE": {"shares": 100, "market_value": 250000},
            "TCS": {"shares": 50, "market_value": 180000},
            "HDFCBANK": {"shares": 75, "market_value": 120000},
        }

        # Mock market data (would come from market data service)
        mock_market_data = {
            "RELIANCE": pd.DataFrame(
                {
                    "close": [2500, 2520, 2480, 2510, 2530],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
            "TCS": pd.DataFrame(
                {
                    "close": [3600, 3620, 3580, 3610, 3640],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
            "HDFCBANK": pd.DataFrame(
                {
                    "close": [1600, 1610, 1590, 1605, 1620],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
        }

        # Calculate portfolio metrics
        portfolio_metrics = await enhanced_risk_manager.calculate_portfolio_metrics(
            mock_positions, mock_market_data
        )

        return PortfolioMetricsResponse(
            total_value=portfolio_metrics.total_value,
            total_exposure=portfolio_metrics.total_exposure,
            leverage=portfolio_metrics.leverage,
            var_1d=portfolio_metrics.var_1d,
            var_5d=portfolio_metrics.var_5d,
            cvar_1d=portfolio_metrics.cvar_1d,
            expected_shortfall=portfolio_metrics.expected_shortfall,
            max_drawdown=portfolio_metrics.max_drawdown,
            current_drawdown=portfolio_metrics.current_drawdown,
            sharpe_ratio=portfolio_metrics.sharpe_ratio,
            sortino_ratio=portfolio_metrics.sortino_ratio,
            calmar_ratio=portfolio_metrics.calmar_ratio,
            beta=portfolio_metrics.beta,
            alpha=portfolio_metrics.alpha,
            volatility=portfolio_metrics.volatility,
            sector_exposures=portfolio_metrics.sector_exposures,
            risk_level=portfolio_metrics.risk_level.value,
            timestamp=portfolio_metrics.timestamp,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate portfolio metrics: {str(e)}"
        )


@router.get("/limits")
async def get_risk_limits(current_user: dict[str, Any] = Depends(get_current_user)):
    """Get current risk limits configuration"""
    try:
        limits = enhanced_risk_manager.risk_limits
        return {
            "success": True,
            "data": {
                "max_portfolio_var": limits.max_portfolio_var,
                "max_sector_exposure": limits.max_sector_exposure,
                "max_single_position": limits.max_single_position,
                "max_correlation": limits.max_correlation,
                "max_leverage": limits.max_leverage,
                "max_drawdown": limits.max_drawdown,
                "min_sharpe_ratio": limits.min_sharpe_ratio,
                "max_beta": limits.max_beta,
                "volatility_adjustment": limits.volatility_adjustment,
                "market_stress_multiplier": limits.market_stress_multiplier,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get risk limits: {str(e)}"
        )


@router.put("/limits")
async def update_risk_limits(
    limits: RiskLimitsModel, current_user: dict[str, Any] = Depends(get_current_user)
):
    """Update risk limits configuration"""
    try:
        # Update the risk manager's limits
        enhanced_risk_manager.risk_limits = RiskLimits(
            max_portfolio_var=limits.max_portfolio_var,
            max_sector_exposure=limits.max_sector_exposure,
            max_single_position=limits.max_single_position,
            max_correlation=limits.max_correlation,
            max_leverage=limits.max_leverage,
            max_drawdown=limits.max_drawdown,
            min_sharpe_ratio=limits.min_sharpe_ratio,
            max_beta=limits.max_beta,
        )

        return {
            "success": True,
            "message": "Risk limits updated successfully",
            "data": limits.dict(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update risk limits: {str(e)}"
        )


@router.post("/position-size")
async def calculate_optimal_position_size(
    request: PositionSizeRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Calculate optimal position size using enhanced risk management

    Takes into account:
    - Portfolio-level VaR constraints
    - Sector concentration limits
    - Dynamic market conditions
    - Correlation adjustments
    """
    try:
        # Prepare portfolio state
        portfolio_state = {
            "total_value": request.portfolio_value,
            "var_1d": request.current_var or 0.02,
            "sector_exposures": request.sector_exposures,
        }

        # Prepare market conditions
        market_conditions = {
            "volatility": request.market_volatility or 0.02,
            "stress_level": request.market_stress or 0.0,
            "volatility_adjustment": 1.0,
            "correlation_adjustment": 1.0,
        }

        # Calculate optimal position size
        position_size = await enhanced_risk_manager.calculate_optimal_position_size(
            request.symbol,
            request.entry_price,
            request.stop_loss,
            portfolio_state,
            market_conditions,
        )

        return {"success": True, "data": position_size}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate position size: {str(e)}"
        )


@router.get("/check")
async def check_risk_limits(current_user: dict[str, Any] = Depends(get_current_user)):
    """Check current portfolio against risk limits"""
    try:
        # Get current portfolio metrics (mock data for now)
        mock_positions = {
            "RELIANCE": {"shares": 100, "market_value": 250000},
            "TCS": {"shares": 50, "market_value": 180000},
            "HDFCBANK": {"shares": 75, "market_value": 120000},
        }

        mock_market_data = {
            "RELIANCE": pd.DataFrame(
                {
                    "close": [2500, 2520, 2480, 2510, 2530],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
            "TCS": pd.DataFrame(
                {
                    "close": [3600, 3620, 3580, 3610, 3640],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
            "HDFCBANK": pd.DataFrame(
                {
                    "close": [1600, 1610, 1590, 1605, 1620],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
        }

        portfolio_metrics = await enhanced_risk_manager.calculate_portfolio_metrics(
            mock_positions, mock_market_data
        )

        risk_check = await enhanced_risk_manager.check_risk_limits(portfolio_metrics)

        return {"success": True, "data": risk_check}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to check risk limits: {str(e)}"
        )


@router.get("/recommendations", response_model=list[RiskRecommendation])
async def get_risk_recommendations(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get actionable risk management recommendations"""
    try:
        # Get portfolio metrics and risk check
        mock_positions = {
            "RELIANCE": {"shares": 100, "market_value": 250000},
            "TCS": {"shares": 50, "market_value": 180000},
            "HDFCBANK": {"shares": 75, "market_value": 120000},
        }

        mock_market_data = {
            "RELIANCE": pd.DataFrame(
                {
                    "close": [2500, 2520, 2480, 2510, 2530],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
            "TCS": pd.DataFrame(
                {
                    "close": [3600, 3620, 3580, 3610, 3640],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
            "HDFCBANK": pd.DataFrame(
                {
                    "close": [1600, 1610, 1590, 1605, 1620],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
        }

        portfolio_metrics = await enhanced_risk_manager.calculate_portfolio_metrics(
            mock_positions, mock_market_data
        )

        risk_check = await enhanced_risk_manager.check_risk_limits(portfolio_metrics)

        recommendations = await enhanced_risk_manager.generate_risk_recommendations(
            portfolio_metrics, risk_check
        )

        return [
            RiskRecommendation(
                priority=rec["priority"],
                category=rec["category"],
                action=rec["action"],
                description=rec["description"],
                timeline=rec["timeline"],
            )
            for rec in recommendations
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get recommendations: {str(e)}"
        )


@router.get("/dashboard")
async def get_risk_dashboard(current_user: dict[str, Any] = Depends(get_current_user)):
    """Get formatted data for risk management dashboard"""
    try:
        # Get portfolio metrics
        mock_positions = {
            "RELIANCE": {"shares": 100, "market_value": 250000},
            "TCS": {"shares": 50, "market_value": 180000},
            "HDFCBANK": {"shares": 75, "market_value": 120000},
        }

        mock_market_data = {
            "RELIANCE": pd.DataFrame(
                {
                    "close": [2500, 2520, 2480, 2510, 2530],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
            "TCS": pd.DataFrame(
                {
                    "close": [3600, 3620, 3580, 3610, 3640],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
            "HDFCBANK": pd.DataFrame(
                {
                    "close": [1600, 1610, 1590, 1605, 1620],
                    "date": pd.date_range("2025-01-01", periods=5),
                }
            ).set_index("date"),
        }

        portfolio_metrics = await enhanced_risk_manager.calculate_portfolio_metrics(
            mock_positions, mock_market_data
        )

        dashboard_data = enhanced_risk_manager.get_risk_dashboard_data(
            portfolio_metrics
        )

        return {"success": True, "data": dashboard_data}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get dashboard data: {str(e)}"
        )


@router.get("/var-analysis")
async def get_var_analysis(
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get detailed Value at Risk analysis"""
    try:
        # Mock portfolio returns for VaR calculation
        returns_data = pd.Series(
            [0.01, -0.02, 0.015, -0.008, 0.012, -0.018, 0.009, -0.015, 0.020, -0.012]
        )

        var_calculator = enhanced_risk_manager.var_calculator

        # Calculate VaR using different methods
        historical_var = var_calculator.calculate_historical_var(
            returns_data, confidence_level, time_horizon
        )
        parametric_var = var_calculator.calculate_parametric_var(
            returns_data, confidence_level, time_horizon
        )
        monte_carlo_var = var_calculator.calculate_monte_carlo_var(
            returns_data, confidence_level, time_horizon
        )

        return {
            "success": True,
            "data": {
                "historical_var": historical_var,
                "parametric_var": parametric_var,
                "monte_carlo_var": monte_carlo_var,
                "confidence_level": confidence_level,
                "time_horizon": time_horizon,
                "analysis_date": datetime.now(),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get VaR analysis: {str(e)}"
        )
