from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.services.websocket_manager import websocket_broadcaster

router = APIRouter()


class PortfolioSummary(BaseModel):
    total_value: float
    day_pnl: float
    day_pnl_percent: float
    total_pnl: float
    total_pnl_percent: float
    cash_balance: float
    invested_value: float
    position_count: int
    timestamp: datetime


class PerformanceMetrics(BaseModel):
    returns: dict[str, float]  # 1d, 1w, 1m, 3m, 6m, 1y
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: dict[str, Any]
    worst_trade: dict[str, Any]
    total_trades: int


class RiskMetrics(BaseModel):
    portfolio_beta: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional VaR
    max_drawdown: float
    current_drawdown: float
    concentration_risk: dict[str, float]
    correlation_matrix: dict[str, dict[str, float]]
    risk_score: int  # 1-10
    warnings: list[str]


@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get real-time portfolio summary from Zerodha"""
    try:
        # Get Kite client from app state
        kite_client = getattr(request.app.state, "kite_client", None)

        if not kite_client:
            # Initialize Kite client if not available
            from services.kite.client import KiteConnectService

            kite_client = KiteConnectService()
            await kite_client.initialize()
            request.app.state.kite_client = kite_client

        # Get live portfolio data from Zerodha
        try:
            # Get portfolio summary
            portfolio_summary = (
                await kite_client.portfolio_manager.get_portfolio_summary()
            )

            # Get current positions
            positions = await kite_client.portfolio_manager.get_positions(refresh=True)

            # Get account margins
            margins = await kite_client.portfolio_manager.get_margins()

            # Calculate real-time metrics
            total_value = portfolio_summary.total_portfolio_value
            cash_balance = margins.get("available", {}).get("cash", 0)
            invested_value = total_value - cash_balance

            # Calculate P&L from live positions
            day_pnl = sum(pos.pnl for pos in positions.get("day", []))
            total_pnl = sum(pos.pnl for pos in positions.get("net", []))

            # Calculate percentage changes
            day_pnl_percent = (day_pnl / total_value * 100) if total_value > 0 else 0
            total_pnl_percent = (
                (total_pnl / invested_value * 100) if invested_value > 0 else 0
            )

            # Create portfolio summary response
            summary = PortfolioSummary(
                total_value=total_value,
                day_pnl=day_pnl,
                day_pnl_percent=day_pnl_percent,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                cash_balance=cash_balance,
                invested_value=invested_value,
                position_count=len(
                    [pos for pos in positions.get("net", []) if pos.quantity != 0]
                ),
                timestamp=datetime.utcnow(),
            )

            # Broadcast real-time portfolio update via WebSocket
            await websocket_broadcaster.broadcast_portfolio_update(
                {
                    "totalValue": total_value,
                    "dayPnL": day_pnl,
                    "dayPnLPercent": day_pnl_percent,
                    "totalPnL": total_pnl,
                    "totalPnLPercent": total_pnl_percent,
                    "positions": [
                        {
                            "symbol": pos.symbol,
                            "quantity": pos.quantity,
                            "avgPrice": pos.average_price,
                            "currentPrice": pos.last_price,
                            "unrealizedPnL": pos.pnl,
                            "unrealizedPnLPercent": pos.day_change_percentage,
                            "value": pos.quantity * pos.last_price,
                            "dayChange": pos.day_change,
                        }
                        for pos in positions.get("net", [])
                        if pos.quantity != 0
                    ],
                    "cash": cash_balance,
                    "availableMargin": margins.get("available", {}).get("cash", 0),
                    "usedMargin": margins.get("utilised", {}).get("debits", 0),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            return summary

        except Exception as api_error:
            logger.error(f"Failed to fetch live portfolio data: {api_error}")
            # Return fallback data if API fails
            return PortfolioSummary(
                total_value=0.0,
                day_pnl=0.0,
                day_pnl_percent=0.0,
                total_pnl=0.0,
                total_pnl_percent=0.0,
                cash_balance=0.0,
                invested_value=0.0,
                position_count=0,
                timestamp=datetime.utcnow(),
            )

        return PortfolioSummary(
            total_value=total_value,
            day_pnl=day_pnl,
            day_pnl_percent=day_pnl_percent,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            cash_balance=cash_balance,
            invested_value=invested_value,
            position_count=len(
                [p for p in positions.get("net", []) if p["quantity"] != 0]
            ),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_portfolio_history(
    request: Request,
    current_user: User = Depends(get_current_user),
    days: int = Query(30, description="Number of days of history"),
):
    """Get portfolio value history"""
    try:
        # In production, this would fetch from a time-series database
        # For now, generate sample data
        history = []
        base_value = 100000
        current_date = datetime.now()

        for i in range(days):
            date = current_date - timedelta(days=days - i - 1)
            # Simulate portfolio value changes
            change = (i * 0.1) + ((-1) ** i * 0.5)  # Simple simulation
            value = base_value + (base_value * change / 100)

            history.append(
                {
                    "date": date.date().isoformat(),
                    "value": round(value, 2),
                    "pnl": round(value - base_value, 2),
                    "pnl_percent": round((value - base_value) / base_value * 100, 2),
                }
            )

        return {
            "history": history,
            "period": f"{days} days",
            "start_date": history[0]["date"] if history else None,
            "end_date": history[-1]["date"] if history else None,
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    period: str = Query("1M", description="Period: 1D, 1W, 1M, 3M, 6M, 1Y, ALL"),
):
    """Get portfolio performance metrics"""
    try:
        crew_manager = request.app.state.crew_manager

        # Calculate performance metrics
        metrics = await crew_manager.calculate_performance_metrics(
            user_id=current_user.id, period=period
        )

        return PerformanceMetrics(
            returns=metrics.get("returns", {}),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            win_rate=metrics.get("win_rate", 0),
            profit_factor=metrics.get("profit_factor", 0),
            avg_win=metrics.get("avg_win", 0),
            avg_loss=metrics.get("avg_loss", 0),
            best_trade=metrics.get("best_trade", {}),
            worst_trade=metrics.get("worst_trade", {}),
            total_trades=metrics.get("total_trades", 0),
        )
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk", response_model=RiskMetrics)
async def get_risk_metrics(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get portfolio risk metrics"""
    try:
        crew_manager = request.app.state.crew_manager
        kite_client = request.app.state.kite_client

        # Get current positions
        positions = await kite_client.get_positions()

        # Calculate risk metrics
        risk_analysis = await crew_manager.analyze_portfolio_risk(
            positions=positions.get("net", []), user_id=current_user.id
        )

        return RiskMetrics(
            portfolio_beta=risk_analysis.get("beta", 0),
            var_95=risk_analysis.get("var_95", 0),
            cvar_95=risk_analysis.get("cvar_95", 0),
            max_drawdown=risk_analysis.get("max_drawdown", 0),
            current_drawdown=risk_analysis.get("current_drawdown", 0),
            concentration_risk=risk_analysis.get("concentration_risk", {}),
            correlation_matrix=risk_analysis.get("correlation_matrix", {}),
            risk_score=risk_analysis.get("risk_score", 5),
            warnings=risk_analysis.get("warnings", []),
        )
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/holdings/analysis")
async def analyze_holdings(
    request: Request, current_user: User = Depends(get_current_user)
):
    """Get detailed analysis of current holdings"""
    try:
        crew_manager = request.app.state.crew_manager
        kite_client = request.app.state.kite_client

        # Get positions
        positions = await kite_client.get_positions()

        # Analyze each holding
        holdings_analysis = []
        for position in positions.get("net", []):
            if position["quantity"] != 0:
                # Get AI analysis for each position
                analysis = await crew_manager.analyze_position(
                    symbol=position["tradingsymbol"],
                    quantity=position["quantity"],
                    avg_price=position["average_price"],
                )

                holdings_analysis.append(
                    {
                        "symbol": position["tradingsymbol"],
                        "quantity": position["quantity"],
                        "avg_price": position["average_price"],
                        "current_price": position.get("last_price", 0),
                        "pnl": position.get("pnl", 0),
                        "pnl_percent": (
                            (
                                position.get("pnl", 0)
                                / (
                                    position["average_price"]
                                    * abs(position["quantity"])
                                )
                            )
                            * 100
                            if position["quantity"] != 0
                            else 0
                        ),
                        "value": position.get("value", 0),
                        "recommendation": analysis.get("recommendation", "hold"),
                        "target_price": analysis.get("target_price"),
                        "stop_loss": analysis.get("stop_loss"),
                        "analysis": analysis.get("analysis", {}),
                    }
                )

        return {
            "holdings": holdings_analysis,
            "total_holdings": len(holdings_analysis),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error analyzing holdings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebalance/suggest")
async def suggest_rebalancing(
    target_allocation: dict[str, float],
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Suggest portfolio rebalancing based on target allocation"""
    try:
        crew_manager = request.app.state.crew_manager
        kite_client = request.app.state.kite_client

        # Get current positions and portfolio value
        positions = await kite_client.get_positions()
        portfolio = await kite_client.get_portfolio()

        # Calculate rebalancing suggestions
        suggestions = await crew_manager.calculate_rebalancing(
            current_positions=positions.get("net", []),
            portfolio_value=portfolio.get("equity", 0),
            target_allocation=target_allocation,
        )

        return {
            "suggestions": suggestions,
            "estimated_cost": sum(s.get("estimated_cost", 0) for s in suggestions),
            "target_allocation": target_allocation,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error calculating rebalancing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tax/summary")
async def get_tax_summary(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    financial_year: str | None = None,
):
    """Get tax summary for capital gains"""
    try:
        # Calculate tax summary from trades
        # This is a simplified version - actual implementation would be more complex

        return {
            "financial_year": financial_year or "2024-25",
            "short_term_gains": 0,
            "long_term_gains": 0,
            "total_gains": 0,
            "tax_liability": {"short_term_tax": 0, "long_term_tax": 0, "total_tax": 0},
            "trades_count": {"short_term": 0, "long_term": 0},
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error calculating tax summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/account-balance")
async def get_account_balance(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Get real-time account balance and trading funds"""
    try:
        kite_client = getattr(request.app.state, "kite_client", None)

        if not kite_client:
            from services.kite.client import KiteConnectService

            kite_client = KiteConnectService()
            await kite_client.initialize()
            request.app.state.kite_client = kite_client

        # Get account margins and funds
        margins = await kite_client.portfolio_manager.get_margins()

        # Get user profile for account details
        profile = await kite_client.get_user_profile()

        account_info = {
            "user_id": profile.get("user_id"),
            "user_name": profile.get("user_name"),
            "broker": profile.get("broker"),
            "account_type": profile.get("user_type"),
            "available_cash": margins.get("available", {}).get("cash", 0),
            "opening_balance": margins.get("available", {}).get("opening_balance", 0),
            "live_balance": margins.get("available", {}).get("live_balance", 0),
            "adhoc_margin": margins.get("available", {}).get("adhoc_margin", 0),
            "collateral": margins.get("available", {}).get("collateral", 0),
            "intraday_payin": margins.get("available", {}).get("intraday_payin", 0),
            "used_margin": margins.get("utilised", {}).get("debits", 0),
            "span": margins.get("utilised", {}).get("span", 0),
            "exposure": margins.get("utilised", {}).get("exposure", 0),
            "option_premium": margins.get("utilised", {}).get("option_premium", 0),
            "hold_sales": margins.get("utilised", {}).get("hold_sales", 0),
            "turnover": margins.get("utilised", {}).get("turnover", 0),
            "net_available": margins.get("net", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return account_info

    except Exception as e:
        logger.error(f"Failed to get account balance: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get account balance: {str(e)}"
        )


@router.get("/live-positions")
async def get_live_positions(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Get real-time positions from Zerodha"""
    try:
        kite_client = getattr(request.app.state, "kite_client", None)

        if not kite_client:
            from services.kite.client import KiteConnectService

            kite_client = KiteConnectService()
            await kite_client.initialize()
            request.app.state.kite_client = kite_client

        # Get live positions
        positions = await kite_client.portfolio_manager.get_positions(refresh=True)

        # Format positions for frontend
        formatted_positions = {
            "day_positions": [
                {
                    "symbol": pos.symbol,
                    "exchange": pos.exchange,
                    "quantity": pos.quantity,
                    "average_price": pos.average_price,
                    "last_price": pos.last_price,
                    "close_price": pos.close_price,
                    "pnl": pos.pnl,
                    "day_change": pos.day_change,
                    "day_change_percentage": pos.day_change_percentage,
                    "value": pos.quantity * pos.last_price,
                    "product": pos.product,
                    "instrument_token": pos.instrument_token,
                }
                for pos in positions.get("day", [])
                if pos.quantity != 0
            ],
            "net_positions": [
                {
                    "symbol": pos.symbol,
                    "exchange": pos.exchange,
                    "quantity": pos.quantity,
                    "average_price": pos.average_price,
                    "last_price": pos.last_price,
                    "close_price": pos.close_price,
                    "pnl": pos.pnl,
                    "day_change": pos.day_change,
                    "day_change_percentage": pos.day_change_percentage,
                    "value": pos.quantity * pos.last_price,
                    "product": pos.product,
                    "instrument_token": pos.instrument_token,
                }
                for pos in positions.get("net", [])
                if pos.quantity != 0
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return formatted_positions

    except Exception as e:
        logger.error(f"Failed to get live positions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get live positions: {str(e)}"
        )
