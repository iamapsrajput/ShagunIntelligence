"""
Database API Routes
Endpoints for database operations and analytics
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.db.session import get_db
from app.services.database_services import (
    MarketDataService,
    PortfolioService,
    RiskMetricsService,
    TradingOrderService,
)

router = APIRouter(prefix="/database", tags=["database"])

# Service instances
market_data_service = MarketDataService()
order_service = TradingOrderService()
portfolio_service = PortfolioService()
risk_service = RiskMetricsService()


class MarketDataRequest(BaseModel):
    """Request to store market data"""

    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange")
    timeframe: str = Field(..., description="Timeframe")
    data: list[dict[str, Any]] = Field(..., description="OHLCV data")


class OrderAnalyticsRequest(BaseModel):
    """Request for order analytics"""

    start_date: datetime | None = Field(None, description="Start date")
    end_date: datetime | None = Field(None, description="End date")
    broker_name: str | None = Field(None, description="Broker name filter")
    symbol: str | None = Field(None, description="Symbol filter")


@router.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    timeframe: str = Query("1d", description="Timeframe"),
    start_date: datetime | None = Query(None, description="Start date"),
    end_date: datetime | None = Query(None, description="End date"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum records"),
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Get historical market data from database

    Returns OHLCV data for the specified symbol and timeframe
    """
    try:
        data = market_data_service.get_market_data(
            db, symbol, timeframe, start_date, end_date, limit
        )

        if data.empty:
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "records": 0,
                    "data": [],
                },
            }

        # Convert DataFrame to JSON-serializable format
        data_dict = data.reset_index().to_dict("records")

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "records": len(data_dict),
                "data": data_dict,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get market data: {str(e)}"
        )


@router.post("/market-data/store")
async def store_market_data(
    request: MarketDataRequest,
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Store market data in database

    Accepts OHLCV data and stores it efficiently using bulk operations
    """
    try:
        import pandas as pd

        # Convert data to DataFrame
        df = pd.DataFrame(request.data)

        # Ensure timestamp column is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        # Store data
        records_stored = market_data_service.store_market_data(
            db, request.symbol, request.exchange, request.timeframe, df
        )

        return {
            "success": True,
            "message": f"Stored {records_stored} market data records",
            "data": {
                "symbol": request.symbol,
                "exchange": request.exchange,
                "timeframe": request.timeframe,
                "records_stored": records_stored,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to store market data: {str(e)}"
        )


@router.get("/market-data/symbols")
async def get_symbols_list(
    exchange: str | None = Query(None, description="Exchange filter"),
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get list of available symbols in database"""
    try:
        symbols = market_data_service.get_symbols_list(db, exchange)

        return {
            "success": True,
            "data": {"symbols": symbols, "count": len(symbols), "exchange": exchange},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get symbols: {str(e)}")


@router.get("/market-data/latest-price/{symbol}")
async def get_latest_price(
    symbol: str,
    timeframe: str = Query("1d", description="Timeframe"),
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get latest price for a symbol"""
    try:
        price = market_data_service.get_latest_price(db, symbol, timeframe)

        if price is None:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "latest_price": price,
                "timestamp": datetime.utcnow(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get latest price: {str(e)}"
        )


@router.delete("/market-data/cleanup")
async def cleanup_old_data(
    days_to_keep: int = Query(365, ge=30, le=3650, description="Days of data to keep"),
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Clean up old market data to manage database size"""
    try:
        deleted_count = market_data_service.cleanup_old_data(db, days_to_keep)

        return {
            "success": True,
            "message": f"Cleaned up {deleted_count} old records",
            "data": {"deleted_records": deleted_count, "days_kept": days_to_keep},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup data: {str(e)}")


@router.get("/orders")
async def get_orders(
    broker_name: str | None = Query(None, description="Broker name filter"),
    symbol: str | None = Query(None, description="Symbol filter"),
    status: str | None = Query(None, description="Status filter"),
    start_date: datetime | None = Query(None, description="Start date"),
    end_date: datetime | None = Query(None, description="End date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records"),
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get trading orders with filtering options"""
    try:
        orders = order_service.get_orders(
            db, broker_name, symbol, status, start_date, end_date, limit
        )

        # Convert to JSON-serializable format
        orders_data = []
        for order in orders:
            order_dict = {
                "id": str(order.id),
                "order_id": order.order_id,
                "broker_name": order.broker_name,
                "symbol": order.symbol,
                "exchange": order.exchange,
                "transaction_type": order.transaction_type,
                "order_type": order.order_type,
                "quantity": order.quantity,
                "filled_quantity": order.filled_quantity,
                "price": order.price,
                "average_price": order.average_price,
                "status": order.status,
                "order_timestamp": order.order_timestamp,
                "slippage": order.slippage,
                "execution_time_ms": order.execution_time_ms,
            }
            orders_data.append(order_dict)

        return {
            "success": True,
            "data": {
                "orders": orders_data,
                "count": len(orders_data),
                "filters": {
                    "broker_name": broker_name,
                    "symbol": symbol,
                    "status": status,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get orders: {str(e)}")


@router.get("/orders/analytics")
async def get_order_analytics(
    start_date: datetime | None = Query(None, description="Start date"),
    end_date: datetime | None = Query(None, description="End date"),
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get order execution analytics"""
    try:
        analytics = order_service.get_order_analytics(db, start_date, end_date)

        return {
            "success": True,
            "data": {
                "analytics": analytics,
                "period": {"start_date": start_date, "end_date": end_date},
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get order analytics: {str(e)}"
        )


@router.get("/portfolio/summary")
async def get_portfolio_summary(
    broker_name: str | None = Query(None, description="Broker name filter"),
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get portfolio summary with P&L"""
    try:
        summary = portfolio_service.get_portfolio_summary(db, broker_name)

        return {
            "success": True,
            "data": {
                "portfolio_summary": summary,
                "broker_name": broker_name,
                "timestamp": datetime.utcnow(),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get portfolio summary: {str(e)}"
        )


@router.get("/risk-metrics/{portfolio_id}")
async def get_risk_metrics(
    portfolio_id: str,
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get latest risk metrics for a portfolio"""
    try:
        risk_metrics = risk_service.get_latest_risk_metrics(db, portfolio_id)

        if not risk_metrics:
            raise HTTPException(
                status_code=404,
                detail=f"No risk metrics found for portfolio {portfolio_id}",
            )

        return {
            "success": True,
            "data": {
                "portfolio_id": portfolio_id,
                "risk_metrics": {
                    "total_value": risk_metrics.total_value,
                    "total_exposure": risk_metrics.total_exposure,
                    "leverage": risk_metrics.leverage,
                    "var_1d": risk_metrics.var_1d,
                    "var_5d": risk_metrics.var_5d,
                    "max_drawdown": risk_metrics.max_drawdown,
                    "current_drawdown": risk_metrics.current_drawdown,
                    "sharpe_ratio": risk_metrics.sharpe_ratio,
                    "sortino_ratio": risk_metrics.sortino_ratio,
                    "risk_level": risk_metrics.risk_level,
                    "risk_score": risk_metrics.risk_score,
                    "sector_exposures": risk_metrics.sector_exposures,
                    "calculation_date": risk_metrics.calculation_date,
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get risk metrics: {str(e)}"
        )


@router.get("/risk-metrics/{portfolio_id}/history")
async def get_risk_history(
    portfolio_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get risk metrics history for a portfolio"""
    try:
        history = risk_service.get_risk_history(db, portfolio_id, days)

        # Convert to JSON-serializable format
        history_data = []
        for metrics in history:
            history_data.append(
                {
                    "calculation_date": metrics.calculation_date,
                    "total_value": metrics.total_value,
                    "var_1d": metrics.var_1d,
                    "max_drawdown": metrics.max_drawdown,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "risk_level": metrics.risk_level,
                    "risk_score": metrics.risk_score,
                }
            )

        return {
            "success": True,
            "data": {
                "portfolio_id": portfolio_id,
                "history": history_data,
                "records": len(history_data),
                "days": days,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get risk history: {str(e)}"
        )


@router.get("/stats")
async def get_database_stats(
    db: Session = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get database statistics"""
    try:
        from app.db.models.trading_models import (
            Holding,
            MarketData,
            Position,
            RiskMetrics,
            TradingOrder,
        )

        # Get record counts
        market_data_count = db.query(MarketData).count()
        orders_count = db.query(TradingOrder).count()
        positions_count = db.query(Position).count()
        holdings_count = db.query(Holding).count()
        risk_metrics_count = db.query(RiskMetrics).count()

        # Get date ranges
        oldest_market_data = (
            db.query(MarketData.timestamp).order_by(MarketData.timestamp.asc()).first()
        )
        latest_market_data = (
            db.query(MarketData.timestamp).order_by(MarketData.timestamp.desc()).first()
        )

        return {
            "success": True,
            "data": {
                "record_counts": {
                    "market_data": market_data_count,
                    "trading_orders": orders_count,
                    "positions": positions_count,
                    "holdings": holdings_count,
                    "risk_metrics": risk_metrics_count,
                },
                "date_ranges": {
                    "oldest_market_data": (
                        oldest_market_data[0] if oldest_market_data else None
                    ),
                    "latest_market_data": (
                        latest_market_data[0] if latest_market_data else None
                    ),
                },
                "timestamp": datetime.utcnow(),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get database stats: {str(e)}"
        )
