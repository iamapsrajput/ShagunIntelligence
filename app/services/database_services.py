"""
Database services for the trading application
Enhanced with time-series data management and trading analytics
"""

from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models.trading_models import (
    Holding,
    MarketData,
    Position,
    RiskMetrics,
    TradingOrder,
)


class MarketDataService:
    """Service for time-series market data operations"""

    def __init__(self):
        pass

    def store_market_data(
        self,
        db: Session,
        symbol: str,
        exchange: str,
        timeframe: str,
        data: pd.DataFrame,
    ) -> int:
        """Store market data efficiently using bulk operations"""
        try:
            records = []
            for timestamp, row in data.iterrows():
                record = MarketData(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    vwap=float(row.get("vwap", 0)) if "vwap" in row else None,
                    turnover=(
                        float(row.get("turnover", 0)) if "turnover" in row else None
                    ),
                    trades_count=(
                        int(row.get("trades_count", 0))
                        if "trades_count" in row
                        else None
                    ),
                )
                records.append(record)

            # Use bulk insert for better performance
            db.bulk_save_objects(records)
            db.commit()

            logger.info(f"Stored {len(records)} market data records for {symbol}")
            return len(records)

        except Exception as e:
            db.rollback()
            logger.error(f"Error storing market data: {str(e)}")
            raise

    def get_market_data(
        self,
        db: Session,
        symbol: str,
        timeframe: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Retrieve market data as pandas DataFrame"""
        try:
            query = db.query(MarketData).filter(
                MarketData.symbol == symbol, MarketData.timeframe == timeframe
            )

            if start_date:
                query = query.filter(MarketData.timestamp >= start_date)
            if end_date:
                query = query.filter(MarketData.timestamp <= end_date)

            query = query.order_by(MarketData.timestamp.desc()).limit(limit)

            results = query.all()

            if not results:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for record in reversed(results):  # Reverse to get chronological order
                data.append(
                    {
                        "timestamp": record.timestamp,
                        "open": record.open,
                        "high": record.high,
                        "low": record.low,
                        "close": record.close,
                        "volume": record.volume,
                        "vwap": record.vwap,
                        "turnover": record.turnover,
                        "trades_count": record.trades_count,
                    }
                )

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            return pd.DataFrame()

    def get_latest_price(
        self, db: Session, symbol: str, timeframe: str = "1d"
    ) -> float | None:
        """Get latest price for a symbol"""
        try:
            latest = (
                db.query(MarketData)
                .filter(MarketData.symbol == symbol, MarketData.timeframe == timeframe)
                .order_by(MarketData.timestamp.desc())
                .first()
            )

            return latest.close if latest else None

        except Exception as e:
            logger.error(f"Error getting latest price: {str(e)}")
            return None

    def get_symbols_list(self, db: Session, exchange: str = None) -> list[str]:
        """Get list of available symbols"""
        try:
            query = db.query(MarketData.symbol).distinct()

            if exchange:
                query = query.filter(MarketData.exchange == exchange)

            results = query.all()
            return [result[0] for result in results]

        except Exception as e:
            logger.error(f"Error getting symbols list: {str(e)}")
            return []

    def cleanup_old_data(self, db: Session, days_to_keep: int = 365) -> int:
        """Clean up old market data to manage database size"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            deleted_count = (
                db.query(MarketData).filter(MarketData.timestamp < cutoff_date).delete()
            )

            db.commit()
            logger.info(f"Cleaned up {deleted_count} old market data records")
            return deleted_count

        except Exception as e:
            db.rollback()
            logger.error(f"Error cleaning up old data: {str(e)}")
            return 0


class TradingOrderService:
    """Service for trading order management"""

    def __init__(self):
        pass

    def create_order(self, db: Session, order_data: dict[str, Any]) -> TradingOrder:
        """Create a new trading order record"""
        try:
            order = TradingOrder(**order_data)
            db.add(order)
            db.commit()
            db.refresh(order)
            return order

        except Exception as e:
            db.rollback()
            logger.error(f"Error creating order: {str(e)}")
            raise

    def update_order(
        self, db: Session, order_id: str, update_data: dict[str, Any]
    ) -> TradingOrder | None:
        """Update an existing order"""
        try:
            order = (
                db.query(TradingOrder).filter(TradingOrder.order_id == order_id).first()
            )

            if not order:
                return None

            for key, value in update_data.items():
                if hasattr(order, key):
                    setattr(order, key, value)

            order.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(order)
            return order

        except Exception as e:
            db.rollback()
            logger.error(f"Error updating order: {str(e)}")
            raise

    def get_orders(
        self,
        db: Session,
        broker_name: str = None,
        symbol: str = None,
        status: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100,
    ) -> list[TradingOrder]:
        """Get orders with filtering options"""
        try:
            query = db.query(TradingOrder)

            if broker_name:
                query = query.filter(TradingOrder.broker_name == broker_name)
            if symbol:
                query = query.filter(TradingOrder.symbol == symbol)
            if status:
                query = query.filter(TradingOrder.status == status)
            if start_date:
                query = query.filter(TradingOrder.order_timestamp >= start_date)
            if end_date:
                query = query.filter(TradingOrder.order_timestamp <= end_date)

            return (
                query.order_by(TradingOrder.order_timestamp.desc()).limit(limit).all()
            )

        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []

    def get_order_analytics(
        self, db: Session, start_date: datetime = None, end_date: datetime = None
    ) -> dict[str, Any]:
        """Get order execution analytics"""
        try:
            query = db.query(TradingOrder)

            if start_date:
                query = query.filter(TradingOrder.order_timestamp >= start_date)
            if end_date:
                query = query.filter(TradingOrder.order_timestamp <= end_date)

            orders = query.all()

            if not orders:
                return {}

            total_orders = len(orders)
            completed_orders = [o for o in orders if o.status == "COMPLETE"]
            cancelled_orders = [o for o in orders if o.status == "CANCELLED"]
            rejected_orders = [o for o in orders if o.status == "REJECTED"]

            # Calculate execution metrics
            execution_times = [
                o.execution_time_ms for o in completed_orders if o.execution_time_ms
            ]
            slippages = [o.slippage for o in completed_orders if o.slippage is not None]

            return {
                "total_orders": total_orders,
                "completed_orders": len(completed_orders),
                "cancelled_orders": len(cancelled_orders),
                "rejected_orders": len(rejected_orders),
                "success_rate": (
                    len(completed_orders) / total_orders * 100
                    if total_orders > 0
                    else 0
                ),
                "average_execution_time_ms": (
                    sum(execution_times) / len(execution_times)
                    if execution_times
                    else 0
                ),
                "average_slippage": sum(slippages) / len(slippages) if slippages else 0,
                "max_slippage": max(slippages) if slippages else 0,
                "min_slippage": min(slippages) if slippages else 0,
            }

        except Exception as e:
            logger.error(f"Error getting order analytics: {str(e)}")
            return {}


class PortfolioService:
    """Service for portfolio and position management"""

    def __init__(self):
        pass

    def store_positions(self, db: Session, positions_data: list[dict[str, Any]]) -> int:
        """Store current positions"""
        try:
            # Clear existing positions for the day
            today = datetime.utcnow().date()
            db.query(Position).filter(
                func.date(Position.position_date) == today
            ).delete()

            # Insert new positions
            positions = [Position(**pos_data) for pos_data in positions_data]
            db.bulk_save_objects(positions)
            db.commit()

            logger.info(f"Stored {len(positions)} positions")
            return len(positions)

        except Exception as e:
            db.rollback()
            logger.error(f"Error storing positions: {str(e)}")
            raise

    def store_holdings(self, db: Session, holdings_data: list[dict[str, Any]]) -> int:
        """Store portfolio holdings"""
        try:
            # Clear existing holdings for the day
            today = datetime.utcnow().date()
            db.query(Holding).filter(func.date(Holding.holding_date) == today).delete()

            # Insert new holdings
            holdings = [Holding(**holding_data) for holding_data in holdings_data]
            db.bulk_save_objects(holdings)
            db.commit()

            logger.info(f"Stored {len(holdings)} holdings")
            return len(holdings)

        except Exception as e:
            db.rollback()
            logger.error(f"Error storing holdings: {str(e)}")
            raise

    def get_portfolio_summary(
        self, db: Session, broker_name: str = None
    ) -> dict[str, Any]:
        """Get portfolio summary with P&L"""
        try:
            # Get latest positions
            positions_query = db.query(Position)
            if broker_name:
                positions_query = positions_query.filter(
                    Position.broker_name == broker_name
                )

            positions = positions_query.filter(
                Position.position_date >= datetime.utcnow().date()
            ).all()

            # Get latest holdings
            holdings_query = db.query(Holding)
            if broker_name:
                holdings_query = holdings_query.filter(
                    Holding.broker_name == broker_name
                )

            holdings = holdings_query.filter(
                Holding.holding_date >= datetime.utcnow().date()
            ).all()

            # Calculate summary
            total_position_value = sum(
                abs(p.quantity * p.last_price) for p in positions
            )
            total_position_pnl = sum(p.pnl for p in positions)
            total_holding_value = sum(h.current_value for h in holdings)
            total_holding_pnl = sum(h.pnl for h in holdings)

            return {
                "positions": {
                    "count": len(positions),
                    "total_value": total_position_value,
                    "total_pnl": total_position_pnl,
                },
                "holdings": {
                    "count": len(holdings),
                    "total_value": total_holding_value,
                    "total_pnl": total_holding_pnl,
                },
                "overall": {
                    "total_portfolio_value": total_position_value + total_holding_value,
                    "total_pnl": total_position_pnl + total_holding_pnl,
                },
            }

        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {}


class RiskMetricsService:
    """Service for risk metrics management"""

    def __init__(self):
        pass

    def store_risk_metrics(self, db: Session, risk_data: dict[str, Any]) -> RiskMetrics:
        """Store portfolio risk metrics"""
        try:
            risk_metrics = RiskMetrics(**risk_data)
            db.add(risk_metrics)
            db.commit()
            db.refresh(risk_metrics)
            return risk_metrics

        except Exception as e:
            db.rollback()
            logger.error(f"Error storing risk metrics: {str(e)}")
            raise

    def get_latest_risk_metrics(
        self, db: Session, portfolio_id: str
    ) -> RiskMetrics | None:
        """Get latest risk metrics for a portfolio"""
        try:
            return (
                db.query(RiskMetrics)
                .filter(RiskMetrics.portfolio_id == portfolio_id)
                .order_by(RiskMetrics.calculation_date.desc())
                .first()
            )

        except Exception as e:
            logger.error(f"Error getting risk metrics: {str(e)}")
            return None

    def get_risk_history(
        self, db: Session, portfolio_id: str, days: int = 30
    ) -> list[RiskMetrics]:
        """Get risk metrics history"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            return (
                db.query(RiskMetrics)
                .filter(
                    RiskMetrics.portfolio_id == portfolio_id,
                    RiskMetrics.calculation_date >= start_date,
                )
                .order_by(RiskMetrics.calculation_date.desc())
                .all()
            )

        except Exception as e:
            logger.error(f"Error getting risk history: {str(e)}")
            return []
