"""
Trade Executor Agent for AlgoHive

This module provides comprehensive trade execution capabilities including:
- Order management with multiple order types
- Real-time position monitoring
- Paper trading simulation
- Trade logging and audit trail
- Order timing optimization
"""

from .agent import TradeExecutorAgent, TradeSignal
from .order_manager import OrderManager, OrderType, OrderStatus
from .position_monitor import PositionMonitor, Position
from .paper_trading_manager import PaperTradingManager, PaperTrade
from .trade_logger import TradeLogger, TradeLog
from .order_timing_optimizer import OrderTimingOptimizer, MarketPhase

__all__ = [
    "TradeExecutorAgent",
    "TradeSignal",
    "OrderManager",
    "OrderType",
    "OrderStatus",
    "PositionMonitor",
    "Position",
    "PaperTradingManager",
    "PaperTrade",
    "TradeLogger",
    "TradeLog",
    "OrderTimingOptimizer",
    "MarketPhase"
]