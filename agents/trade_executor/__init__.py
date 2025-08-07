"""
Trade Executor Agent for Shagun Intelligence

This module provides comprehensive trade execution capabilities including:
- Order management with multiple order types
- Real-time position monitoring
- Paper trading simulation
- Trade logging and audit trail
- Order timing optimization
"""

from .agent import TradeExecutorAgent, TradeSignal
from .order_manager import OrderManager, OrderStatus, OrderType
from .order_timing_optimizer import MarketPhase, OrderTimingOptimizer
from .paper_trading_manager import PaperTrade, PaperTradingManager
from .position_monitor import Position, PositionMonitor
from .trade_logger import TradeLog, TradeLogger

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
    "MarketPhase",
]
