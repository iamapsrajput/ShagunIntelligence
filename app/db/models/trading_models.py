"""
Trading-specific database models
Enhanced models for advanced trading system with time-series optimization
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class MarketData(Base):
    """Time-series market data with optimized indexing"""

    __tablename__ = "market_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(10), nullable=False)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 1d, etc.
    timestamp = Column(DateTime, nullable=False, index=True)

    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)

    # Additional market data
    vwap = Column(Float)  # Volume Weighted Average Price
    turnover = Column(Float)
    trades_count = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite indexes for time-series queries
    __table_args__ = (
        Index("idx_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
        Index("idx_timestamp_symbol", "timestamp", "symbol"),
        Index("idx_symbol_exchange", "symbol", "exchange"),
    )


class TradingOrder(Base):
    """Enhanced trading orders with execution tracking"""

    __tablename__ = "trading_orders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    broker_name = Column(String(50), nullable=False)

    # Order details
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(10), nullable=False)
    transaction_type = Column(String(10), nullable=False)  # BUY, SELL
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT, SL, SL-M
    product_type = Column(String(10), default="MIS")  # MIS, CNC, NRML

    # Quantities and prices
    quantity = Column(Integer, nullable=False)
    filled_quantity = Column(Integer, default=0)
    pending_quantity = Column(Integer, default=0)
    cancelled_quantity = Column(Integer, default=0)

    price = Column(Float)
    trigger_price = Column(Float)
    average_price = Column(Float, default=0.0)

    # Order status and timing
    status = Column(String(20), nullable=False, index=True)
    validity = Column(String(10), default="DAY")
    disclosed_quantity = Column(Integer, default=0)

    # Timestamps
    order_timestamp = Column(DateTime, nullable=False)
    exchange_timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Metadata
    tag = Column(String(100))
    parent_order_id = Column(String(50))  # For bracket/cover orders
    strategy_name = Column(String(100))
    notes = Column(Text)

    # Execution analytics
    slippage = Column(Float)  # Difference from expected price
    execution_time_ms = Column(Integer)  # Time to execute in milliseconds
    market_impact = Column(Float)  # Estimated market impact

    __table_args__ = (
        Index("idx_symbol_timestamp", "symbol", "order_timestamp"),
        Index("idx_status_timestamp", "status", "order_timestamp"),
        Index("idx_broker_timestamp", "broker_name", "order_timestamp"),
    )


class Position(Base):
    """Trading positions with P&L tracking"""

    __tablename__ = "positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    broker_name = Column(String(50), nullable=False)

    # Position details
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(10), nullable=False)
    product_type = Column(String(10), nullable=False)

    # Position data
    quantity = Column(Integer, nullable=False)
    average_price = Column(Float, nullable=False)
    last_price = Column(Float, nullable=False)

    # P&L calculations
    pnl = Column(Float, nullable=False)
    pnl_percent = Column(Float, nullable=False)
    day_pnl = Column(Float, default=0.0)

    # Risk metrics
    value_at_risk = Column(Float)
    max_loss = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float, default=0.0)

    # Timestamps
    position_date = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_symbol_date", "symbol", "position_date"),
        Index("idx_broker_date", "broker_name", "position_date"),
    )


class Holding(Base):
    """Portfolio holdings with performance tracking"""

    __tablename__ = "holdings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    broker_name = Column(String(50), nullable=False)

    # Holding details
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(10), nullable=False)
    isin = Column(String(20))  # International Securities Identification Number

    # Holding data
    quantity = Column(Integer, nullable=False)
    average_price = Column(Float, nullable=False)
    last_price = Column(Float, nullable=False)

    # Valuation
    current_value = Column(Float, nullable=False)
    investment_value = Column(Float, nullable=False)
    pnl = Column(Float, nullable=False)
    pnl_percent = Column(Float, nullable=False)

    # Additional data
    collateral_quantity = Column(Integer, default=0)
    collateral_type = Column(String(20))
    haircut = Column(Float, default=0.0)

    # Timestamps
    holding_date = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_symbol_date", "symbol", "holding_date"),
        Index("idx_broker_date", "broker_name", "holding_date"),
    )


class RiskMetrics(Base):
    """Portfolio risk metrics with time-series tracking"""

    __tablename__ = "risk_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(String(50), nullable=False, index=True)

    # Portfolio values
    total_value = Column(Float, nullable=False)
    total_exposure = Column(Float, nullable=False)
    leverage = Column(Float, nullable=False)

    # VaR metrics
    var_1d = Column(Float, nullable=False)
    var_5d = Column(Float, nullable=False)
    cvar_1d = Column(Float, nullable=False)
    expected_shortfall = Column(Float, nullable=False)

    # Drawdown metrics
    max_drawdown = Column(Float, nullable=False)
    current_drawdown = Column(Float, nullable=False)
    drawdown_duration_days = Column(Integer, default=0)

    # Performance ratios
    sharpe_ratio = Column(Float, nullable=False)
    sortino_ratio = Column(Float, nullable=False)
    calmar_ratio = Column(Float, nullable=False)

    # Market exposure
    beta = Column(Float, nullable=False)
    alpha = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)

    # Sector exposures (JSON field)
    sector_exposures = Column(JSON)
    correlation_matrix = Column(JSON)

    # Risk level and alerts
    risk_level = Column(String(20), nullable=False)
    risk_score = Column(Float, nullable=False)
    violations_count = Column(Integer, default=0)

    # Timestamps
    calculation_date = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_portfolio_date", "portfolio_id", "calculation_date"),
        Index("idx_risk_level_date", "risk_level", "calculation_date"),
    )


class TradingStrategy(Base):
    """Trading strategies with performance tracking"""

    __tablename__ = "trading_strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)

    # Strategy configuration
    strategy_type = Column(String(50), nullable=False)  # momentum, mean_reversion, etc.
    parameters = Column(JSON)  # Strategy-specific parameters
    risk_parameters = Column(JSON)  # Risk management parameters

    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)

    total_pnl = Column(Float, default=0.0)
    max_profit = Column(Float, default=0.0)
    max_loss = Column(Float, default=0.0)
    average_profit = Column(Float, default=0.0)
    average_loss = Column(Float, default=0.0)

    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    calmar_ratio = Column(Float, default=0.0)

    # Status and control
    is_active = Column(Boolean, default=True)
    is_live = Column(Boolean, default=False)
    last_signal_time = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    signals = relationship("TradingSignal", back_populates="strategy")


class TradingSignal(Base):
    """Trading signals generated by strategies"""

    __tablename__ = "trading_signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=False
    )

    # Signal details
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(10), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    signal_strength = Column(Float, nullable=False)  # 0.0 to 1.0
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0

    # Price and timing
    signal_price = Column(Float, nullable=False)
    target_price = Column(Float)
    stop_loss_price = Column(Float)
    timeframe = Column(String(10), nullable=False)

    # Signal metadata
    reasoning = Column(JSON)  # List of reasons for the signal
    technical_indicators = Column(JSON)  # Indicator values at signal time
    market_conditions = Column(JSON)  # Market condition assessment

    # Execution tracking
    is_executed = Column(Boolean, default=False)
    execution_price = Column(Float)
    execution_time = Column(DateTime)
    order_id = Column(String(50))

    # Performance tracking
    pnl = Column(Float)
    pnl_percent = Column(Float)
    max_favorable_excursion = Column(Float)
    max_adverse_excursion = Column(Float)

    # Timestamps
    signal_time = Column(DateTime, nullable=False, index=True)
    expiry_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    strategy = relationship("TradingStrategy", back_populates="signals")

    __table_args__ = (
        Index("idx_symbol_signal_time", "symbol", "signal_time"),
        Index("idx_strategy_signal_time", "strategy_id", "signal_time"),
        Index("idx_signal_type_time", "signal_type", "signal_time"),
    )


class BacktestResult(Base):
    """Backtest results for strategy validation"""

    __tablename__ = "backtest_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=False
    )

    # Backtest configuration
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    symbols = Column(JSON)  # List of symbols tested

    # Performance results
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    total_return_percent = Column(Float, nullable=False)
    annualized_return = Column(Float, nullable=False)

    # Trade statistics
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)
    losing_trades = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)

    # Risk metrics
    max_drawdown = Column(Float, nullable=False)
    max_drawdown_percent = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    sortino_ratio = Column(Float, nullable=False)
    calmar_ratio = Column(Float, nullable=False)

    # Additional metrics
    profit_factor = Column(Float, nullable=False)
    average_trade = Column(Float, nullable=False)
    largest_win = Column(Float, nullable=False)
    largest_loss = Column(Float, nullable=False)

    # Detailed results (JSON)
    equity_curve = Column(JSON)  # Daily equity values
    trade_log = Column(JSON)  # Individual trade details
    monthly_returns = Column(JSON)  # Monthly return breakdown

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_strategy_backtest_date", "strategy_id", "created_at"),)
