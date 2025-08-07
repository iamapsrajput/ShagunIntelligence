import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Metrics for a single trade"""

    symbol: str
    entry_time: datetime
    exit_time: datetime | None
    entry_price: float
    exit_price: float | None
    quantity: int
    side: str  # BUY or SELL
    pnl: float = 0.0
    pnl_percent: float = 0.0
    holding_period: timedelta | None = None
    max_profit: float = 0.0
    max_loss: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED


class PerformanceMonitor:
    """Monitors and analyzes trading performance in real-time"""

    def __init__(self):
        # Trade tracking
        self.active_trades: dict[str, TradeMetrics] = {}
        self.closed_trades: list[TradeMetrics] = []
        self.trade_history = deque(maxlen=5000)

        # Performance metrics
        self.daily_metrics = defaultdict(
            lambda: {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "commission": 0.0,
                "max_drawdown": 0.0,
                "peak_capital": 100000.0,
            }
        )

        # Real-time metrics
        self.current_metrics = {
            "open_positions": 0,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "current_streak": 0,
        }

        # Capital tracking
        self.initial_capital = 100000.0
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.capital_history = deque(maxlen=1000)

        # Risk metrics
        self.risk_metrics = {
            "value_at_risk": 0.0,
            "conditional_var": 0.0,
            "beta": 0.0,
            "correlation_matrix": {},
            "position_concentration": {},
        }

        # Time-based performance
        self.hourly_performance = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
        self.daily_returns = deque(maxlen=252)  # One year of trading days

        # Benchmarking
        self.benchmark_returns = deque(maxlen=252)
        self.benchmark_symbol = "NIFTY50"

    def record_trade_entry(
        self, trade_id: str, symbol: str, side: str, quantity: int, entry_price: float
    ) -> None:
        """Record a new trade entry"""
        trade = TradeMetrics(
            symbol=symbol,
            entry_time=datetime.now(),
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            side=side,
        )

        self.active_trades[trade_id] = trade
        self.current_metrics["open_positions"] = len(self.active_trades)

        logger.info(
            f"Recorded trade entry: {trade_id} - {symbol} {side} "
            f"{quantity} @ {entry_price}"
        )

    def record_trade_exit(
        self, trade_id: str, exit_price: float, commission: float = 0.0
    ) -> TradeMetrics | None:
        """Record a trade exit and calculate metrics"""
        if trade_id not in self.active_trades:
            logger.warning(f"Trade {trade_id} not found in active trades")
            return None

        trade = self.active_trades[trade_id]
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.holding_period = trade.exit_time - trade.entry_time

        # Calculate P&L
        if trade.side == "BUY":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity - commission
        else:  # SELL
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity - commission

        trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
        trade.status = "CLOSED"

        # Move to closed trades
        self.closed_trades.append(trade)
        del self.active_trades[trade_id]

        # Update metrics
        self._update_metrics_on_trade_close(trade, commission)

        logger.info(
            f"Recorded trade exit: {trade_id} - P&L: {trade.pnl:.2f} "
            f"({trade.pnl_percent:.2f}%)"
        )

        return trade

    def update_position_metrics(self, trade_id: str, current_price: float) -> None:
        """Update metrics for an open position"""
        if trade_id not in self.active_trades:
            return

        trade = self.active_trades[trade_id]

        # Calculate unrealized P&L
        if trade.side == "BUY":
            unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
        else:
            unrealized_pnl = (trade.entry_price - current_price) * trade.quantity

        # Update max profit/loss
        trade.max_profit = max(trade.max_profit, unrealized_pnl)
        trade.max_loss = min(trade.max_loss, unrealized_pnl)

    def _update_metrics_on_trade_close(
        self, trade: TradeMetrics, commission: float
    ) -> None:
        """Update performance metrics when a trade closes"""
        # Update daily metrics
        today = datetime.now().date()
        daily = self.daily_metrics[today]

        daily["trades"] += 1
        daily["commission"] += commission

        if trade.pnl > 0:
            daily["wins"] += 1
            self.current_metrics["consecutive_wins"] += 1
            self.current_metrics["consecutive_losses"] = 0
        else:
            daily["losses"] += 1
            self.current_metrics["consecutive_losses"] += 1
            self.current_metrics["consecutive_wins"] = 0

        daily["gross_pnl"] += trade.pnl + commission
        daily["net_pnl"] += trade.pnl

        # Update capital
        self.current_capital += trade.pnl
        self.capital_history.append(
            {"timestamp": datetime.now(), "capital": self.current_capital}
        )

        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        self.current_metrics["max_drawdown"] = max(
            self.current_metrics["max_drawdown"], drawdown
        )

        # Update time-based performance
        hour = trade.entry_time.hour
        self.hourly_performance[hour]["trades"] += 1
        self.hourly_performance[hour]["pnl"] += trade.pnl

        # Update aggregate metrics
        self._update_aggregate_metrics()

    def _update_aggregate_metrics(self) -> None:
        """Update aggregate performance metrics"""
        if not self.closed_trades:
            return

        # Calculate win rate
        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]

        total_trades = len(self.closed_trades)
        self.current_metrics["win_rate"] = (
            (len(wins) / total_trades * 100) if total_trades > 0 else 0
        )

        # Average win/loss
        self.current_metrics["avg_win"] = np.mean([t.pnl for t in wins]) if wins else 0
        self.current_metrics["avg_loss"] = (
            np.mean([t.pnl for t in losses]) if losses else 0
        )

        # Profit factor
        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 1
        self.current_metrics["profit_factor"] = (
            total_wins / total_losses if total_losses > 0 else 0
        )

        # Calculate returns for different periods
        self._calculate_period_returns()

        # Calculate Sharpe ratio
        self._calculate_sharpe_ratio()

        # Update risk metrics
        self._update_risk_metrics()

    def _calculate_period_returns(self) -> None:
        """Calculate returns for different time periods"""
        now = datetime.now()

        # Daily P&L
        today = now.date()
        self.current_metrics["daily_pnl"] = self.daily_metrics[today]["net_pnl"]

        # Weekly P&L
        week_start = now - timedelta(days=7)
        weekly_trades = [
            t for t in self.closed_trades if t.exit_time and t.exit_time >= week_start
        ]
        self.current_metrics["weekly_pnl"] = sum(t.pnl for t in weekly_trades)

        # Monthly P&L
        month_start = now - timedelta(days=30)
        monthly_trades = [
            t for t in self.closed_trades if t.exit_time and t.exit_time >= month_start
        ]
        self.current_metrics["monthly_pnl"] = sum(t.pnl for t in monthly_trades)

    def _calculate_sharpe_ratio(self) -> None:
        """Calculate Sharpe ratio based on daily returns"""
        if len(self.daily_returns) < 30:  # Need at least 30 days
            return

        returns = np.array(list(self.daily_returns))

        # Assuming risk-free rate of 5% annually
        risk_free_rate = 0.05 / 252  # Daily risk-free rate

        excess_returns = returns - risk_free_rate

        if len(excess_returns) > 0 and np.std(excess_returns) > 0:
            # Annualized Sharpe ratio
            sharpe = np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))
            self.current_metrics["sharpe_ratio"] = sharpe

    def _update_risk_metrics(self) -> None:
        """Update risk-related metrics"""
        if len(self.daily_returns) < 10:
            return

        returns = np.array(list(self.daily_returns))

        # Value at Risk (95% confidence)
        if len(returns) > 0:
            self.risk_metrics["value_at_risk"] = np.percentile(returns, 5)

            # Conditional VaR (expected loss beyond VaR)
            var_threshold = self.risk_metrics["value_at_risk"]
            beyond_var = returns[returns <= var_threshold]
            self.risk_metrics["conditional_var"] = (
                np.mean(beyond_var) if len(beyond_var) > 0 else var_threshold
            )

        # Position concentration
        if self.active_trades:
            total_value = sum(
                t.entry_price * t.quantity for t in self.active_trades.values()
            )

            for _trade_id, trade in self.active_trades.items():
                position_value = trade.entry_price * trade.quantity
                concentration = position_value / total_value if total_value > 0 else 0
                self.risk_metrics["position_concentration"][
                    trade.symbol
                ] = concentration

    def get_current_performance(self) -> dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.current_metrics,
            "total_trades": len(self.closed_trades),
            "active_trades": len(self.active_trades),
            "current_capital": self.current_capital,
            "total_return": (
                (self.current_capital - self.initial_capital)
                / self.initial_capital
                * 100
            ),
            "risk_metrics": self.risk_metrics,
            "current_streak_type": (
                "wins" if self.current_metrics["consecutive_wins"] > 0 else "losses"
            ),
        }

    def get_trade_statistics(self) -> dict[str, Any]:
        """Get detailed trade statistics"""
        if not self.closed_trades:
            return {"message": "No closed trades available"}

        # Calculate various statistics
        holding_periods = [
            t.holding_period.total_seconds() / 3600  # Convert to hours
            for t in self.closed_trades
            if t.holding_period
        ]

        pnl_values = [t.pnl for t in self.closed_trades]
        pnl_percents = [t.pnl_percent for t in self.closed_trades]

        return {
            "total_closed_trades": len(self.closed_trades),
            "average_holding_period_hours": (
                np.mean(holding_periods) if holding_periods else 0
            ),
            "median_holding_period_hours": (
                np.median(holding_periods) if holding_periods else 0
            ),
            "best_trade": {
                "pnl": max(pnl_values) if pnl_values else 0,
                "pnl_percent": max(pnl_percents) if pnl_percents else 0,
            },
            "worst_trade": {
                "pnl": min(pnl_values) if pnl_values else 0,
                "pnl_percent": min(pnl_percents) if pnl_percents else 0,
            },
            "average_trade_pnl": np.mean(pnl_values) if pnl_values else 0,
            "trade_pnl_std": np.std(pnl_values) if pnl_values else 0,
            "positive_trades": len([t for t in self.closed_trades if t.pnl > 0]),
            "negative_trades": len([t for t in self.closed_trades if t.pnl <= 0]),
        }

    def get_time_based_analysis(self) -> dict[str, Any]:
        """Get performance analysis by time"""
        hourly_stats = {}

        for hour, data in self.hourly_performance.items():
            if data["trades"] > 0:
                hourly_stats[hour] = {
                    "trades": data["trades"],
                    "total_pnl": data["pnl"],
                    "avg_pnl": data["pnl"] / data["trades"],
                    "win_rate": self._calculate_hourly_win_rate(hour),
                }

        return {
            "hourly_performance": hourly_stats,
            "best_hour": (
                max(hourly_stats.items(), key=lambda x: x[1]["avg_pnl"])[0]
                if hourly_stats
                else None
            ),
            "worst_hour": (
                min(hourly_stats.items(), key=lambda x: x[1]["avg_pnl"])[0]
                if hourly_stats
                else None
            ),
        }

    def _calculate_hourly_win_rate(self, hour: int) -> float:
        """Calculate win rate for a specific hour"""
        hourly_trades = [t for t in self.closed_trades if t.entry_time.hour == hour]

        if not hourly_trades:
            return 0.0

        wins = len([t for t in hourly_trades if t.pnl > 0])
        return (wins / len(hourly_trades)) * 100

    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        perf = self.get_current_performance()
        stats = self.get_trade_statistics()

        report = f"""
=== Shagun Intelligence Performance Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ACCOUNT SUMMARY
--------------
Initial Capital: ${self.initial_capital:,.2f}
Current Capital: ${self.current_capital:,.2f}
Total Return: {perf['total_return']:.2f}%
Peak Capital: ${self.peak_capital:,.2f}
Max Drawdown: {perf['max_drawdown']*100:.2f}%

TRADING METRICS
--------------
Total Trades: {perf['total_trades']}
Active Positions: {perf['active_trades']}
Win Rate: {perf['win_rate']:.2f}%
Profit Factor: {perf['profit_factor']:.2f}
Sharpe Ratio: {perf['sharpe_ratio']:.2f}

P&L SUMMARY
----------
Daily P&L: ${perf['daily_pnl']:.2f}
Weekly P&L: ${perf['weekly_pnl']:.2f}
Monthly P&L: ${perf['monthly_pnl']:.2f}
Average Win: ${perf['avg_win']:.2f}
Average Loss: ${perf['avg_loss']:.2f}

RISK METRICS
-----------
Value at Risk (95%): {self.risk_metrics['value_at_risk']:.2f}%
Conditional VaR: {self.risk_metrics['conditional_var']:.2f}%
Current Streak: {perf['consecutive_wins'] or perf['consecutive_losses']} {perf['current_streak_type']}

TRADE STATISTICS
---------------
Best Trade: ${stats.get('best_trade', {}).get('pnl', 0):.2f} ({stats.get('best_trade', {}).get('pnl_percent', 0):.2f}%)
Worst Trade: ${stats.get('worst_trade', {}).get('pnl', 0):.2f} ({stats.get('worst_trade', {}).get('pnl_percent', 0):.2f}%)
Avg Holding Period: {stats.get('average_holding_period_hours', 0):.2f} hours
"""

        return report

    def export_performance_data(self, filepath: str) -> bool:
        """Export performance data for analysis"""
        try:
            # Prepare data for export
            trade_data = []
            for trade in self.closed_trades[-1000:]:  # Last 1000 trades
                trade_data.append(
                    {
                        "symbol": trade.symbol,
                        "entry_time": trade.entry_time.isoformat(),
                        "exit_time": (
                            trade.exit_time.isoformat() if trade.exit_time else None
                        ),
                        "side": trade.side,
                        "quantity": trade.quantity,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "pnl": trade.pnl,
                        "pnl_percent": trade.pnl_percent,
                        "holding_period_hours": (
                            trade.holding_period.total_seconds() / 3600
                            if trade.holding_period
                            else None
                        ),
                    }
                )

            # Create DataFrame
            df = pd.DataFrame(trade_data)

            # Save to CSV
            df.to_csv(filepath, index=False)

            logger.info(f"Performance data exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting performance data: {str(e)}")
            return False
