import json
import logging
import random
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Represents a paper trade"""

    trade_id: str
    symbol: str
    action: str
    quantity: int
    entry_price: float
    exit_price: float | None = None
    timestamp: datetime = None
    exit_timestamp: datetime | None = None
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class PaperTradingManager:
    """Manages paper trading simulation for testing strategies"""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: dict[str, PaperTrade] = {}
        self.trade_history: list[PaperTrade] = []

        # Market simulation parameters
        self.slippage_range = (0.0001, 0.0005)  # 0.01% to 0.05%
        self.commission_rate = 0.0003  # 0.03% per trade
        self.market_impact = 0.0001  # 0.01% for large orders

        # Price simulation
        self.price_data: dict[str, float] = {}
        self.price_volatility = 0.02  # 2% daily volatility
        self.price_update_thread = None
        self.is_running = False

        # Performance metrics
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "max_drawdown": 0.0,
            "peak_capital": initial_capital,
        }

        # Start price simulation
        self._start_price_simulation()

    def _start_price_simulation(self):
        """Start simulating price movements"""
        self.is_running = True
        self.price_update_thread = threading.Thread(
            target=self._price_update_loop, daemon=True
        )
        self.price_update_thread.start()
        logger.info("Started price simulation for paper trading")

    def _price_update_loop(self):
        """Continuously update simulated prices"""
        while self.is_running:
            try:
                # Update prices for all tracked symbols
                for symbol in list(self.price_data.keys()):
                    self._update_price(symbol)

                time.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Error in price update loop: {str(e)}")

    def _update_price(self, symbol: str):
        """Update simulated price for a symbol"""
        if symbol not in self.price_data:
            return

        current_price = self.price_data[symbol]

        # Simulate random walk with drift
        drift = random.uniform(-0.0001, 0.0001)  # Small drift
        volatility = random.gauss(0, self.price_volatility / 100)  # Daily vol scaled

        new_price = current_price * (1 + drift + volatility)
        self.price_data[symbol] = max(new_price, 0.01)  # Ensure positive price

    def execute_trade(self, signal: Any) -> dict[str, Any]:
        """Execute a paper trade based on signal"""
        try:
            # Generate trade ID
            trade_id = f"PT_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

            # Get current price
            entry_price = self.get_price(signal.symbol)

            # Calculate slippage
            slippage_pct = random.uniform(*self.slippage_range)
            if signal.action == "BUY":
                entry_price *= 1 + slippage_pct
            else:
                entry_price *= 1 - slippage_pct

            # Calculate commission
            trade_value = entry_price * signal.quantity
            commission = trade_value * self.commission_rate

            # Check if we have enough capital
            required_capital = trade_value + commission
            if signal.action == "BUY" and required_capital > self.current_capital:
                return {
                    "status": "rejected",
                    "reason": "insufficient_capital",
                    "required": required_capital,
                    "available": self.current_capital,
                }

            # Create paper trade
            trade = PaperTrade(
                trade_id=trade_id,
                symbol=signal.symbol,
                action=signal.action,
                quantity=signal.quantity,
                entry_price=entry_price,
                commission=commission,
                slippage=entry_price * slippage_pct,
                metadata={
                    "signal_price": signal.price,
                    "strategy": signal.strategy_name,
                    "confidence": signal.confidence,
                },
            )

            # Update capital
            if signal.action == "BUY":
                self.current_capital -= required_capital

            # Store trade
            self.trades[trade_id] = trade
            self.metrics["total_trades"] += 1
            self.metrics["total_commission"] += commission
            self.metrics["total_slippage"] += trade.slippage

            logger.info(
                f"Paper trade executed: {trade_id} - {signal.symbol} "
                f"{signal.action} {signal.quantity} @ {entry_price:.2f}"
            )

            return {
                "status": "success",
                "order_id": trade_id,
                "trade": asdict(trade),
                "execution_price": entry_price,
                "commission": commission,
                "slippage": trade.slippage,
            }

        except Exception as e:
            logger.error(f"Error executing paper trade: {str(e)}")
            return {"status": "error", "message": str(e)}

    def close_trade(self, trade_id: str, exit_price: float | None = None) -> dict:
        """Close a paper trade"""
        if trade_id not in self.trades:
            return {"status": "error", "message": "Trade not found"}

        trade = self.trades[trade_id]

        if trade.status != "OPEN":
            return {"status": "error", "message": "Trade already closed"}

        # Get exit price
        if exit_price is None:
            exit_price = self.get_price(trade.symbol)

        # Apply slippage to exit
        slippage_pct = random.uniform(*self.slippage_range)
        if trade.action == "BUY":
            exit_price *= 1 - slippage_pct  # Selling, so worse price
        else:
            exit_price *= 1 + slippage_pct  # Buying back, so worse price

        # Calculate commission for exit
        exit_commission = exit_price * trade.quantity * self.commission_rate

        # Calculate P&L
        if trade.action == "BUY":
            gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # SELL
            gross_pnl = (trade.entry_price - exit_price) * trade.quantity

        net_pnl = gross_pnl - trade.commission - exit_commission

        # Update trade
        trade.exit_price = exit_price
        trade.exit_timestamp = datetime.now()
        trade.status = "CLOSED"
        trade.pnl = net_pnl
        trade.commission += exit_commission

        # Update capital
        if trade.action == "BUY":
            self.current_capital += exit_price * trade.quantity - exit_commission
        else:
            # For short positions, return the initial sale proceeds minus buyback cost
            self.current_capital -= exit_price * trade.quantity + exit_commission

        # Update metrics
        self.metrics["total_pnl"] += net_pnl
        self.metrics["total_commission"] += exit_commission

        if net_pnl > 0:
            self.metrics["winning_trades"] += 1
        else:
            self.metrics["losing_trades"] += 1

        # Update drawdown
        if self.current_capital > self.metrics["peak_capital"]:
            self.metrics["peak_capital"] = self.current_capital

        drawdown = (self.metrics["peak_capital"] - self.current_capital) / self.metrics[
            "peak_capital"
        ]
        self.metrics["max_drawdown"] = max(self.metrics["max_drawdown"], drawdown)

        # Move to history
        self.trade_history.append(trade)
        del self.trades[trade_id]

        logger.info(f"Paper trade closed: {trade_id} - P&L: {net_pnl:.2f}")

        return {
            "status": "success",
            "trade_id": trade_id,
            "pnl": net_pnl,
            "exit_price": exit_price,
            "commission": exit_commission,
        }

    def get_price(self, symbol: str) -> float:
        """Get simulated price for a symbol"""
        # Initialize price if not exists
        if symbol not in self.price_data:
            # Use a base price with some randomness
            base_price = 100 + random.uniform(-20, 20)
            self.price_data[symbol] = base_price

        return self.price_data[symbol]

    def get_open_trades(self) -> list[dict]:
        """Get all open paper trades"""
        open_trades = []

        for trade in self.trades.values():
            current_price = self.get_price(trade.symbol)

            # Calculate unrealized P&L
            if trade.action == "BUY":
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl = (trade.entry_price - current_price) * trade.quantity

            unrealized_pnl -= trade.commission  # Subtract entry commission

            open_trades.append(
                {
                    **asdict(trade),
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                }
            )

        return open_trades

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = self.metrics.copy()

        # Calculate additional metrics
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = (
                metrics["winning_trades"] / metrics["total_trades"]
            ) * 100
            metrics["average_trade"] = metrics["total_pnl"] / metrics["total_trades"]
        else:
            metrics["win_rate"] = 0
            metrics["average_trade"] = 0

        # Calculate returns
        metrics["total_return"] = (
            (self.current_capital - self.initial_capital) / self.initial_capital
        ) * 100
        metrics["current_capital"] = self.current_capital
        metrics["initial_capital"] = self.initial_capital

        # Add trade statistics
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t.pnl > 0]
            losing_trades = [t for t in self.trade_history if t.pnl < 0]

            metrics["average_win"] = (
                sum(t.pnl for t in winning_trades) / len(winning_trades)
                if winning_trades
                else 0
            )
            metrics["average_loss"] = (
                sum(t.pnl for t in losing_trades) / len(losing_trades)
                if losing_trades
                else 0
            )

            # Profit factor
            total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
            total_losses = (
                abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
            )
            metrics["profit_factor"] = (
                total_wins / total_losses if total_losses > 0 else 0
            )

        return metrics

    def get_trade_history(self) -> list[dict]:
        """Get complete trade history"""
        return [asdict(trade) for trade in self.trade_history]

    def export_results(self, filepath: str) -> bool:
        """Export paper trading results to file"""
        try:
            results = {
                "metrics": self.get_performance_metrics(),
                "trade_history": self.get_trade_history(),
                "open_trades": self.get_open_trades(),
                "timestamp": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Paper trading results exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return False

    def reset(self):
        """Reset paper trading account"""
        self.current_capital = self.initial_capital
        self.trades.clear()
        self.trade_history.clear()
        self.price_data.clear()

        # Reset metrics
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "max_drawdown": 0.0,
            "peak_capital": self.initial_capital,
        }

        logger.info("Paper trading account reset")

    def stop(self):
        """Stop paper trading simulation"""
        self.is_running = False
        if self.price_update_thread:
            self.price_update_thread.join()
        logger.info("Paper trading simulation stopped")
