import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeLog:
    """Represents a trade log entry"""

    timestamp: datetime
    log_type: str  # SIGNAL, ORDER, EXECUTION, EXIT, ERROR
    trade_id: str | None
    symbol: str
    action: str
    quantity: int
    price: float | None
    status: str
    details: dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary with serializable timestamp"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class TradeLogger:
    """Comprehensive trade logging and audit trail system"""

    def __init__(self, log_dir: str = "logs/trades"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current session logs
        self.session_logs: list[TradeLog] = []
        self.trade_summary: dict[str, dict] = {}

        # Statistics
        self.stats = {
            "signals_received": 0,
            "orders_placed": 0,
            "orders_executed": 0,
            "orders_failed": 0,
            "orders_rejected": 0,
            "trades_closed": 0,
        }

        # Initialize daily log file
        self.daily_log_file = self._get_daily_log_file()

    def _get_daily_log_file(self) -> Path:
        """Get path for today's log file"""
        today = date.today()
        filename = f"trades_{today.strftime('%Y%m%d')}.json"
        return self.log_dir / filename

    def log_signal(self, signal: Any) -> None:
        """Log incoming trade signal"""
        log_entry = TradeLog(
            timestamp=datetime.now(),
            log_type="SIGNAL",
            trade_id=None,
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            price=signal.price,
            status="RECEIVED",
            details={
                "order_type": signal.order_type,
                "stop_loss": signal.stop_loss,
                "target": signal.target,
                "confidence": signal.confidence,
                "strategy": signal.strategy_name,
                "metadata": signal.metadata,
            },
        )

        self._add_log(log_entry)
        self.stats["signals_received"] += 1

        logger.info(
            f"Signal logged: {signal.symbol} {signal.action} "
            f"{signal.quantity} @ {signal.price}"
        )

    def log_order(
        self,
        order_id: str,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        status: str,
        **details,
    ) -> None:
        """Log order placement"""
        log_entry = TradeLog(
            timestamp=datetime.now(),
            log_type="ORDER",
            trade_id=order_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=details.get("price"),
            status=status,
            details=details,
        )

        self._add_log(log_entry)
        self.stats["orders_placed"] += 1

        # Track order in summary
        self.trade_summary[order_id] = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "status": status,
            "events": [log_entry.to_dict()],
        }

    def log_trade(self, result: dict[str, Any]) -> None:
        """Log trade execution result"""
        if result.get("status") == "success":
            order_id = result.get("order_id")

            log_entry = TradeLog(
                timestamp=datetime.now(),
                log_type="EXECUTION",
                trade_id=order_id,
                symbol=result.get("order_params", {}).get("tradingsymbol", "UNKNOWN"),
                action=result.get("order_params", {}).get(
                    "transaction_type", "UNKNOWN"
                ),
                quantity=result.get("order_params", {}).get("quantity", 0),
                price=result.get("confirmation", {}).get("fill_price"),
                status="EXECUTED",
                details={
                    "fill_price": result.get("confirmation", {}).get("fill_price"),
                    "commission": result.get("commission"),
                    "slippage": result.get("slippage"),
                    "execution_time": result.get("execution_time"),
                },
            )

            self._add_log(log_entry)
            self.stats["orders_executed"] += 1

            # Update summary
            if order_id in self.trade_summary:
                self.trade_summary[order_id]["status"] = "EXECUTED"
                self.trade_summary[order_id]["events"].append(log_entry.to_dict())

        elif result.get("status") == "failed":
            self.stats["orders_failed"] += 1
            self._log_error("ORDER_FAILED", result)

        elif result.get("status") == "rejected":
            self.stats["orders_rejected"] += 1
            self._log_error("ORDER_REJECTED", result)

    def log_exit(
        self, order_id: str, exit_price: float, pnl: float, reason: str, **details
    ) -> None:
        """Log position exit"""
        log_entry = TradeLog(
            timestamp=datetime.now(),
            log_type="EXIT",
            trade_id=order_id,
            symbol=details.get("symbol", "UNKNOWN"),
            action=details.get("action", "UNKNOWN"),
            quantity=details.get("quantity", 0),
            price=exit_price,
            status="CLOSED",
            details={
                "exit_reason": reason,
                "pnl": pnl,
                "exit_price": exit_price,
                **details,
            },
        )

        self._add_log(log_entry)
        self.stats["trades_closed"] += 1

        # Update summary
        if order_id in self.trade_summary:
            self.trade_summary[order_id]["status"] = "CLOSED"
            self.trade_summary[order_id]["pnl"] = pnl
            self.trade_summary[order_id]["events"].append(log_entry.to_dict())

    def _log_error(self, error_type: str, details: dict) -> None:
        """Log error events"""
        log_entry = TradeLog(
            timestamp=datetime.now(),
            log_type="ERROR",
            trade_id=details.get("order_id"),
            symbol=details.get("symbol", "UNKNOWN"),
            action=details.get("action", "UNKNOWN"),
            quantity=details.get("quantity", 0),
            price=None,
            status=error_type,
            details=details,
        )

        self._add_log(log_entry)

    def _add_log(self, log_entry: TradeLog) -> None:
        """Add log entry and persist"""
        self.session_logs.append(log_entry)

        # Write to daily log file
        self._append_to_daily_log(log_entry)

    def _append_to_daily_log(self, log_entry: TradeLog) -> None:
        """Append log entry to daily file"""
        try:
            # Read existing logs
            logs = []
            if self.daily_log_file.exists():
                with open(self.daily_log_file) as f:
                    logs = json.load(f)

            # Append new log
            logs.append(log_entry.to_dict())

            # Write back
            with open(self.daily_log_file, "w") as f:
                json.dump(logs, f, indent=2)

        except Exception as e:
            logger.error(f"Error writing to daily log: {str(e)}")

    def get_trade_count(self) -> int:
        """Get total number of trades"""
        return self.stats["orders_executed"]

    def get_successful_trades(self) -> int:
        """Get number of successful trades"""
        return self.stats["orders_executed"]

    def get_failed_trades(self) -> int:
        """Get number of failed trades"""
        return self.stats["orders_failed"] + self.stats["orders_rejected"]

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of current trading session"""
        closed_trades = [
            trade
            for trade in self.trade_summary.values()
            if trade.get("status") == "CLOSED"
        ]

        total_pnl = sum(trade.get("pnl", 0) for trade in closed_trades)
        winning_trades = [t for t in closed_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in closed_trades if t.get("pnl", 0) < 0]

        return {
            "session_start": (
                self.session_logs[0].timestamp if self.session_logs else None
            ),
            "total_signals": self.stats["signals_received"],
            "total_orders": self.stats["orders_placed"],
            "executed_orders": self.stats["orders_executed"],
            "failed_orders": self.stats["orders_failed"],
            "rejected_orders": self.stats["orders_rejected"],
            "closed_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (
                len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
            ),
            "total_pnl": total_pnl,
            "average_win": (
                sum(t.get("pnl", 0) for t in winning_trades) / len(winning_trades)
                if winning_trades
                else 0
            ),
            "average_loss": (
                sum(t.get("pnl", 0) for t in losing_trades) / len(losing_trades)
                if losing_trades
                else 0
            ),
        }

    def generate_trade_report(self, output_file: str | None = None) -> str:
        """Generate comprehensive trade report"""
        summary = self.get_session_summary()

        report = f"""
=== Shagun Intelligence Trade Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SESSION SUMMARY
--------------
Session Start: {summary['session_start']}
Total Signals: {summary['total_signals']}
Orders Placed: {summary['total_orders']}
Orders Executed: {summary['executed_orders']}
Orders Failed: {summary['failed_orders']}
Orders Rejected: {summary['rejected_orders']}

PERFORMANCE METRICS
------------------
Closed Trades: {summary['closed_trades']}
Winning Trades: {summary['winning_trades']}
Losing Trades: {summary['losing_trades']}
Win Rate: {summary['win_rate']:.2f}%
Total P&L: {summary['total_pnl']:.2f}
Average Win: {summary['average_win']:.2f}
Average Loss: {summary['average_loss']:.2f}

DETAILED TRADES
--------------
"""

        # Add individual trade details
        for trade_id, trade in self.trade_summary.items():
            if trade.get("status") == "CLOSED":
                report += f"\n{trade_id}:"
                report += f"\n  Symbol: {trade['symbol']}"
                report += f"\n  Action: {trade['action']}"
                report += f"\n  Quantity: {trade['quantity']}"
                report += f"\n  P&L: {trade.get('pnl', 0):.2f}"
                report += f"\n  Events: {len(trade['events'])}"

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"Trade report saved to {output_file}")

        return report

    def export_to_csv(self, output_file: str) -> bool:
        """Export trades to CSV for analysis"""
        try:
            # Convert logs to DataFrame
            if not self.session_logs:
                logger.warning("No trades to export")
                return False

            logs_data = [log.to_dict() for log in self.session_logs]
            df = pd.DataFrame(logs_data)

            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Trades exported to {output_file}")

            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return False

    def get_trade_timeline(self, trade_id: str) -> list[dict]:
        """Get complete timeline of events for a trade"""
        if trade_id in self.trade_summary:
            return self.trade_summary[trade_id]["events"]
        return []

    def search_logs(
        self,
        symbol: str | None = None,
        log_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict]:
        """Search logs with filters"""
        results = []

        for log in self.session_logs:
            # Apply filters
            if symbol and log.symbol != symbol:
                continue
            if log_type and log.log_type != log_type:
                continue
            if start_time and log.timestamp < start_time:
                continue
            if end_time and log.timestamp > end_time:
                continue

            results.append(log.to_dict())

        return results
