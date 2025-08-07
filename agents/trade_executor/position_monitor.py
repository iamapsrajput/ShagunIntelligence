import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position"""

    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    entry_price: float
    current_price: float = 0.0
    stop_loss: float | None = None
    target: float | None = None
    entry_time: datetime = field(default_factory=datetime.now)
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: str = "OPEN"
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_pnl(self, current_price: float):
        """Update P&L based on current price"""
        self.current_price = current_price

        if self.side == "BUY":
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:  # SELL
            self.pnl = (self.entry_price - current_price) * self.quantity

        self.pnl_percent = (self.pnl / (self.entry_price * self.quantity)) * 100


class PositionMonitor:
    """Monitors and manages open positions in real-time"""

    def __init__(self, kite_client=None, paper_trading=False):
        self.kite_client = kite_client
        self.paper_trading = paper_trading
        self.positions: dict[str, Position] = {}
        self.closed_positions: list[Position] = []

        # Monitoring configuration
        self.monitor_interval = 5  # seconds
        self.is_monitoring = False
        self.monitor_thread = None

        # Risk metrics
        self.max_positions = 10
        self.max_loss_per_position = 0.02  # 2%
        self.trailing_stop_activation = 0.01  # 1% profit to activate trailing stop
        self.trailing_stop_distance = 0.005  # 0.5% trailing distance

    def add_position(self, order_id: str, signal: Any) -> None:
        """Add a new position to monitor"""
        try:
            # Get order details
            order_details = self._get_order_details(order_id)

            if order_details and order_details.get("filled_quantity", 0) > 0:
                position = Position(
                    order_id=order_id,
                    symbol=signal.symbol,
                    side=signal.action.upper(),
                    quantity=order_details["filled_quantity"],
                    entry_price=order_details["average_price"],
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    metadata={
                        "strategy": signal.strategy_name,
                        "confidence": signal.confidence,
                    },
                )

                self.positions[order_id] = position
                logger.info(
                    f"Added position to monitor: {position.symbol} - {position.side}"
                )

                # Start monitoring if not already running
                if not self.is_monitoring:
                    self.start_monitoring()

        except Exception as e:
            logger.error(f"Error adding position: {str(e)}")

    def _get_order_details(self, order_id: str) -> dict | None:
        """Get order details from broker"""
        try:
            if self.paper_trading:
                # Return simulated order details
                return {
                    "order_id": order_id,
                    "filled_quantity": 100,
                    "average_price": 100.0,
                    "status": "COMPLETE",
                }

            order_history = self.kite_client.order_history(order_id)
            if order_history:
                return order_history[-1]  # Latest status

        except Exception as e:
            logger.error(f"Error getting order details: {str(e)}")

        return None

    def start_monitoring(self) -> None:
        """Start real-time position monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitor_thread.start()
            logger.info("Position monitoring started")

    def stop_monitoring(self) -> None:
        """Stop position monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Position monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update all positions
                for _order_id, position in list(self.positions.items()):
                    self._update_position(position)

                    # Check for exit conditions
                    if self._should_exit_position(position):
                        self._close_position(position)

                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(self.monitor_interval)

    def _update_position(self, position: Position) -> None:
        """Update position with current market data"""
        try:
            current_price = self._get_current_price(position.symbol)
            if current_price > 0:
                position.update_pnl(current_price)

                # Update trailing stop if applicable
                self._update_trailing_stop(position)

                # Log significant P&L changes
                if abs(position.pnl_percent) > 1.0:
                    logger.info(
                        f"Position {position.symbol}: P&L {position.pnl_percent:.2f}%"
                    )

        except Exception as e:
            logger.error(f"Error updating position {position.symbol}: {str(e)}")

    def _get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            if self.paper_trading:
                # Simulate price movement
                import random

                base_price = 100
                return base_price * (1 + random.uniform(-0.02, 0.02))

            quote = self.kite_client.quote([symbol])
            return quote[symbol]["last_price"]

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return 0.0

    def _should_exit_position(self, position: Position) -> bool:
        """Check if position should be exited"""
        # Check stop loss
        if position.stop_loss:
            if position.side == "BUY" and position.current_price <= position.stop_loss:
                logger.info(f"Stop loss hit for {position.symbol}")
                return True
            elif (
                position.side == "SELL" and position.current_price >= position.stop_loss
            ):
                logger.info(f"Stop loss hit for {position.symbol}")
                return True

        # Check target
        if position.target:
            if position.side == "BUY" and position.current_price >= position.target:
                logger.info(f"Target hit for {position.symbol}")
                return True
            elif position.side == "SELL" and position.current_price <= position.target:
                logger.info(f"Target hit for {position.symbol}")
                return True

        # Check max loss
        if position.pnl_percent <= -self.max_loss_per_position * 100:
            logger.warning(f"Max loss reached for {position.symbol}")
            return True

        return False

    def _update_trailing_stop(self, position: Position) -> None:
        """Update trailing stop loss if conditions are met"""
        if not position.stop_loss:
            return

        # Check if trailing stop should be activated
        if position.pnl_percent >= self.trailing_stop_activation * 100:
            if position.side == "BUY":
                # Calculate new stop loss
                new_stop = position.current_price * (1 - self.trailing_stop_distance)

                # Only update if new stop is higher than current
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    logger.info(
                        f"Updated trailing stop for {position.symbol} to {new_stop}"
                    )

            else:  # SELL
                # Calculate new stop loss
                new_stop = position.current_price * (1 + self.trailing_stop_distance)

                # Only update if new stop is lower than current
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    logger.info(
                        f"Updated trailing stop for {position.symbol} to {new_stop}"
                    )

    def _close_position(self, position: Position) -> None:
        """Close a position"""
        position.status = "CLOSED"
        self.closed_positions.append(position)
        del self.positions[position.order_id]
        logger.info(f"Closed position: {position.symbol} - P&L: {position.pnl:.2f}")

    def get_open_positions(self) -> list[dict]:
        """Get all open positions"""
        return [
            {
                "order_id": pos.order_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "stop_loss": pos.stop_loss,
                "target": pos.target,
                "pnl": pos.pnl,
                "pnl_percent": pos.pnl_percent,
                "duration": (datetime.now() - pos.entry_time).total_seconds()
                / 60,  # minutes
                "metadata": pos.metadata,
            }
            for pos in self.positions.values()
        ]

    def get_position_by_symbol(self, symbol: str) -> Position | None:
        """Get position by symbol"""
        for position in self.positions.values():
            if position.symbol == symbol:
                return position
        return None

    def get_total_pnl(self) -> float:
        """Get total P&L across all positions"""
        open_pnl = sum(pos.pnl for pos in self.positions.values())
        closed_pnl = sum(pos.pnl for pos in self.closed_positions)
        return open_pnl + closed_pnl

    def get_position_summary(self) -> dict[str, Any]:
        """Get summary of all positions"""
        open_positions = list(self.positions.values())

        return {
            "open_positions": len(open_positions),
            "closed_positions": len(self.closed_positions),
            "total_pnl": self.get_total_pnl(),
            "open_pnl": sum(pos.pnl for pos in open_positions),
            "closed_pnl": sum(pos.pnl for pos in self.closed_positions),
            "winning_trades": len([p for p in self.closed_positions if p.pnl > 0]),
            "losing_trades": len([p for p in self.closed_positions if p.pnl < 0]),
            "win_rate": (
                len([p for p in self.closed_positions if p.pnl > 0])
                / len(self.closed_positions)
                * 100
                if self.closed_positions
                else 0
            ),
            "average_win": (
                sum(p.pnl for p in self.closed_positions if p.pnl > 0)
                / len([p for p in self.closed_positions if p.pnl > 0])
                if [p for p in self.closed_positions if p.pnl > 0]
                else 0
            ),
            "average_loss": (
                sum(p.pnl for p in self.closed_positions if p.pnl < 0)
                / len([p for p in self.closed_positions if p.pnl < 0])
                if [p for p in self.closed_positions if p.pnl < 0]
                else 0
            ),
        }

    def close_position_manually(self, order_id: str) -> bool:
        """Manually close a position"""
        if order_id in self.positions:
            position = self.positions[order_id]
            self._close_position(position)
            return True
        return False

    def update_stop_loss(self, order_id: str, new_stop_loss: float) -> bool:
        """Update stop loss for a position"""
        if order_id in self.positions:
            self.positions[order_id].stop_loss = new_stop_loss
            logger.info(
                f"Updated stop loss for {self.positions[order_id].symbol} to {new_stop_loss}"
            )
            return True
        return False

    def update_target(self, order_id: str, new_target: float) -> bool:
        """Update target for a position"""
        if order_id in self.positions:
            self.positions[order_id].target = new_target
            logger.info(
                f"Updated target for {self.positions[order_id].symbol} to {new_target}"
            )
            return True
        return False
