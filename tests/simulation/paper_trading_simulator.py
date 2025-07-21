import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
import json
from pathlib import Path
from loguru import logger


@dataclass
class PaperPosition:
    """Represents a paper trading position"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    position_type: str  # "long" or "short"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    status: str = "open"  # "open", "closed"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    fees: float = 0
    
    def update_price(self, current_price: float):
        """Update position with current price"""
        self.current_price = current_price
        if self.position_type == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity


@dataclass
class PaperOrder:
    """Represents a paper trading order"""
    order_id: str = field(default_factory=lambda: f"PAPER_{uuid.uuid4().hex[:8]}")
    symbol: str = ""
    order_type: str = ""  # "BUY", "SELL"
    quantity: int = 0
    price_type: str = "MARKET"  # "MARKET", "LIMIT"
    price: Optional[float] = None
    status: str = "PENDING"  # "PENDING", "FILLED", "CANCELLED", "REJECTED"
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None
    created_time: datetime = field(default_factory=datetime.now)
    

class PaperTradingSimulator:
    """Simulates paper trading with realistic execution and portfolio management"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.current_prices: Dict[str, float] = {}
        
        # Trading parameters
        self.commission_rate = 0.0003  # 0.03% per trade
        self.slippage_factor = 0.0005  # 0.05% slippage
        self.max_position_size = 0.2  # Max 20% per position
        self.margin_requirement = 0.2  # 20% margin for intraday
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = initial_capital
        
    async def place_order(self, order: Dict[str, Any]) -> PaperOrder:
        """Place a paper trading order"""
        paper_order = PaperOrder(
            symbol=order["symbol"],
            order_type=order["order_type"],
            quantity=order["quantity"],
            price_type=order.get("price_type", "MARKET"),
            price=order.get("price")
        )
        
        # Validate order
        validation_result = self._validate_order(paper_order)
        if not validation_result["valid"]:
            paper_order.status = "REJECTED"
            logger.warning(f"Order rejected: {validation_result['reason']}")
            return paper_order
            
        # Store order
        self.orders[paper_order.order_id] = paper_order
        
        # Execute order based on type
        if paper_order.price_type == "MARKET":
            await self._execute_market_order(paper_order)
        else:
            await self._execute_limit_order(paper_order)
            
        return paper_order
        
    async def _execute_market_order(self, order: PaperOrder):
        """Execute a market order immediately"""
        current_price = self.current_prices.get(order.symbol, 2500)
        
        # Apply slippage
        if order.order_type == "BUY":
            execution_price = current_price * (1 + self.slippage_factor)
        else:
            execution_price = current_price * (1 - self.slippage_factor)
            
        # Calculate commission
        commission = abs(execution_price * order.quantity * self.commission_rate)
        self.total_commission += commission
        
        # Update order
        order.filled_price = execution_price
        order.filled_time = datetime.now()
        order.status = "FILLED"
        
        # Update positions
        if order.order_type == "BUY":
            await self._open_position(order, commission)
        else:
            await self._close_position(order, commission)
            
        # Record trade
        self._record_trade(order, commission)
        
    async def _execute_limit_order(self, order: PaperOrder):
        """Execute a limit order when price conditions are met"""
        # For simulation, check if current price allows execution
        current_price = self.current_prices.get(order.symbol, 2500)
        
        can_execute = False
        if order.order_type == "BUY" and current_price <= order.price:
            can_execute = True
        elif order.order_type == "SELL" and current_price >= order.price:
            can_execute = True
            
        if can_execute:
            # Execute at limit price
            commission = abs(order.price * order.quantity * self.commission_rate)
            self.total_commission += commission
            
            order.filled_price = order.price
            order.filled_time = datetime.now()
            order.status = "FILLED"
            
            if order.order_type == "BUY":
                await self._open_position(order, commission)
            else:
                await self._close_position(order, commission)
                
            self._record_trade(order, commission)
        else:
            order.status = "PENDING"
            
    async def _open_position(self, order: PaperOrder, commission: float):
        """Open a new position"""
        position = PaperPosition(
            symbol=order.symbol,
            quantity=order.quantity,
            entry_price=order.filled_price,
            entry_time=order.filled_time,
            position_type="long" if order.order_type == "BUY" else "short",
            fees=commission
        )
        
        # Update available capital
        position_cost = order.filled_price * order.quantity + commission
        if self.margin_requirement < 1:
            # Intraday with margin
            required_margin = position_cost * self.margin_requirement
        else:
            required_margin = position_cost
            
        self.available_capital -= required_margin
        
        # Store position
        self.positions[order.symbol] = position
        logger.info(f"Opened position: {order.symbol} @ {order.filled_price}")
        
    async def _close_position(self, order: PaperOrder, commission: float):
        """Close an existing position"""
        if order.symbol not in self.positions:
            logger.warning(f"No position found for {order.symbol}")
            return
            
        position = self.positions[order.symbol]
        
        # Calculate P&L
        if position.position_type == "long":
            pnl = (order.filled_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - order.filled_price) * position.quantity
            
        # Subtract commissions
        total_commission = position.fees + commission
        realized_pnl = pnl - total_commission
        
        # Update position
        position.exit_price = order.filled_price
        position.exit_time = order.filled_time
        position.realized_pnl = realized_pnl
        position.status = "closed"
        
        # Update metrics
        self.total_trades += 1
        if realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        # Return capital
        if self.margin_requirement < 1:
            returned_margin = position.entry_price * position.quantity * self.margin_requirement
        else:
            returned_margin = position.entry_price * position.quantity
            
        self.available_capital += returned_margin + realized_pnl
        
        # Remove from active positions
        del self.positions[order.symbol]
        
        logger.info(f"Closed position: {order.symbol} P&L: {realized_pnl:.2f}")
        
    def _validate_order(self, order: PaperOrder) -> Dict[str, Any]:
        """Validate order before execution"""
        # Check if symbol has price data
        if order.symbol not in self.current_prices:
            return {"valid": False, "reason": "No price data for symbol"}
            
        current_price = self.current_prices[order.symbol]
        order_value = current_price * order.quantity
        
        # Check position size limit
        if order_value > self.initial_capital * self.max_position_size:
            return {"valid": False, "reason": "Position size exceeds limit"}
            
        # Check available capital
        required_margin = order_value * self.margin_requirement
        if required_margin > self.available_capital:
            return {"valid": False, "reason": "Insufficient capital"}
            
        # Check if closing position exists
        if order.order_type == "SELL" and order.symbol not in self.positions:
            return {"valid": False, "reason": "No position to close"}
            
        return {"valid": True}
        
    def _record_trade(self, order: PaperOrder, commission: float):
        """Record trade in history"""
        trade = {
            "order_id": order.order_id,
            "timestamp": order.filled_time,
            "symbol": order.symbol,
            "order_type": order.order_type,
            "quantity": order.quantity,
            "price": order.filled_price,
            "commission": commission,
            "portfolio_value": self.get_portfolio_value()
        }
        
        self.trade_history.append(trade)
        
    async def update_prices(self, price_updates: Dict[str, float]):
        """Update current market prices"""
        self.current_prices.update(price_updates)
        
        # Update position values
        for symbol, position in self.positions.items():
            if symbol in price_updates:
                position.update_price(price_updates[symbol])
                
                # Check stop loss and take profit
                await self._check_exit_conditions(position)
                
        # Update portfolio metrics
        self._update_portfolio_metrics()
        
    async def _check_exit_conditions(self, position: PaperPosition):
        """Check if position should be exited based on stop loss or take profit"""
        if position.status != "open":
            return
            
        should_exit = False
        exit_reason = ""
        
        # Check stop loss
        if position.stop_loss:
            if position.position_type == "long" and position.current_price <= position.stop_loss:
                should_exit = True
                exit_reason = "Stop loss hit"
            elif position.position_type == "short" and position.current_price >= position.stop_loss:
                should_exit = True
                exit_reason = "Stop loss hit"
                
        # Check take profit
        if position.take_profit and not should_exit:
            if position.position_type == "long" and position.current_price >= position.take_profit:
                should_exit = True
                exit_reason = "Take profit hit"
            elif position.position_type == "short" and position.current_price <= position.take_profit:
                should_exit = True
                exit_reason = "Take profit hit"
                
        if should_exit:
            # Create exit order
            exit_order = {
                "symbol": position.symbol,
                "order_type": "SELL" if position.position_type == "long" else "BUY",
                "quantity": position.quantity,
                "price_type": "MARKET"
            }
            
            logger.info(f"Auto-exit triggered for {position.symbol}: {exit_reason}")
            await self.place_order(exit_order)
            
    def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        current_value = self.get_portfolio_value()
        
        # Update peak value
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
            
        # Calculate drawdown
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
        # Record portfolio snapshot
        snapshot = {
            "timestamp": datetime.now(),
            "total_value": current_value,
            "available_capital": self.available_capital,
            "positions_value": sum(p.current_price * p.quantity for p in self.positions.values()),
            "unrealized_pnl": sum(p.unrealized_pnl for p in self.positions.values()),
            "realized_pnl": current_value - self.initial_capital,
            "open_positions": len(self.positions),
            "total_trades": self.total_trades
        }
        
        self.portfolio_history.append(snapshot)
        
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            position.current_price * position.quantity 
            for position in self.positions.values()
        )
        
        return self.available_capital + positions_value
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        total_pnl = self.get_portfolio_value() - self.initial_capital
        total_return = (total_pnl / self.initial_capital) * 100
        
        win_rate = 0
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            
        # Calculate Sharpe ratio from portfolio history
        if len(self.portfolio_history) > 1:
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_value = self.portfolio_history[i-1]["total_value"]
                curr_value = self.portfolio_history[i]["total_value"]
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
                    
            if returns:
                import numpy as np
                sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0
            
        return {
            "total_pnl": total_pnl,
            "total_return_percent": total_return,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "max_drawdown_percent": self.max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "total_commission": self.total_commission,
            "current_portfolio_value": self.get_portfolio_value(),
            "open_positions": len(self.positions)
        }
        
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get complete trade history"""
        return self.trade_history
        
    def get_open_positions(self) -> Dict[str, PaperPosition]:
        """Get all open positions"""
        return self.positions.copy()
        
    def save_results(self, filepath: Path):
        """Save simulation results to file"""
        results = {
            "performance_metrics": self.get_performance_metrics(),
            "trade_history": self.trade_history,
            "portfolio_history": self.portfolio_history,
            "final_positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl
                }
                for symbol, pos in self.positions.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Simulation results saved to {filepath}")
        
    def reset(self):
        """Reset simulator to initial state"""
        self.available_capital = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trade_history.clear()
        self.portfolio_history.clear()
        self.current_prices.clear()
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = self.initial_capital