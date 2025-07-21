from crewai import Agent
from typing import Dict, Any, List, Optional
from datetime import datetime, time
import logging
from dataclasses import dataclass

from .order_manager import OrderManager
from .position_monitor import PositionMonitor
from .paper_trading_manager import PaperTradingManager
from .trade_logger import TradeLogger
from .order_timing_optimizer import OrderTimingOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trade signal with all necessary information"""
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET, LIMIT, SL, SL-M
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    confidence: float = 0.0
    strategy_name: str = ""
    metadata: Dict[str, Any] = None


class TradeExecutorAgent(Agent):
    """Agent responsible for executing trades with optimal timing and handling"""
    
    def __init__(self, kite_client=None, paper_trading=False):
        super().__init__(
            name="Trade Executor",
            role="Execute trades efficiently with proper risk management",
            goal="Execute trades at optimal prices while managing orders and positions",
            backstory="""You are an expert trade execution specialist with deep knowledge
            of order types, market microstructure, and execution algorithms. You ensure
            trades are executed efficiently while minimizing slippage and managing risk.""",
            verbose=True,
            allow_delegation=False
        )
        
        self.kite_client = kite_client
        self.paper_trading = paper_trading
        
        # Initialize components
        self.order_manager = OrderManager(kite_client, paper_trading)
        self.position_monitor = PositionMonitor(kite_client, paper_trading)
        self.paper_trading_manager = PaperTradingManager() if paper_trading else None
        self.trade_logger = TradeLogger()
        self.timing_optimizer = OrderTimingOptimizer()
        
        # Trading session times (NSE)
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
        self.intraday_square_off = time(15, 15)  # Square off 15 mins before close
        
    def execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade based on the provided signal"""
        try:
            # Log the incoming signal
            self.trade_logger.log_signal(signal)
            
            # Check if market is open
            if not self._is_market_open():
                logger.warning(f"Market is closed. Cannot execute trade for {signal.symbol}")
                return {"status": "rejected", "reason": "market_closed"}
            
            # Check paper trading mode
            if self.paper_trading:
                result = self.paper_trading_manager.execute_trade(signal)
                self.trade_logger.log_trade(result)
                return result
            
            # Optimize order timing
            timing_params = self.timing_optimizer.get_optimal_timing(
                signal.symbol,
                signal.action,
                signal.quantity
            )
            
            # Execute based on order type
            if signal.order_type == "BRACKET":
                result = self._execute_bracket_order(signal, timing_params)
            elif signal.order_type == "LIMIT":
                result = self._execute_limit_order(signal, timing_params)
            else:  # MARKET
                result = self._execute_market_order(signal, timing_params)
            
            # Log the execution result
            self.trade_logger.log_trade(result)
            
            # Start monitoring the position if order was successful
            if result.get("status") == "success":
                self.position_monitor.add_position(result["order_id"], signal)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _execute_market_order(self, signal: TradeSignal, timing_params: Dict) -> Dict:
        """Execute a market order"""
        return self.order_manager.place_market_order(
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            **timing_params
        )
    
    def _execute_limit_order(self, signal: TradeSignal, timing_params: Dict) -> Dict:
        """Execute a limit order with smart pricing"""
        # Get optimal limit price
        limit_price = signal.price
        if not limit_price:
            limit_price = self._calculate_limit_price(signal.symbol, signal.action)
        
        return self.order_manager.place_limit_order(
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            price=limit_price,
            **timing_params
        )
    
    def _execute_bracket_order(self, signal: TradeSignal, timing_params: Dict) -> Dict:
        """Execute a bracket order with stop loss and target"""
        return self.order_manager.place_bracket_order(
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            price=signal.price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            **timing_params
        )
    
    def monitor_positions(self) -> List[Dict]:
        """Monitor all open positions and handle exits"""
        positions = self.position_monitor.get_open_positions()
        
        for position in positions:
            # Check exit conditions
            exit_signal = self._check_exit_conditions(position)
            
            if exit_signal:
                # Execute exit trade
                exit_result = self.execute_trade(exit_signal)
                logger.info(f"Exit trade executed for {position['symbol']}: {exit_result}")
        
        return positions
    
    def _check_exit_conditions(self, position: Dict) -> Optional[TradeSignal]:
        """Check if position should be exited"""
        current_time = datetime.now().time()
        
        # Force square off near market close for intraday
        if current_time >= self.intraday_square_off:
            return TradeSignal(
                symbol=position["symbol"],
                action="SELL" if position["side"] == "BUY" else "BUY",
                quantity=position["quantity"],
                order_type="MARKET",
                strategy_name="intraday_square_off"
            )
        
        # Check stop loss hit
        if self._is_stop_loss_hit(position):
            return TradeSignal(
                symbol=position["symbol"],
                action="SELL" if position["side"] == "BUY" else "BUY",
                quantity=position["quantity"],
                order_type="MARKET",
                strategy_name="stop_loss_hit"
            )
        
        # Check target hit
        if self._is_target_hit(position):
            return TradeSignal(
                symbol=position["symbol"],
                action="SELL" if position["side"] == "BUY" else "BUY",
                quantity=position["quantity"],
                order_type="MARKET",
                strategy_name="target_hit"
            )
        
        return None
    
    def _is_stop_loss_hit(self, position: Dict) -> bool:
        """Check if stop loss is hit for a position"""
        if not position.get("stop_loss"):
            return False
        
        current_price = self._get_current_price(position["symbol"])
        
        if position["side"] == "BUY":
            return current_price <= position["stop_loss"]
        else:
            return current_price >= position["stop_loss"]
    
    def _is_target_hit(self, position: Dict) -> bool:
        """Check if target is hit for a position"""
        if not position.get("target"):
            return False
        
        current_price = self._get_current_price(position["symbol"])
        
        if position["side"] == "BUY":
            return current_price >= position["target"]
        else:
            return current_price <= position["target"]
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        if self.paper_trading:
            return self.paper_trading_manager.get_price(symbol)
        
        try:
            quote = self.kite_client.quote([symbol])
            return quote[symbol]["last_price"]
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return 0.0
    
    def _calculate_limit_price(self, symbol: str, action: str) -> float:
        """Calculate optimal limit price based on order book"""
        try:
            quote = self.kite_client.quote([symbol])
            bid = quote[symbol]["depth"]["buy"][0]["price"]
            ask = quote[symbol]["depth"]["sell"][0]["price"]
            
            # Place limit order slightly better than best bid/ask
            if action == "BUY":
                return bid + 0.05  # 5 paise above best bid
            else:
                return ask - 0.05  # 5 paise below best ask
                
        except Exception as e:
            logger.error(f"Error calculating limit price: {str(e)}")
            return 0.0
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        current_time = datetime.now().time()
        
        # Check if it's a weekday
        if datetime.now().weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check market hours
        return self.market_open <= current_time <= self.market_close
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions for the day"""
        return {
            "total_trades": self.trade_logger.get_trade_count(),
            "successful_trades": self.trade_logger.get_successful_trades(),
            "failed_trades": self.trade_logger.get_failed_trades(),
            "open_positions": len(self.position_monitor.get_open_positions()),
            "pnl": self.position_monitor.get_total_pnl(),
            "execution_stats": self.order_manager.get_execution_stats()
        }
    
    def close_all_positions(self) -> List[Dict]:
        """Close all open positions (emergency or end of day)"""
        positions = self.position_monitor.get_open_positions()
        results = []
        
        for position in positions:
            exit_signal = TradeSignal(
                symbol=position["symbol"],
                action="SELL" if position["side"] == "BUY" else "BUY",
                quantity=position["quantity"],
                order_type="MARKET",
                strategy_name="close_all_positions"
            )
            
            result = self.execute_trade(exit_signal)
            results.append(result)
        
        return results