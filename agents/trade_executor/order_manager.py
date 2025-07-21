import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import time
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Supported order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"  # Stop Loss
    SL_M = "SL-M"  # Stop Loss Market


class OrderStatus(Enum):
    """Order status types"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class OrderManager:
    """Manages order placement, modifications, and confirmations"""
    
    def __init__(self, kite_client=None, paper_trading=False):
        self.kite_client = kite_client
        self.paper_trading = paper_trading
        self.orders = {}  # Track all orders
        self.execution_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "rejected_orders": 0,
            "avg_execution_time": 0,
            "slippage_stats": {}
        }
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
    def place_market_order(self, symbol: str, action: str, quantity: int, **kwargs) -> Dict:
        """Place a market order with retry logic"""
        order_params = {
            "exchange": "NSE",
            "tradingsymbol": symbol,
            "transaction_type": action.upper(),
            "quantity": quantity,
            "order_type": OrderType.MARKET.value,
            "product": "MIS",  # Intraday
            "validity": "DAY"
        }
        
        return self._place_order_with_retry(order_params)
    
    def place_limit_order(self, symbol: str, action: str, quantity: int, 
                         price: float, **kwargs) -> Dict:
        """Place a limit order with smart pricing"""
        order_params = {
            "exchange": "NSE",
            "tradingsymbol": symbol,
            "transaction_type": action.upper(),
            "quantity": quantity,
            "order_type": OrderType.LIMIT.value,
            "price": price,
            "product": "MIS",
            "validity": "DAY"
        }
        
        return self._place_order_with_retry(order_params)
    
    def place_bracket_order(self, symbol: str, action: str, quantity: int,
                           price: float, stop_loss: float, target: float, **kwargs) -> Dict:
        """Place a bracket order with stop loss and target"""
        # Calculate trailing stop loss value
        trailing_sl = abs(price - stop_loss)
        
        order_params = {
            "exchange": "NSE",
            "tradingsymbol": symbol,
            "transaction_type": action.upper(),
            "quantity": quantity,
            "order_type": OrderType.LIMIT.value,
            "price": price,
            "product": "MIS",
            "variety": "bo",
            "squareoff": abs(target - price),
            "stoploss": trailing_sl,
            "trailing_stoploss": trailing_sl * 0.5,  # 50% of stop loss as trailing
            "validity": "DAY"
        }
        
        return self._place_order_with_retry(order_params)
    
    def place_stop_loss_order(self, symbol: str, action: str, quantity: int,
                             trigger_price: float, price: Optional[float] = None) -> Dict:
        """Place a stop loss order"""
        order_params = {
            "exchange": "NSE",
            "tradingsymbol": symbol,
            "transaction_type": action.upper(),
            "quantity": quantity,
            "order_type": OrderType.SL.value if price else OrderType.SL_M.value,
            "trigger_price": trigger_price,
            "product": "MIS",
            "validity": "DAY"
        }
        
        if price:
            order_params["price"] = price
        
        return self._place_order_with_retry(order_params)
    
    def _place_order_with_retry(self, order_params: Dict) -> Dict:
        """Place order with retry logic for handling failures"""
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                if self.paper_trading:
                    # Simulate order placement
                    order_id = f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    result = {
                        "order_id": order_id,
                        "status": "success",
                        "order_params": order_params,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Place actual order
                    order_id = self.kite_client.place_order(**order_params)
                    result = {
                        "order_id": order_id,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Update execution stats
                execution_time = time.time() - start_time
                self._update_execution_stats(True, execution_time)
                
                # Track order
                self.orders[order_id] = {
                    "params": order_params,
                    "status": OrderStatus.PENDING.value,
                    "result": result,
                    "attempts": attempt + 1
                }
                
                # Confirm order placement
                confirmation = self._confirm_order_placement(order_id)
                result["confirmation"] = confirmation
                
                return result
                
            except Exception as e:
                logger.error(f"Order placement attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    # Final attempt failed
                    self._update_execution_stats(False, time.time() - start_time)
                    return {
                        "status": "failed",
                        "error": str(e),
                        "attempts": attempt + 1,
                        "order_params": order_params
                    }
    
    def _confirm_order_placement(self, order_id: str) -> Dict:
        """Confirm order was placed successfully"""
        try:
            if self.paper_trading:
                return {
                    "confirmed": True,
                    "status": OrderStatus.COMPLETE.value,
                    "fill_price": self.orders[order_id]["params"].get("price", 100)
                }
            
            # Check order status
            order_info = self.kite_client.order_history(order_id)
            
            if order_info:
                latest_status = order_info[-1]
                return {
                    "confirmed": True,
                    "status": latest_status["status"],
                    "fill_price": latest_status.get("average_price"),
                    "filled_quantity": latest_status.get("filled_quantity"),
                    "pending_quantity": latest_status.get("pending_quantity")
                }
            
        except Exception as e:
            logger.error(f"Error confirming order {order_id}: {str(e)}")
        
        return {"confirmed": False, "error": "Could not confirm order"}
    
    def modify_order(self, order_id: str, price: Optional[float] = None,
                    quantity: Optional[int] = None, trigger_price: Optional[float] = None) -> Dict:
        """Modify an existing order"""
        try:
            params = {}
            if price is not None:
                params["price"] = price
            if quantity is not None:
                params["quantity"] = quantity
            if trigger_price is not None:
                params["trigger_price"] = trigger_price
            
            if self.paper_trading:
                return {"status": "success", "order_id": order_id, "modifications": params}
            
            self.kite_client.modify_order(variety="regular", order_id=order_id, **params)
            
            return {"status": "success", "order_id": order_id, "modifications": params}
            
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an existing order"""
        try:
            if self.paper_trading:
                if order_id in self.orders:
                    self.orders[order_id]["status"] = OrderStatus.CANCELLED.value
                return {"status": "success", "order_id": order_id}
            
            self.kite_client.cancel_order(variety="regular", order_id=order_id)
            
            if order_id in self.orders:
                self.orders[order_id]["status"] = OrderStatus.CANCELLED.value
            
            return {"status": "success", "order_id": order_id}
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get current status of an order"""
        try:
            if self.paper_trading:
                return self.orders.get(order_id, {"status": "not_found"})
            
            order_info = self.kite_client.order_history(order_id)
            
            if order_info:
                latest = order_info[-1]
                return {
                    "order_id": order_id,
                    "status": latest["status"],
                    "filled_quantity": latest.get("filled_quantity", 0),
                    "pending_quantity": latest.get("pending_quantity", 0),
                    "average_price": latest.get("average_price", 0)
                }
            
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
        
        return {"status": "error", "order_id": order_id}
    
    def get_pending_orders(self) -> List[Dict]:
        """Get all pending orders"""
        try:
            if self.paper_trading:
                return [
                    order for order_id, order in self.orders.items()
                    if order["status"] == OrderStatus.PENDING.value
                ]
            
            orders = self.kite_client.orders()
            return [
                order for order in orders
                if order["status"] in ["PENDING", "OPEN"]
            ]
            
        except Exception as e:
            logger.error(f"Error getting pending orders: {str(e)}")
            return []
    
    def _update_execution_stats(self, success: bool, execution_time: float):
        """Update execution statistics"""
        self.execution_stats["total_orders"] += 1
        
        if success:
            self.execution_stats["successful_orders"] += 1
        else:
            self.execution_stats["failed_orders"] += 1
        
        # Update average execution time
        total = self.execution_stats["total_orders"]
        current_avg = self.execution_stats["avg_execution_time"]
        self.execution_stats["avg_execution_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
    
    def calculate_slippage(self, order_id: str, expected_price: float) -> float:
        """Calculate slippage for an executed order"""
        try:
            status = self.get_order_status(order_id)
            
            if status.get("average_price"):
                actual_price = status["average_price"]
                slippage = abs(actual_price - expected_price)
                slippage_pct = (slippage / expected_price) * 100
                
                # Track slippage stats
                symbol = self.orders[order_id]["params"]["tradingsymbol"]
                if symbol not in self.execution_stats["slippage_stats"]:
                    self.execution_stats["slippage_stats"][symbol] = []
                
                self.execution_stats["slippage_stats"][symbol].append(slippage_pct)
                
                return slippage_pct
                
        except Exception as e:
            logger.error(f"Error calculating slippage: {str(e)}")
        
        return 0.0
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        stats = self.execution_stats.copy()
        
        # Calculate average slippage per symbol
        avg_slippage = {}
        for symbol, slippages in stats["slippage_stats"].items():
            if slippages:
                avg_slippage[symbol] = sum(slippages) / len(slippages)
        
        stats["average_slippage"] = avg_slippage
        stats["success_rate"] = (
            stats["successful_orders"] / stats["total_orders"] * 100
            if stats["total_orders"] > 0 else 0
        )
        
        return stats